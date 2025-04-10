import pandas as pd
import torch
from transformers import pipeline
import json
import gc
import re
from run_baseline import clean_text, mistral_setup, ministral_setup, llama_setup
from run_questions import label_flipped_answers, label_answers, load_jsonl
import math
import os
from typing import List, Tuple, Optional

CHUNK_LENGTH = 2000


def apply_prompt(chunk_text, question):
    full_prompt = (
        f"Context information is below.\n---------------------\n{chunk_text}\n---------------------\nGiven the context information and not prior knowledge, answer the query.\nQuery: {question}\nAnswer:"
    )

    return full_prompt



# Global tokenizer variable that will be set in identify_allegations
tokenizer = None


def tokenize(text):
    """Tokenize text using the global tokenizer"""
    if tokenizer is None:
        raise ValueError("Tokenizer not initialized. Call identify_allegations first.")
    return tokenizer.encode(text, add_special_tokens=True)


def recursively_split_chunk(chunk: str, max_tokens: int, chunk_delimiter="\n\n") -> List[str]:
    """
    Recursively split a chunk that is too large until all pieces are below max_tokens.
    Splits by delimiter or sentences, but avoids character-level splitting.
    """
    # If the chunk is already small enough, return it as is
    if len(tokenize(chunk)) <= max_tokens:
        return [chunk]

    # Try to split by the delimiter first
    if chunk_delimiter in chunk:
        sub_chunks = chunk.split(chunk_delimiter)
        # If after splitting, we still have chunks that are too large
        result = []
        for sub_chunk in sub_chunks:
            # Recursively split any large sub-chunks
            result.extend(recursively_split_chunk(sub_chunk, max_tokens, chunk_delimiter))
        return result

    # If there's no delimiter, try to split by sentences
    sentences = re.split(r'(?<=[.!?])\s+', chunk)
    if len(sentences) > 1:
        result = []
        current_chunk = ""
        for sentence in sentences:
            # If adding this sentence would make the chunk too large, start a new chunk
            if len(tokenize(current_chunk + sentence)) > max_tokens:
                if current_chunk:  # Don't add empty chunks
                    result.append(current_chunk)
                # If a single sentence is too large, we'll have to take it as is
                # but warn the user
                if len(tokenize(sentence)) > max_tokens:
                    print(
                        f"warning: sentence with {len(tokenize(sentence))} tokens exceeds max_tokens ({max_tokens}). Taking it as is.")
                    result.append(sentence)
                else:
                    current_chunk = sentence
            else:
                if current_chunk:
                    current_chunk += " " + sentence
                else:
                    current_chunk = sentence

        if current_chunk:  # Don't forget to add the last chunk
            result.append(current_chunk)
        return result

    # If we get here, it means we have a large chunk with no delimiters and no sentence breaks
    # Just return it as is and warn the user
    print(
        f"warning: chunk with {len(tokenize(chunk))} tokens exceeds max_tokens ({max_tokens}) and has no natural splits. Taking it as is.")
    return [chunk]


def chunk_on_delimiter(input_string: str, max_tokens: int, delimiter: str) -> List[str]:
    """
    Split input_string on delimiter and combine chunks to maximize token usage.
    Never drops chunks - instead splits large chunks at natural boundaries.
    """
    chunks = input_string.split(delimiter)
    processed_chunks = []

    for chunk in chunks:
        # Check if this chunk alone is too large
        if len(tokenize(chunk)) > max_tokens:
            # If too large, recursively split it
            split_chunks = recursively_split_chunk(chunk, max_tokens, delimiter)
            processed_chunks.extend(split_chunks)
        else:
            processed_chunks.append(chunk)

    # Now combine small chunks when possible
    combined_chunks = []
    current_combined = ""

    for chunk in processed_chunks:
        # If adding this chunk would exceed max_tokens, start a new combined chunk
        if len(tokenize(current_combined + delimiter + chunk if current_combined else chunk)) > max_tokens:
            if current_combined:  # Don't add empty chunks
                combined_chunks.append(current_combined)
            current_combined = chunk
        else:
            if current_combined:
                current_combined += delimiter + chunk
            else:
                current_combined = chunk

    # Don't forget to add the last combined chunk
    if current_combined:
        combined_chunks.append(current_combined)

    # Add the delimiter at the end of each chunk as requested in the original function
    combined_chunks = [f"{chunk}{delimiter}" for chunk in combined_chunks]

    return combined_chunks


def combine_chunks_with_no_minimum(
        chunks: List[str],
        max_tokens: int,
        chunk_delimiter="\n\n",
        header: Optional[str] = None,
        add_ellipsis_for_overflow=False,
) -> Tuple[List[str], List[int], int]:
    """Combine chunks into larger chunks that don't exceed max_tokens."""
    dropped_chunk_count = 0  # This will always be 0 with our approach
    output = []  # list to hold the final combined chunks
    output_indices = []  # list to hold the indices of the final combined chunks
    candidate = (
        [] if header is None else [header]
    )  # list to hold the current combined chunk candidate
    candidate_indices = []

    for chunk_i, chunk in enumerate(chunks):
        chunk_with_header = [chunk] if header is None else [header, chunk]

        # Handle the case where a single chunk is too large
        if len(tokenize(chunk_delimiter.join(chunk_with_header))) > max_tokens:
            # Instead of dropping, try to split the chunk
            if header is None:
                split_chunks = recursively_split_chunk(chunk, max_tokens, chunk_delimiter)
                for split_chunk in split_chunks:
                    output.append(split_chunk)
                    output_indices.append([chunk_i])  # Associate with original chunk index
            else:
                # If we have a header, we need to ensure it fits with at least some content
                header_tokens = len(tokenize(header))
                if header_tokens >= max_tokens:
                    print(f"warning: header is too large ({header_tokens} tokens)")
                    # Include the header alone and warn
                    output.append(header)
                    output_indices.append([])

                    # Then process the chunk without the header
                    split_chunks = recursively_split_chunk(chunk, max_tokens, chunk_delimiter)
                    for split_chunk in split_chunks:
                        output.append(split_chunk)
                        output_indices.append([chunk_i])
                else:
                    # Split the chunk to fit with the header
                    remaining_tokens = max_tokens - header_tokens - len(tokenize(chunk_delimiter))
                    if remaining_tokens <= 0:
                        # Just include the header
                        output.append(header)
                        output_indices.append([])

                        # Process chunk separately
                        split_chunks = recursively_split_chunk(chunk, max_tokens, chunk_delimiter)
                        for split_chunk in split_chunks:
                            output.append(split_chunk)
                            output_indices.append([chunk_i])
                    else:
                        # We can fit some content with the header
                        split_chunks = recursively_split_chunk(chunk, remaining_tokens, chunk_delimiter)
                        output.append(chunk_delimiter.join([header, split_chunks[0]]))
                        output_indices.append([chunk_i])

                        # Process remaining split chunks if any
                        for split_chunk in split_chunks[1:]:
                            output.append(split_chunk)
                            output_indices.append([chunk_i])
            continue

        # Normal case - try to add to current candidate
        extended_candidate_token_count = len(tokenize(chunk_delimiter.join(candidate + [chunk])))
        if extended_candidate_token_count > max_tokens:
            # Current candidate is full, start a new one
            output.append(chunk_delimiter.join(candidate))
            output_indices.append(candidate_indices)
            candidate = chunk_with_header  # re-initialize candidate
            candidate_indices = [chunk_i]
        else:
            # Add to current candidate
            candidate.append(chunk)
            candidate_indices.append(chunk_i)

    # Add the remaining candidate to output if it's not empty
    if (header is not None and len(candidate) > 1) or (header is None and len(candidate) > 0):
        output.append(chunk_delimiter.join(candidate))
        output_indices.append(candidate_indices)

    return output, output_indices, dropped_chunk_count


def identify_allegations(batch_size=4, question=None, label=None, max_tokens=500):
    """Process and identify allegations in text data using a language model."""
    gc.collect()
    torch.cuda.empty_cache()
    model, global_tokenizer = ministral_setup()

    # Update the global tokenizer
    global tokenizer
    tokenizer = global_tokenizer

    question_jsonl = load_jsonl("dnms_aoe_none_olmocr.jsonl")

    pipe = pipeline(
        "text-generation",
        model=model,
        torch_dtype=torch.bfloat16,
        device_map='cuda',
        tokenizer=tokenizer
    )

    results = []

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    test_len = 5

    for batch_start in range(0, test_len, batch_size):
        batch_end = min(batch_start + batch_size, test_len)
        batch = question_jsonl.iloc[batch_start:batch_end]

        for i, content in enumerate(batch.olmocr_text.values):
            # Use the improved chunking method
            chunked_content = chunk_on_delimiter(content, max_tokens, delimiter="\n\n")

            all_prompts = [apply_prompt(chunk, question) for chunk in chunked_content]
            all_messages = [[{"role": "user", "content": x}] for x in all_prompts]

            num_chunks = min(5, len(chunked_content))

            example_outputs = []
            for j in range(0, len(all_messages), num_chunks):
                start = j
                end = min(j + num_chunks, len(all_messages))
                batch_results = pipe(
                    all_messages[start:end],
                    max_new_tokens=300,
                    do_sample=False
                )
                example_outputs += batch_results

            for k, result in enumerate(example_outputs):
                results.append({
                    "Index": batch.Index.iloc[i],
                    "Response": result[0]["generated_text"][1]["content"],
                    "Text": result[0]['generated_text'][0]['content']
                })

            if batch_start % (batch_size * 5) == 0:
                gc.collect()
                torch.cuda.empty_cache()

            print(f"Processed up to sample {batch_end}")

    results_df = pd.DataFrame(results)
    results_df.to_csv(f"./ministral_olmocr_questions/{label}.csv", index=False)
    gc.collect()
    torch.cuda.empty_cache()
    if torch.cuda.is_available():
        torch.cuda.synchronize()


if __name__ == "__main__":
    # aoe_none_question = "In the text above, is there any mention of prosecutorial misconduct, misconduct by the prosecutor or misconduct by the state? Answer with only a 'Yes' or 'No'.  If you cannot determine the answer, provide your best yes or no guess."
    # identify_allegations(question=aoe_none_question, label = "aoe_none", label_func=label_flipped_answers)

    # aoe_procbar_question = "If there is an alleged assignment of error in the text above, was it procedurally barred? For example, is it barred by res judicata because it was not raised during original trial and now it’s too late? Answer with only a 'Yes' or 'No'.  If you cannot determine the answer, provide your best yes or no guess."
    # identify_allegations(question=aoe_procbar_question, label = "aoe_procbar", label_func=label_answers)

    # aoe_prochist_question = "Is the assignment of error in the procedural history, i.e., if there is prosecutorial misconduct mentioned, was it raised in a previous appeal? Answer with only a 'Yes' or 'No'.  If you cannot determine the answer, provide your best yes or no guess."
    # identify_allegations(question=aoe_prochist_question, label="aoe_prochist", label_func=label_answers)

    aoe_noneprocbar_question = "If the text above is describing an assignment of error, is the error alleging that the prosecutor (sometimes referred to as the state) committed misconduct? Respond 'Yes' or 'No'."
    identify_allegations(question=aoe_noneprocbar_question, label="aoe_test")