import os
import gc
import torch
import json
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    pipeline
)
from datasets import load_dataset
from peft import LoraConfig, PeftModel

from run_baseline import clean_text, mistral_setup, ministral_setup, llama_setup
from run_case_questions import label_flipped_answers, label_answers, load_jsonl


from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import JSONLoader
from langchain_core.vectorstores import InMemoryVectorStore

from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_community.document_loaders.json_loader import JSONLoader


os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

CHUNK_LEN = 2200

def length_function(text: str, tokenizer) -> int:
    return len(tokenizer(text)["input_ids"])

def metadata_func(record: dict, metadata: dict) -> dict:
    metadata["index"] = record.get("Index")
    return metadata

def filter_by_index(document, index):
    if document.metadata["index"] == index: 
        return True
    return False

def setup_vector_store(filepath, tokenizer):

    loader = JSONLoader(
        file_path=filepath,
        jq_schema='.',
        content_key='Context',
        json_lines=True,
        metadata_func=metadata_func
    )
    documents = loader.load()
    print(f'document count: {len(documents)}')
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_LEN,
        chunk_overlap=200,
        length_function=lambda text: length_function(text, tokenizer=tokenizer),
        is_separator_regex=False,
        strip_whitespace=True,
    )
    chunked_documents = text_splitter.split_documents(documents)
    print(f'chunked document count: {len(chunked_documents)}')

    embeddings = HuggingFaceEmbeddings(model_name ='sentence-transformers/all-MiniLM-L6-v2')
    vector_store = InMemoryVectorStore(embedding=embeddings)
    vector_store.add_documents(documents=chunked_documents)
    return vector_store

def format_docs(docs):
    return "\n\n----------NEW DOC----------\n\n".join(doc.page_content for doc in docs)

if __name__ == "__main__":
    # from huggingface_hub import login
    # login()
    gc.collect()
    torch.cuda.empty_cache()
    filepath = './list_of_allegations_20250520.jsonl'
    all_jsonl = load_jsonl(filepath)
    print(all_jsonl)

    model, tokenizer = ministral_setup()
    pipe = pipeline("text-generation", model=model, max_new_tokens=1200, torch_dtype=torch.bfloat16, device_map='cuda', tokenizer=tokenizer)
    pipe.model = pipe.model.to('cuda')

    bad_indices = set(["253-FORTUNE II", "420-Quinn", "527-Brown II", "993-SPAULDING", "675-Freed", "371-Myers", "284-Armengau", "287-Mack", "052-D'Ambrosio", "925-The State ex rel. The Cincinnati Enquirer, A Division of Gannett GP Media, Inc.,"])

    vector_store = setup_vector_store("../cases_olmocr/all.jsonl", tokenizer)
    for i, row in all_jsonl.iterrows():
        key = row['index']
        allegations = row['allegations']
        allegations_obj = allegations.replace("```", "").replace("json", "")
        if key in bad_indices:
            continue
        allegations_dict = json.loads(allegations_obj)
        for allegation_num, allegation in allegations_dict.items():
            docs = vector_store.similarity_search(allegation, k=5,filter=lambda doc: filter_by_index(doc, index=key))
            retrieved_docs = format_docs(docs)
            with open("allegations_evidence_20250220.jsonl", "a") as f:
                json_record = json.dumps({"index": key, "allegation": allegation, "evidence": retrieved_docs})
                f.write(json_record + "\n")
        if i == 10:
            break
        gc.collect()
        torch.cuda.empty_cache()