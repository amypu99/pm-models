import os
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

from langchain.text_splitter import CharacterTextSplitter


from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import JSONLoader
from langchain_core.vectorstores import InMemoryVectorStore

from langchain.prompts import PromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain_text_splitters import CharacterTextSplitter
from langchain_core.output_parsers import StrOutputParser
from run_baseline import clean_text, mistral_setup, ministral_setup, llama_setup
from run_questions import label_flipped_answers, label_answers, load_jsonl


os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

from langchain_community.document_loaders.json_loader import JSONLoader
from langchain_community.document_loaders import DirectoryLoader

from langchain.vectorstores import Chroma

def metadata_func(record: dict, metadata: dict) -> dict:
    metadata["index"] = record.get("metadata")["Source-File"].replace(".pdf", "")
    return metadata


def setup_vector_store(document_folder, glob):
    loader = DirectoryLoader(document_folder, glob=glob, show_progress=True, loader_cls=JSONLoader, loader_kwargs = {'jq_schema':'.', 'content_key': 'text', 'json_lines': True, 'metadata_func': metadata_func})
    documents = loader.load()
    print(f'document count: {len(documents)}')
    # print(documents[0])
    # print(len(documents[0].page_content))
    
    text_splitter = CharacterTextSplitter(separator="\n\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
        is_separator_regex=False,
    )
    chunked_documents = text_splitter.split_documents(documents)

    # # db = FAISS.from_documents(chunked_documents, HuggingFaceEmbeddings(model_name='BAAI/bge-base-en-v1.5'))
    # # retriever = db.as_retriever()

    embeddings = HuggingFaceEmbeddings(model_name ='sentence-transformers/all-MiniLM-L6-v2')
    vector_store = InMemoryVectorStore(embedding=embeddings)
    vector_store.add_documents(documents=chunked_documents)
    return vector_store


def apply_prompt(chunk_text, question):
    full_prompt = (
        f"Context information is below.\n---------------------\n{chunk_text}\n---------------------\nGiven the context information and not prior knowledge, answer the query.\nQuery: {question}\nAnswer:"
    )

    return full_prompt

prompt_template = """
Context information is below.\n---------------------\n{context}\n---------------------\nGiven the context information and not prior knowledge, answer the query.\nQuery: {question}\n"
"""

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def invoke_ragchain(llm, prompt_template, retriever, question):
    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template=prompt_template,
    )

    rag_chain = ( 
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    ans = rag_chain.invoke(question)
    return ans

def apply_prompt(chunk_text, question):
        full_prompt = (
            f"Context information is below.\n---------------------\n{chunk_text}\n---------------------\nGiven the context information and not prior knowledge, answer the query.\nQuery: {question}\nAnswer:"
        )

        return full_prompt


def query_model(model, tokenizer, query, docs):
    pipe = pipeline("text-generation", model=model, max_new_tokens=200, torch_dtype=torch.bfloat16, device_map='cuda', tokenizer=tokenizer)
    pipe.model = pipe.model.to('cuda')

    all_chunk_text = format_docs(docs)
    full_prompt = apply_prompt(all_chunk_text, query)
    messages = [{"role": "user", "content": full_prompt},]
            
    results = pipe(messages, max_new_tokens=256)
    return results

def filter(document, index):
    if document.metadata["index"] == index: 
        return True
    return False

if __name__ == "__main__":
    document_folder = "../cases_olmocr/DNMS/"
    glob = "dnms_olmocr.jsonl"
    vector_store = setup_vector_store(document_folder, glob=glob)

    retrieval_query = "Find and return all of mentions of assignments of errors in the case or any information related to any claims of miscondut or allegations of error."
    
    all_jsonl = load_jsonl(document_folder+glob)
    results = {}
    temp_results = {}

    model, tokenizer = ministral_setup()
    for i, key in enumerate([x['Source-File'].replace(".pdf", "") for x in all_jsonl.metadata]):
        docs = vector_store.similarity_search(retrieval_query, filter=lambda doc: filter(doc, index=key))
        query = "Name and list all the assignments of error claimed by the appellant/defendant. Please format your response as the following: Error 1: ...\n\nError 2:..."
        # print(query_model(query, docs)[0]['generated_text'][1]['content'])
        results[key] = query_model(model, tokenizer, query, docs)[0]['generated_text'][1]['content']
        temp_results[key] = query_model(model, tokenizer, query, docs)[0]['generated_text'][1]['content']
        if i % 20 == 0:
            print(i)
            with open("list_of_allegations_temp.jsonl", "a") as f:
                for index, allegations in temp_results.items():
                    json_record = json.dumps({"index": index, "allegations": allegations})
                    f.write(json_record + "\n")
            temp_results = {}

    
    with open("list_of_allegations.jsonl", "w") as f:
        for index, allegations in results.items():
            json_record = json.dumps({"index": index, "allegations": allegations})
            f.write(json_record + "\n")
