import os
import json
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_community.vectorstores import FAISS

# Using the new, dedicated package for Ollama
from langchain_ollama import OllamaLLM

from langchain_huggingface import HuggingFaceEmbeddings

# --- 1. SETUP ---
class StructuredCommand(BaseModel):
    command: str = Field(description="The primary command to be executed (e.g., 'create_calendar_event', 'send_email').")
    parameters: dict = Field(description="A dictionary of parameters for the command.")

# --- 2. DATA PREPARATION ---
with open('examples.json', 'r') as f:
    examples = json.load(f)
example_queries = [example['query'] for example in examples]

def format_examples(examples_list):
    formatted_examples = []
    for example in examples_list:
        query = example['query']
        command_str = json.dumps(example['command'], indent=2)
        formatted_examples.append(f"User Query: {query}\nStructured Command:\n{command_str}")
    return "\n\n---\n\n".join(formatted_examples)

# --- 3. CORE RAG PIPELINE COMPONENTS ---
print("Loading local embedding model...")
embedding_model = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"},
    encode_kwargs={'normalize_embeddings': True}
)
print("Embedding model loaded.")

# --- CORRECTED MODEL NAME ---
# Use the correct model name 'llama3', which defaults to the 'latest' tag you have.
llm = OllamaLLM(
    model="llama3", 
    format="json",
    temperature=0
)

print("Creating vector store from examples...")
vectorstore = FAISS.from_texts(example_queries, embedding=embedding_model)
retriever = vectorstore.as_retriever(k=2)
print("Vector store ready.")

# Adjusted prompt for Llama3
prompt_template = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are an expert at translating natural language requests into structured JSON commands. Your goal is to parse the user's request and generate a JSON object that follows the specified format. Based on the following examples, please process the new user query. Generate ONLY the JSON object.

--- EXAMPLES ---
{context}<|eot_id|><|start_header_id|>user<|end_header_id|>
USER QUERY:
{question}<|eot_id|><|start_header_id|>assistant<|end_header_id|>
"""
prompt = ChatPromptTemplate.from_template(prompt_template)

# --- 4. ASSEMBLE THE CHAIN ---
def format_retrieved_docs(docs):
    retrieved_examples = []
    for doc in docs:
        for example in examples:
            if example['query'] == doc.page_content:
                retrieved_examples.append(example)
                break
    return format_examples(retrieved_examples)

parser = JsonOutputParser(pantic_object=StructuredCommand)
chain = (
    { "context": retriever | format_retrieved_docs, "question": RunnablePassthrough() }
    | prompt
    | llm
    | parser
)

# --- 5. RUN AND TEST THE SYSTEM ---
if __name__ == "__main__":
    print("\n--- Natural Language to Structured Command Translator (Final Version) ---")
    print("Type 'exit' to quit.")
    while True:
        user_input = input("\n> ")
        if user_input.lower() in ['exit', 'quit']:
            break
        try:
            result = chain.invoke(user_input)
            print("\n--- Structured Command ---")
            print(json.dumps(result, indent=2))
            print("--------------------------")
        except Exception as e:
            print(f"\nAn error occurred: {e}")