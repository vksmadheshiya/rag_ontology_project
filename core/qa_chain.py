from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import Chroma # For type hinting
from typing import Dict

from core.query_analyzer import analyze_query_llm

# --- Configuration ---
QA_MODEL_NAME = "gpt-3.5-turbo-0125" # Can be same or different from ontology model

# --- RAG Prompt Template ---
rag_template = """You are an expert assistant knowledgeable about the provided text.
Use the following retrieved context passages to answer the user's question.
If the answer cannot be found in the context, state that clearly. Do not make up information.
Be concise and directly answer the question based *only* on the provided context.

Context:
{context}

Question:
{question}

Answer:"""

RAG_PROMPT = PromptTemplate.from_template(rag_template)

# --- QA Chain Creation ---
def create_qa_chain(vector_store: Chroma, query: str, ontology: dict): # Pass query and ontology if using keyword method
    """Creates the RetrievalQA chain."""
    print("Creating QA chain...")
    if vector_store is None:
        print("Error: Vector store is not available to create QA chain.")
        return None

    llm = ChatOpenAI(model_name=QA_MODEL_NAME, temperature=0.1) # Low temp for factual answers


    # 1. Analyze the query
    analysis_result = analyze_query_llm(query) # Or analyze_query_keywords(query, ontology)

    # 2. Build the filter based on analysis
    search_filter = None
    if analysis_result and analysis_result.mentioned_entities:
        # Example: Filter for chunks containing ALL mentioned entities
        # Adjust based on your metadata format (list vs CSV, "Type:Name" vs just "Name")
        # ASSUMING metadata.entities is a list like ["Person:Elizabeth Bennet", "Place:Pemberley"]
        required_entities = [e for e in analysis_result.mentioned_entities] # Use the direct output if format matches
        if required_entities:
             search_filter = {
                 "entities": {"$all": required_entities} # Chroma syntax for list contains all elements
                 # Or "$contains" for any, check Chroma docs exact syntax
             }
             print(f"Applying metadata filter: {search_filter}")

    # 3. Create retriever with filter (if applicable)
    retriever_kwargs = {"k": 5}
    if search_filter:
        retriever_kwargs["k"] = 10 # Retrieve more candidates when filtering
        retriever_kwargs["filter"] = search_filter

    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs=retriever_kwargs
    )

   # 4. Create the QA chain (as before)
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": RAG_PROMPT},
    )

    print("QA chain created successfully.")
    return qa_chain


def create_qa_chain(vector_store: Chroma): # <-- REMOVED query and ontology arguments
    """Creates the RetrievalQA chain with simple semantic retrieval."""
    print("Creating QA chain...")
    if vector_store is None:
        print("Error: Vector store is not available to create QA chain.")
        return None

    llm = ChatOpenAI(model_name=QA_MODEL_NAME, temperature=0.1)

    # --- Use simple semantic retriever ---
    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 5} # Retrieve top 5 relevant chunks semantically
    )
    # --- REMOVED query analysis and filter building ---

    # Create the RetrievalQA chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever, # Use the simple semantic retriever
        return_source_documents=True,
        chain_type_kwargs={"prompt": RAG_PROMPT},
    )

    print("QA chain created successfully.")
    return qa_chain



def get_answer(query: str, qa_chain: RetrievalQA) -> Dict:
    """Gets an answer from the QA chain."""
    if qa_chain is None:
        return {"error": "QA chain is not initialized."}
    try:
        print(f"Invoking QA chain for query: '{query}'")
        # The RetrievalQA chain internally handles passing the query to the retriever and LLM
        result = qa_chain.invoke({"query": query})
        print("QA chain invocation complete.")
        return result
    except Exception as e:
        print(f"Error during QA chain invocation: {e}")
        return {"error": f"An error occurred: {e}"}
    

# --- Query Function ---
def get_answer(query: str, qa_chain: RetrievalQA) -> Dict:
    """Gets an answer from the QA chain."""
    if qa_chain is None:
        return {"error": "QA chain is not initialized."}
    try:
        print(f"Invoking QA chain for query: '{query}'")
        result = qa_chain.invoke({"query": query})
        print("QA chain invocation complete.")
        return result
    except Exception as e:
        print(f"Error during QA chain invocation: {e}")
        return {"error": f"An error occurred: {e}"}
    


