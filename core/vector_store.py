import os
import chromadb
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_openai import OpenAIEmbeddings

from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from typing import List, Dict
import re # Ensure re is imported

# --- Configuration ---
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
EMBEDDING_MODEL_NAME = "text-embedding-3-large"
CHUNK_SIZE_VECTOR = 1000  # Smaller chunks for better retrieval granularity
CHUNK_OVERLAP_VECTOR = 200
DB_DIR = "db"

# --- Helper Function for Metadata ---
def enrich_metadata(chunk_text: str, ontology: Dict) -> Dict:
    """Checks for ontology concepts in the chunk and adds them to metadata."""
    metadata = {"source_chunk": chunk_text[:200] + "..."} # Add snippet for context
    present_concepts = []

    # Simple string matching (case-insensitive)
    text_lower = chunk_text.lower()
    for concept_type in ["characters", "places", "themes"]:
        if concept_type in ontology:
            for concept in ontology[concept_type]:
                 # Use word boundaries for better matching, avoid partial matches within words
                 # Adding checks to ensure concept['name'] exists and is a string
                 concept_name = concept.get('name')
                 if concept_name and isinstance(concept_name, str) and re.search(r'\b' + re.escape(concept_name.lower()) + r'\b', text_lower):
                     present_concepts.append(f"{concept_type}:{concept_name}") # e.g., "character:Elizabeth Bennet"

    if present_concepts:
        metadata["ontology_concepts"] = ", ".join(present_concepts) # Store as comma-separated string

    # Add relationships? More complex - maybe find sentences containing related entities.
    # For simplicity, we'll stick to entity presence for now.

    return metadata

# --- Vector Store Logic ---
def create_vector_store(book_identifier: str, full_text: str, ontology: Dict):
    """Creates and persists a Chroma vector store with ontology-enriched metadata."""
    print("Creating vector store...")
    os.makedirs(DB_DIR, exist_ok=True)
    persist_directory = os.path.join(DB_DIR, book_identifier)
    collection_name = book_identifier # Use identifier for collection name

    # 1. Initialize Embeddings
    print(f"Initializing embedding model: {EMBEDDING_MODEL_NAME}")
    # embeddings = SentenceTransformerEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-large",
        # With the `text-embedding-3` class
        # of models, you can specify the size
        # of the embeddings you want returned.
        # dimensions=1024
    )

    # 2. Chunk Text
    print("Chunking text for vector store...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE_VECTOR,
        chunk_overlap=CHUNK_OVERLAP_VECTOR,
        length_function=len,
    )
    chunks = text_splitter.split_text(full_text)
    print(f"Split text into {len(chunks)} chunks for vector store.")

    # 3. Create LangChain Documents with Metadata
    print("Enriching chunks with ontology metadata...")
    documents = []
    for i, chunk in enumerate(chunks):
        metadata = enrich_metadata(chunk, ontology)
        # Ensure metadata values are suitable for Chroma (strings, numbers, bools)
        # Our current metadata should be fine.
        doc = Document(page_content=chunk, metadata=metadata)
        documents.append(doc)
        if (i + 1) % 100 == 0:
            print(f"Processed {i+1}/{len(chunks)} chunks for metadata...")

    # 4. Create and Persist Chroma DB
    print(f"Creating Chroma vector store at: {persist_directory}")
    vector_store = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        collection_name=collection_name,
        persist_directory=persist_directory
    )
    print(f"Persisting vector store...")
    vector_store.persist()
    print("Vector store creation complete.")
    return vector_store

def load_vector_store(book_identifier: str):
    """Loads an existing Chroma vector store."""
    persist_directory = os.path.join(DB_DIR, book_identifier)
    if not os.path.exists(persist_directory):
        print(f"Vector store directory not found: {persist_directory}")
        return None

    print(f"Loading vector store from: {persist_directory}")
    try:
        # embeddings = SentenceTransformerEmbeddings(model_name=EMBEDDING_MODEL_NAME)
        embeddings = OpenAIEmbeddings(
            model="text-embedding-3-large",
            # With the `text-embedding-3` class
            # of models, you can specify the size
            # of the embeddings you want returned.
            # dimensions=1024
        )
        vector_store = Chroma(
            collection_name=book_identifier,
            persist_directory=persist_directory,
            embedding_function=embeddings
        )
        print("Vector store loaded successfully.")
        return vector_store
    except Exception as e:
        # Catch specific exceptions if possible, e.g., related to Chroma loading
        print(f"Error loading vector store from {persist_directory}: {e}")
        # It might be corrupted, consider deleting or handling recovery
        return None

def get_or_create_vector_store(book_identifier: str, full_text: str, ontology: Dict):
    """Loads the vector store if it exists, otherwise creates it."""
    persist_directory = os.path.join(DB_DIR, book_identifier)
    # A simple check for existence. Chroma might have more robust ways.
    # Check specifically for files Chroma creates, like 'chroma.sqlite3'
    db_file_path = os.path.join(persist_directory, 'chroma.sqlite3')

    if os.path.exists(db_file_path):
        print(f"Found existing vector store data for {book_identifier}. Loading...")
        vs = load_vector_store(book_identifier)
        if vs:
            return vs
        else:
            print("Failed to load existing store, attempting to recreate.")
            # Potentially delete corrupted store here before recreating
            # import shutil
            # shutil.rmtree(persist_directory)
    else:
         print(f"No existing vector store found for {book_identifier}. Creating new one.")

    # If load failed or dir didn't exist, create it
    return create_vector_store(book_identifier, full_text, ontology)

# core/vector_store.py (Modified Sections)

# ... (Imports and Config remain similar) ...
import re # Ensure re is imported

# --- Updated Helper Function for Metadata ---
def enrich_metadata(chunk_text: str, ontology: Dict) -> Dict:
    """Adds metadata based on fixed ontology schema instances found in the chunk."""
    metadata = {"source_chunk_snippet": chunk_text[:150] + "..."} # Example snippet
    present_entities = []
    present_relations = [] # Store a simplified representation

    text_lower = chunk_text.lower()

    # Check for Entities
    if "entities" in ontology:
        for entity in ontology["entities"]:
            entity_name = entity.get('name')
            entity_type = entity.get('entity_type')
            if entity_name and isinstance(entity_name, str):
                 # Use word boundaries for better matching
                 if re.search(r'\b' + re.escape(entity_name.lower()) + r'\b', text_lower):
                     # Store as "Type:Name" for clarity
                     present_entities.append(f"{entity_type}:{entity_name}")

    # Check for Relations (more complex - does the chunk contain context for a relation?)
    # Simpler approach: Add relation if *both* source and target entities are present in the chunk
    if "relations" in ontology:
        for relation in ontology["relations"]:
            source_name = relation.get('source_entity_name')
            target_name = relation.get('target_entity_name')
            rel_type = relation.get('relationship_type')

            if source_name and target_name and rel_type and \
               isinstance(source_name, str) and isinstance(target_name, str) and \
               re.search(r'\b' + re.escape(source_name.lower()) + r'\b', text_lower) and \
               re.search(r'\b' + re.escape(target_name.lower()) + r'\b', text_lower):
                # Store simplified relation string
                present_relations.append(f"{rel_type}({source_name},{target_name})")

    if present_entities:
        # Use distinct keys for easier filtering if DB supports it, else join
        metadata["entities"] = list(set(present_entities)) # Store unique list
    if present_relations:
        metadata["relations"] = list(set(present_relations)) # Store unique list

    return metadata

# --- Vector Store Logic (create_vector_store, load_vector_store, get_or_create_vector_store) ---
# The core logic remains the same, as it just passes the metadata dict created above
# Ensure the keys used ('entities', 'relations') are consistent
# **Important**: When creating Chroma, ensure metadata keys are handled correctly.
# If Chroma has issues with lists directly in filters, consider storing comma-separated strings:
# metadata["entities_csv"] = ",".join(list(set(present_entities)))
# metadata["relations_csv"] = ",".join(list(set(present_relations)))
# And adjust filtering logic later accordingly. Assuming lists work for now.

# In create_vector_store function, inside the loop:
# ...
# for i, chunk in enumerate(chunks):
#     metadata = enrich_metadata(chunk, ontology) # Call updated function
#     doc = Document(page_content=chunk, metadata=metadata)
#     documents.append(doc)
# ...

# (Rest of the vector_store.py remains structurally similar)