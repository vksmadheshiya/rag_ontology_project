import streamlit as st
import os
from dotenv import load_dotenv

# --- Load API Key ---
# Load from .env file if it exists
load_dotenv()

# Load core modules AFTER potentially setting environment variables
from core.utils import load_text_from_url, load_text_from_upload, clean_text, get_book_identifier
from core.ontology_builder import get_or_create_ontology
from core.vector_store import get_or_create_vector_store
from core.qa_chain import create_qa_chain, get_answer
from core.ontology_schema import PREDEFINED_ENTITY_TYPES, PREDEFINED_RELATIONSHIP_TYPES # Optional: Display schema info

# --- Page Configuration ---
st.set_page_config(page_title="Book Ontology RAG", layout="wide")


# --- Caching Functions (Resource-intensive objects) ---

# Cache LLM and Embedding models (less critical as LangChain might cache internally, but good practice)
# @st.cache_resource
# def get_llm_instance(model_name):
#     # Potentially initialize LLM here if needed outside chains
#     pass

# Cache the vector store resource based on the book identifier
@st.cache_resource(show_spinner="Loading Vector Store...")
def load_vector_store_cached(book_identifier, book_text, ontology_data):
    # This function ensures vector store is created only once per book identifier
    # during the app's lifecycle or until cache is cleared.
    if not book_text or not ontology_data:
         st.warning("Cannot load vector store without book text and ontology.")
         return None
    return get_or_create_vector_store(book_identifier, book_text, ontology_data)

# Cache the QA chain resource, dependent on the vector store object ID
# Using the vector store object itself implicitly makes it dependent
@st.cache_resource(show_spinner="Creating QA Chain...")
def load_qa_chain_cached(_vector_store):
    # Pass the actual vector store object. Streamlit's caching handles object identity.
    if _vector_store is None:
        st.error("Vector store is not loaded, cannot create QA chain.")
        return None
        
    return create_qa_chain(_vector_store)

# Cache ontology data (serializable JSON)
@st.cache_data(show_spinner="Loading/Building Ontology...")
def load_ontology_cached(book_identifier, book_text):
     if not book_text:
         st.warning("Cannot build ontology without book text.")
         return None
     return get_or_create_ontology(book_identifier, book_text)

# --- Initialize Session State ---
if 'book_processed' not in st.session_state:
    st.session_state.book_processed = False
if 'book_identifier' not in st.session_state:
    st.session_state.book_identifier = None
if 'book_text' not in st.session_state:
     st.session_state.book_text = None # Store book text to avoid reloading
if 'ontology' not in st.session_state:
     st.session_state.ontology = None
if 'vector_store' not in st.session_state:
    st.session_state.vector_store = None
if 'qa_chain' not in st.session_state:
    st.session_state.qa_chain = None


# --- Sidebar ---
with st.sidebar:
    st.header("Configuration")
    # Allow setting API key via sidebar input if not found in .env
    api_key_input = st.text_input("OpenAI API Key", value=os.getenv("OPENAI_API_KEY", ""), type="password",
                                  help="Enter your OpenAI API Key or set it in the .env file.")

    if api_key_input:
        os.environ["OPENAI_API_KEY"] = api_key_input # Set it for the current session

    if not os.getenv("OPENAI_API_KEY"):
        st.warning("OpenAI API Key not found. Please enter it above or set it in the .env file.")
        st.stop() # Stop execution if no API key is available

    st.subheader("Load Knowledge Base")
    book_url = st.text_input("Enter Book URL (.txt)", "https://www.gutenberg.org/cache/epub/1342/pg1342.txt")
    uploaded_file = st.file_uploader("Or Upload a Book (.txt)", type=['txt'])

    process_button = st.button("Load and Process Book")

    if process_button:
        st.session_state.book_processed = False # Reset status on new load attempt
        st.session_state.book_identifier = None
        st.session_state.book_text = None
        st.session_state.ontology = None
        st.session_state.vector_store = None
        st.session_state.qa_chain = None
        # Clear specific cache entries if necessary (more complex, usually clearing all works)
        # st.cache_data.clear()
        # st.cache_resource.clear()

        source_text = None
        identifier = None

        if uploaded_file is not None:
            source_text = load_text_from_upload(uploaded_file)
            identifier = get_book_identifier(uploaded_file.name)
            st.info(f"Processing uploaded file: {uploaded_file.name}")
        elif book_url:
            source_text = load_text_from_url(book_url)
            identifier = get_book_identifier(book_url)
            st.info(f"Processing book from URL: {book_url}")
        else:
            st.error("Please provide a URL or upload a file.")

        if source_text and identifier:
            with st.spinner("Cleaning text..."):
                cleaned_text = clean_text(source_text)
                st.session_state.book_text = cleaned_text # Store cleaned text
                st.session_state.book_identifier = identifier
                st.success(f"Text loaded and cleaned successfully. Identifier: {identifier}")

                # --- Trigger processing via cached functions ---
                st.session_state.ontology = load_ontology_cached(identifier, cleaned_text)

                # --- Trigger processing (calls updated functions) ---
                st.session_state.ontology = load_ontology_cached(identifier, cleaned_text) # Uses updated builder implicitly
                        
                if st.session_state.ontology:
                    st.success("Fixed Ontology loaded/built.")
                    st.session_state.vector_store = load_vector_store_cached(identifier, cleaned_text, st.session_state.ontology)
                    if st.session_state.vector_store:
                        st.success("Vector Store loaded/built.")
                        st.session_state.qa_chain = load_qa_chain_cached(st.session_state.vector_store)
                        if st.session_state.qa_chain:
                            st.success("QA Chain ready.")
                            st.session_state.book_processed = True
                        else:
                            st.error("Failed to create QA Chain.")
                    else:
                        st.error("Failed to load/build Vector Store.")
                else:
                    st.error("Failed to load/build Ontology.")
        else:
            st.error("Failed to load text from the provided source.")


# --- Main Content Area ---
st.title("📚 RAG with Ontology Enhancement")
st.markdown("Ask questions about the loaded knowledge base.")

if not st.session_state.book_processed:
    st.warning("Please load and process a book using the sidebar.")
else:
    st.success(f"Book '{st.session_state.book_identifier}' processed and ready.")

    st.subheader("Ask a Question")
    query = st.text_input("Enter your question:", key="query_input")

    if query and st.session_state.qa_chain:
        with st.spinner("Thinking..."):
            answer_data = get_answer(query, st.session_state.qa_chain)

        if "error" in answer_data:
            st.error(f"Error generating answer: {answer_data['error']}")
        elif "result" in answer_data:
            st.markdown("### Answer")
            st.markdown(answer_data["result"])

            if "source_documents" in answer_data and answer_data["source_documents"]:
                with st.expander("Show Context / Sources"):
                    for i, doc in enumerate(answer_data["source_documents"]):
                        st.markdown(f"**Source {i+1}**")
                        # Display the new metadata structure
                        st.caption("Metadata:")
                        meta_entities = doc.metadata.get('entities', [])
                        meta_relations = doc.metadata.get('relations', [])
                        if meta_entities:
                            st.write(f"**Entities Present:** {', '.join(meta_entities)}")
                        if meta_relations:
                            st.write(f"**Relations Mentioned:** {', '.join(meta_relations)}")
                        # Display source snippet or full text
                        st.text(doc.page_content)
                        st.divider()
        else:
            st.error("Received an unexpected response structure.")

    # Optional: Display the generated ontology (using the fixed structure)
    if st.session_state.ontology:
        with st.expander("View Extracted Knowledge (Fixed Schema)"):
            st.json(st.session_state.ontology) # Displays the entities/relations lists


    # Optional: Display the fixed schema definition for user reference
    with st.sidebar.expander("View Fixed Ontology Schema"):
        st.markdown("**Target Entity Types:**")
        st.json(PREDEFINED_ENTITY_TYPES)
        st.markdown("**Target Relationship Types:**")
        st.json(PREDEFINED_RELATIONSHIP_TYPES)