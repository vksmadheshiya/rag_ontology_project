# RAG with Ontology Enhancement Prototype

This project implements a Retrieval-Augmented Generation (RAG) system that uses a dynamically generated ontology to enhance information retrieval from a large text knowledge base (e.g., a book).

**Business Goal:** Navigate large knowledge bases more effectively.
**Problem:** Manually creating ontologies is difficult and doesn't scale.
**Solution:** Use GenAI to create an ontology, use it to enrich text chunks stored in a vector database, and answer questions using RAG.

## Architecture

1.  **Load Text:** Download book text from URL or upload file.
2.  **Clean Text:** Remove headers/footers (e.g., Gutenberg).
3.  **Build Ontology:**
    *   Process text in chunks.
    *   Use an LLM (e.g., GPT-3.5/4) with structured output prompting to extract entities (characters, places, themes) and basic relationships.
    *   Aggregate results and cache the ontology as JSON.
4.  **Chunk & Embed with Metadata:**
    *   Chunk the cleaned text into smaller pieces suitable for retrieval.
    *   For each chunk, identify relevant concepts from the generated ontology and add them as metadata.
    *   Embed text chunks using a sentence transformer model (`all-MiniLM-L6-v2`).
    *   Store chunks, embeddings, and metadata in a persistent Chroma vector database.
5.  **RAG Query:**
    *   User enters a query via Streamlit UI.
    *   Query the vector store for relevant chunks (semantic similarity). The metadata *could* be used for filtering/boosting in more advanced setups, but this prototype primarily uses it for context display.
    *   Feed retrieved chunks and the query to another LLM (e.g., GPT-3.5) using a specific RAG prompt.
    *   Display the generated answer and source chunks (with their ontology metadata).

## Technology Stack

*   **Backend:** Python 3.9+
*   **UI:** Streamlit
*   **Core AI/Orchestration:** LangChain
*   **LLMs:** OpenAI API (Configurable: `gpt-3.5-turbo-0125` used by default)
*   **Embeddings:** `sentence-transformers` (`all-MiniLM-L6-v2`)
*   **Vector DB:** ChromaDB (persistent local storage)
*   **Ontology Storage:** JSON files (cached)
*   **Dependencies:** See `requirements.txt`

## Setup

1.  **Clone:** `git clone <your-repo-url>`
2.  **Navigate:** `cd rag_ontology_project`
3.  **Create Virtual Environment:** `python -m venv venv`
4.  **Activate Environment:**
    *   Windows: `.\venv\Scripts\activate`
    *   macOS/Linux: `source venv/bin/activate`
5.  **Install Dependencies:** `pip install -r requirements.txt`
6.  **Configure API Key:**
    *   Create a file named `.env` in the project root.
    *   Add your OpenAI API key: `OPENAI_API_KEY="sk-..."`
    *   Alternatively, you can enter the key directly in the Streamlit sidebar when running the app.

## Running the Application

1.  Ensure your virtual environment is activated.
2.  Run the Streamlit app: `streamlit run app.py`
3.  Open the URL provided by Streamlit in your browser.
4.  Use the sidebar to:
    *   Confirm/Enter your OpenAI API key.
    *   Keep the default book URL or provide another `.txt` URL / Upload a `.txt` file.
    *   Click "Load and Process Book". Wait for processing (ontology building and vector store creation can take time, especially the first time for a book).
5.  Once processing is complete, ask questions in the main area.

## Assumptions

*   Access to OpenAI API (or another configured LLM provider).
*   Input text is in plain text format (.txt) and reasonably clean (Gutenberg format handled).
*   The generated "ontology" is a simplified, LLM-extracted structure, not a formally validated one. Extraction quality depends on the LLM.
*   Sufficient local resources (RAM for models/embeddings, disk space for cache/DB).

## AI Assistant Usage

*   **Conceptualization & Planning:** Used LLM (Claude 3 Opus in initial discussion) to refine the understanding of the problem statement, brainstorm architectural options, and outline the implementation steps based on the prompt.
*   **Code Generation (Example Prompts & Refinement):**
    *   Used code assistants (like GitHub Copilot or directly prompting models like GPT-4/Claude) for boilerplate code generation and specific function implementations.
    *   *Example Prompt (Ontology Extraction Function):* "Write a Python function using LangChain and OpenAI's `gpt-3.5-turbo` to extract characters, places, and themes from a text chunk. The function should take the text chunk as input and return a dictionary. Use structured output prompting with Pydantic for reliability. Include basic error handling."
    *   *Example Prompt (Streamlit Caching):* "Show how to use `st.cache_data` and `st.cache_resource` in Streamlit to cache ontology JSON data and a Chroma vector store object, respectively. Ensure the caching depends on a unique book identifier."
*   **Refinement & Debugging:** Pasted code snippets and error messages into LLMs to get suggestions for fixes, improvements, or alternative approaches (e.g., improving regex, handling API errors, structuring LangChain chains).
*   **Assistant Selection:** Primarily used models known for strong coding and instruction-following (GPT-4, Claude 3). Selected based on access and perceived quality for the specific task (e.g., GPT-4 often better for complex code generation, Claude better for explanation/brainstorming). Output was always reviewed, tested, and modified significantly. The generated code serves as a starting point or solution to a specific sub-problem, not a copy-paste final product.

## Future Improvements

*   More robust text cleaning for diverse sources.
*   Advanced ontology extraction (e.g., event extraction, detailed relationship attributes).
*   Using ontology metadata for filtered/boosted retrieval in the vector store.
*   Support for other document formats (PDF, DOCX).
*   Option to choose different LLMs/embedding models via UI.
*   More sophisticated ontology visualization (e.g., using NetworkX).
*   Error handling and user feedback improvements.
*   Deployment instructions (e.g., Streamlit Cloud).