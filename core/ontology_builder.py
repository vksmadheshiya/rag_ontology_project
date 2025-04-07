# core/ontology_builder.py (Modified Sections)

import json
import os
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from typing import List, Dict, Any # Added Any

# Import the fixed schema definitions
from core.ontology_schema import FixedExtractedKnowledge, PREDEFINED_ENTITY_TYPES, PREDEFINED_RELATIONSHIP_TYPES

# --- Configuration --- (Keep Cache Dir, Model Name, Chunk Size)
ONTOLOGY_MODEL_NAME = "gpt-4-turbo-preview" # May need a stronger model for mapping
CHUNK_SIZE_ONTOLOGY = 8000 # Adjust based on model context
CHUNK_OVERLAP_ONTOLOGY = 400
CACHE_DIR = "cache"

# --- Updated Prompt Template ---
# Convert lists to strings for embedding in the prompt
entity_types_str = ", ".join(PREDEFINED_ENTITY_TYPES)
relationship_types_str = ", ".join(PREDEFINED_RELATIONSHIP_TYPES)

ontology_prompt_template = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            f"""You are an expert knowledge extraction system tasked with mapping information from text onto a FIXED ontology schema.
Your goal is to identify instances of predefined entity types and relationships between them within the provided text excerpt.

FIXED SCHEMA:
- Entity Types: {entity_types_str}
- Relationship Types: {relationship_types_str}

Instructions:
1. Read the text excerpt carefully.
2. Identify specific mentions that represent instances of the predefined Entity Types. Assign the most appropriate type. Extract a name and optionally a brief description or key attributes mentioned.
3. Identify explicit relationships between the identified entities that match one of the predefined Relationship Types.
4. Focus *only* on information present in the excerpt.
5. If no relevant entities or relationships fitting the schema are found, return empty lists for 'entities' and/or 'relations'.
6. Format your output *strictly* as JSON conforming to the provided `FixedExtractedKnowledge` schema.
""",
        ),
        ("human", "Text Excerpt:\n```\n{text_excerpt}\n```\n\nJSON Schema for Output:\n```json\n{schema}\n```\nPlease extract knowledge based on the fixed schema and the text."),
    ]
)

# --- Ontology Building Logic ---
def extract_ontology_from_chunk(text_chunk, llm_structured) -> FixedExtractedKnowledge:
    """Extracts ontology from a single text chunk using the LLM and fixed schema."""
    try:
        # Pass the actual schema definition (as JSON string) to the prompt
        response = llm_structured.invoke({
            "text_excerpt": text_chunk,
            "schema": FixedExtractedKnowledge.schema_json()
        })
        # Ensure the response conforms to the Pydantic model
        if isinstance(response, FixedExtractedKnowledge):
            return response
        else:
             # Handle cases where the LLM might return a dict that needs parsing
             # This depends on the LangChain version and structured output behavior
             print(f"Warning: LLM output type mismatch. Expected FixedExtractedKnowledge, got {type(response)}. Attempting parse.")
             try:
                 return FixedExtractedKnowledge.parse_obj(response) # For older Pydantic v1 style
                 # Or FixedExtractedKnowledge.model_validate(response) # For Pydantic v2 style
             except Exception as parse_error:
                 print(f"Error parsing LLM output into FixedExtractedKnowledge: {parse_error}")
                 return FixedExtractedKnowledge(entities=[], relations=[])

    except Exception as e:
        print(f"Error during LLM invocation for ontology chunk: {e}")
        return FixedExtractedKnowledge(entities=[], relations=[]) # Return empty structure

def aggregate_ontologies(chunk_results: List[FixedExtractedKnowledge]) -> Dict[str, List[Dict[str, Any]]]:
    """Aggregates ontology results, deduplicating based on entity name/type and relationship details."""
    aggregated_entities: Dict[tuple, Dict] = {} # Use (name, type) as key for uniqueness
    aggregated_relations: set = set() # Use a tuple representation of relation for uniqueness

    for result in chunk_results:
        if not isinstance(result, FixedExtractedKnowledge):
             print(f"Warning: Skipping invalid result type during aggregation: {type(result)}")
             continue

        for entity in result.entities:
             key = (entity.name.strip().lower(), entity.entity_type) # Normalize name slightly
             if key not in aggregated_entities:
                 aggregated_entities[key] = entity.dict()
             else:
                 # Optional: Merge attributes/descriptions if needed
                 pass # Simple approach: keep the first encountered

        for relation in result.relations:
             # Create a tuple representation for hashing/set membership
             rel_key = (
                 relation.source_entity_name.strip().lower(),
                 relation.source_entity_type,
                 relation.relationship_type,
                 relation.target_entity_name.strip().lower(),
                 relation.target_entity_type
             )
             # Store the full dict representation if the key is new
             if rel_key not in aggregated_relations:
                 aggregated_relations.add(rel_key)
                 # Store the dict for reconstruction later (a bit inefficient but simple)
                 # A better way might be to store the dict in a parallel structure mapped by rel_key
                 # For simplicity now, we'll rebuild the list from unique keys after processing all chunks.

    # Reconstruct the final list of unique relations (this step is needed because sets don't store the original dicts)
    # We need to efficiently retrieve the full relation dict corresponding to each unique key.
    # Let's refine the aggregation logic slightly:
    final_relations_map = {}
    aggregated_relations_keys = set()
    for result in chunk_results:
         if not isinstance(result, FixedExtractedKnowledge): continue
         for relation in result.relations:
             rel_key = (
                 relation.source_entity_name.strip().lower(),
                 relation.source_entity_type,
                 relation.relationship_type,
                 relation.target_entity_name.strip().lower(),
                 relation.target_entity_type
             )
             if rel_key not in aggregated_relations_keys:
                 aggregated_relations_keys.add(rel_key)
                 final_relations_map[rel_key] = relation.dict() # Keep first occurrence


    final_ontology = {
        "entities": list(aggregated_entities.values()),
        "relations": list(final_relations_map.values())
    }
    return final_ontology


def build_ontology(full_text: str) -> Dict:
    """Builds the ontology using the fixed schema."""
    print("Building ontology using fixed schema...")
    # Use a model potentially better at instruction following for fixed schema mapping
    llm = ChatOpenAI(model=ONTOLOGY_MODEL_NAME, temperature=0)
    structured_llm = llm.with_structured_output(FixedExtractedKnowledge) # Use the correct Pydantic model

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE_ONTOLOGY,
        chunk_overlap=CHUNK_OVERLAP_ONTOLOGY,
        length_function=len,
    )
    chunks = text_splitter.split_text(full_text)
    print(f"Split text into {len(chunks)} chunks for fixed ontology extraction.")

    results = []
    for i, chunk in enumerate(chunks):
        print(f"Processing chunk {i+1}/{len(chunks)} for ontology...")
        # Pass the schema definition string for the LLM to see
        chunk_ontology = extract_ontology_from_chunk(chunk, structured_llm)
        results.append(chunk_ontology)

    print("Aggregating ontology results...")
    final_ontology = aggregate_ontologies(results)
    print(f"Ontology building complete. Found {len(final_ontology['entities'])} unique entities and {len(final_ontology['relations'])} unique relations.")
    return final_ontology

# --- Caching function get_or_create_ontology remains the same ---
# Just ensure it calls the updated build_ontology function.
def get_or_create_ontology(book_identifier: str, full_text: str) -> Dict:
    """Loads ontology from cache or builds it if not found, using the fixed schema."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    # Append schema version/name to cache filename if schema might change later
    cache_file = os.path.join(CACHE_DIR, f"fixed_ontology_{book_identifier}.json")

    if os.path.exists(cache_file):
        print(f"Loading fixed ontology from cache: {cache_file}")
        try:
            with open(cache_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading ontology from cache: {e}. Rebuilding...")

    # Build and cache
    ontology = build_ontology(full_text) # Calls the updated function
    try:
        with open(cache_file, 'w', encoding='utf-8') as f:
            json.dump(ontology, f, indent=4)
        print(f"Saved fixed ontology to cache: {cache_file}")
    except Exception as e:
        print(f"Error saving ontology to cache: {e}")

    return ontology