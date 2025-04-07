# Potential addition to core/qa_chain.py or a new core/query_analyzer.py

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from typing import List, Optional

# Import your fixed schema types for reference in the prompt
from core.ontology_schema import PREDEFINED_ENTITY_TYPES, PREDEFINED_RELATIONSHIP_TYPES

class AnalyzedQuery(BaseModel):
    """Structured analysis of the user query."""
    mentioned_entities: List[str] = Field(description="List of entity names mentioned in the query potentially matching the ontology (e.g., 'Elizabeth Bennet', 'Pemberley').")
    target_entity_types: List[str] = Field(description="List of entity types explicitly or implicitly asked about (e.g., 'Person', 'Place').")
    target_relationship_types: List[str] = Field(description="List of relationship types explicitly or implicitly asked about (e.g., 'WORKS_FOR', 'LOCATED_AT').")

# Convert lists to strings for the prompt
entity_types_str = ", ".join(PREDEFINED_ENTITY_TYPES)
relationship_types_str = ", ".join(PREDEFINED_RELATIONSHIP_TYPES)

query_analysis_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", f"""Analyze the user's query to identify key components relevant to a knowledge base structured with a fixed ontology.
        Focus on extracting names that likely represent entities and identifying the types of entities or relationships the user is asking about.
        The known entity types are: {entity_types_str}.
        The known relationship types are: {relationship_types_str}.
        Return ONLY the names/types mentioned or implied by the query. If nothing relevant is found, return empty lists.
        Format the output strictly as JSON conforming to the provided `AnalyzedQuery` schema.
        """),
        ("human", "User Query: \"{query}\"\n\nJSON Schema for Output:\n```json\n{schema}\n```\nAnalyze the query based on the fixed ontology concepts.")
    ]
)

# Initialize a separate, potentially faster/cheaper LLM for this task if desired
query_llm = ChatOpenAI(model="gpt-3.5-turbo-0125", temperature=0) # Example
structured_query_llm = query_llm.with_structured_output(AnalyzedQuery)

def analyze_query_llm(query: str) -> Optional[AnalyzedQuery]:
    """Analyzes the user query using an LLM to extract relevant ontology concepts."""
    if not query:
        return None
    try:
        print(f"Analyzing query: '{query}'")
        response = structured_query_llm.invoke({
            "query": query,
            "schema": AnalyzedQuery.schema_json()
        })
        if isinstance(response, AnalyzedQuery):
            print(f"Query analysis result: {response}")
            return response
        else:
            print(f"Warning: Query analysis LLM output type mismatch: {type(response)}")
            # Attempt parse like in ontology builder if needed
            return None
    except Exception as e:
        print(f"Error during query analysis LLM invocation: {e}")
        return None
    


# Alternative implementation (simpler, less accurate)
# Assumes you have access to the 'ontology' dictionary built earlier

def analyze_query_keywords(query: str, ontology: dict) -> dict:
    """Analyzes query by simple keyword matching against known entities."""
    analysis = {"mentioned_entities": [], "target_entity_types": [], "target_relationship_types": []}
    query_lower = query.lower()

    if not ontology or "entities" not in ontology:
        return analysis

    # Find mentioned entity names
    all_entity_names = {entity.get('name', '').lower(): entity.get('entity_type')
                        for entity in ontology['entities'] if entity.get('name')}

    for name_lower, ent_type in all_entity_names.items():
        if name_lower and re.search(r'\b' + re.escape(name_lower) + r'\b', query_lower):
             # Store as "Type:Name" to match potential metadata format
             analysis["mentioned_entities"].append(f"{ent_type}:{name_lower.capitalize()}") # Or just name

    # Simple check for relationship keywords (very basic)
    for rel_type in PREDEFINED_RELATIONSHIP_TYPES:
        # Check if words related to the relationship type are in the query
        # Example: Check if "interacts" or "interaction" is in query for INTERACTS_WITH
        # This needs more sophisticated mapping (e.g., 'who works for' -> WORKS_FOR)
        if rel_type.lower().replace("_", " ") in query_lower:
             analysis["target_relationship_types"].append(rel_type)

    print(f"Query keyword analysis result: {analysis}")
    return analysis # Return a dictionary compatible with filtering logic