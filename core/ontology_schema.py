# core/ontology_schema.py
from pydantic import BaseModel, Field
from typing import List, Dict, Optional

# --- Fixed Schema Definition ---

class FixedEntity(BaseModel):
    """Represents an instance of a predefined class found in the text."""
    name: str = Field(description="Unique name or identifier of the entity instance.")
    entity_type: str = Field(description="The predefined class this instance belongs to (e.g., Person, Place, Organization, Concept, Event, Artifact).")
    description: Optional[str] = Field(description="Brief description based on the text.", default=None)
    # Example properties - adapt as needed for your fixed schema
    attributes: Optional[Dict[str, str]] = Field(description="Key attributes mentioned (e.g., 'Status': 'Main Character', 'LocationType': 'Estate').", default={})

class FixedRelation(BaseModel):
    """Represents a relationship instance between two identified entities."""
    source_entity_name: str = Field(description="Name of the source entity instance.")
    source_entity_type: str = Field(description="Type of the source entity instance.")
    relationship_type: str = Field(description="The predefined type of relationship (e.g., INTERACTS_WITH, LOCATED_AT, PART_OF, WORKS_FOR, TOPIC_OF, CAUSES, MENTIONED_IN).")
    target_entity_name: str = Field(description="Name of the target entity instance.")
    target_entity_type: str = Field(description="Type of the target entity instance.")
    context: Optional[str] = Field(description="Sentence providing context for the relationship.", default=None)

class FixedExtractedKnowledge(BaseModel):
    """Structured output containing instances conforming to the fixed schema."""
    entities: List[FixedEntity] = Field(description="List of distinct entity instances identified conforming to the schema.")
    relations: List[FixedRelation] = Field(description="List of relationship instances identified between entities, conforming to the schema.")

# --- Predefined Classes and Relationship Types (Informational) ---
# These are the *target* types the LLM should try to map text onto.
# This list can be passed to the LLM in the prompt for clarity.

PREDEFINED_ENTITY_TYPES = [
    "Person", "Place", "Organization", "Event", "Concept", "Artifact", "Miscellaneous"
]

PREDEFINED_RELATIONSHIP_TYPES = [
    "INTERACTS_WITH", # General interaction between Persons, Orgs, etc.
    "LOCATED_AT",     # An entity is at a Place
    "PART_OF",        # An entity is part of another (e.g., Person in Org, Place in larger Place)
    "WORKS_FOR",      # Person works for Organization
    "TOPIC_OF",       # Concept/Event/Person is the topic of discussion/document section
    "MENTIONED_IN",   # Entity mentioned in the context of another
    "CAUSES",         # Event/Action causes another
    "CHARACTERIZED_BY", # Person/Place characterized by Concept/Attribute
    "FAMILY_RELATION", # Specific type for Persons if needed
    "OWNS",           # Person/Org owns Artifact/Place
]