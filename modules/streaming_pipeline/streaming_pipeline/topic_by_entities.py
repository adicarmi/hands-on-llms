import spacy
from fuzzywuzzy import fuzz
from fuzzywuzzy import process
import re

def normalize_text(text):
    """
    Normalize text for entity resolution while maintaining readability.
    - Convert to lowercase.
    - Strip leading/trailing whitespace.
    - Remove non-essential punctuation.
    - Keep alphanumeric characters and important symbols intact.
    """
    return re.sub(r'[^\w\s\-&]', '', text).strip()

def extract_and_resolve_entities(text, similarity_threshold=80, entity_types=None):
    """
    Extracts and resolves entities from text, normalizes them, and groups similar entities.

    Args:
        text (str): The input text to process.
        similarity_threshold (int): The minimum similarity score to consider two entities as matching.
        entity_types (list): List of entity types to include in the results. If None, include all.

    Returns:
        dict: A dictionary with resolved entities and their counts, sorted by count.
    """
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(text)

    # Extract entities with normalization
    entities_by_type = {}
    for ent in doc.ents:
        if entity_types and ent.label_ not in entity_types:
            continue

        normalized_text = normalize_text(ent.text)
        if ent.label_ not in entities_by_type:
            entities_by_type[ent.label_] = {}
        if normalized_text in entities_by_type[ent.label_]:
            entities_by_type[ent.label_][normalized_text]['originals'].append(ent.text)
            entities_by_type[ent.label_][normalized_text]['count'] += 1
        else:
            entities_by_type[ent.label_][normalized_text] = {
                'originals': [ent.text],
                'count': 1
            }

    # Resolve entities
    resolved_entities = {}
    for entity_type, entities in entities_by_type.items():
        resolved_entities[entity_type] = {}
        canonical_map = {}

        for normalized_entity, data in entities.items():
            # Check if the entity matches an existing canonical entity
            match_data = process.extractOne(normalized_entity, list(canonical_map.keys()), scorer=fuzz.token_set_ratio)
            if match_data:
                match, score = match_data
                if score >= similarity_threshold:
                    canonical_map[match]['variants'].extend(data['originals'])
                    canonical_map[match]['count'] += data['count']
                else:
                    # Add as a new canonical entity
                    canonical_map[normalized_entity] = {
                        'variants': data['originals'],
                        'count': data['count']
                    }
            else:
                # Add as a new canonical entity if no matches are found
                canonical_map[normalized_entity] = {
                    'variants': data['originals'],
                    'count': data['count']
                }

        # Flatten canonical_map into the resolved_entities structure
        resolved_entities[entity_type] = {canonical: {
            'count': data['count'],
            'original_variants': list(set(data['variants']))
        } for canonical, data in canonical_map.items()}

    # Sort canonical entities by count
    for entity_type in resolved_entities:
        resolved_entities[entity_type] = dict(sorted(
            resolved_entities[entity_type].items(),
            key=lambda item: item[1]['count'],
            reverse=True
        ))

    return resolved_entities

def extract_top_entities_as_string(text, similarity_threshold=80, entity_types=None, max_n=3):
    """
    Extracts entities from text, resolves them, and concatenates the top N canonical entities by count into a string.

    Args:
        text (str): The input text to process.
        similarity_threshold (int): The minimum similarity score to consider two entities as matching.
        entity_types (list): List of entity types to include in the results. If None, include all.
        max_n (int): The maximum number of top entities to include for each entity type.

    Returns:
        str: A concatenated string of top canonical entities.
    """
    resolved_entities = extract_and_resolve_entities(text, similarity_threshold, entity_types)

    top_entities = []
    for entity_type, entities in resolved_entities.items():
        count = 0
        for canonical, data in entities.items():
            if count >= max_n:
                break
            top_entities.append(canonical)
            count += 1

    return ", ".join(top_entities)

