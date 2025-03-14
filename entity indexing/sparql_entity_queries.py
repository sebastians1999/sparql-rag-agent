


SPARQL_QUERY_ALL_CLASSES = """
PREFIX owl: <http://www.w3.org/2002/07/owl#>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>

SELECT DISTINCT ?class ?label ?comment ?superClass ?superClassLabel 
       (GROUP_CONCAT(DISTINCT ?subClass; SEPARATOR="|") AS ?subClasses)
       (GROUP_CONCAT(DISTINCT ?subClassLabel; SEPARATOR="|") AS ?subClassLabels)
       (GROUP_CONCAT(DISTINCT ?property; SEPARATOR="|") AS ?properties)
       (GROUP_CONCAT(DISTINCT ?propertyLabel; SEPARATOR="|") AS ?propertyLabels)
       (COUNT(DISTINCT ?instance) AS ?instanceCount)
WHERE {
  # Get all OWL classes
  ?class a owl:Class .
  
  # Get labels (will be used for embeddings)
  OPTIONAL { ?class rdfs:label ?label . }
  
  # Get comments/descriptions
  OPTIONAL { ?class rdfs:comment ?comment . }
  
  # Get superclasses to understand the hierarchy
  OPTIONAL { 
    ?class rdfs:subClassOf ?superClass . 
    FILTER(?superClass != owl:Thing)
    OPTIONAL { ?superClass rdfs:label ?superClassLabel . }
  }
  
  # Get subclasses of this class
  OPTIONAL {
    ?subClass rdfs:subClassOf ?class .
    FILTER(?subClass != ?class) # Avoid self-references
    OPTIONAL { ?subClass rdfs:label ?subClassLabel . }
  }
  
  # Get properties associated with this class (domain)
  OPTIONAL { 
    ?property rdfs:domain ?class .
    OPTIONAL { ?property rdfs:label ?propertyLabel . }
  }
  
  # Get instance count for the class
  OPTIONAL { ?instance a ?class . }
  
  # Filter out OWL/RDFS built-in classes
  FILTER (!STRSTARTS(STR(?class), "http://www.w3.org/2002/07/owl#"))
  FILTER (!STRSTARTS(STR(?class), "http://www.w3.org/2000/01/rdf-schema#"))
}
GROUP BY ?class ?label ?comment ?superClass ?superClassLabel
ORDER BY ?class
"""