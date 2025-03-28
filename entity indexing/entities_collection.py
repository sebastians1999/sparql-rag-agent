entities_list = [
        {
            "uniprot_disease": {
                "uri": "http://purl.uniprot.org/core/Disease",
                "label": "Disease",
                "description": "The preferred names of diseases.",
                "endpoint": "https://sparql.uniprot.org/sparql/",
                "pagination": True,
                "query": """PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
                            PREFIX skos: <http://www.w3.org/2004/02/skos/core#>
                            PREFIX up: <http://purl.uniprot.org/core/>
                            SELECT ?uri ?label ?type WHERE {
                                ?uri a up:Disease ;
                                    skos:prefLabel ?label .
                            }"""
            }
        },
        {
            "uniprot_taxon": {
                "uri": "http://purl.uniprot.org/core/Taxon",
                "label": "species",
                "description": "taxon scientific names",
                "endpoint": "https://sparql.uniprot.org/sparql/",
                "pagination": True,
                "query": """PREFIX up: <http://purl.uniprot.org/core/>
                            SELECT ?uri ?label
                            WHERE {
                                ?uri a up:Taxon ;
                                    up:scientificName ?label .
                            }"""
            }
        },
        {
            "rhea_reaction": {
                "uri": "http://rdf.rhea-db.org/",
                "label": "reactions",
                "description": "Reactions in RHEA.",
                "endpoint": "https://sparql.uniprot.org/sparql/",
                "pagination": True,
                "query": """PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
                            PREFIX rh:   <http://rdf.rhea-db.org/>
                            SELECT DISTINCT ?uri ?label
                            WHERE {
                                ?uri rdfs:subClassOf rh:Reaction .
                                ?uri rdfs:label ?label .
                            }"""
            }
        },
        {
            "chebi_chemical_entities": {
                "uri": "http://purl.obolibrary.org/obo/CHEBI",
                "label": "chemical entities",
                "description": "Chemical entities in ChEBI.",
                "endpoint": "https://sparql.uniprot.org/sparql/",
                "pagination": True,
                "query": """
                        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
                        PREFIX rh:   <http://rdf.rhea-db.org/>
                        SELECT DISTINCT ?uri ?label
                        WHERE {
                            # Restrict to things that are recognized as Rhea reactions
                            ?reaction rdfs:subClassOf rh:Reaction ;
                                    rh:side        ?side .
                            # Each reaction side contains one or more participants
                            ?side rh:contains ?participant .
                            # Each participant is linked to a ChEBI entity via 'rh:compound' 
                            ?participant rh:compound ?compound .
                            # 'rh:chebi' points to the ChEBI identifier (e.g., http://purl.obolibrary.org/obo/CHEBI_15377)
                            ?compound rh:chebi ?uri .
                            ?uri rdfs:label ?label .
                        }"""
            }
        }
    ]