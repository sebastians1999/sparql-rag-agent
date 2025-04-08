import requests
from bs4 import BeautifulSoup


def validate_sparql_syntax(query: str) -> tuple[bool, str | None]:
    """
    Validates the syntax of a SPARQL query using the SPARQLer Query Validator.
    
    Parameters:
      query (str): The SPARQL query string to validate.
    
    Returns:
      tuple: (True, None) if the query is syntactically correct,
             (False, error_message) if there is a syntax error.
    """
    base_url = "http://www.sparql.org/"
    endpoint = base_url + "$/validate/query"
    
    data = {
        "query": query,
        "languageSyntax": "SPARQL",
        "outputFormat": "sparql",
        "linenumbers": "true"
    }
    
    headers = {
        "Referer": "http://www.sparql.org/query-validator.html",
    }
    
    try:
        response = requests.post(endpoint, data=data, headers=headers)
        response.raise_for_status()
    except requests.RequestException as e:
        return False, f"Request failed: {e}"
    

    soup = BeautifulSoup(response.text, 'html.parser')
    
    error_section = soup.find('p', text='Syntax error:')
    
    if error_section:
        error_box = error_section.find_next('pre', class_='box')
        if error_box:
            return False, error_box.text.strip()
        else:
            return False, "Unknown syntax error"
    else:
        return True, None