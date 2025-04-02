from scr.agent.state.state import State, StepOutput
from langchain_together import Together
from langchain_core.messages import AIMessage
from langchain_core.prompts import ChatPromptTemplate
from scr.agent.prompts.prompts import QUERY_GENERATION_PROMPT
from typing import List, Dict, Any
from scr.agent.utils.config import Configuration
from langchain_core.runnables import RunnableConfig
from scr.agent.utils.llm_utils import get_llm
import re



async def query_generator(state: State, config: RunnableConfig) -> Dict[str, List[AIMessage]]:
    """Generate a SPARQL query based on the structured question and retrieved documents.
    
    Args:
        state: The current state containing structured question and retrieved documents
        config: Configuration for the runner
        
    Returns:
        Dict containing structured_output and steps
    """
    
    try:

        configuration = Configuration.from_runnable_config(config)


        llm = get_llm(configuration)

        prompt_template = ChatPromptTemplate.from_messages(
            [
                ("system", QUERY_GENERATION_PROMPT),
                ("placeholder", "{question} {potential_entities} {retrieved_documents}"),
            ]
        )

        message = prompt_template.invoke(
            {
                "question": state.messages[0].content if state.messages else "Generate a SPARQL query",
                "potential_entities": state.extracted_entities,
            }
        )
        
        response_message = await llm.ainvoke(message)

        extracted_queries = extract_sparql_queries(response_message.content)

        return {
            "structured_output": extracted_queries,
            "steps": [
                StepOutput(
                    label="Generated SPARQL query",
                    details=response_message.content,
                )
            ]
        }
    except Exception as e:
        return {
            "error": str(e),
            "steps": [
                StepOutput(
                    label="Error in SPARQL query generation",
                    details=f"Failed to generate query: {str(e)}",
                    type="fix-message"
                )
            ]
        }



queries_pattern = re.compile(r"```sparql(.*?)```", re.DOTALL)
endpoint_pattern = re.compile(r"^#.*(https?://[^\s]+)", re.MULTILINE)


def extract_sparql_queries(md_resp: str) -> list[dict[str, Optional[str]]]:
    """Extract SPARQL queries and endpoint URL from a markdown response."""
    extracted_queries = []
    queries = queries_pattern.findall(md_resp)
    for query in queries:
        extracted_endpoint = endpoint_pattern.search(query.strip())
        if extracted_endpoint:
            extracted_queries.append(
                {
                    "query": str(query).strip(),
                    "endpoint_url": str(extracted_endpoint.group(1)) if extracted_endpoint else None,
                }
            )
    return extracted_queries