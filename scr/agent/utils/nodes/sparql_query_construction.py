from scr.agent.utils.state.state import State, StepOutput
from scr.agent.utils.config import RunnerConfig
from langchain_together import Together
from langchain_core.messages import AIMessage
from langchain_core.prompts import ChatPromptTemplate
from scr.agent.utils.prompts.prompts import QUERY_GENERATION_PROMPT
from typing import List, Dict, Any
from scr.agent.utils.config import Configuration
from langchain_core.runnables import RunnableConfig



async def query_generator(state: State, config: RunnableConfig) -> Dict[str, Any]:
    """Generate a SPARQL query based on the structured question and retrieved documents.
    
    Args:
        state: The current state containing structured question and retrieved documents
        config: Configuration for the runner
        
    Returns:
        Dict containing structured_output and steps
    """
    
    try:

        configuration = Configuration.from_runnable_config(config)


        llm = Together(model=configuration.llm_config.model_name, temperature=configuration.llm_config.temperature)

        prompt_template = ChatPromptTemplate.from_messages(
            [
                ("system", QUERY_GENERATION_PROMPT),
                ("placeholder", "{question} {potential_entities} {retrieved_documents}"),
            ]
        )

        message = await prompt_template.invoke(
            {
                "question": state.structured_question.question_steps[0] if state.structured_question.question_steps else "Generate a SPARQL query",
                "potential_entities": state.extracted_entities,
                "retrieved_documents": [doc.page_content for doc in state.retrieved_docs],
            }
        )
        
        response_message = await llm.invoke(message)

        return {
            "structured_output": response_message.content,
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

