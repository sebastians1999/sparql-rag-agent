from scr.agent.utils.state.state import State, StructuredQuestion, StepOutput
from langchain_together import Together
from langchain_core.prompts import ChatPromptTemplate
from scr.agent.utils.prompts.prompts import EXTRACTION_PROMPT
from typing import Dict, Any, List
from langchain_core.messages import AIMessage
import json
from pydantic import ValidationError
from scr.agent.utils.config import Configuration
from langchain_core.runnables import RunnableConfig

async def question_understanding(state: State, config: RunnableConfig) -> Dict[str, List[AIMessage]]:
    """Extract structured information from the user question.
    
    Args:
        state: The current state containing messages
        config: Configuration for the runner
        
    Returns:
        Dict containing structured_question and steps
        
    Raises:
        ValueError: If messages are empty or invalid
        ValidationError: If LLM response cannot be parsed into StructuredQuestion
    """


    configuration = Configuration.from_runnable_config(config)

    if not state.messages:
        raise ValueError("No messages found in state")

    try:
        llm = Together(model=configuration.llm_config.model_name, temperature=configuration.llm_config.temperature)

        prompt_template = ChatPromptTemplate.from_messages(
            [
                ("system", EXTRACTION_PROMPT),
                ("placeholder", "{messages}"),
            ]
        )

        chain = prompt_template | llm

        response = chain.invoke(state.messages)

        # Ensure the response is a valid JSON
        try:
            structured_question = StructuredQuestion.model_validate_json(response.content)
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to parse LLM response as JSON: {e}")
        except ValidationError as e:
            raise ValueError(f"Failed to validate structured question: {e}")

        return {
            "structured_question": structured_question,
            "steps": [
                StepOutput(
                    label=f"Extracted {len(structured_question.question_steps)} steps and {len(structured_question.extracted_classes)} classes",
                    details=f"""Intent: {structured_question.intent.replace("_", " ")}

    Steps to answer the user question:

    {chr(10).join(f"- {step}" for step in structured_question.question_steps)}

    Potential classes:

    {chr(10).join(f"- {cls}" for cls in structured_question.extracted_classes)}

    Potential entities:

    {chr(10).join(f"- {entity}" for entity in structured_question.extracted_entities)}""",
                )
            ],
        }
    except Exception as e:
        return {
            "error": str(e),
            "steps": [
                StepOutput(
                    label="Error in question understanding",
                    details=f"Failed to process question: {str(e)}",
                    type="fix-message"
                )
            ]
        }    