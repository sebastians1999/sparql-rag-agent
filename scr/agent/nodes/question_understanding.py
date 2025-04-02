from scr.agent.state.state import State, StructuredQuestion, StepOutput
from langchain_together import Together
from langchain_core.prompts import ChatPromptTemplate
from scr.agent.prompts.prompts import EXTRACTION_PROMPT
from typing import Dict, Any, List
from langchain_core.messages import AIMessage
import json
from pydantic import ValidationError
from scr.agent.utils.config import Configuration
from langchain_core.runnables import RunnableConfig
from scr.agent.utils.llm_utils import get_llm

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


    if not state.messages:
        raise ValueError("No messages found in state")

    try:
        configuration = Configuration.from_runnable_config(config)
        llm = get_llm(configuration).with_structured_output(StructuredQuestion)

        prompt_template = ChatPromptTemplate.from_messages(
            [
                ("system", EXTRACTION_PROMPT),
                ("placeholder", "{messages}"),
            ]
        )

        message_value = await prompt_template.ainvoke(
            {
                "messages": state.messages,
            },
        )

        structured_question: StructuredQuestion = await llm.ainvoke(message_value)

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