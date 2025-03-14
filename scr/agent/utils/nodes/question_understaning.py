from state import State, StructuredQuestion, StepOutput
from config import RunnerConfig
from langchain_together import Together
from langchain_core.prompts import ChatPromptTemplate
from prompts import EXTRACTION_PROMPT
from typing import Dict, List
from langchain_core.messages import AIMessage




async def question_understanding(state: State, config:RunnerConfig) -> Dict[str, List[AIMessage]]:
    """Extract structured information from the user question."""

    llm = Together(model=config.llm_config.model_name, temperature=config.llm_config.temperature)

    prompt_template = ChatPromptTemplate.from_messages(
        [
            ("system", EXTRACTION_PROMPT),
            ("placeholder", "{messages}"),
        ]
    )

    chain = prompt_template | llm

    response = chain.invoke(state["messages"])

    #Ensure the response is a valid JSON
    structured_question = StructuredQuestion.model_validate_json(response.content)

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