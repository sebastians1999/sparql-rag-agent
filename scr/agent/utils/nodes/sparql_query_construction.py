
from state import State
from config import RunnerConfig
from langchain_together import Together
from langchain_core.messages import AIMessage
from langchain_core.prompts import ChatPromptTemplate
from prompts import QUERY_GENERATION_PROMPT


async def query_generator(state: State, config: RunnerConfig) -> dict[str, List[AIMessage]]:


 llm = Together(model=config.llm_config.model_name, temperature=config.llm_config.temperature)

 prompt_template = ChatPromptTemplate.from_messages(
        [
            ("system", QUERY_GENERATION_PROMPT),
            ("placeholder", "{question} {potential_entities} {retrieved_documents}"),
        ]
    )

 message = await prompt_template.invoke(
        {
            "question": state["structured_question"].question,
            "potential_entities": state["structured_question"].extracted_entities,
            "retrieved_documents": state["retrieved_documents"],
        }
    )
 






#need a 