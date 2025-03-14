from dataclasses import dataclass, field
from pydantic import BaseModel, Field
from typing import Literal, Optional, List, Dict, Any
from langgraph.managed import IsLastStep
from langchain_core.documents import Document
from typing import Annotated
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class StepOutput(BaseModel):
    """Represents a step the agent went through to generate the answer."""

    label: str
    """The human-readable title for this step to be displayed to the user."""

    details: str = Field(default="")
    """Details of the steps results in markdown to be displayed to the user. It can be either a markdown string or a list of StepOutput."""

    substeps: Optional[List["StepOutput"]] = Field(default_factory=list)
    """Optional substeps for a step."""

    type: Literal["context", "fix-message", "recall"] = Field(default="context")
    """The type of the step."""

    fixed_message: Optional[str] = None
    """The fixed message to replace the last message sent to the user."""


class StructuredQuestion(BaseModel):
    """Structured informations extracted from the user question."""

    intent: Literal["general_information", "access_resources"] = Field(
        default="access_resources",
        description="Intent extracted from the user question",
    )
    extracted_classes: List[str] = Field(
        default_factory=list,
        description="List of classes extracted from the user question",
    )
    extracted_entities: List[str] = Field(
        default_factory=list,
        description="List of entities extracted from the user question",
    )
    question_steps: List[str] = Field(
        default_factory=list,
        description="List of steps extracted from the user question",
    )


@dataclass
class BaseState:
    """Base state with messages field."""
    messages: Annotated[List[BaseMessage], add_messages]


@dataclass
class State(BaseState):
    """Represents the complete state of the agent, extending InputState with additional attributes.

    This class can be used to store any information needed throughout the agent's lifecycle.
    """

    is_last_step: IsLastStep = field(default=False)
    """Indicates whether the current step is the last one before the graph raises an error.

    This is a 'managed' variable, controlled by the state machine rather than user code.
    It is set to 'True' when the step count reaches recursion_limit - 1.
    """

    structured_question: StructuredQuestion = field(default_factory=StructuredQuestion)
    """Structured information extracted from the user's question."""

    retrieved_docs: List[Document] = field(default_factory=list)
    """Documents retrieved from the knowledge base."""

    extracted_entities: List[str] = field(default_factory=list)
    """List of entities extracted and resolved from the question."""

    structured_output: str = field(default="")
    """The final structured output (e.g., SPARQL query)."""

    steps: List[StepOutput] = field(default_factory=list)
    """List of steps taken during the agent's execution."""