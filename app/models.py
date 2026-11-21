from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class SentimentEnum(str, Enum):
    """Sentiment categories for call transcripts."""
    POSITIVE = "Positive"
    NEUTRAL = "Neutral"
    NEGATIVE = "Negative"


# ==============================================================================
# Health Check Models
# ==============================================================================
class HealthResponse(BaseModel):
    status: str = Field(..., description="Service health status", examples=["healthy"])
    service: str = Field(..., description="Service identifier", examples=["ai-crm-voice-agent-api"])
    version: str = Field(..., description="Service version", examples=["1.0.0"])
    primary_model: str = Field(..., description="Primary Gemini model configured")
    fallback_model: str = Field(..., description="Fallback Gemini model configured")
    api_key_configured: bool = Field(..., description="Whether a live Gemini API key is set")


# ==============================================================================
# Call Transcript Analyzer Models
# ==============================================================================
class CallAnalysisRequest(BaseModel):
    transcript: str = Field(
        ...,
        min_length=10,
        description="The customer call audio transcript text to analyze.",
        examples=[
            "Agent: Hello, thank you for calling Fortivex CRM. How can I help you today?\n"
            "Customer: Hi! I am Sarah from TechCorp. We currently have 35 sales reps on HubSpot, "
            "and our call logging is a mess. We are looking for an AI voice solution that automatically "
            "transcribes calls and scores leads. We'd love to see a demo tomorrow at 2 PM if possible."
        ]
    )


class CallAnalysisSchema(BaseModel):
    """Pydantic schema used for Gemini structured JSON output."""
    summary: str = Field(
        ...,
        description="A concise 2-3 sentence executive summary of the conversation and customer needs."
    )
    sentiment: SentimentEnum = Field(
        ...,
        description="Overall customer sentiment: Positive, Neutral, or Negative."
    )
    lead_score: int = Field(
        ...,
        ge=1,
        le=10,
        description="Lead qualification score on a scale from 1 (very cold/unqualified) to 10 (hot prospect/ready to buy)."
    )
    recommended_action: str = Field(
        ...,
        description="Specific actionable next step for the sales or customer success team."
    )
    key_pain_points: List[str] = Field(
        default_factory=list,
        description="List of primary pain points or requirements mentioned by the caller."
    )
    followup_required: bool = Field(
        default=True,
        description="Flag indicating if immediate follow-up is necessary."
    )


class CallAnalysisResponse(CallAnalysisSchema):
    """API response model for call transcript analysis."""
    model_used: str = Field(..., description="Gemini model that performed the analysis.")


# ==============================================================================
# Knowledge Base (RAG) Models
# ==============================================================================
class KnowledgeQueryRequest(BaseModel):
    question: str = Field(
        ...,
        min_length=3,
        description="Customer or voice agent question to answer using the Fortivex CRM knowledge base.",
        examples=["What are the differences between the Starter and Growth plans, and what are the voice minute overage fees?"]
    )


class KnowledgeQueryResponse(BaseModel):
    question: str = Field(..., description="Original user question.")
    answer: str = Field(..., description="Concise, grounded voice-agent answer.")
    sources: List[str] = Field(..., description="Retrieved sections or context chunks used to ground the answer.")
    model_used: str = Field(..., description="Model used to formulate the grounded voice response.")


# ==============================================================================
# Agentic Tool Calling Models
# ==============================================================================
class AgentChatRequest(BaseModel):
    message: str = Field(
        ...,
        min_length=2,
        description="Customer voice or chat message.",
        examples=["Hi, I would like to schedule a product demo for Sarah Connor tomorrow at 3 PM to discuss Enterprise pricing."]
    )
    customer_name: Optional[str] = Field(
        None,
        description="Optional customer name if already identified by caller ID or telephony session."
    )


class ToolExecutionDetails(BaseModel):
    tool_name: str = Field(..., description="Name of the invoked Python tool.")
    tool_args: Dict[str, Any] = Field(..., description="Arguments extracted and passed into the tool.")
    tool_result: Dict[str, Any] = Field(..., description="Structured response returned by the Python tool.")


class AgentChatResponse(BaseModel):
    reply: str = Field(..., description="Conversational voice-ready response to the user.")
    tool_called: bool = Field(default=False, description="Whether an agentic tool was executed.")
    tool_execution: Optional[ToolExecutionDetails] = Field(
        None,
        description="Details of the tool execution if a function was invoked."
    )
    model_used: str = Field(..., description="Gemini model that executed the agent conversation.")
