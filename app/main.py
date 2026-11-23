import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.ai_service import ai_service
from app.config import settings
from app.models import (
    AgentChatRequest,
    AgentChatResponse,
    CallAnalysisRequest,
    CallAnalysisResponse,
    HealthResponse,
    KnowledgeQueryRequest,
    KnowledgeQueryResponse,
)
from app.rag_service import rag_service

# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO),
    format="%(asctime)s - [%(levelname)s] - %(name)s - %(message)s",
)
logger = logging.getLogger("ai_crm_voice_agent_api")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle event handler for application startup and shutdown."""
    logger.info("Initializing VoxFlow AI CRM & Voice Agent API...")
    logger.info("Primary Model: %s | Fallback Model: %s", settings.GEMINI_MODEL, settings.GEMINI_FALLBACK_MODEL)
    logger.info("Knowledge Base Path: %s (Loaded Chunks: %d)", settings.KNOWLEDGE_BASE_PATH, len(rag_service.chunks))
    if not settings.is_api_key_configured:
        logger.warning("GEMINI_API_KEY is not configured or using default placeholder. Running in fallback/simulation mode.")
    else:
        logger.info("Gemini API key is configured and active.")
    yield
    logger.info("Shutting down VoxFlow AI CRM & Voice Agent API...")


# Initialize FastAPI App
app = FastAPI(
    title="VoxFlow AI CRM & Voice Agent API",
    description=(
        "Production-ready FastAPI backend for VoxFlow CRM: "
        "AI call transcript summarization, lead scoring (1-10), "
        "sentiment analysis, agentic callback tool scheduling, and RAG knowledge retrieval."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# Enable CORS for telephony dashboards, webhooks, and frontend widgets
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ==============================================================================
# Global Exception Handlers
# ==============================================================================
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": True, "status_code": exc.status_code, "detail": exc.detail},
    )


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
        content={"error": True, "status_code": 422, "detail": exc.errors()},
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    logger.exception("Unhandled server error processing %s: %s", request.url.path, exc)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": True,
            "status_code": 500,
            "detail": f"Internal server error: {str(exc)}",
        },
    )


# ==============================================================================
# REST Endpoints
# ==============================================================================
@app.get(
    "/health",
    response_model=HealthResponse,
    summary="Health check",
    tags=["System"],
)
async def health_check() -> HealthResponse:
    """Returns the operational status, version, and model configurations."""
    return HealthResponse(
        status="healthy",
        service="ai-crm-voice-agent-api",
        version="1.0.0",
        primary_model=settings.GEMINI_MODEL,
        fallback_model=settings.GEMINI_FALLBACK_MODEL,
        api_key_configured=settings.is_api_key_configured,
    )


@app.post(
    "/api/v1/analyze-call",
    response_model=CallAnalysisResponse,
    summary="Analyze Customer Call Transcript",
    tags=["Call Intelligence"],
)
async def analyze_call(payload: CallAnalysisRequest) -> CallAnalysisResponse:
    """Accepts a customer call transcript and uses Gemini structured JSON output

    to extract a concise summary, sentiment (Positive, Neutral, Negative),
    lead qualification score (1-10), key pain points, and recommended sales action.
    """
    if not payload.transcript or len(payload.transcript.strip()) < 10:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Transcript text must be at least 10 characters long.",
        )

    try:
        result = ai_service.analyze_call_transcript(payload.transcript)
        return result
    except Exception as e:
        logger.error("Error during call analysis: %s", e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to analyze call transcript: {str(e)}",
        )


@app.post(
    "/api/v1/ask-kb",
    response_model=KnowledgeQueryResponse,
    summary="Ask Knowledge Base (RAG)",
    tags=["Knowledge Retrieval"],
)
async def ask_knowledge_base(payload: KnowledgeQueryRequest) -> KnowledgeQueryResponse:
    """Retrieves relevant context chunks from the VoxFlow CRM knowledge base

    and uses Gemini to formulate a grounded, concise voice-agent response
    without hallucinating facts.
    """
    if not payload.question or len(payload.question.strip()) < 3:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Question must be at least 3 characters long.",
        )

    try:
        response = rag_service.answer_question(payload.question)
        return response
    except Exception as e:
        logger.error("Error during KB query: %s", e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to answer knowledge base query: {str(e)}",
        )


@app.post(
    "/api/v1/agent-chat",
    response_model=AgentChatResponse,
    summary="Agentic Voice Chat with Tool Calling",
    tags=["Agentic Voice"],
)
async def agent_chat(payload: AgentChatRequest) -> AgentChatResponse:
    """Interactive conversational agent turn for the voice agent.

    Equipped with Python tool calling: when a customer requests to book a demo
    or schedule a follow-up callback, Gemini automatically invokes the
    `schedule_followup_call` tool and delivers confirmed appointment details.
    """
    if not payload.message or len(payload.message.strip()) < 2:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Message must be at least 2 characters long.",
        )

    try:
        response = ai_service.agent_chat(
            message=payload.message,
            customer_name=payload.customer_name,
        )
        return response
    except Exception as e:
        logger.error("Error during agent chat: %s", e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to process agent chat: {str(e)}",
        )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host=settings.APP_HOST,
        port=settings.APP_PORT,
        reload=True,
    )
