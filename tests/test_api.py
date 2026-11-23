import pytest
from fastapi.testclient import TestClient

from app.ai_service import schedule_followup_call, BOOKINGS_REGISTRY
from app.main import app
from app.rag_service import rag_service

client = TestClient(app)


def test_health_check():
    """Verify health endpoint returns status 200 and expected metadata."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["service"] == "ai-crm-voice-agent-api"
    assert "gemini-2.5-flash" in data["primary_model"]
    assert "gemini-2.5-flash-lite" in data["fallback_model"]


def test_rag_service_chunks_loaded():
    """Verify knowledge base chunks were parsed from data/knowledge.txt."""
    assert len(rag_service.chunks) > 0
    chunk_titles = [c.title for c in rag_service.chunks]
    assert any("PRICING" in title or "PLAN" in title for title in chunk_titles)


def test_ask_kb_endpoint():
    """Verify POST /api/v1/ask-kb retrieves context and returns a grounded answer."""
    payload = {
        "question": "What is the price of the Starter plan and how many voice minutes are included?"
    }
    response = client.post("/api/v1/ask-kb", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["question"] == payload["question"]
    assert len(data["answer"]) > 10
    assert len(data["sources"]) > 0
    assert "model_used" in data


def test_ask_kb_validation_error():
    """Verify POST /api/v1/ask-kb rejects empty or too-short questions with 422."""
    response = client.post("/api/v1/ask-kb", json={"question": "hi"})
    assert response.status_code == 422
    assert "detail" in response.json()


def test_analyze_call_endpoint():
    """Verify POST /api/v1/analyze-call extracts structured summary, sentiment, and lead score."""
    transcript = (
        "Agent: Welcome to Fortivex CRM sales. My name is Alex.\n"
        "Customer: Hi Alex, I am Marcus from Apex Logistics. We are managing 50 reps and currently using "
        "spreadsheets and basic VoIP. We urgently need an AI voice agent to qualify inbound leads "
        "and push them into HubSpot. Can we book a deep dive demo tomorrow afternoon?"
    )
    response = client.post("/api/v1/analyze-call", json={"transcript": transcript})
    assert response.status_code == 200
    data = response.json()
    assert "summary" in data and len(data["summary"]) > 10
    assert data["sentiment"] in ["Positive", "Neutral", "Negative"]
    assert 1 <= data["lead_score"] <= 10
    assert "recommended_action" in data and len(data["recommended_action"]) > 0
    assert isinstance(data["key_pain_points"], list)
    assert "model_used" in data


def test_analyze_call_validation_error():
    """Verify POST /api/v1/analyze-call rejects invalid/empty transcripts with 422."""
    response = client.post("/api/v1/analyze-call", json={"transcript": "too short"})
    assert response.status_code == 422


def test_direct_python_tool_execution():
    """Verify schedule_followup_call directly registers a booking."""
    initial_count = len(BOOKINGS_REGISTRY)
    result = schedule_followup_call(
        customer_name="Alice Smith",
        preferred_time="Friday at 11:00 AM",
        topic="Enterprise HIPAA BAA Review"
    )
    assert result["status"] == "confirmed"
    assert result["customer_name"] == "Alice Smith"
    assert result["booking_id"].startswith("BK-")
    assert result["calendar_event_created"] is True
    assert len(BOOKINGS_REGISTRY) == initial_count + 1


def test_agent_chat_trigger_tool():
    """Verify POST /api/v1/agent-chat triggers schedule_followup_call when demo/callback requested."""
    payload = {
        "message": "Hi, I am David Miller. Can we schedule a demo call tomorrow at 3 PM to review pricing?",
        "customer_name": "David Miller"
    }
    response = client.post("/api/v1/agent-chat", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "reply" in data and len(data["reply"]) > 10
    assert data["tool_called"] is True
    assert data["tool_execution"] is not None
    assert data["tool_execution"]["tool_name"] == "schedule_followup_call"
    assert data["tool_execution"]["tool_result"]["status"] == "confirmed"
    assert "BK-" in data["tool_execution"]["tool_result"]["booking_id"]


def test_agent_chat_general_inquiry():
    """Verify POST /api/v1/agent-chat answers without tool calling for general greeting."""
    payload = {
        "message": "Hello, what does your service do?"
    }
    response = client.post("/api/v1/agent-chat", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "reply" in data and len(data["reply"]) > 5
    assert "model_used" in data
