import json
import logging
import re
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from google import genai
from google.genai import types

from app.config import settings
from app.models import (
    AgentChatResponse,
    CallAnalysisResponse,
    CallAnalysisSchema,
    SentimentEnum,
    ToolExecutionDetails,
)

logger = logging.getLogger(__name__)

# In-memory booking registry for confirmed follow-up calls
BOOKINGS_REGISTRY: List[Dict[str, Any]] = []


# ==============================================================================
# Agentic Python Tool Definition
# ==============================================================================
def schedule_followup_call(
    customer_name: str = "Valued Prospect",
    preferred_time: str = "Tomorrow at 10:00 AM EST",
    topic: str = "Fortivex CRM Consultation",
) -> dict:
    """Schedules a followup call or demo for a customer.

    Args:
        customer_name: Full name of the customer or lead.
        preferred_time: Requested date, time, or window for the call (e.g. 'Tomorrow at 3 PM', '2026-11-22 14:00').
        topic: Purpose or topic of the followup meeting (e.g. 'Product Demo', 'Enterprise Pricing', 'Growth Plan').

    Returns:
        dict: Confirmed booking details including booking_id, customer, scheduled time, and calendar event status.
    """
    booking_id = f"BK-{uuid.uuid4().hex[:8].upper()}"
    booking_record = {
        "status": "confirmed",
        "booking_id": booking_id,
        "customer_name": customer_name,
        "preferred_time": preferred_time,
        "topic": topic,
        "calendar_event_created": True,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "confirmation_message": (
            f"Follow-up call confirmed for {customer_name} on {preferred_time} "
            f"regarding '{topic}'. Reference ID: {booking_id}."
        ),
    }
    BOOKINGS_REGISTRY.append(booking_record)
    logger.info("Executed schedule_followup_call: %s", booking_record)
    return booking_record


# ==============================================================================
# AI Service Class
# ==============================================================================
class AIService:
    """Service wrapping Google Gemini models for structured call analysis

    and agentic tool execution using the official google-genai SDK.
    """

    def __init__(self):
        self.primary_model = settings.GEMINI_MODEL
        self.fallback_model = settings.GEMINI_FALLBACK_MODEL

    def _get_genai_client(self, timeout_ms: Optional[int] = None) -> genai.Client:
        """Initialize Google GenAI client with configured API key and timeout/retry options."""
        effective_timeout = timeout_ms or settings.GEMINI_TIMEOUT_MS
        return genai.Client(
            api_key=settings.GEMINI_API_KEY,
            http_options=types.HttpOptions(
                timeout=effective_timeout,
                retry_options=types.HttpRetryOptions(attempts=settings.GEMINI_MAX_RETRIES),
            ),
        )

    # --------------------------------------------------------------------------
    # 1. Call Transcript Analyzer (Structured Output)
    # --------------------------------------------------------------------------
    def analyze_call_transcript(self, transcript: str) -> CallAnalysisResponse:
        """Analyzes a call transcript to extract structured JSON data via Gemini.

        Extracts summary, sentiment (Positive, Neutral, Negative), lead_score (1-10),
        and recommended_action.
        """
        if not settings.is_api_key_configured:
            logger.info("GEMINI_API_KEY not configured. Running offline deterministic analysis.")
            return self._analyze_transcript_offline(transcript)

        models_to_try = [self.primary_model, self.fallback_model, "gemini-3.1-flash-lite"]
        last_error = None
        client = self._get_genai_client()

        system_prompt = (
            "You are an expert revenue operations and sales enablement AI for Fortivex CRM. "
            "Analyze the provided telephone call transcript between an agent and a prospective lead or customer. "
            "Extract a concise summary, classify overall sentiment (Positive, Neutral, Negative), "
            "evaluate lead qualification score on a strict scale from 1 (unqualified/lost) to 10 (hot prospect/ready to buy), "
            "and suggest the immediate recommended action for the sales or customer success team."
        )

        user_content = f"TRANSCRIPT TO ANALYZE:\n{transcript}"

        for model_name in models_to_try:
            try:
                logger.info("Analyzing transcript with model: %s", model_name)
                response = client.models.generate_content(
                    model=model_name,
                    contents=user_content,
                    config=types.GenerateContentConfig(
                        system_instruction=system_prompt,
                        temperature=0.1,
                        response_mime_type="application/json",
                        response_schema=CallAnalysisSchema,
                        http_options=types.HttpOptions(
                            timeout=settings.GEMINI_TIMEOUT_MS,
                            retry_options=types.HttpRetryOptions(attempts=settings.GEMINI_MAX_RETRIES),
                        ),
                    ),
                )

                if response and response.text:
                    raw_text = response.text.strip()
                    if raw_text.startswith("```"):
                        raw_text = re.sub(r"^```(?:json)?\s*", "", raw_text)
                        raw_text = re.sub(r"\s*```$", "", raw_text)
                    parsed_data = json.loads(raw_text)
                    return CallAnalysisResponse(
                        summary=parsed_data.get("summary", ""),
                        sentiment=SentimentEnum(parsed_data.get("sentiment", "Neutral")),
                        lead_score=int(parsed_data.get("lead_score", 5)),
                        recommended_action=parsed_data.get("recommended_action", ""),
                        key_pain_points=parsed_data.get("key_pain_points", []),
                        followup_required=parsed_data.get("followup_required", True),
                        model_used=model_name,
                    )
            except Exception as e:
                logger.warning("Transcript analysis failed on model %s: %s. Trying fallback...", model_name, e)
                last_error = e

        logger.error("All Gemini models failed for transcript analysis: %s. Falling back to local analyzer.", last_error)
        return self._analyze_transcript_offline(transcript, error_note=str(last_error))

    def _analyze_transcript_offline(self, transcript: str, error_note: str = "") -> CallAnalysisResponse:
        """Deterministic offline rule-based analysis for unit tests or fallback."""
        lower_text = transcript.lower()

        # Sentiment heuristics
        pos_words = ["love", "great", "interested", "demo", "awesome", "perfect", "excited", "definitely", "yes"]
        neg_words = ["cancel", "unhappy", "bad", "expensive", "hate", "competitor", "problem", "broken", "no"]

        pos_count = sum(1 for w in pos_words if w in lower_text)
        neg_count = sum(1 for w in neg_words if w in lower_text)

        if pos_count > neg_count + 1:
            sentiment = SentimentEnum.POSITIVE
            lead_score = min(10, 7 + pos_count)
            action = "Schedule high-priority product demo and assign senior account executive."
        elif neg_count > pos_count + 1:
            sentiment = SentimentEnum.NEGATIVE
            lead_score = max(1, 4 - neg_count)
            action = "Escalate to Customer Success Manager for churn mitigation and feedback review."
        else:
            sentiment = SentimentEnum.NEUTRAL
            lead_score = 5
            action = "Send follow-up product overview email and schedule standard discovery call."

        # Extract pain points
        pain_points = []
        if "sync" in lower_text or "hubspot" in lower_text or "salesforce" in lower_text:
            pain_points.append("CRM integration and manual call logging friction")
        if "pricing" in lower_text or "cost" in lower_text:
            pain_points.append("Pricing tiers and voice minute overage concerns")
        if "ai" in lower_text or "transcrib" in lower_text:
            pain_points.append("Automated transcription and call sentiment tracking")
        if not pain_points:
            pain_points.append("General inquiry regarding Fortivex CRM voice agent capabilities")

        first_sentence = transcript.strip().split("\n")[0][:120]
        summary = (
            f"Customer engaged in discussion regarding Fortivex CRM capabilities. "
            f"Key focus on voice features and CRM workflow integration: {first_sentence}..."
        )

        model_name = "offline-structured-analyzer" if not error_note else f"offline-analyzer (fallback: {error_note[:40]})"

        return CallAnalysisResponse(
            summary=summary,
            sentiment=sentiment,
            lead_score=lead_score,
            recommended_action=action,
            key_pain_points=pain_points,
            followup_required=True,
            model_used=model_name,
        )

    # --------------------------------------------------------------------------
    # 2. Agentic Function / Tool Calling
    # --------------------------------------------------------------------------
    def agent_chat(self, message: str, customer_name: Optional[str] = None) -> AgentChatResponse:
        """Executes conversational agent turn with automatic tool calling.

        If the user requests a callback or demo booking, Gemini invokes
        `schedule_followup_call` and produces the confirmed response.
        """
        if not settings.is_api_key_configured:
            logger.info("GEMINI_API_KEY not configured. Running offline agent simulation.")
            return self._agent_chat_offline(message, customer_name)

        models_to_try = [self.primary_model, self.fallback_model, "gemini-3.1-flash-lite"]
        last_error = None
        client = self._get_genai_client()

        system_instruction = (
            "You are an AI Voice & CRM Agent for Fortivex CRM. "
            "You speak directly with prospective customers and existing clients over the telephone. "
            "Always be courteous, concise, and helpful. "
            "If the customer expresses interest in scheduling a demo, booking a call, meeting with a sales rep, "
            "or receiving a callback, you MUST invoke the `schedule_followup_call` function with their name, "
            "preferred date/time, and topic. "
            "If the customer's name is not explicitly in the message, check if it was provided in the session or ask politely, "
            "or use 'Valued Prospect' as the default customer_name. "
            "Once the function executes, confirm the appointment clearly with the customer in 1 to 2 spoken sentences."
        )

        full_prompt = message
        if customer_name:
            full_prompt = f"[Known Telephony Caller Name: {customer_name}]\nCustomer says: {message}"

        for model_name in models_to_try:
            try:
                logger.info("Running agent turn with model: %s", model_name)
                # Initiate a chat session with automatic function calling enabled for the tool
                chat = client.chats.create(
                    model=model_name,
                    config=types.GenerateContentConfig(
                        system_instruction=system_instruction,
                        temperature=0.2,
                        tools=[schedule_followup_call],
                        http_options=types.HttpOptions(
                            timeout=settings.GEMINI_TIMEOUT_MS,
                            retry_options=types.HttpRetryOptions(attempts=settings.GEMINI_MAX_RETRIES),
                        ),
                    ),
                )
                response = chat.send_message(full_prompt)

                tool_called = False
                tool_details: Optional[ToolExecutionDetails] = None

                # 1. Inspect chat history for executed tool calls and responses (via Automatic Function Calling)
                history = chat.get_history() if hasattr(chat, "get_history") else []
                for content in history:
                    for part in content.parts:
                        if part.function_call and part.function_call.name == "schedule_followup_call":
                            tool_called = True
                            call_args = dict(part.function_call.args) if part.function_call.args else {}
                            if not call_args.get("customer_name") and customer_name:
                                call_args["customer_name"] = customer_name
                            if not tool_details:
                                tool_details = ToolExecutionDetails(
                                    tool_name="schedule_followup_call",
                                    tool_args=call_args,
                                    tool_result={},
                                )
                            else:
                                tool_details.tool_name = "schedule_followup_call"
                                tool_details.tool_args = call_args

                        if part.function_response and part.function_response.name == "schedule_followup_call":
                            resp_data = dict(part.function_response.response) if part.function_response.response else {}
                            res_val = resp_data.get("result", resp_data)
                            if tool_details:
                                tool_details.tool_result = res_val
                            else:
                                tool_called = True
                                tool_details = ToolExecutionDetails(
                                    tool_name="schedule_followup_call",
                                    tool_args={},
                                    tool_result=res_val,
                                )

                # 2. Inspect automatic_function_calling_history on response if present
                afc_history = getattr(response, "automatic_function_calling_history", None)
                if not tool_called and afc_history:
                    for content in afc_history:
                        for part in content.parts:
                            if part.function_call and part.function_call.name == "schedule_followup_call":
                                tool_called = True
                                call_args = dict(part.function_call.args) if part.function_call.args else {}
                                if not call_args.get("customer_name") and customer_name:
                                    call_args["customer_name"] = customer_name
                                if not tool_details:
                                    tool_details = ToolExecutionDetails(
                                        tool_name="schedule_followup_call",
                                        tool_args=call_args,
                                        tool_result={},
                                    )
                                else:
                                    tool_details.tool_name = "schedule_followup_call"
                                    tool_details.tool_args = call_args

                            if part.function_response and part.function_response.name == "schedule_followup_call":
                                resp_data = dict(part.function_response.response) if part.function_response.response else {}
                                res_val = resp_data.get("result", resp_data)
                                if tool_details:
                                    tool_details.tool_result = res_val
                                else:
                                    tool_called = True
                                    tool_details = ToolExecutionDetails(
                                        tool_name="schedule_followup_call",
                                        tool_args={},
                                        tool_result=res_val,
                                    )

                # 3. Handle manual function calls if AFC was bypassed or unhandled
                if not tool_called and response.function_calls:
                    for function_call in response.function_calls:
                        if function_call.name == "schedule_followup_call":
                            tool_called = True
                            call_args = dict(function_call.args) if function_call.args else {}
                            if not call_args.get("customer_name") and customer_name:
                                call_args["customer_name"] = customer_name
                            elif not call_args.get("customer_name"):
                                call_args["customer_name"] = "Valued Prospect"
                            if not call_args.get("preferred_time"):
                                call_args["preferred_time"] = "Tomorrow at 10:00 AM EST"
                            if not call_args.get("topic"):
                                call_args["topic"] = "Fortivex CRM Product Demo"

                            tool_result = schedule_followup_call(
                                customer_name=call_args["customer_name"],
                                preferred_time=call_args["preferred_time"],
                                topic=call_args["topic"],
                            )
                            tool_details = ToolExecutionDetails(
                                tool_name="schedule_followup_call",
                                tool_args=call_args,
                                tool_result=tool_result,
                            )

                # 4. Fallback for tool_result if tool was called but result dict was empty
                if tool_called and tool_details and not tool_details.tool_result and BOOKINGS_REGISTRY:
                    tool_details.tool_result = BOOKINGS_REGISTRY[-1]

                # Determine the final spoken reply
                if response and response.text:
                    final_reply = response.text.strip()
                elif tool_details and tool_details.tool_result and "confirmation_message" in tool_details.tool_result:
                    final_reply = tool_details.tool_result["confirmation_message"]
                else:
                    final_reply = "Thank you for reaching out to Fortivex CRM. How can I assist you further?"

                return AgentChatResponse(
                    reply=final_reply,
                    tool_called=tool_called,
                    tool_execution=tool_details,
                    model_used=model_name,
                )

            except Exception as e:
                logger.warning("Agent chat failed on model %s: %s. Trying fallback...", model_name, e)
                last_error = e

        logger.error("All Gemini models failed for agent chat: %s. Falling back to offline simulator.", last_error)
        return self._agent_chat_offline(message, customer_name, error_note=str(last_error))

    def _agent_chat_offline(
        self,
        message: str,
        customer_name: Optional[str] = None,
        error_note: str = ""
    ) -> AgentChatResponse:
        """Deterministic agent fallback that simulates intent detection and tool calling."""
        lower_msg = message.lower()
        booking_triggers = ["schedule", "book", "demo", "callback", "call back", "follow up", "follow-up", "appointment", "meeting"]

        should_book = any(trigger in lower_msg for trigger in booking_triggers)

        if should_book:
            extracted_name = customer_name or "Valued Prospect"
            name_match = re.search(r"(?:for|i am|i'm|name is)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)", message)
            if name_match and not customer_name:
                extracted_name = name_match.group(1)

            # Extract time
            time_match = re.search(r"(tomorrow(?:\s+at\s+[0-9]+(?::[0-9]+)?\s*(?:am|pm)?)?|next\s+\w+|at\s+[0-9]+(?::[0-9]+)?\s*(?:am|pm)?)", lower_msg)
            pref_time = time_match.group(1).title() if time_match else "Tomorrow at 2:00 PM EST"

            # Extract topic
            topic = "Fortivex CRM Product Demo"
            if "pricing" in lower_msg:
                topic = "Enterprise Pricing & Plan Consultation"
            elif "support" in lower_msg:
                topic = "Technical Integration & Support Review"

            tool_result = schedule_followup_call(
                customer_name=extracted_name,
                preferred_time=pref_time,
                topic=topic,
            )

            reply = (
                f"I have successfully scheduled your follow-up meeting for {extracted_name} "
                f"at {pref_time} regarding '{topic}'. A calendar invitation and confirmation have been dispatched. "
                f"Booking reference: {tool_result['booking_id']}."
            )

            model_name = "offline-agentic-simulator" if not error_note else f"offline-simulator (fallback: {error_note[:40]})"

            return AgentChatResponse(
                reply=reply,
                tool_called=True,
                tool_execution=ToolExecutionDetails(
                    tool_name="schedule_followup_call",
                    tool_args={
                        "customer_name": extracted_name,
                        "preferred_time": pref_time,
                        "topic": topic,
                    },
                    tool_result=tool_result,
                ),
                model_used=model_name,
            )

        reply = (
            "Hello! I am the Fortivex CRM AI Voice Agent. I can assist you with product information, "
            "pricing plan details, or book a live product demo and callback with our sales team. "
            "How can I assist you today?"
        )
        model_name = "offline-agentic-simulator" if not error_note else f"offline-simulator (fallback: {error_note[:40]})"

        return AgentChatResponse(
            reply=reply,
            tool_called=False,
            tool_execution=None,
            model_used=model_name,
        )


# Global singleton instance
ai_service = AIService()
