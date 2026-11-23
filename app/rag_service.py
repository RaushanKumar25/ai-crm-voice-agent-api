import logging
import re
from pathlib import Path
from typing import List, Tuple

from google import genai
from google.genai import types

from app.config import settings
from app.models import KnowledgeQueryResponse

logger = logging.getLogger(__name__)


class KnowledgeBaseChunk:
    """Represents a chunk of the Fortivex knowledge base."""

    def __init__(self, title: str, content: str):
        self.title = title
        self.content = content.strip()

    @property
    def full_text(self) -> str:
        return f"[{self.title}]\n{self.content}"


class RAGService:
    """Retrieval-Augmented Generation (RAG) service for Fortivex CRM.

    Loads the knowledge base, indexes chunks, performs keyword/semantic
    relevance scoring, and queries Gemini to produce grounded voice responses.
    """

    def __init__(self, kb_path: Path = settings.KNOWLEDGE_BASE_PATH):
        self.kb_path = kb_path
        self.chunks: List[KnowledgeBaseChunk] = []
        self._load_and_chunk_knowledge()

    def _load_and_chunk_knowledge(self) -> None:
        """Parse the knowledge base file into structured semantic sections."""
        if not self.kb_path.exists():
            logger.warning("Knowledge base file not found at %s. Creating empty index.", self.kb_path)
            return

        try:
            raw_text = self.kb_path.read_text(encoding="utf-8")
        except Exception as e:
            logger.error("Failed to read knowledge base file: %s", e)
            return

        # Split on major numbered or bulleted sections (e.g., "1. COMPANY OVERVIEW", "* STARTER PLAN")
        sections = re.split(r"\n(?=[0-9]+\.\s+[A-Z\s,]+|\*\s+[A-Z\s\(\)]+)", raw_text)
        for section in sections:
            section = section.strip()
            if not section or section.startswith("==="):
                continue

            lines = section.split("\n", 1)
            title = lines[0].strip().strip("*").strip()
            content = lines[1].strip() if len(lines) > 1 else lines[0].strip()
            self.chunks.append(KnowledgeBaseChunk(title=title, content=content))

        logger.info("Loaded %d knowledge base chunks from %s", len(self.chunks), self.kb_path)

    def retrieve_relevant_chunks(self, question: str, top_k: int = 3) -> List[Tuple[KnowledgeBaseChunk, float]]:
        """Retrieve the top-k most relevant chunks for a customer question."""
        if not self.chunks:
            return []

        # Tokenize and normalize question keywords (filter short punctuation/stop words)
        stop_words = {
            "a", "an", "the", "in", "on", "at", "for", "to", "of", "and", "or",
            "is", "are", "was", "were", "what", "how", "why", "when", "where",
            "who", "can", "could", "would", "do", "does", "did", "have", "has",
            "i", "you", "we", "they", "our", "my", "your", "it", "with", "about"
        }
        tokens = [
            t.lower() for t in re.findall(r"\b[A-Za-z0-9\$\.]+\b", question)
            if t.lower() not in stop_words and len(t) > 1
        ]

        scored_chunks: List[Tuple[KnowledgeBaseChunk, float]] = []

        for chunk in self.chunks:
            chunk_lower = chunk.full_text.lower()
            score = 0.0

            # Match exact question phrase
            if question.lower() in chunk_lower:
                score += 5.0

            # Title matching gives higher relevance
            title_lower = chunk.title.lower()
            for token in tokens:
                if token in title_lower:
                    score += 3.0
                if token in chunk_lower:
                    score += 1.0

            if score > 0:
                scored_chunks.append((chunk, score))

        # Sort descending by relevance score
        scored_chunks.sort(key=lambda x: x[1], reverse=True)

        # If no direct token match, return the first few general overview chunks
        if not scored_chunks:
            return [(chunk, 0.5) for chunk in self.chunks[:top_k]]

        return scored_chunks[:top_k]

    def _call_gemini_grounded(self, prompt: str, system_instruction: str) -> Tuple[str, str]:
        """Calls Gemini with primary model, falling back to flash-lite if needed."""
        models_to_try = [settings.GEMINI_MODEL, settings.GEMINI_FALLBACK_MODEL, "gemini-3.1-flash-lite"]
        last_error = None

        client = genai.Client(
            api_key=settings.GEMINI_API_KEY,
            http_options=types.HttpOptions(
                timeout=settings.GEMINI_TIMEOUT_MS,
                retry_options=types.HttpRetryOptions(attempts=settings.GEMINI_MAX_RETRIES),
            ),
        )

        for model_name in models_to_try:
            try:
                logger.info("Querying Gemini model: %s for KB retrieval", model_name)
                response = client.models.generate_content(
                    model=model_name,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        system_instruction=system_instruction,
                        temperature=0.1,  # Low temperature for strict grounding
                        max_output_tokens=300,
                        http_options=types.HttpOptions(
                            timeout=settings.GEMINI_TIMEOUT_MS,
                            retry_options=types.HttpRetryOptions(attempts=settings.GEMINI_MAX_RETRIES),
                        ),
                    ),
                )
                if response and response.text:
                    return response.text.strip(), model_name
            except Exception as e:
                logger.warning("Call to model %s failed: %s. Trying fallback...", model_name, e)
                last_error = e

        raise RuntimeError(f"All Gemini models failed: {last_error}")

    def answer_question(self, question: str) -> KnowledgeQueryResponse:
        """Finds relevant context and generates a grounded voice agent answer."""
        relevant_chunks_with_scores = self.retrieve_relevant_chunks(question, top_k=3)
        relevant_chunks = [c[0] for c in relevant_chunks_with_scores]
        sources = [c.title for c in relevant_chunks]

        context_text = "\n\n---\n\n".join([c.full_text for c in relevant_chunks])

        system_instruction = (
            "You are the voice assistant for Fortivex CRM. "
            "Use ONLY the provided retrieved context chunks to answer the caller's question. "
            "Your response will be spoken over a telephone line, so make it natural, crisp, polite, "
            "and concise (2 to 3 sentences maximum). Strictly ground your answer in the provided facts. "
            "Do NOT make up features, pricing, or policies. "
            "If the retrieved context does not contain the answer, say: "
            "'I don't have that specific information in my system right now, but I would be glad to "
            "connect you with our specialist team or schedule a callback to help you.'"
        )

        user_prompt = f"RETRIEVED CONTEXT:\n{context_text}\n\nCALLER QUESTION:\n{question}"

        # If API key is configured, call Gemini
        if settings.is_api_key_configured:
            try:
                answer, model_used = self._call_gemini_grounded(
                    prompt=user_prompt,
                    system_instruction=system_instruction,
                )
                return KnowledgeQueryResponse(
                    question=question,
                    answer=answer,
                    sources=sources,
                    model_used=model_used,
                )
            except Exception as e:
                logger.error("Error generating answer via Gemini: %s", e)
                # Fall back to grounded offline snippet if API fails
                return self._generate_offline_grounded_answer(question, relevant_chunks, sources, error_note=str(e))

        # Offline / dry-run mode
        return self._generate_offline_grounded_answer(question, relevant_chunks, sources)

    def _generate_offline_grounded_answer(
        self,
        question: str,
        relevant_chunks: List[KnowledgeBaseChunk],
        sources: List[str],
        error_note: str = ""
    ) -> KnowledgeQueryResponse:
        """Deterministic grounded fallback when live API key is unavailable or offline."""
        if not relevant_chunks:
            answer = (
                "I don't have that specific information in my system right now, but I would be glad to "
                "connect you with our specialist team or schedule a callback to help you."
            )
        else:
            top_chunk = relevant_chunks[0]
            summary_sentences = [s.strip() for s in top_chunk.content.split("\n") if s.strip() and not s.startswith("-")]
            sample = summary_sentences[0] if summary_sentences else top_chunk.content[:200]
            answer = f"According to Fortivex CRM documentation on {top_chunk.title}: {sample}"

        model_name = "offline-grounded-engine" if not error_note else f"offline-fallback-due-to-error: {error_note}"
        return KnowledgeQueryResponse(
            question=question,
            answer=answer,
            sources=sources,
            model_used=model_name,
        )


# Global singleton instance
rag_service = RAGService()
