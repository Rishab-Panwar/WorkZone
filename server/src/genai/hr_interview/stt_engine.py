from src.core.logger import logger
from google.genai import types
from src.genai.llm_client import llm_client

STT_MODEL = "gemini-2.5-flash"


async def speech_to_text(audio_data: bytes) -> str:
    if not llm_client.client:
        raise Exception("Vertex AI client not initialized.")

    try:
        logger.info(f"Starting Gemini transcription for {len(audio_data)} bytes")

        response = await llm_client.client.aio.models.generate_content(
            model=STT_MODEL,
            contents=[
                types.Part.from_bytes(data=audio_data, mime_type="audio/webm"),
                "Transcribe this audio clearly and accurately. Return only the transcribed text, nothing else.",
            ],
        )

        result = (response.text or "").strip()

        if len(result) < 1:
            logger.warning("No speech detected in audio")
            return "No speech detected"

        logger.info(f"Gemini transcription completed: {len(result)} chars")
        return result

    except Exception as e:
        logger.error(f"Gemini transcription failed: {e}")
        raise
