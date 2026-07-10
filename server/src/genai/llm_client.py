from google import genai
from google.genai import types
from google.oauth2 import service_account
from typing import List, Dict, Optional
from src.core.config import Config


class LLMClient:

    def __init__(self):
        self.model_name = "gemini-2.5-flash"
        self.embedding_model_name = "gemini-embedding-001"
        self.client = None

        try:
            credentials = service_account.Credentials.from_service_account_info(
                {
                    "type": "service_account",
                    "project_id": Config.GOOGLE_PROJECT_ID,
                    "private_key": Config.GOOGLE_PRIVATE_KEY,
                    "client_email": Config.GOOGLE_CLIENT_EMAIL,
                    "token_uri": "https://oauth2.googleapis.com/token",
                },
                scopes=["https://www.googleapis.com/auth/cloud-platform"],
            )
            self.client = genai.Client(
                vertexai=True,
                project=Config.GOOGLE_PROJECT_ID,
                location=Config.GOOGLE_CLOUD_LOCATION,
                credentials=credentials,
            )
            print(f"✓ LLM Client initialized successfully (Vertex AI, {self.model_name})")
        except Exception as e:
            print(f"!!! ERROR: Failed to initialize Vertex AI client: {e}")

        # Back-compat truthiness flag used elsewhere as "is the LLM available".
        self.text_model = self.client

    def call_llm(self, messages: List[Dict], temperature: float = 0.7) -> str:
        if not self.client:
            raise Exception("LLM client not initialized. Check Vertex AI config.")

        try:
            system_instruction = None
            contents = []
            for msg in messages:
                role = msg["role"]
                content = msg["content"]
                if role == "system":
                    system_instruction = content
                elif role == "user":
                    contents.append(types.Content(role="user", parts=[types.Part(text=content)]))
                elif role == "assistant":
                    contents.append(types.Content(role="model", parts=[types.Part(text=content)]))

            config = types.GenerateContentConfig(
                temperature=temperature,
                top_p=0.95,
                top_k=40,
                max_output_tokens=8192,
                system_instruction=system_instruction,
                # gemini-2.5-flash "thinks" by default, adding several seconds.
                # These are grounded/structured tasks, so disable it for speed.
                thinking_config=types.ThinkingConfig(thinking_budget=0),
            )

            response = self.client.models.generate_content(
                model=self.model_name,
                contents=contents,
                config=config,
            )
            return (response.text or "").strip()

        except Exception as e:
            print(f"LLMClient Error in call_llm: {e}")
            raise Exception(f"LLM generation failed: {str(e)}")

    def generate_text(self, prompt: str, temperature: float = 0.7) -> str:
        if not self.client:
            return "Error: LLM model not available."

        try:
            config = types.GenerateContentConfig(
                temperature=temperature,
                top_p=0.95,
                top_k=40,
                max_output_tokens=8192,
                # Disable gemini-2.5-flash thinking to cut RAG answer latency.
                thinking_config=types.ThinkingConfig(thinking_budget=0),
            )
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config=config,
            )
            return (response.text or "").strip()

        except Exception as e:
            print(f"LLMClient Error during text generation: {e}")
            return "Error: Could not generate a response."

    async def generate_text_async(self, prompt: str) -> str:
        if not self.client:
            return "Error: LLM model not available."
        try:
            response = await self.client.aio.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    thinking_config=types.ThinkingConfig(thinking_budget=0),
                ),
            )
            return (response.text or "").strip()
        except Exception as e:
            return f"Error: {e}"

    def generate_embedding(self, text: str) -> Optional[List[float]]:
        if not self.client:
            print("Cannot generate embedding: Vertex AI client not configured.")
            return None

        if not text or not text.strip():
            return None

        try:
            result = self.client.models.embed_content(
                model=self.embedding_model_name,
                contents=text,
                config=types.EmbedContentConfig(task_type="RETRIEVAL_DOCUMENT"),
            )
            return result.embeddings[0].values

        except Exception as e:
            print(f"LLMClient Error during embedding generation: {e}")
            return None


# Create a single, shared instance
llm_client = LLMClient()
