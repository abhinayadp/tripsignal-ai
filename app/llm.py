import json
import os
import urllib.request


class RuleBasedComposer:
    """Default no-key composer used for deterministic local demos."""

    name = "rule-based"

    def compose(self, system_prompt: str, user_prompt: str, draft: str) -> str:
        return draft


class OpenAICompatibleClient:
    """Minimal OpenAI-compatible chat client using only the standard library.

    Set LLM_API_KEY, LLM_BASE_URL, and LLM_MODEL to use a hosted or local
    model server. The rest of TripSignal keeps working without a key through
    RuleBasedComposer, which is important for reproducible evaluation.
    """

    name = "openai-compatible"

    def __init__(self):
        self.api_key = os.getenv("LLM_API_KEY") or os.getenv("OPENAI_API_KEY")
        self.base_url = os.getenv("LLM_BASE_URL", "https://api.openai.com/v1").rstrip("/")
        self.model = os.getenv("LLM_MODEL", "gpt-4o-mini")
        if not self.api_key:
            raise RuntimeError("LLM_API_KEY or OPENAI_API_KEY is not configured")

    def compose(self, system_prompt: str, user_prompt: str, draft: str) -> str:
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt + "\n\nDraft answer to polish without adding facts:\n" + draft},
            ],
            "temperature": 0.2,
        }
        request = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {self.api_key}"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=30) as response:
            data = json.loads(response.read().decode("utf-8"))
        return data["choices"][0]["message"]["content"]


def get_llm_client():
    if os.getenv("LLM_API_KEY") or os.getenv("OPENAI_API_KEY"):
        return OpenAICompatibleClient()
    return RuleBasedComposer()
