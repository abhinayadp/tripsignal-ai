try:
    from fastapi import FastAPI
    from pydantic import BaseModel
except ImportError:  # pragma: no cover - CLI and stdlib server still work.
    FastAPI = None
    BaseModel = object

from .chatbot import TravelDisruptionAssistant

assistant = TravelDisruptionAssistant()


if FastAPI is not None:

    class ChatRequest(BaseModel):
        query: str

    app = FastAPI(title="TripSignal AI", version="0.2.0")

    @app.get("/health")
    def health():
        return {"status": "ok", "documents": len(assistant.documents)}

    @app.post("/chat")
    def chat(request: ChatRequest):
        return {"answer": assistant.answer(request.query)}

else:
    app = None
