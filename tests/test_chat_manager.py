import pytest
from unittest.mock import MagicMock
from app.chat import ChatManager


@pytest.mark.asyncio
async def test_chat_with_groq_success(monkeypatch):
    config = MagicMock()
    config.groq_api_key = "test_key"
    config.rps_limit = 1
    modes = MagicMock()
    modes.modes = {"friend": "system prompt"}
    logger = MagicMock()

    class MockCompletion:
        def __init__(self):
            self.choices = [MagicMock(message=MagicMock(content="Hello!"))]

    class MockCompletions:
        @staticmethod
        def create(**kwargs):  # <-- changed to sync method here
            return MockCompletion()

    class MockChat:
        completions = MockCompletions()

    class MockClient:
        chat = MockChat()

    monkeypatch.setattr("app.chat.OpenAI", lambda **kwargs: MockClient())

    chat_manager = ChatManager(config, modes, logger)
    reply = await chat_manager.chat_with_groq("user1", "friend", "Hi!")

    assert reply == "Hello!"
