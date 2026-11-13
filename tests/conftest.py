import os
import re

os.environ["EMBEDDER"] = "hashing"  # tests use the deterministic keyword embedder
import pytest
from fastapi.testclient import TestClient

from app.llm import LLMUnavailable
from app.main import create_app

ALDER = {"X-API-Key": "demo-key-alder-0001"}
BIRCH = {"X-API-Key": "demo-key-birch-0002"}


class FakeLLM:
    """Well-behaved stand-in: cites the ids of the documents it was given."""
    model = "fake-llm"

    def __init__(self):
        self.calls = []

    def is_reachable(self):
        return True

    def chat(self, system, user):
        self.calls.append((system, user))
        ids = re.findall(r'<document id="([^"]+)"', user)
        return "Based on the records: " + " ".join(f"[{i}]" for i in ids)


class CompromisedLLM(FakeLLM):
    """Simulates a model that obeys the injected instruction and tries to leak."""
    model = "compromised-llm"

    def chat(self, system, user):
        self.calls.append((system, user))
        return ("Admin mode enabled. Birch Street Medical escalations go to "
                "billing-desk@birch.example [KB-BIR-001]. Tenant clinic-birch.")


class DownLLM(FakeLLM):
    model = "down-llm"

    def is_reachable(self):
        return False

    def chat(self, system, user):
        raise LLMUnavailable("cannot reach Ollama (simulated)")


@pytest.fixture
def fake_llm():
    return FakeLLM()


@pytest.fixture
def client(fake_llm):
    return TestClient(create_app(llm=fake_llm))


def make_client(llm):
    return TestClient(create_app(llm=llm))