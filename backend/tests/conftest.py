import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

SAMPLES = Path(__file__).resolve().parents[2] / "samples"


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("DOCMIND_DATA_DIR", str(tmp_path))
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    from app.main import app

    with TestClient(app) as c:
        yield c


@pytest.fixture
def samples():
    return SAMPLES
