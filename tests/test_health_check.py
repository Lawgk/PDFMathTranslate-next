"""The translator health check runs beside parsing instead of before it.

It is one LLM round trip, seconds long, and the only thing that turns a dead
translator (bad key, no quota) into a failed task: BabelDOC keeps the original
text of every paragraph it cannot translate and reports success.
"""

import asyncio
import json
import threading
import time
from concurrent.futures import Future
from http.server import BaseHTTPRequestHandler
from http.server import ThreadingHTTPServer
from types import SimpleNamespace

import pytest
from pdf2zh_next.config.model import SettingsModel
from pdf2zh_next.config.translate_engine_model import OpenAICompatibleSettings
from pdf2zh_next.high_level import _failed_health_check
from pdf2zh_next.high_level import _initialization_error
from pdf2zh_next.translator import get_translator

COMPLETION = {
    "id": "c",
    "object": "chat.completion",
    "created": 0,
    "model": "m",
    "choices": [
        {
            "index": 0,
            "finish_reason": "stop",
            "message": {"role": "assistant", "content": "你好"},
        }
    ],
    "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
}
NO_QUOTA = {"error": {"message": "无可用订阅或额度", "code": "quota_exceeded"}}


@pytest.fixture
def llm(monkeypatch):
    """A local OpenAI-compatible endpoint; tests set its delay and status."""
    monkeypatch.setenv("no_proxy", "127.0.0.1,localhost")
    monkeypatch.setenv("NO_PROXY", "127.0.0.1,localhost")
    state = SimpleNamespace(delay=0.0, status=200)

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            self.rfile.read(int(self.headers["Content-Length"]))
            time.sleep(state.delay)
            data = json.dumps(COMPLETION if state.status == 200 else NO_QUOTA).encode()
            self.send_response(state.status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    state.url = f"http://127.0.0.1:{server.server_address[1]}/v1"
    yield state
    server.shutdown()


def _settings(url: str) -> SettingsModel:
    settings = SettingsModel(
        translate_engine_settings=OpenAICompatibleSettings(
            openai_compatible_model="m",
            openai_compatible_base_url=url,
            openai_compatible_api_key="k",
        )
    )
    settings.validate_settings()
    return settings


def test_the_check_runs_before_returning_by_default(llm):
    llm.delay = 0.5
    start = time.monotonic()
    translator = get_translator(_settings(llm.url))
    assert time.monotonic() - start >= 0.5
    assert getattr(translator, "health_check", None) is None


def test_a_background_check_answers_later(llm):
    llm.delay = 1.0
    start = time.monotonic()
    translator = get_translator(_settings(llm.url), background_check=True)
    assert time.monotonic() - start < 0.5
    assert not translator.health_check.done()
    assert translator.health_check.result(timeout=10) is None


def test_a_failed_check_keeps_the_message_callers_classify(llm):
    llm.status = 402
    translator = get_translator(_settings(llm.url), background_check=True)
    error = translator.health_check.exception(timeout=10)
    assert error is not None
    # pdf-service reads "Error code: 402" out of this text as QUOTA_EXCEEDED.
    message = _initialization_error(error).raw_message
    assert message.startswith("Translation subprocess initialization error: ")
    assert "Error code: 402" in message


def _config(check: Future | None) -> SimpleNamespace:
    translator = SimpleNamespace(health_check=check)
    return SimpleNamespace(translator=translator, term_extraction_translator=translator)


def test_a_pending_check_holds_up_only_the_outcome():
    check = Future()
    config = _config(check)
    assert asyncio.run(_failed_health_check(config, wait=False)) is None

    def fail_soon():
        time.sleep(0.2)
        check.set_exception(ConnectionError("Error code: 401"))

    threading.Thread(target=fail_soon).start()
    error = asyncio.run(_failed_health_check(config, wait=True))
    assert isinstance(error, ConnectionError)


def test_a_passed_or_absent_check_is_no_error():
    passed = Future()
    passed.set_result(None)
    assert asyncio.run(_failed_health_check(_config(passed), wait=True)) is None
    assert asyncio.run(_failed_health_check(_config(None), wait=True)) is None
