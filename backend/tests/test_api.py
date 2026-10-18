"""REST surface and the WebSocket handshake."""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


class TestRest:
    def test_health_reports_active_backends(self, client):
        r = client.get("/api/health")
        assert r.status_code == 200
        body = r.json()
        assert body["status"] == "ok"
        assert set(body["backends"]) >= {"asr", "brain", "tts", "delivery_scorer"}
        assert body["latency_budget_ms"] > 0

    def test_create_session(self, client):
        r = client.post("/api/sessions", json={"track": "dsa"})
        assert r.status_code == 200
        body = r.json()
        assert body["track"] == "dsa"
        assert body["ws_url"].startswith("/ws/")

    def test_invalid_track_rejected(self, client):
        assert client.post("/api/sessions", json={"track": "underwater"}).status_code == 400

    def test_jd_track_requires_text(self, client):
        assert client.post("/api/sessions", json={"track": "jd"}).status_code == 400
        r = client.post("/api/sessions", json={"track": "jd", "jd_text": "Senior backend role."})
        assert r.status_code == 200

    def test_missing_report_404s(self, client):
        assert client.get("/api/sessions/nope/report").status_code == 404

    def test_list_sessions(self, client):
        client.post("/api/sessions", json={"track": "behavioral"})
        r = client.get("/api/sessions")
        assert r.status_code == 200
        assert isinstance(r.json()["sessions"], list)


class TestWebSocket:
    def test_handshake_emits_ready_then_opening(self, client):
        sid = client.post("/api/sessions", json={"track": "dsa"}).json()["session_id"]
        with client.websocket_connect(f"/ws/{sid}?track=dsa") as ws:
            ready = json.loads(ws.receive_text())
            assert ready["type"] == "ready"
            assert ready["session_id"] == sid
            assert "backends" in ready

            kinds = []
            for _ in range(8):
                kinds.append(json.loads(ws.receive_text())["type"])
                if "speech_end" in kinds:
                    break
            assert "speech" in kinds
            assert "speech_end" in kinds

    def test_ping_pong(self, client):
        sid = client.post("/api/sessions", json={"track": "dsa"}).json()["session_id"]
        with client.websocket_connect(f"/ws/{sid}") as ws:
            for _ in range(8):
                if json.loads(ws.receive_text())["type"] == "speech_end":
                    break
            ws.send_text(json.dumps({"type": "ping"}))
            for _ in range(10):
                if json.loads(ws.receive_text())["type"] == "pong":
                    return
            pytest.fail("no pong received")

    def test_unknown_message_type_errors_without_closing(self, client):
        sid = client.post("/api/sessions", json={"track": "dsa"}).json()["session_id"]
        with client.websocket_connect(f"/ws/{sid}") as ws:
            for _ in range(8):
                if json.loads(ws.receive_text())["type"] == "speech_end":
                    break
            ws.send_text(json.dumps({"type": "banana"}))
            for _ in range(10):
                msg = json.loads(ws.receive_text())
                if msg["type"] == "error":
                    assert "banana" in msg["message"]
                    return
            pytest.fail("no error message for unknown type")

    def test_invalid_track_closes_socket(self, client):
        from starlette.websockets import WebSocketDisconnect

        with pytest.raises(WebSocketDisconnect):
            with client.websocket_connect("/ws/xyz?track=nonsense") as ws:
                ws.receive_text()
