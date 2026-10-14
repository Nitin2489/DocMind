def upload(client, samples, *names):
    files = [("files", (n, (samples / n).read_bytes(), "text/plain")) for n in names]
    return client.post("/api/documents", files=files)


def test_health_reports_offline_mode(client):
    body = client.get("/api/health").json()
    assert body["status"] == "ok" and body["llm_mode"] == "extractive"


def test_full_flow_upload_ask_delete(client, samples):
    r = upload(client, samples, "leave_policy.md", "engineering_handbook.md")
    assert r.status_code == 201 and len(r.json()["documents"]) == 2
    assert len(client.get("/api/documents").json()) == 2

    ans = client.post("/api/ask", json={"question": "How many days of annual leave do employees get?"}).json()
    assert "24 days" in ans["answer"]
    assert ans["sources"][0]["doc_name"] == "leave_policy.md"
    assert "[1]" in ans["answer"] or "[2]" in ans["answer"]

    doc_id = client.get("/api/documents").json()[0]["id"]
    assert client.delete(f"/api/documents/{doc_id}").status_code == 204
    assert client.delete(f"/api/documents/{doc_id}").status_code == 404
    assert len(client.get("/api/documents").json()) == 1


def test_doc_filter_limits_sources(client, samples):
    docs = upload(client, samples, "leave_policy.md", "engineering_handbook.md").json()["documents"]
    handbook = next(d for d in docs if d["name"] == "engineering_handbook.md")["id"]
    ans = client.post("/api/ask", json={"question": "annual leave days", "doc_ids": [handbook]}).json()
    assert all(s["doc_id"] == handbook for s in ans["sources"])


def test_unanswerable_question_says_not_found(client, samples):
    upload(client, samples, "leave_policy.md")
    ans = client.post("/api/ask", json={"question": "zebra quantum submarine"}).json()
    assert "couldn't find" in ans["answer"] and ans["sources"] == []


def test_ask_with_no_documents(client):
    ans = client.post("/api/ask", json={"question": "anything?"}).json()
    assert "couldn't find" in ans["answer"]


def test_validation_errors(client):
    assert client.post("/api/ask", json={"question": ""}).status_code == 422
    assert client.post("/api/ask", json={"question": "x" * 2001}).status_code == 422
    assert client.post("/api/ask", json={"question": "hi", "top_k": 50}).status_code == 422


def test_bad_upload_is_rejected_with_reason(client):
    r = client.post("/api/documents", files=[("files", ("evil.exe", b"MZ", "application/octet-stream"))])
    assert r.status_code == 422 and "Unsupported" in str(r.json()["detail"])


def test_mixed_upload_reports_partial_success(client, samples):
    files = [
        ("files", ("ok.txt", b"Plain text about refunds and returns policy.", "text/plain")),
        ("files", ("bad.exe", b"MZ", "application/octet-stream")),
    ]
    body = client.post("/api/documents", files=files).json()
    assert len(body["documents"]) == 1 and len(body["errors"]) == 1


def test_oversized_upload_rejected(client, monkeypatch):
    from app import config

    monkeypatch.setattr(config, "MAX_UPLOAD_BYTES", 10)
    r = client.post("/api/documents", files=[("files", ("big.txt", b"x" * 50, "text/plain"))])
    assert r.status_code == 422


def test_frontend_served(client):
    assert "DocMind" in client.get("/").text
    assert client.get("/static/app.js").status_code == 200
