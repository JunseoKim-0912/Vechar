import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import Session
from sqlalchemy.pool import StaticPool

from app import llm, main
from app.auth import issue_token
from app.database import Base, get_db
from app.models import Character, TrainingJob, User, World
from app.services import training_pipeline
from app import training_source


class TrainingUploadTests(unittest.TestCase):
    def setUp(self):
        self.engine = create_engine(
            "sqlite:///:memory:", connect_args={"check_same_thread": False}, poolclass=StaticPool
        )
        Base.metadata.create_all(self.engine)
        with Session(self.engine) as db:
            owner = User(email="training-owner@example.invalid", password_hash="unused")
            other = User(email="training-other@example.invalid", password_hash="unused")
            db.add_all([owner, other])
            db.commit()
            self.owner_id, self.other_id = owner.id, other.id

        def test_db():
            with Session(self.engine) as db:
                yield db

        main.app.dependency_overrides[get_db] = test_db
        self.engine_patch = patch.object(main, "engine", self.engine)
        self.engine_patch.start()
        self.no_network = patch.object(llm, "_get_client", side_effect=AssertionError("Unexpected OpenAI call"))
        self.no_network.start()
        # Submission does not call the provider; worker tests cover processing.
        self.counter_patch = patch.object(training_pipeline, "make_training_text_counter",
                                          return_value=lambda text: 1)
        self.counter_patch.start()
        self.client = TestClient(main.app)
        self.client.__enter__()

    def tearDown(self):
        self.client.__exit__(None, None, None)
        self.no_network.stop()
        self.counter_patch.stop()
        self.engine_patch.stop()
        main.app.dependency_overrides.clear()
        self.engine.dispose()

    def _target(self, kind):
        with Session(self.engine) as db:
            if kind == "character":
                target = Character(user_id=self.owner_id, name="Training target")
                suffix = "training-sources"
            else:
                target = World(user_id=self.owner_id, name="Training world")
                suffix = "sources"
            db.add(target)
            db.commit()
            return f"/{kind}s/{target.id}/{suffix}"

    def _post(self, path, *, text=None, filename=None, content=None, user_id=None):
        source_type = "STORY" if path.startswith("/characters/") else "DESCRIPTION"
        data = {"source_type": source_type}
        if text is not None:
            data["text"] = text
        files = {"file": (filename, content, "text/plain")} if filename is not None else None
        headers = {"Authorization": f"Bearer {issue_token(user_id or self.owner_id)}"}
        if filename is None:
            return self.client.post(path, json=data, headers=headers)
        return self.client.post(path, data=data, files=files, headers=headers)

    def test_direct_text_accepts_15001_and_300000_characters_for_both_entities(self):
        for kind in ("character", "world"):
            for length in (15_001, 300_000):
                with self.subTest(kind=kind, length=length):
                    path = self._target(kind)
                    response = self._post(path, text="가" * length)
                    self.assertEqual(response.status_code, 202, response.text)
                    with Session(self.engine) as db:
                        row = db.get(TrainingJob, response.json()["job_id"])
                        self.assertEqual((row.source_char_count, len(row.source_text)), (length, length))

    def test_txt_upload_accepts_300000_characters_and_takes_precedence_over_text(self):
        for kind in ("character", "world"):
            for content in ("가" * 300_000, "😀" * 300_000):
                with self.subTest(kind=kind, bytes=len(content.encode("utf-8"))):
                    path = self._target(kind)
                    response = self._post(path, text="ignored", filename="source.TXT", content=content.encode("utf-8"))
                    self.assertEqual(response.status_code, 202, response.text)
                    with Session(self.engine) as db:
                        self.assertEqual(db.get(TrainingJob, response.json()["job_id"]).source_text, content)

    def test_legacy_multipart_text_accepts_300000_four_byte_characters(self):
        for kind in ("character", "world"):
            with self.subTest(kind=kind):
                path = self._target(kind)
                source_type = "STORY" if kind == "character" else "DESCRIPTION"
                response = self.client.post(
                    path,
                    data={"source_type": source_type},
                    files={"text": (None, "😀" * 300_000)},
                    headers={"Authorization": f"Bearer {issue_token(self.owner_id)}"},
                )
                self.assertEqual(response.status_code, 202, response.text)

    def test_maximum_utf8_txt_remains_in_memory(self):
        path = self._target("character")
        actual_reader = training_source.read_training_source
        observed = []

        async def inspect_file(text, file):
            observed.append(file.file._rolled)
            return await actual_reader(text, file)

        with patch.object(training_source, "read_training_source", side_effect=inspect_file):
            response = self._post(path, filename="large.txt", content=("😀" * 300_000).encode("utf-8"))
        self.assertEqual(response.status_code, 202, response.text)
        self.assertEqual(observed, [False])

    def test_invalid_sources_return_readable_codes_without_llm_calls(self):
        cases = (
            ({}, "training_source_missing"),
            ({"text": "x" * 300_001}, "source_too_long"),
            ({"filename": "empty.txt", "content": b""}, "empty_training_file"),
            ({"filename": "bad.txt", "content": b"\xff"}, "source_invalid_utf8"),
            ({"filename": "bad.pdf", "content": b"text"}, "invalid_source_file_type"),
            ({"filename": "too-long.txt", "content": b"x" * 300_001}, "source_too_long"),
        )
        for kind in ("character", "world"):
            for kwargs, code in cases:
                with self.subTest(kind=kind, code=code):
                    path = self._target(kind)
                    with Session(self.engine) as db:
                        before = db.query(TrainingJob).count()
                    response = self._post(path, **kwargs)
                    self.assertEqual(response.status_code, 400, response.text)
                    self.assertEqual(response.json()["detail"]["code"], code)
                    with Session(self.engine) as db:
                        self.assertEqual(db.query(TrainingJob).count(), before)

    def test_malformed_multipart_and_auth_ownership_still_reject(self):
        for kind in ("character", "world"):
            with self.subTest(kind=kind):
                path = self._target(kind)
                headers = {"Authorization": f"Bearer {issue_token(self.owner_id)}"}
                malformed = self.client.post(
                    path, content=b"broken", headers={**headers, "Content-Type": "multipart/form-data; boundary=abc"}
                )
                self.assertEqual(malformed.status_code, 400)
                self.assertEqual(malformed.json()["detail"]["code"], "malformed_training_request")
                self.assertEqual(self.client.post(path, data={"source_type": "DESCRIPTION"}).status_code, 401)
                with Session(self.engine) as db:
                    before = db.query(TrainingJob).count()
                self.assertEqual(
                    self._post(path, filename="source.txt", content=b"private", user_id=self.other_id).status_code,
                    404,
                )
                with Session(self.engine) as db:
                    self.assertEqual(db.query(TrainingJob).count(), before)

    def test_duplicate_submission_returns_same_active_job_without_llm(self):
        for kind in ("character", "world"):
            with self.subTest(kind=kind):
                path = self._target(kind)
                with Session(self.engine) as db:
                    before = db.query(TrainingJob).count()
                response = self._post(path, text="first")
                duplicate = self._post(path, text="second")
                self.assertEqual(response.status_code, 202)
                self.assertEqual(response.json()["job_id"], duplicate.json()["job_id"])
                with Session(self.engine) as db:
                    self.assertEqual(db.query(TrainingJob).count(), before + 1)

    def test_job_status_is_private_and_excludes_source(self):
        path = self._target("character")
        job_id = self._post(path, text="private story").json()["job_id"]
        headers = {"Authorization": f"Bearer {issue_token(self.owner_id)}"}
        status = self.client.get(f"/training-jobs/{job_id}", headers=headers)
        self.assertEqual(status.status_code, 200)
        self.assertEqual(status.json()["status"], "queued")
        self.assertNotIn("source_text", status.json())
        other = {"Authorization": f"Bearer {issue_token(self.other_id)}"}
        self.assertEqual(self.client.get(f"/training-jobs/{job_id}", headers=other).status_code, 404)


if __name__ == "__main__":
    unittest.main()
