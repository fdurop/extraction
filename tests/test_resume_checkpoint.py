import json

from src2.utils.knowledge_exporter import KnowledgeExporter


def _write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")


def test_load_existing_restores_identity_and_records(tmp_path):
    _write_json(
        tmp_path / "course_manifest.json",
        {
            "user_id": "user-a",
            "username": "teacher-a",
            "course_id": "course-a",
            "course_name": "Course A",
            "job_id": "job-a",
            "created_at": "2026-01-01T00:00:00",
        },
    )
    _write_json(tmp_path / "documents.json", [{"document_id": "doc-1"}])
    _write_json(tmp_path / "pages.json", [{"page_id": "page-1"}])
    _write_json(tmp_path / "content_list.json", [{"element_id": "text-1"}])
    _write_json(tmp_path / "multimodal_nodes.json", [{"node_id": "node-1"}])

    exporter = KnowledgeExporter(base_dir=str(tmp_path))
    counts = exporter.load_existing()

    assert counts == {"documents": 1, "pages": 1, "content_items": 1, "nodes": 1}
    assert exporter.user_id == "user-a"
    assert exporter.username == "teacher-a"
    assert exporter.course_id == "course-a"
    assert exporter.course_name == "Course A"
    assert exporter.job_id == "job-a"
    assert exporter.created_at == "2026-01-01T00:00:00"


def test_write_json_uses_atomic_replace(tmp_path):
    exporter = KnowledgeExporter(base_dir=str(tmp_path))
    exporter._write_json("checkpoint.json", {"status": "ok"})

    assert json.loads((tmp_path / "checkpoint.json").read_text(encoding="utf-8")) == {
        "status": "ok"
    }
    assert not (tmp_path / "checkpoint.json.tmp").exists()
