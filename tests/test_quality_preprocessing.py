import os
import sys
import tempfile
import unittest
from pathlib import Path

from PIL import Image

EXTRACTION_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXTRACTION_DIR))

from src2.core.table_extractor import TableExtractor
from src2.core.table_image_preprocessor import TableImagePreprocessor
from src2.core.formula_validator import FormulaValidator
from src2.utils.output_manager import OutputManager
from src2.utils.embedding_text_cleaner import clean_embedding_text, compose_embedding_text
from src2.utils.path_config import resolve_workspace_path, workspace_relative


class FakeTableClient:
    provider = "qwen"
    model = "test-vlm"

    def __init__(self, responses=None, error=None):
        self.responses = list(responses or [])
        self.error = error
        self.calls = []

    def recognize_table(self, image_path, context=""):
        self.calls.append(image_path)
        if self.error:
            raise self.error
        return self.responses.pop(0)


class EmbeddingTextCleanerTests(unittest.TestCase):
    def test_figure_noise_is_removed_but_subject_is_kept(self):
        raw = (
            "图中展示直流电机驱动电路。右上角印有复旦大学校徽。"
            "联系邮箱 tangguoan@fudan.edu.cn。该图片与课程主题相关。"
        )
        cleaned = clean_embedding_text(raw, node_type="Figure")
        self.assertIn("直流电机驱动电路", cleaned)
        self.assertNotIn("校徽", cleaned)
        self.assertNotIn("tangguoan", cleaned)
        self.assertNotIn("课程主题相关", cleaned)

    def test_compose_deduplicates_parts(self):
        result = compose_embedding_text(["步进电机", "步进电机", "细分控制"])
        self.assertEqual(result, "步进电机 细分控制")

    def test_corner_attribution_is_removed(self):
        raw = (
            "图片展示无刷直流电机散热风扇。图片左上角叠加黄色文字"
            "@YouTube精选字幕，左下角标注网址 www.LearnEngineering.org。"
        )
        cleaned = clean_embedding_text(raw, node_type="Figure")
        self.assertIn("无刷直流电机散热风扇", cleaned)
        self.assertNotIn("YouTube", cleaned)
        self.assertNotIn("LearnEngineering.org", cleaned)


class TableImagePreprocessorTests(unittest.TestCase):
    def test_small_image_is_upscaled_and_rotation_is_recorded(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            source = os.path.join(temp_dir, "small-table.png")
            Image.new("RGB", (800, 400), "white").save(source)
            processor = TableImagePreprocessor()

            normal = processor.prepare(source, 0)
            rotated = processor.prepare(source, 90)

            with Image.open(normal.path) as image:
                self.assertEqual(image.size, (3200, 1600))
            with Image.open(rotated.path) as image:
                self.assertEqual(image.size, (1600, 3200))
            self.assertEqual(rotated.rotation, 90)

    def test_rotation_retry_stops_after_a_valid_table(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            source = os.path.join(temp_dir, "rotated-table.png")
            Image.new("RGB", (900, 500), "white").save(source)
            client = FakeTableClient(
                responses=[
                    {"has_table": True, "headers": [], "cells": [], "confidence": 0.4},
                    {
                        "has_table": True,
                        "visible_table_count": 1,
                        "coverage_complete": True,
                        "tables": [{
                            "headers": [["参数", "数值"]],
                            "cells": [["电压", "24V"]],
                            "confidence": 0.95,
                        }],
                    },
                ]
            )
            result = TableExtractor(vlm_client=client).extract_tables_from_vlm_image(source)

            self.assertEqual(len(client.calls), 2)
            self.assertFalse(result[0]["review_required"])
            self.assertEqual(result[0]["preprocessing"]["rotation"], 90)

    def test_api_failure_does_not_retry_rotated_images(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            source = os.path.join(temp_dir, "table.png")
            Image.new("RGB", (900, 500), "white").save(source)
            client = FakeTableClient(error=TimeoutError("network timeout"))
            result = TableExtractor(vlm_client=client).extract_tables_from_vlm_image(source)

            self.assertEqual(len(client.calls), 1)
            self.assertEqual(result[0]["review_reason"], "api_failed")

    def test_multiple_tables_are_preserved_and_sent_to_review(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            source = os.path.join(temp_dir, "two-tables.png")
            Image.new("RGB", (1200, 900), "white").save(source)
            client = FakeTableClient(responses=[{
                "has_table": True,
                "visible_table_count": 2,
                "coverage_complete": True,
                "tables": [
                    {"headers": [["电流", "峰值"]], "cells": [["1.5A", "2.1A"]], "confidence": 0.98},
                    {"headers": [["细分", "脉冲数"]], "cells": [["2", "400"]], "confidence": 0.98},
                ],
            }])

            result = TableExtractor(vlm_client=client).extract_tables_from_vlm_image(source)

            self.assertEqual(len(result), 2)
            self.assertTrue(all(item["review_required"] for item in result))
            self.assertTrue(all("multiple_tables_require_review" in item["validation_issues"] for item in result))

    def test_incomplete_table_coverage_is_sent_to_review(self):
        validator = TableExtractor(vlm_client=FakeTableClient(responses=[])).validator
        result = validator.validate({
            "backend": "api",
            "headers": [["参数", "数值"]],
            "cells": [["电流", "2.1A"]],
            "confidence": 0.99,
            "visible_table_count": 2,
            "extracted_table_count": 1,
            "coverage_complete": False,
        })
        self.assertTrue(result["review_required"])
        self.assertIn("table_coverage_mismatch", result["validation_issues"])


class FormulaValidatorTests(unittest.TestCase):
    def setUp(self):
        self.validator = FormulaValidator()

    def test_isolated_formula_fragments_are_sent_to_review(self):
        for latex in (r"\mp 1", r"\Delta\theta/2"):
            result = self.validator.validate({
                "latex": latex,
                "source_type": "image_vlm",
                "backend": "api",
                "confidence": 1.0,
                "is_complete_formula": True,
            })
            self.assertTrue(result["review_required"])
            self.assertIn("isolated_formula_fragment", result["validation_issues"])

    def test_possible_case_collision_is_sent_to_review(self):
        result = self.validator.validate({
            "latex": r"C = C / \tau^2",
            "source_type": "image_vlm",
            "backend": "api",
            "confidence": 1.0,
            "is_complete_formula": True,
        })
        self.assertTrue(result["review_required"])
        self.assertIn("possible_symbol_case_confusion", result["validation_issues"])

    def test_substantive_equation_still_passes(self):
        result = self.validator.validate({
            "latex": r"C = c / \tau^2",
            "source_type": "image_vlm",
            "backend": "api",
            "confidence": 0.98,
            "is_complete_formula": True,
        })
        self.assertFalse(result["review_required"])


class CodeOutputTests(unittest.TestCase):
    def test_repeated_page_saves_use_new_indices(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = OutputManager(base_dir=temp_dir)
            manager.save_code([{"language": "cpp", "code": "int first;"}], "lesson", 0)
            manager.save_code([{"language": "cpp", "code": "int second;"}], "lesson", 0)

            code_dir = Path(temp_dir) / "debug" / "code"
            self.assertEqual((code_dir / "lesson_page_1_code_1.cpp").read_text(), "int first;")
            self.assertEqual((code_dir / "lesson_page_1_code_2.cpp").read_text(), "int second;")
            code_items = [
                item for item in manager.knowledge_exporter.content_items.values()
                if item.get("type") == "code"
            ]
            self.assertEqual(len(code_items), 2)

    def test_resumed_run_continues_code_indices(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            first = OutputManager(base_dir=temp_dir)
            first.save_code([{"language": "cpp", "code": "int before_resume;"}], "lesson", 0)
            first.knowledge_exporter.finalize(build_vectors=False)

            resumed = OutputManager(base_dir=temp_dir)
            resumed.load_existing()
            resumed.save_code([{"language": "cpp", "code": "int after_resume;"}], "lesson", 0)

            code_dir = Path(temp_dir) / "debug" / "code"
            self.assertEqual((code_dir / "lesson_page_1_code_1.cpp").read_text(), "int before_resume;")
            self.assertEqual((code_dir / "lesson_page_1_code_2.cpp").read_text(), "int after_resume;")


class IdentityPropagationTests(unittest.TestCase):
    def test_different_users_get_different_document_ids(self):
        with tempfile.TemporaryDirectory() as first_dir, tempfile.TemporaryDirectory() as second_dir:
            first = OutputManager(base_dir=first_dir, user_id="user-a", course_id="course-1")
            second = OutputManager(base_dir=second_dir, user_id="user-b", course_id="course-1")
            self.assertNotEqual(
                first.knowledge_exporter.document_id("same-name.pdf"),
                second.knowledge_exporter.document_id("same-name.pdf"),
            )

    def test_identity_is_written_to_manifest_and_nodes(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = OutputManager(
                base_dir=temp_dir,
                user_id="user-7",
                username="teacher-zhang",
                course_id="course-9",
                course_name="机器人基础",
                job_id="job-11",
            )
            manager.save_text({"text": "步进电机", "key_points": [], "technical_terms": []}, "lesson", 0)
            manager.knowledge_exporter.finalize(build_vectors=False)

            kg_dir = Path(temp_dir) / "kg_data"
            manifest = __import__("json").loads((kg_dir / "course_manifest.json").read_text(encoding="utf-8"))
            nodes = __import__("json").loads((kg_dir / "multimodal_nodes.json").read_text(encoding="utf-8"))
            expected = {
                "user_id": "user-7",
                "username": "teacher-zhang",
                "course_id": "course-9",
                "course_name": "机器人基础",
                "job_id": "job-11",
            }
            self.assertTrue(all(manifest[key] == value for key, value in expected.items()))
            self.assertTrue(nodes)
            self.assertTrue(all(node[key] == value for key, value in expected.items() for node in nodes))


class WorkspacePathTests(unittest.TestCase):
    def test_legacy_and_workspace_paths_resolve_to_same_output(self):
        legacy = resolve_workspace_path("output/qwen_qwen3.7-plus_7")
        current = resolve_workspace_path("extraction/output/qwen_qwen3.7-plus_7")
        self.assertEqual(legacy, current)
        self.assertEqual(
            workspace_relative(legacy),
            "extraction/output/qwen_qwen3.7-plus_7",
        )


if __name__ == "__main__":
    unittest.main()
