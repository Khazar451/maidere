"""Tests for multimodal vision, attachment processing, and vision routing."""

import base64
import io
import unittest
import zlib
from unittest.mock import AsyncMock, patch

from api.schemas import ChatRequest
from core.attachments import (
    clean_base64,
    compose_turn_prompt_with_attachments,
    extract_pdf_text,
    process_attachment,
)
from core.llm import _normalize_messages_for_cloud
from core.router import TaskComplexity, classify_complexity, select_model


class TestAttachmentsProcessor(unittest.TestCase):
    """Test attachment parsing, base64 extraction, and document synthesis."""

    def test_clean_base64_data_uri(self):
        raw = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
        b64, mime = clean_base64(raw)
        self.assertEqual(mime, "image/png")
        self.assertTrue(b64.startswith("iVBORw0KGgo"))

    def test_clean_base64_raw_string(self):
        raw = "SGVsbG8gV29ybGQ="
        b64, mime = clean_base64(raw)
        self.assertEqual(b64, "SGVsbG8gV29ybGQ=")
        self.assertEqual(mime, "application/octet-stream")

    def test_clean_base64_empty(self):
        b64, mime = clean_base64("")
        self.assertEqual(b64, "")
        self.assertEqual(mime, "")

    def test_process_image_attachment(self):
        # 1x1 transparent PNG in base64
        sample_b64 = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
        item = {
            "name": "screenshot.png",
            "type": "image/png",
            "data": sample_b64,
            "size": 120,
        }
        res = process_attachment(item)
        self.assertEqual(res["kind"], "image")
        self.assertEqual(res["name"], "screenshot.png")
        self.assertEqual(res["mime_type"], "image/png")
        self.assertTrue(len(res["base64"]) > 10)

    def test_process_text_code_attachment(self):
        code_content = "def hello():\n    return 'world'\n"
        b64_code = base64.b64encode(code_content.encode("utf-8")).decode("utf-8")
        item = {
            "name": "solution.py",
            "type": "text/x-python",
            "data": f"data:text/x-python;base64,{b64_code}",
            "size": len(code_content),
        }
        res = process_attachment(item)
        self.assertEqual(res["kind"], "text")
        self.assertEqual(res["name"], "solution.py")
        self.assertIn("def hello():", res["text"])

    def test_process_pdf_attachment(self):
        # Simple PDF structure containing stream with text
        stream_content = b"BT /F1 12 Tf (Maidere Vision Report) Tj ET"
        compressed = zlib.compress(stream_content)
        fake_pdf = (
            b"%PDF-1.4\n1 0 obj\n<< /Length "
            + str(len(compressed)).encode()
            + b" /Filter /FlateDecode >>\nstream\n"
            + compressed
            + b"\nendstream\nendobj\ntrailer\n<< /Root 1 0 R >>\n%%EOF"
        )
        b64_pdf = base64.b64encode(fake_pdf).decode("utf-8")

        item = {
            "name": "report.pdf",
            "type": "application/pdf",
            "data": f"data:application/pdf;base64,{b64_pdf}",
            "size": len(fake_pdf),
        }
        res = process_attachment(item)
        self.assertEqual(res["kind"], "pdf")
        self.assertEqual(res["name"], "report.pdf")
        self.assertIn("Maidere Vision Report", res["text"])

    def test_compose_turn_prompt_with_attachments(self):
        sample_img = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
        code_text = "print('hello from maidere')"
        b64_code = base64.b64encode(code_text.encode("utf-8")).decode("utf-8")

        attachments = [
            {"name": "mockup.png", "type": "image/png", "data": sample_img},
            {"name": "script.py", "type": "text/x-python", "data": f"data:text/x-python;base64,{b64_code}"},
        ]

        query = "Explain this mockup and script"
        composed_text, images = compose_turn_prompt_with_attachments(query, attachments)

        # Image base64 must be separated into image list
        self.assertEqual(len(images), 1)
        self.assertTrue(images[0].startswith("iVBORw0KGgo"))

        # Script text must be injected into the composed query
        self.assertIn("Explain this mockup and script", composed_text)
        self.assertIn("### [Attached File: script.py]", composed_text)
        self.assertIn("print('hello from maidere')", composed_text)

    def test_compose_turn_prompt_no_attachments(self):
        query = "Hello Maidere"
        composed_text, images = compose_turn_prompt_with_attachments(query, None)
        self.assertEqual(composed_text, "Hello Maidere")
        self.assertEqual(images, [])


class TestRouterVisionSelection(unittest.IsolatedAsyncioTestCase):
    """Test model routing with multimodal vision inputs."""

    def test_classify_complexity_with_images(self):
        comp = classify_complexity("look at this", has_images=True)
        self.assertEqual(comp, TaskComplexity.VISION)

    async def test_select_model_auto_routes_to_vision(self):
        with patch("core.router.get_available_models", new=AsyncMock(return_value=["qwen2.5:7b", "qwen2.5-vl:7b"])):
            model, reason, complexity = await select_model(
                requested_model="auto",
                query="What is in this UI screenshot?",
                has_images=True,
            )
            self.assertEqual(complexity, TaskComplexity.VISION)
            self.assertIn("vl", model.lower())
            self.assertEqual(reason, "multimodal_vision")

    async def test_select_model_respects_explicit_user_override_with_images(self):
        with patch("core.router.get_available_models", new=AsyncMock(return_value=["qwen2.5:7b", "qwen2.5-vl:7b"])):
            model, reason, complexity = await select_model(
                requested_model="my-custom-vision:latest",
                query="Check this",
                has_images=True,
            )
            self.assertEqual(model, "my-custom-vision:latest")
            self.assertEqual(reason, "user_override_vision")
            self.assertEqual(complexity, TaskComplexity.VISION)


class TestCloudMultimodalNormalization(unittest.TestCase):
    """Test message normalization for OpenAI / Cloud vision APIs."""

    def test_normalize_user_message_with_images(self):
        messages = [
            {
                "role": "user",
                "content": "Describe this image",
                "images": ["iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="],
            }
        ]
        normalized = _normalize_messages_for_cloud(messages)
        self.assertEqual(len(normalized), 1)
        user_msg = normalized[0]
        self.assertEqual(user_msg["role"], "user")
        self.assertIsInstance(user_msg["content"], list)

        # First part: text
        self.assertEqual(user_msg["content"][0]["type"], "text")
        self.assertEqual(user_msg["content"][0]["text"], "Describe this image")

        # Second part: image_url
        self.assertEqual(user_msg["content"][1]["type"], "image_url")
        self.assertTrue(user_msg["content"][1]["image_url"]["url"].startswith("data:image/jpeg;base64,"))


class TestApiChatRequestSchema(unittest.TestCase):
    """Test ChatRequest schema accepts images and attachments."""

    def test_chat_request_schema_with_multimodal_fields(self):
        req = ChatRequest(
            message="Please analyze this screenshot",
            images=["data:image/png;base64,ABCDEF123456"],
            attachments=[{"name": "spec.pdf", "type": "application/pdf", "data": "XYZ", "size": 100}],
        )
        self.assertEqual(len(req.images), 1)
        self.assertEqual(len(req.attachments), 1)
        self.assertEqual(req.attachments[0]["name"], "spec.pdf")


if __name__ == "__main__":
    unittest.main()
