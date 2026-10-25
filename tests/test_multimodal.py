"""Tests for multimodal vision, attachment processing, and vision routing."""

import base64
import unittest
import zlib
from unittest.mock import AsyncMock, patch

from maidere.api.schemas import ChatRequest
from maidere.core.attachments import (
    clean_base64,
    compose_turn_prompt_with_attachments,
    process_attachment,
)
from maidere.core.llm import _normalize_messages_for_cloud
from maidere.core.router import TaskComplexity, classify_complexity, select_model


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
        with patch(
            "maidere.core.router.get_available_models", new=AsyncMock(return_value=["qwen2.5:7b", "qwen2.5vl:7b"])
        ):
            model, reason, complexity = await select_model(
                requested_model="auto",
                query="What is in this UI screenshot?",
                has_images=True,
            )
            self.assertEqual(complexity, TaskComplexity.VISION)
            self.assertIn("vl", model.lower())
            self.assertEqual(reason, "multimodal_vision")

    async def test_default_text_model_is_not_treated_as_a_user_choice_for_images(self):
        """The API sends the default text model when the client picked Auto; images must still reach a vision model."""
        from maidere.core.config import settings

        with (
            patch(
                "maidere.core.router.get_available_models",
                new=AsyncMock(return_value=[settings.ollama_model, "qwen2.5vl:3b"]),
            ),
            patch.object(settings, "ollama_vision_model", "qwen2.5vl:3b"),
        ):
            model, reason, _ = await select_model(
                requested_model=settings.ollama_model, query="What is in this screenshot?", has_images=True
            )
        self.assertEqual(model, "qwen2.5vl:3b")
        self.assertEqual(reason, "multimodal_vision")

    async def test_vision_model_is_found_under_its_real_ollama_name(self):
        """Ollama's tag is `qwen2.5vl` (no hyphen); the hyphenated name does not exist."""
        from maidere.core.config import settings

        self.assertEqual(settings.ollama_vision_model.split(":")[0], "qwen2.5vl")
        with patch(
            "maidere.core.router.get_available_models", new=AsyncMock(return_value=["qwen2.5:7b", "qwen2.5vl:7b"])
        ):
            model, _, _ = await select_model(requested_model=None, query="Check this", has_images=True)
        self.assertEqual(model, "qwen2.5vl:7b")

    async def test_select_model_respects_explicit_user_override_with_images(self):
        with patch(
            "maidere.core.router.get_available_models", new=AsyncMock(return_value=["qwen2.5:7b", "qwen2.5vl:7b"])
        ):
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
                "images": [
                    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
                ],
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


def _png_b64(width: int, height: int, color=(200, 30, 30)) -> str:
    import io

    from PIL import Image

    buf = io.BytesIO()
    Image.new("RGB", (width, height), color).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _decoded_size(b64: str) -> tuple[int, int]:
    import io

    from PIL import Image

    with Image.open(io.BytesIO(base64.b64decode(b64.split(",", 1)[-1]))) as img:
        return img.size


class TestVisualDeixis(unittest.TestCase):
    """Follow-ups that point back at an earlier image."""

    def test_detects_references_to_prior_images(self):
        from maidere.core.visual_memory import references_prior_image

        for q in (
            "What changes should I make to the primary button in that design?",
            "Can you describe the screenshot again?",
            "In the UI mockup, is the header too tall?",
            "what about the diagram I uploaded",
            "Compare it with the previous image",
        ):
            self.assertTrue(references_prior_image(q), q)

    def test_ignores_unrelated_follow_ups(self):
        from maidere.core.visual_memory import references_prior_image

        for q in ("Thanks!", "Now write a Python function that sorts a list", "", "Explain asyncio to me"):
            self.assertFalse(references_prior_image(q), q)


class TestImageDownscale(unittest.IsolatedAsyncioTestCase):
    async def test_large_image_fits_within_1080p_and_keeps_aspect(self):
        from maidere.core.visual_memory import downscale_image

        out = await downscale_image(_png_b64(3840, 2160))
        w, h = _decoded_size(out)
        self.assertLessEqual(w, 1920)
        self.assertLessEqual(h, 1080)
        self.assertAlmostEqual(w / h, 3840 / 2160, places=1)

    async def test_small_image_is_returned_untouched(self):
        from maidere.core.visual_memory import downscale_image

        small = _png_b64(64, 64)
        self.assertEqual(await downscale_image(small), small)

    async def test_data_uri_prefix_is_preserved(self):
        from maidere.core.visual_memory import downscale_image

        out = await downscale_image("data:image/png;base64," + _png_b64(2500, 1500))
        self.assertTrue(out.startswith("data:image/png;base64,"))
        self.assertLessEqual(_decoded_size(out)[0], 1920)

    async def test_unreadable_image_is_kept_as_uploaded(self):
        from maidere.core.visual_memory import downscale_image

        self.assertEqual(await downscale_image("not-an-image"), "not-an-image")


class TestImageHistoryWindow(unittest.IsolatedAsyncioTestCase):
    """Sliding window and cap on remembered images."""

    @staticmethod
    def _msgs(n: int):
        from langchain_core.messages import AIMessage, HumanMessage

        out = []
        for i in range(n):
            out.append(HumanMessage(content=f"q{i}", id=f"h{i}"))
            out.append(AIMessage(content=f"a{i}", id=f"a{i}"))
        return out[:-1]  # last user message has no reply yet

    async def test_image_survives_one_follow_up_then_expires(self):
        from maidere.core.visual_memory import remember_images

        img = _png_b64(8, 8)
        history = await remember_images(None, self._msgs(1), [img])
        self.assertEqual([e["message_id"] for e in history], ["h0"])

        history = await remember_images(history, self._msgs(2), [])  # turn 2: still available
        self.assertEqual([e["message_id"] for e in history], ["h0"])

        history = await remember_images(history, self._msgs(3), [])  # turn 3: pruned
        self.assertEqual(history, [])

    async def test_max_images_cap_keeps_newest(self):
        from maidere.core.visual_memory import remember_images

        imgs = [_png_b64(8 + i, 8) for i in range(6)]
        with patch("maidere.core.visual_memory.settings.image_memory_max_images", 3):
            history = await remember_images(None, self._msgs(1), imgs)
        kept = [i for e in history for i in e["images"]]
        self.assertEqual(kept, imgs[-3:])

    async def test_same_image_is_not_stored_twice(self):
        from maidere.core.visual_memory import remember_images

        img = _png_b64(8, 8)
        history = await remember_images(None, self._msgs(1), [img])
        history = await remember_images(history, self._msgs(2), [img])  # stale re-send
        self.assertEqual(sum(len(e["images"]) for e in history), 1)

    async def test_disabled_when_turns_is_zero(self):
        from maidere.core.visual_memory import remember_images

        with patch("maidere.core.visual_memory.settings.image_memory_turns", 0):
            self.assertEqual(await remember_images(None, self._msgs(1), [_png_b64(8, 8)]), [])

    def test_recall_excludes_current_turn_and_requires_deixis(self):
        from maidere.core.visual_memory import recall_images

        history = [{"message_id": "h0", "images": ["A"]}, {"message_id": "h1", "images": ["B"]}]
        self.assertEqual(recall_images(history, "h1", "what is in that screenshot?"), {"h0": ["A"]})
        self.assertEqual(recall_images(history, "h1", "thanks"), {})


class TestMultiTurnVisualMemory(unittest.IsolatedAsyncioTestCase):
    """End to end through the graph and checkpointer: ask about an image on the next turn."""

    async def asyncSetUp(self):
        import tempfile

        self.temp_db = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        self.temp_db_path = self.temp_db.name
        self.temp_db.close()

    async def asyncTearDown(self):
        import os

        if os.path.exists(self.temp_db_path):
            os.remove(self.temp_db_path)

    async def test_follow_up_sees_image_then_it_expires(self):
        from langchain_core.messages import HumanMessage

        from maidere.core.agent import create_graph

        seen: list[tuple[str, list[str]]] = []

        async def fake_chat(messages, tools=None, model=None, num_ctx=None, on_token=None):
            images = [img for m in messages if m.get("role") == "user" for img in (m.get("images") or [])]
            last_user = next(m["content"] for m in reversed(messages) if m.get("role") == "user")
            seen.append((last_user, images))
            return {"message": {"role": "assistant", "content": "ok", "tool_calls": []}}

        img = _png_b64(32, 32)
        graph, ctx = await create_graph(self.temp_db_path)
        config = {"configurable": {"thread_id": "visual-memory-thread"}}

        async def turn(text: str, images: list[str]):
            await graph.ainvoke(
                {
                    "messages": [HumanMessage(content=text)],
                    "memory_context": "No relevant memories.",
                    "thread_id": "visual-memory-thread",
                    "images": images,
                },
                config=config,
            )

        try:
            with (
                patch("maidere.core.llm.chat", side_effect=fake_chat),
                patch("maidere.core.agent.recall", new_callable=AsyncMock, return_value=[]),
                patch("maidere.core.agent.emit_agent_event", new_callable=AsyncMock),
            ):
                await turn("Here is my login screen mockup", [img])
                await turn("What should I change on the primary button in that design?", [])
                await turn("Thanks. Now explain Python decorators", [])
                await turn("Go back to that screenshot: is the header too tall?", [])
        finally:
            await ctx.__aexit__(None, None, None)

        by_text = dict(seen)
        self.assertEqual(len(by_text["Here is my login screen mockup"]), 1)
        self.assertEqual(
            len(by_text["What should I change on the primary button in that design?"]),
            1,
            "follow-up turn must still see the uploaded image",
        )
        self.assertEqual(by_text["Thanks. Now explain Python decorators"], [], "unrelated turn stays text-only")
        self.assertEqual(
            by_text["Go back to that screenshot: is the header too tall?"],
            [],
            "image is pruned after the 2-turn window",
        )


if __name__ == "__main__":
    unittest.main()
