"""
Chat API Route Tests
====================

Tests chat request validation and inference integration
through the isolated FastAPI chat router.
"""

import unittest

from uuid import uuid4
from unittest.mock import Mock

from fastapi import FastAPI
from fastapi.testclient import TestClient

from backend.app.api.routes.chat import router


class TestChatRoutes(unittest.TestCase):

    @classmethod
    def setUpClass(cls):

        cls.app = FastAPI()

        # Create a mock inference pipeline.
        cls.pipeline = Mock()

        cls.pipeline.analyze_text.return_value = {
            "modules": {
                "emotion": {"status": "evaluated"},
                "sentiment": {"status": "evaluated"},
            },
            "confidence_report": {
                "status": "completed",
            },
            "human_review_required": False,
        }

        # Attach the pipeline to application state.
        cls.app.state.inference_pipeline = cls.pipeline

        cls.app.include_router(
            router,
            prefix="/api/v1"
        )

        cls.client = TestClient(cls.app)

    def setUp(self):

        self.pipeline = type(self).pipeline
        self.pipeline.reset_mock()

        self.pipeline.analyze_text.return_value = {
            "modules": {
                "emotion": {"status": "evaluated"},
                "sentiment": {"status": "evaluated"},
            },
            "confidence_report": {
                "status": "completed",
            },
            "human_review_required": False,
        }

    # --------------------------------------------------------
    # TEST 1: Valid chat request and inference integration
    # --------------------------------------------------------

    def test_valid_chat_request(self):

        message = "I have been feeling stressed lately."

        response = self.client.post(
            "/api/v1/chat/messages",
            json={
                "message": message
            }
        )

        self.assertEqual(response.status_code, 200)

        data = response.json()

        self.assertTrue(data["success"])

        self.assertEqual(
            data["status"],
            "analysis_completed"
        )

        self.assertEqual(
            data["message_length"],
            len(message)
        )

        self.assertIn("analysis", data)

        self.pipeline.analyze_text.assert_called_once_with(
            message
        )

        # Original message must not be returned.
        self.assertNotIn("original_message", data)

    # --------------------------------------------------------
    # TEST 2: Valid conversation identifier
    # --------------------------------------------------------

    def test_valid_conversation_id(self):

        conversation_id = str(uuid4())

        response = self.client.post(
            "/api/v1/chat/messages",
            json={
                "message": "Hello",
                "conversation_id": conversation_id
            }
        )

        self.assertEqual(response.status_code, 200)

        self.assertEqual(
            response.json()["conversation_id"],
            conversation_id
        )

    # --------------------------------------------------------
    # TEST 3: Empty message rejected
    # --------------------------------------------------------

    def test_empty_message_rejected(self):

        response = self.client.post(
            "/api/v1/chat/messages",
            json={
                "message": ""
            }
        )

        self.assertEqual(response.status_code, 422)

    # --------------------------------------------------------
    # TEST 4: Invalid UUID rejected
    # --------------------------------------------------------

    def test_invalid_uuid_rejected(self):

        response = self.client.post(
            "/api/v1/chat/messages",
            json={
                "message": "Hello",
                "conversation_id": "invalid-uuid"
            }
        )

        self.assertEqual(response.status_code, 422)

    # --------------------------------------------------------
    # TEST 5: Unexpected field rejected
    # --------------------------------------------------------

    def test_unexpected_field_rejected(self):

        response = self.client.post(
            "/api/v1/chat/messages",
            json={
                "message": "Hello",
                "student_name": "Unknown"
            }
        )

        self.assertEqual(response.status_code, 422)

    # --------------------------------------------------------
    # TEST 6: Whitespace-only message rejected
    # --------------------------------------------------------

    def test_whitespace_message_rejected(self):

        response = self.client.post(
            "/api/v1/chat/messages",
            json={
                "message": "       "
            }
        )

        self.assertEqual(response.status_code, 422)

    # --------------------------------------------------------
    # TEST 7: Oversized message rejected
    # --------------------------------------------------------

    def test_oversized_message_rejected(self):

        response = self.client.post(
            "/api/v1/chat/messages",
            json={
                "message": "A" * 20001
            }
        )

        self.assertEqual(response.status_code, 422)

    # --------------------------------------------------------
    # TEST 8: Valid metadata accepted
    # --------------------------------------------------------

    def test_valid_metadata(self):

        response = self.client.post(
            "/api/v1/chat/messages",
            json={
                "message": "Hello",
                "metadata": {
                    "platform": "web",
                    "app_version": "1.0.0"
                }
            }
        )

        self.assertEqual(response.status_code, 200)

        self.assertEqual(
            response.json()["status"],
            "analysis_completed"
        )

    # --------------------------------------------------------
    # TEST 9: Inference result returned
    # --------------------------------------------------------

    def test_inference_result_returned(self):

        response = self.client.post(
            "/api/v1/chat/messages",
            json={
                "message": "I feel overwhelmed."
            }
        )

        self.assertEqual(response.status_code, 200)

        data = response.json()

        self.assertIn("modules", data["analysis"])

        self.assertIn(
            "confidence_report",
            data["analysis"]
        )


if __name__ == "__main__":

    unittest.main(verbosity=2)