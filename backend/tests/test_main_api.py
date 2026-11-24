"""
Main API Integration Tests
===========================

Tests endpoints through the actual FastAPI application.

Covers:
- Application startup and shutdown
- Health endpoint
- Root endpoint
- Chat inference integration
- Request validation
- Global exception-handler integration
"""

import unittest

from uuid import uuid4
from unittest.mock import Mock, patch

from fastapi.testclient import TestClient

from backend.app.main import app


class TestMainAPI(unittest.TestCase):

    @classmethod
    def setUpClass(cls):

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

        # Prevent actual model initialization during tests.
        cls.pipeline_patch = patch(
            "backend.app.main.InferencePipeline",
            return_value=cls.pipeline
        )

        cls.pipeline_factory = cls.pipeline_patch.start()

        # Start TestClient as a context manager.
        # This executes FastAPI's lifespan startup.
        cls.client = TestClient(app)

        cls.client.__enter__()

    @classmethod
    def tearDownClass(cls):

        # Execute FastAPI's lifespan shutdown.
        cls.client.__exit__(None, None, None)

        cls.pipeline_patch.stop()

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
    # TEST 1: Health endpoint
    # --------------------------------------------------------

    def test_health_endpoint(self):

        response = self.client.get(
            "/api/v1/health"
        )

        self.assertEqual(response.status_code, 200)

        data = response.json()

        self.assertEqual(
            data["status"],
            "ok"
        )

    # --------------------------------------------------------
    # TEST 2: Root endpoint
    # --------------------------------------------------------

    def test_root_endpoint(self):

        response = self.client.get("/")

        self.assertEqual(response.status_code, 200)

        self.assertIn(
            "version",
            response.json()
        )

    # --------------------------------------------------------
    # TEST 3: Valid chat request and inference integration
    # --------------------------------------------------------

    def test_valid_chat_request(self):

        message = "I have been feeling stressed lately."

        response = self.client.post(
            "/api/v1/chat/messages",
            json={
                "message": message,
                "metadata": {
                    "platform": "web",
                    "app_version": "1.0.0"
                }
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

    # --------------------------------------------------------
    # TEST 4: Valid conversation UUID
    # --------------------------------------------------------

    def test_valid_conversation_uuid(self):

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
    # TEST 5: Empty message rejected
    # --------------------------------------------------------

    def test_empty_message_validation(self):

        response = self.client.post(
            "/api/v1/chat/messages",
            json={
                "message": ""
            }
        )

        self.assertEqual(response.status_code, 422)

        data = response.json()

        self.assertFalse(data["success"])

        self.assertIn("error", data)

    # --------------------------------------------------------
    # TEST 6: Invalid UUID rejected
    # --------------------------------------------------------

    def test_invalid_uuid_validation(self):

        response = self.client.post(
            "/api/v1/chat/messages",
            json={
                "message": "Hello",
                "conversation_id": "invalid-uuid"
            }
        )

        self.assertEqual(response.status_code, 422)

        self.assertFalse(
            response.json()["success"]
        )

    # --------------------------------------------------------
    # TEST 7: Unexpected field rejected
    # --------------------------------------------------------

    def test_unexpected_field_validation(self):

        response = self.client.post(
            "/api/v1/chat/messages",
            json={
                "message": "Hello",
                "student_name": "Unknown"
            }
        )

        self.assertEqual(response.status_code, 422)

    # --------------------------------------------------------
    # TEST 8: Oversized message rejected
    # --------------------------------------------------------

    def test_oversized_message_validation(self):

        response = self.client.post(
            "/api/v1/chat/messages",
            json={
                "message": "A" * 20001
            }
        )

        self.assertEqual(response.status_code, 422)

    # --------------------------------------------------------
    # TEST 9: Unknown endpoint returns 404
    # --------------------------------------------------------

    def test_unknown_endpoint(self):

        response = self.client.get(
            "/api/v1/unknown-endpoint"
        )

        self.assertEqual(response.status_code, 404)

        self.assertFalse(
            response.json()["success"]
        )

    # --------------------------------------------------------
    # TEST 10: Inference pipeline initialized at startup
    # --------------------------------------------------------

    def test_inference_pipeline_initialized(self):

        self.assertIs(
            app.state.inference_pipeline,
            self.pipeline
        )

        self.pipeline_factory.assert_called_once()

    # --------------------------------------------------------
    # TEST 11: Inference output included in response
    # --------------------------------------------------------

    def test_inference_output_returned(self):

        response = self.client.post(
            "/api/v1/chat/messages",
            json={
                "message": "I feel overwhelmed."
            }
        )

        self.assertEqual(response.status_code, 200)

        analysis = response.json()["analysis"]

        self.assertIn("modules", analysis)

        self.assertIn("confidence_report", analysis)


if __name__ == "__main__":

    unittest.main(verbosity=2)