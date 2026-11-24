
"""
Exception Handler Automated Tests
=================================

Tests the centralized API exception handlers.

Test scenarios:
1. HTTP 404 response
2. Request validation 422 response
3. Unexpected exception 500 response
4. Sensitive information protection

Uses a separate temporary FastAPI application.
Does not modify the production application's routes.
"""

import unittest

from fastapi import FastAPI
from fastapi.testclient import TestClient

from backend.app.core.exception_handlers import (
    register_exception_handlers,
)


# ============================================================
# 1. CREATE TEMPORARY TEST APPLICATION
# ============================================================

def create_test_app():

    test_app = FastAPI(
        debug=False
    )

    register_exception_handlers(test_app)

    # --------------------------------------------------------
    # Route for testing request validation
    # --------------------------------------------------------

    @test_app.get("/test-validation")
    async def validation_route(count: int):

        return {
            "count": count
        }

    # --------------------------------------------------------
    # Route for testing unexpected exceptions
    # --------------------------------------------------------

    @test_app.get("/test-error")
    async def error_route():

        raise RuntimeError(
            "PRIVATE_STUDENT_INFORMATION"
        )

    return test_app


# ============================================================
# 2. EXCEPTION HANDLER TEST CASES
# ============================================================

class TestExceptionHandlers(unittest.TestCase):

    @classmethod
    def setUpClass(cls):

        cls.app = create_test_app()

    def setUp(self):

        self.client = TestClient(
            self.app,
            raise_server_exceptions=False
        )

    def tearDown(self):

        self.client.close()

    # --------------------------------------------------------
    # TEST 1: HTTP 404
    # --------------------------------------------------------

    def test_http_404_response(self):

        response = self.client.get(
            "/nonexistent"
        )

        self.assertEqual(
            response.status_code,
            404
        )

        data = response.json()

        self.assertFalse(
            data["success"]
        )

        self.assertEqual(
            data["error"]["code"],
            "NOT_FOUND"
        )

        self.assertEqual(
            data["error"]["details"],
            []
        )

        self.assertIn(
            "timestamp",
            data
        )

    # --------------------------------------------------------
    # TEST 2: REQUEST VALIDATION 422
    # --------------------------------------------------------

    def test_validation_422_response(self):

        response = self.client.get(

            "/test-validation",

            params={
                "count": "invalid"
            }

        )

        self.assertEqual(
            response.status_code,
            422
        )

        data = response.json()

        self.assertFalse(
            data["success"]
        )

        self.assertEqual(
            data["error"]["code"],
            "VALIDATION_ERROR"
        )

        self.assertGreater(
            len(data["error"]["details"]),
            0
        )

        # Submitted invalid input must not be exposed.
        self.assertNotIn(
            "invalid",
            response.text
        )

    # --------------------------------------------------------
    # TEST 3: UNEXPECTED EXCEPTION 500
    # --------------------------------------------------------

    def test_unexpected_exception_500_response(self):

        response = self.client.get(
            "/test-error"
        )

        self.assertEqual(
            response.status_code,
            500
        )

        data = response.json()

        self.assertFalse(
            data["success"]
        )

        self.assertEqual(
            data["error"]["code"],
            "INTERNAL_SERVER_ERROR"
        )

        # Internal exception details must not leak.
        self.assertNotIn(
            "PRIVATE_STUDENT_INFORMATION",
            response.text
        )

        self.assertNotIn(
            "RuntimeError",
            response.text
        )

        self.assertNotIn(
            "Traceback",
            response.text
        )

    # --------------------------------------------------------
    # TEST 4: ERROR RESPONSE STRUCTURE
    # --------------------------------------------------------

    def test_standard_error_response_structure(self):

        response = self.client.get(
            "/nonexistent"
        )

        data = response.json()

        self.assertIn(
            "success",
            data
        )

        self.assertIn(
            "error",
            data
        )

        self.assertIn(
            "timestamp",
            data
        )

        self.assertIn(
            "code",
            data["error"]
        )

        self.assertIn(
            "message",
            data["error"]
        )

        self.assertIn(
            "details",
            data["error"]
        )


# ============================================================
# 3. TEST EXECUTION
# ============================================================

if __name__ == "__main__":

    unittest.main(verbosity=2)
