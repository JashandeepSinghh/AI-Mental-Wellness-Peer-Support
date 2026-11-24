
"""
Chat Request Schema Tests
=========================

Tests input validation for chatbot requests.
"""

import unittest

from uuid import uuid4

from pydantic import ValidationError

from backend.app.schemas.chat import (
    ChatRequest,
    ClientMetadata,
    ClientPlatform,
    MAX_MESSAGE_LENGTH,
)


class TestChatRequestSchema(unittest.TestCase):

    # --------------------------------------------------------
    # TEST 1: Valid message
    # --------------------------------------------------------

    def test_valid_message(self):

        request = ChatRequest(
            message="I have been feeling stressed lately."
        )

        self.assertEqual(
            request.message,
            "I have been feeling stressed lately."
        )

        self.assertIsNone(
            request.conversation_id
        )

    # --------------------------------------------------------
    # TEST 2: Empty message
    # --------------------------------------------------------

    def test_empty_message_rejected(self):

        with self.assertRaises(ValidationError):

            ChatRequest(
                message=""
            )

    # --------------------------------------------------------
    # TEST 3: Whitespace-only message
    # --------------------------------------------------------

    def test_whitespace_message_rejected(self):

        with self.assertRaises(ValidationError):

            ChatRequest(
                message="     "
            )

    # --------------------------------------------------------
    # TEST 4: Maximum message length
    # --------------------------------------------------------

    def test_maximum_message_length(self):

        request = ChatRequest(

            message="A" * MAX_MESSAGE_LENGTH

        )

        self.assertEqual(
            len(request.message),
            MAX_MESSAGE_LENGTH
        )

    # --------------------------------------------------------
    # TEST 5: Message exceeding maximum length
    # --------------------------------------------------------

    def test_excessive_message_length_rejected(self):

        with self.assertRaises(ValidationError):

            ChatRequest(

                message="A" * (MAX_MESSAGE_LENGTH + 1)

            )

    # --------------------------------------------------------
    # TEST 6: Valid conversation UUID
    # --------------------------------------------------------

    def test_valid_conversation_id(self):

        conversation_id = uuid4()

        request = ChatRequest(

            message="I need someone to talk to.",

            conversation_id=conversation_id

        )

        self.assertEqual(
            request.conversation_id,
            conversation_id
        )

    # --------------------------------------------------------
    # TEST 7: Invalid conversation UUID
    # --------------------------------------------------------

    def test_invalid_conversation_id(self):

        with self.assertRaises(ValidationError):

            ChatRequest(

                message="Hello",

                conversation_id="invalid-uuid"

            )

    # --------------------------------------------------------
    # TEST 8: Valid client metadata
    # --------------------------------------------------------

    def test_valid_client_metadata(self):

        metadata = ClientMetadata(

            platform=ClientPlatform.WEB,

            app_version="1.0.0"

        )

        request = ChatRequest(

            message="Hello",

            metadata=metadata

        )

        self.assertEqual(
            request.metadata.platform,
            ClientPlatform.WEB
        )

    # --------------------------------------------------------
    # TEST 9: Invalid platform
    # --------------------------------------------------------

    def test_invalid_platform(self):

        with self.assertRaises(ValidationError):

            ClientMetadata(
                platform="television"
            )

    # --------------------------------------------------------
    # TEST 10: Unexpected request field
    # --------------------------------------------------------

    def test_unexpected_field_rejected(self):

        with self.assertRaises(ValidationError):

            ChatRequest(

                message="Hello",

                student_name="Student"

            )

    # --------------------------------------------------------
    # TEST 11: Original message preserved
    # --------------------------------------------------------

    def test_original_message_preserved(self):

        original_message = "  I feel anxious today.  "

        request = ChatRequest(
            message=original_message
        )

        self.assertEqual(
            request.message,
            original_message
        )

    # --------------------------------------------------------
    # TEST 12: Optional metadata omitted
    # --------------------------------------------------------

    def test_metadata_optional(self):

        request = ChatRequest(
            message="Hello"
        )

        self.assertIsNone(
            request.metadata
        )


if __name__ == "__main__":

    unittest.main(verbosity=2)
