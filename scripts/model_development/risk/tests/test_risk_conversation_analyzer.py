
"""
Comprehensive Regression Tests
===============================

Tests the conversation-level risk analyzer.

These are software behavior tests using constructed examples.
They do not establish clinical validity or real-world accuracy.
"""

import unittest

from scripts.model_development.risk import risk_config as config

from scripts.model_development.risk.risk_conversation_analyzer import (
    analyze_conversation,
    validate_conversation,
    determine_conversation_category,
    identify_recurring_indicators,
    analyze_conversation_progression
)


CATEGORY = config.RISK_CATEGORIES


class TestRiskConversationAnalyzer(unittest.TestCase):

    # ========================================================
    # 1. NORMAL CONVERSATION
    # ========================================================

    def test_normal_conversation(self):

        result = analyze_conversation([

            "I enjoyed studying today.",

            "I completed my assignment.",

            "Tomorrow I will attend my lecture."
        ])

        self.assertEqual(
            result["conversation_category"],
            CATEGORY["NO_EXPLICIT_RISK_SIGNAL"]
        )

        self.assertFalse(result["requires_human_review"])

    # ========================================================
    # 2. SINGLE POTENTIAL-CONCERN MESSAGE
    # ========================================================

    def test_single_potential_concern(self):

        result = analyze_conversation([

            "I feel completely hopeless."
        ])

        self.assertEqual(
            result["conversation_category"],
            CATEGORY["POTENTIAL_CONCERN"]
        )

        self.assertTrue(result["requires_human_review"])

    # ========================================================
    # 3. REPEATED POTENTIAL CONCERNS
    # ========================================================

    def test_repeated_potential_concerns_do_not_upgrade_to_urgent(self):

        result = analyze_conversation([

            "I feel completely hopeless.",

            "I feel hopeless again.",

            "Nothing seems to get better."
        ])

        self.assertNotEqual(
            result["conversation_category"],
            CATEGORY["URGENT_LANGUAGE_SIGNAL"]
        )

        self.assertTrue(result["requires_human_review"])

    # ========================================================
    # 4. DIRECT URGENT EXPRESSION
    # ========================================================

    def test_direct_urgent_expression(self):

        result = analyze_conversation([

            "I am feeling stressed.",

            "I am going to kill myself."
        ])

        self.assertEqual(
            result["conversation_category"],
            CATEGORY["URGENT_LANGUAGE_SIGNAL"]
        )

        self.assertTrue(result["requires_human_review"])

    # ========================================================
    # 5. HYPOTHETICAL EXPRESSION
    # ========================================================

    def test_hypothetical_expression(self):

        result = analyze_conversation([

            "What should someone do if they want to kill themselves?"
        ])

        self.assertEqual(
            result["conversation_category"],
            CATEGORY["UNCERTAIN_OR_AMBIGUOUS"]
        )

        self.assertTrue(result["requires_human_review"])

    # ========================================================
    # 6. HISTORICAL EXPRESSION
    # ========================================================

    def test_historical_expression(self):

        result = analyze_conversation([

            "I used to want to hurt myself, but that was in the past."
        ])

        self.assertEqual(
            result["conversation_category"],
            CATEGORY["UNCERTAIN_OR_AMBIGUOUS"]
        )

        self.assertTrue(result["requires_human_review"])

    # ========================================================
    # 7. THIRD-PERSON EXPRESSION
    # ========================================================

    def test_third_person_expression(self):

        result = analyze_conversation([

            "My friend says he wants to hurt himself."
        ])

        self.assertEqual(
            result["conversation_category"],
            CATEGORY["UNCERTAIN_OR_AMBIGUOUS"]
        )

        self.assertTrue(result["requires_human_review"])

    # ========================================================
    # 8. QUOTED EXPRESSION
    # ========================================================

    def test_quoted_expression(self):

        result = analyze_conversation([

            'My friend said, "I want to kill myself."'
        ])

        self.assertEqual(
            result["conversation_category"],
            CATEGORY["UNCERTAIN_OR_AMBIGUOUS"]
        )

        self.assertTrue(result["requires_human_review"])

    # ========================================================
    # 9. MIXED CONVERSATION WITH URGENT MESSAGE
    # ========================================================

    def test_urgent_message_dominates_conversation_category(self):

        result = analyze_conversation([

            "I enjoyed my day.",

            "I feel completely hopeless.",

            "I am going to kill myself."
        ])

        self.assertEqual(
            result["conversation_category"],
            CATEGORY["URGENT_LANGUAGE_SIGNAL"]
        )

    # ========================================================
    # 10. RECURRING INDICATOR DETECTION
    # ========================================================

    def test_recurring_indicators(self):

        result = analyze_conversation([

            "I feel completely hopeless.",

            "I feel hopeless again."
        ])

        self.assertIn(
            "hopelessness_language",
            result["recurring_indicators"]
        )

        recurring = result["recurring_indicators"][
            "hopelessness_language"
        ]

        self.assertGreaterEqual(
            recurring["message_count"],
            2
        )

    # ========================================================
    # 11. PROGRESSION LENGTH
    # ========================================================

    def test_progression_contains_all_analyzed_messages(self):

        messages = [

            "I enjoyed my day.",

            "I feel completely hopeless.",

            "I am going to kill myself."
        ]

        result = analyze_conversation(messages)

        progression = result["progression"]["message_progression"]

        self.assertEqual(
            len(progression),
            len(messages)
        )

        self.assertEqual(
            [item["message_index"] for item in progression],
            [1, 2, 3]
        )

    # ========================================================
    # 12. ASSISTANT MESSAGE FILTERING
    # ========================================================

    def test_assistant_messages_are_excluded(self):

        result = analyze_conversation([

            {
                "role": "user",
                "content": "I feel completely hopeless."
            },

            {
                "role": "assistant",
                "content": "I am here to listen."
            },

            {
                "role": "user",
                "content": "Nothing seems to get better."
            }
        ])

        self.assertEqual(
            result["summary"]["total_messages_analyzed"],
            2
        )

        self.assertEqual(
            len(result["message_results"]),
            2
        )

    # ========================================================
    # 13. SAFETY OUTPUT FLAGS
    # ========================================================

    def test_output_never_claims_confirmed_diagnosis_or_risk(self):

        result = analyze_conversation([

            "I am going to kill myself."
        ])

        self.assertFalse(result["clinical_diagnosis"])

        self.assertFalse(result["actual_risk_confirmed"])

        for message_result in result["message_results"]:

            self.assertFalse(
                message_result["clinical_diagnosis"]
            )

            self.assertFalse(
                message_result["actual_risk_confirmed"]
            )

    # ========================================================
    # 14. EMPTY CONVERSATION VALIDATION
    # ========================================================

    def test_empty_conversation_rejected(self):

        with self.assertRaises(ValueError):

            validate_conversation([])

    # ========================================================
    # 15. NON-LIST INPUT VALIDATION
    # ========================================================

    def test_invalid_conversation_type_rejected(self):

        with self.assertRaises(TypeError):

            validate_conversation("I feel stressed.")

    # ========================================================
    # 16. EMPTY MESSAGE VALIDATION
    # ========================================================

    def test_empty_message_rejected(self):

        with self.assertRaises(ValueError):

            validate_conversation([

                "I feel stressed.",

                "   "
            ])

    # ========================================================
    # 17. INVALID MESSAGE TYPE
    # ========================================================

    def test_invalid_message_type_rejected(self):

        with self.assertRaises(TypeError):

            validate_conversation([

                "I feel stressed.",

                123
            ])

    # ========================================================
    # 18. INVALID ROLE VALIDATION
    # ========================================================

    def test_invalid_role_rejected(self):

        with self.assertRaises(ValueError):

            validate_conversation([

                {
                    "role": "moderator",
                    "content": "I feel stressed."
                }
            ])

    # ========================================================
    # 19. ASSISTANT-ONLY CONVERSATION
    # ========================================================

    def test_assistant_only_conversation_rejected(self):

        with self.assertRaises(ValueError):

            analyze_conversation([

                {
                    "role": "assistant",
                    "content": "How can I help you?"
                }
            ])

    # ========================================================
    # 20. MESSAGE LIMIT VALIDATION
    # ========================================================

    def test_message_limit_enforced(self):

        messages = [

            "I feel stressed."

            for _ in range(101)
        ]

        with self.assertRaises(ValueError):

            validate_conversation(messages)


# ============================================================
# TEST EXECUTION
# ============================================================

if __name__ == "__main__":

    print("=" * 75)
    print("RISK CONVERSATION ANALYZER - REGRESSION TESTS")
    print("=" * 75)

    suite = unittest.defaultTestLoader.loadTestsFromTestCase(
        TestRiskConversationAnalyzer
    )

    runner = unittest.TextTestRunner(
        verbosity=2
    )

    result = runner.run(suite)

    total_tests = result.testsRun

    passed_tests = total_tests - len(result.failures) - len(result.errors)

    print("\n" + "=" * 75)

    print(f"Total tests : {total_tests}")
    print(f"Passed      : {passed_tests}")
    print(f"Failed      : {len(result.failures)}")
    print(f"Errors      : {len(result.errors)}")

    if result.wasSuccessful():

        print("\nRISK CONVERSATION REGRESSION TESTS: PASSED")

    else:

        print("\nRISK CONVERSATION REGRESSION TESTS: FAILED")

    print("=" * 75)
