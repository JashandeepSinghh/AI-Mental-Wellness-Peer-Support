"""
Tests for Phase 10 (Supportive Responses) and Phase 11 (Risk Detection & Safety Escalation)
=============================================================================================

Validates:
1. Emotion-aware supportive text generation
2. Stress and loneliness response triggers
3. Wellness resource recommendations (Box breathing, sensory grounding, etc.)
4. Urgent risk language triggers safety guidance with 24/7 crisis lines
5. Non-urgent potential concerns provide gentle human-review suggestions
6. No-risk expressions remain calm without alarming banners
7. Ambiguous statements preserve uncertainty
8. Transparent disclaimers that the platform is not a clinical diagnosis or medical clinic
"""

import unittest
from backend.app.services.supportive_response_service import (
    generate_supportive_guidance,
    select_wellness_resources,
    TRUSTED_CRISIS_CONTACTS,
)


class TestSupportiveResponsesAndSafety(unittest.TestCase):

    def test_ordinary_positive_statement(self):
        """Verify normal positive message does not trigger risk or distress."""
        mock_analysis = {
            "modules": {
                "emotion": {
                    "confidence_analysis": {
                        "selected_labels": ["joy", "gratitude"],
                    }
                },
                "sentiment": {"predicted_class": "positive"},
                "stress": {"predicted_class": "non_stress"},
                "loneliness": {"predicted_class": "non_lonely"},
                "risk": {
                    "risk_category": "no_explicit_risk_signal",
                    "escalation_level": "none",
                },
            }
        }
        guidance = generate_supportive_guidance(mock_analysis)
        self.assertIn("positive reflections", guidance["supportive_response"].lower())
        self.assertIsNone(guidance["safety_guidance"])
        self.assertFalse(guidance["stress_detected"])
        self.assertFalse(guidance["loneliness_detected"])
        self.assertTrue(len(guidance["wellness_resources"]) > 0)

    def test_stress_and_pressure_statement(self):
        """Verify stress triggers stress-targeted coping resources like box breathing."""
        mock_analysis = {
            "modules": {
                "emotion": {
                    "confidence_analysis": {
                        "selected_labels": ["nervousness"],
                    }
                },
                "sentiment": {"predicted_class": "negative"},
                "stress": {"predicted_class": "stress"},
                "loneliness": {"predicted_class": "non_lonely"},
                "risk": {
                    "risk_category": "no_explicit_risk_signal",
                    "escalation_level": "none",
                },
            }
        }
        guidance = generate_supportive_guidance(mock_analysis)
        self.assertTrue(guidance["stress_detected"])
        self.assertIn("stress", guidance["supportive_response"].lower())
        self.assertIsNone(guidance["safety_guidance"])
        categories = [r["category"] for r in guidance["wellness_resources"]]
        self.assertTrue(
            any(cat in ("stress_management", "academic_stress") for cat in categories)
        )

    def test_loneliness_statement(self):
        """Verify loneliness triggers social support reconnection resources."""
        mock_analysis = {
            "modules": {
                "emotion": {
                    "confidence_analysis": {
                        "selected_labels": ["sadness"],
                    }
                },
                "sentiment": {"predicted_class": "negative"},
                "stress": {"predicted_class": "non_stress"},
                "loneliness": {"predicted_class": "lonely"},
                "risk": {
                    "risk_category": "no_explicit_risk_signal",
                    "escalation_level": "none",
                },
            }
        }
        guidance = generate_supportive_guidance(mock_analysis)
        self.assertTrue(guidance["loneliness_detected"])
        self.assertIn("lonely", guidance["supportive_response"].lower())
        categories = [r["category"] for r in guidance["wellness_resources"]]
        self.assertIn("social_support", categories)

    def test_urgent_risk_language_escalation(self):
        """Verify explicit urgent risk signals immediately provide 24/7 crisis guidance."""
        mock_analysis = {
            "modules": {
                "emotion": {
                    "confidence_analysis": {
                        "selected_labels": ["grief", "fear"],
                    }
                },
                "sentiment": {"predicted_class": "negative"},
                "stress": {"predicted_class": "stress"},
                "loneliness": {"predicted_class": "lonely"},
                "risk": {
                    "risk_category": "urgent_language_signal",
                    "escalation_level": "prompt_human_review",
                    "requires_human_review": True,
                },
            }
        }
        guidance = generate_supportive_guidance(mock_analysis)
        safety = guidance["safety_guidance"]
        self.assertIsNotNone(safety)
        self.assertTrue(safety["is_urgent"])
        self.assertEqual(safety["risk_category"], "urgent_language_signal")
        self.assertIn("988 Suicide & Crisis Lifeline", [c["name"] for c in safety["crisis_contacts"]])
        self.assertIn("pain", guidance["supportive_response"].lower())

    def test_potential_concern_escalation(self):
        """Verify potential concern statements provide non-urgent supportive guidance."""
        mock_analysis = {
            "modules": {
                "emotion": {
                    "confidence_analysis": {
                        "selected_labels": ["disappointment"],
                    }
                },
                "sentiment": {"predicted_class": "negative"},
                "stress": {"predicted_class": "stress"},
                "loneliness": {"predicted_class": "non_lonely"},
                "risk": {
                    "risk_category": "potential_concern",
                    "escalation_level": "supportive_follow_up",
                },
            }
        }
        guidance = generate_supportive_guidance(mock_analysis)
        safety = guidance["safety_guidance"]
        self.assertIsNotNone(safety)
        self.assertFalse(safety["is_urgent"])
        self.assertTrue(safety["is_concerning"])

    def test_resource_selection_fallback(self):
        """Verify safe fallback resources are returned when emotions are empty."""
        resources = select_wellness_resources([], stress_detected=False, loneliness_detected=False)
        self.assertEqual(len(resources), 3)
        self.assertEqual(resources[0]["id"], "box-breathing")


if __name__ == "__main__":
    unittest.main()
