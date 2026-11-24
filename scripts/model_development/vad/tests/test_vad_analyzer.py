
"""
VAD Analyzer Regression Tests
=============================

Tests lexical matching, phrase handling, score aggregation,
coverage, warnings, and invalid input.

These tests validate software behavior, not psychological
or clinical accuracy.
"""

import unittest

from scripts.model_development.vad.vad_analyzer import (
    tokenize_text,
    validate_text,
    build_phrase_index,
    find_longest_match,
    match_lexicon_terms,
    calculate_vad_scores,
    calculate_lexical_coverage,
    generate_warnings,
    analyze_vad
)

from scripts.model_development.vad.vad_lexicon import (
    load_vad_lexicon
)


def make_entry(term, valence, arousal, dominance):

    return {

        "term": term,

        "term_type": (
            "multi_word" if " " in term else "unigram"
        ),

        "valence": valence,

        "arousal": arousal,

        "dominance": dominance
    }


class TestVADAnalyzer(unittest.TestCase):

    @classmethod
    def setUpClass(cls):

        cls.lookup, cls.lexicon_summary = load_vad_lexicon()

        cls.phrase_index = build_phrase_index(
            cls.lookup
        )

        cls.synthetic_lookup = {

            "fear": make_entry(
                "fear", -0.8, 0.7, -0.6
            ),

            "fear of failure": make_entry(
                "fear of failure", -0.9, 0.8, -0.7
            ),

            "failure": make_entry(
                "failure", -0.7, 0.6, -0.5
            ),

            "happy": make_entry(
                "happy", 0.9, 0.6, 0.7
            ),

            "confident": make_entry(
                "confident", 0.8, 0.5, 0.9
            )
        }

        cls.synthetic_phrase_index = build_phrase_index(
            cls.synthetic_lookup
        )

    # ========================================================
    # 1. TOKENIZATION
    # ========================================================

    def test_basic_tokenization(self):

        tokens = tokenize_text(
            "I feel happy today."
        )

        self.assertEqual(
            tokens,
            ["i", "feel", "happy", "today"]
        )

    def test_contractions_are_preserved(self):

        tokens = tokenize_text(
            "I don't feel happy."
        )

        self.assertIn("don't", tokens)

    def test_punctuation_is_excluded(self):

        tokens = tokenize_text(
            "Happy! Confident, hopeful."
        )

        self.assertEqual(
            tokens,
            ["happy", "confident", "hopeful"]
        )

    # ========================================================
    # 2. INPUT VALIDATION
    # ========================================================

    def test_valid_text(self):

        text = validate_text(
            "I feel happy."
        )

        self.assertEqual(
            text,
            "I feel happy."
        )

    def test_non_string_input_rejected(self):

        with self.assertRaises(TypeError):

            validate_text(123)

    def test_empty_text_rejected(self):

        with self.assertRaises(ValueError):

            validate_text("   ")

    def test_excessively_long_text_rejected(self):

        with self.assertRaises(ValueError):

            validate_text(
                "a" * 20001
            )

    # ========================================================
    # 3. PHRASE INDEX
    # ========================================================

    def test_multiword_phrase_index_created(self):

        self.assertIn(
            "fear",
            self.synthetic_phrase_index
        )

        self.assertIn(
            ("fear", "of", "failure"),
            self.synthetic_phrase_index["fear"]
        )

    def test_longer_phrases_checked_first(self):

        phrases = self.synthetic_phrase_index["fear"]

        self.assertEqual(
            phrases[0],
            ("fear", "of", "failure")
        )

    # ========================================================
    # 4. LONGEST-MATCH-FIRST
    # ========================================================

    def test_longest_phrase_preferred_over_unigram(self):

        tokens = [
            "fear",
            "of",
            "failure"
        ]

        matches, unmatched = match_lexicon_terms(

            tokens,

            self.synthetic_lookup,

            self.synthetic_phrase_index
        )

        self.assertEqual(
            len(matches),
            1
        )

        self.assertEqual(
            matches[0]["term"],
            "fear of failure"
        )

        self.assertEqual(
            unmatched,
            []
        )

    def test_phrase_does_not_double_count_component_words(self):

        tokens = [
            "fear",
            "of",
            "failure"
        ]

        matches, _ = match_lexicon_terms(

            tokens,

            self.synthetic_lookup,

            self.synthetic_phrase_index
        )

        self.assertEqual(
            len(matches),
            1
        )

        self.assertEqual(
            matches[0]["token_length"],
            3
        )

    def test_separate_unigrams_match_when_phrase_absent(self):

        lookup = {

            "fear": self.synthetic_lookup["fear"],

            "failure": self.synthetic_lookup["failure"]
        }

        phrase_index = build_phrase_index(
            lookup
        )

        matches, unmatched = match_lexicon_terms(

            ["fear", "of", "failure"],

            lookup,

            phrase_index
        )

        self.assertEqual(
            [match["term"] for match in matches],
            ["fear", "failure"]
        )

        self.assertEqual(
            unmatched,
            ["of"]
        )

    # ========================================================
    # 5. NON-OVERLAPPING MATCHES
    # ========================================================

    def test_repeated_unigrams_are_counted_separately(self):

        tokens = [
            "happy",
            "happy"
        ]

        matches, unmatched = match_lexicon_terms(

            tokens,

            self.synthetic_lookup,

            self.synthetic_phrase_index
        )

        self.assertEqual(
            len(matches),
            2
        )

        self.assertEqual(
            unmatched,
            []
        )

    def test_overlapping_phrase_matches_are_prevented(self):

        tokens = [
            "fear",
            "of",
            "failure",
            "happy"
        ]

        matches, _ = match_lexicon_terms(

            tokens,

            self.synthetic_lookup,

            self.synthetic_phrase_index
        )

        self.assertEqual(
            [match["term"] for match in matches],
            ["fear of failure", "happy"]
        )

    # ========================================================
    # 6. VAD SCORE CALCULATION
    # ========================================================

    def test_mean_vad_scores(self):

        matches = [

            {
                "scores": self.synthetic_lookup["happy"],
                "token_length": 1
            },

            {
                "scores": self.synthetic_lookup["confident"],
                "token_length": 1
            }
        ]

        scores = calculate_vad_scores(
            matches
        )

        self.assertEqual(
            scores["valence"],
            0.85
        )

        self.assertEqual(
            scores["arousal"],
            0.55
        )

        self.assertEqual(
            scores["dominance"],
            0.8
        )

    def test_no_matches_return_none_scores(self):

        scores = calculate_vad_scores([])

        self.assertIsNone(scores["valence"])

        self.assertIsNone(scores["arousal"])

        self.assertIsNone(scores["dominance"])

    # ========================================================
    # 7. LEXICAL COVERAGE
    # ========================================================

    def test_coverage_calculation(self):

        matches = [

            {
                "token_length": 2
            },

            {
                "token_length": 1
            }
        ]

        coverage = calculate_lexical_coverage(
            total_tokens=5,
            matched_terms=matches
        )

        self.assertEqual(
            coverage["matched_token_count"],
            3
        )

        self.assertEqual(
            coverage["unmatched_token_count"],
            2
        )

        self.assertEqual(
            coverage["coverage"],
            0.6
        )

    def test_zero_token_coverage(self):

        coverage = calculate_lexical_coverage(
            total_tokens=0,
            matched_terms=[]
        )

        self.assertEqual(
            coverage["coverage"],
            0.0
        )

    # ========================================================
    # 8. WARNING GENERATION
    # ========================================================

    def test_no_match_warning(self):

        warnings = generate_warnings(
            matched_terms=[],
            coverage=0.0,
            total_tokens=5
        )

        self.assertTrue(
            any(
                "No lexicon terms matched" in warning
                for warning in warnings
            )
        )

    def test_low_coverage_warning(self):

        matches = [

            {
                "token_length": 1,
                "scores": self.synthetic_lookup["happy"]
            }
        ]

        warnings = generate_warnings(
            matched_terms=matches,
            coverage=0.05,
            total_tokens=20
        )

        self.assertTrue(
            any(
                "Low lexical coverage" in warning
                for warning in warnings
            )
        )

    # ========================================================
    # 9. COMPLETE ANALYSIS
    # ========================================================

    def test_complete_analysis_output(self):

        result = analyze_vad(

            "I feel happy and confident.",

            lookup=self.synthetic_lookup,

            phrase_index=self.synthetic_phrase_index
        )

        self.assertIn(
            "vad_scores",
            result
        )

        self.assertIn(
            "coverage",
            result
        )

        self.assertIn(
            "matched_terms",
            result
        )

        self.assertEqual(
            result["vad_scores"]["valence"],
            0.85
        )

    def test_unknown_words_do_not_receive_zero_scores(self):

        result = analyze_vad(

            "qzxv plmok",

            lookup=self.synthetic_lookup,

            phrase_index=self.synthetic_phrase_index
        )

        self.assertIsNone(
            result["vad_scores"]["valence"]
        )

        self.assertEqual(
            result["matched_term_count"],
            0
        )

    # ========================================================
    # 10. REAL LEXICON INTEGRATION
    # ========================================================

    def test_real_lexicon_loads(self):

        self.assertEqual(
            len(self.lookup),
            54800
        )

    def test_real_lexicon_has_valid_vad_dimensions(self):

        entry = self.lookup["happy"]

        for dimension in [
            "valence",
            "arousal",
            "dominance"
        ]:

            self.assertGreaterEqual(
                entry[dimension],
                -1.0
            )

            self.assertLessEqual(
                entry[dimension],
                1.0
            )

    def test_phrase_lookup_is_consistent(self):

        # This is an integration check, not an assertion
        # that a specific phrase must exist in NRC VAD.

        phrase = "fear of failure"

        phrase_exists = phrase in self.lookup

        print(
            "\nActual lexicon contains 'fear of failure':",
            phrase_exists
        )

        if phrase_exists:

            self.assertIn(
                ("fear", "of", "failure"),
                self.phrase_index["fear"]
            )

    # ========================================================
    # 11. SAFETY OUTPUT
    # ========================================================

    def test_vad_does_not_claim_clinical_diagnosis(self):

        result = analyze_vad(

            "I feel happy.",

            lookup=self.synthetic_lookup,

            phrase_index=self.synthetic_phrase_index
        )

        self.assertFalse(
            result["clinical_diagnosis"]
        )

        self.assertFalse(
            result["actual_psychological_state_confirmed"]
        )


# ============================================================
# TEST EXECUTION
# ============================================================

if __name__ == "__main__":

    print("=" * 75)

    print("VAD ANALYZER REGRESSION TESTS")

    print("=" * 75)

    suite = unittest.defaultTestLoader.loadTestsFromTestCase(
        TestVADAnalyzer
    )

    runner = unittest.TextTestRunner(
        verbosity=2
    )

    result = runner.run(suite)

    total = result.testsRun

    failed = len(result.failures)

    errors = len(result.errors)

    passed = total - failed - errors

    print("\n" + "=" * 75)

    print(f"Total tests : {total}")

    print(f"Passed      : {passed}")

    print(f"Failed      : {failed}")

    print(f"Errors      : {errors}")

    if result.wasSuccessful():

        print("\nVAD ANALYZER REGRESSION TESTS: PASSED")

    else:

        print("\nVAD ANALYZER REGRESSION TESTS: FAILED")

    print("=" * 75)
