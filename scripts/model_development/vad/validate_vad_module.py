
"""
VAD Module - End-to-End Validation
==================================

Validates:
1. Configuration
2. NRC VAD lexicon
3. Lexicon lookup
4. Tokenization
5. Multiword matching
6. Zero-vector handling
7. VAD score calculation
8. Coverage calculation
9. Empty-match behavior
10. Input validation
11. Structured output
12. Safety-related output flags

These are engineering validation checks, not clinical validation.
"""

import json
from pathlib import Path

from scripts.model_development.vad import vad_config as config

from scripts.model_development.vad.vad_lexicon import (
    load_vad_lexicon
)

from scripts.model_development.vad.vad_analyzer import (
    analyze_vad,
    build_phrase_index,
    tokenize_text,
    calculate_vad_scores,
    is_zero_vector_term
)


# ============================================================
# 1. VALIDATION RESULT TRACKER
# ============================================================

validation_results = []


def record_check(name, passed, details=""):

    validation_results.append({

        "check": name,

        "passed": bool(passed),

        "details": details
    })

    status = "PASSED" if passed else "FAILED"

    print(f"[{status}] {name}")

    if details:

        print(f"         {details}")


# ============================================================
# 2. CONFIGURATION VALIDATION
# ============================================================

def validate_configuration():

    try:

        config.validate_config()

        record_check(
            "Configuration validation",
            True
        )

    except Exception as error:

        record_check(
            "Configuration validation",
            False,
            str(error)
        )


# ============================================================
# 3. LEXICON VALIDATION
# ============================================================

def validate_lexicon(lookup, summary):

    record_check(

        "Lexicon row count",

        summary["total_rows"] == 54800,

        f"Loaded {summary['total_rows']:,} entries"
    )

    record_check(

        "Unique lexicon terms",

        summary["unique_terms"] == 54800,

        f"Unique terms: {summary['unique_terms']:,}"
    )

    record_check(

        "Lookup dictionary size",

        len(lookup) == 54800,

        f"Lookup entries: {len(lookup):,}"
    )

    record_check(

        "Lexicon score ranges",

        all(

            -1.0 <= entry[dimension] <= 1.0

            for entry in lookup.values()

            for dimension in (
                "valence",
                "arousal",
                "dominance"
            )
        )
    )

    record_check(

        "Multiword entries retained",

        summary["term_structure"]["multiword_terms"] == 10073,

        (
            f"Multiword entries: "
            f"{summary['term_structure']['multiword_terms']:,}"
        )
    )


# ============================================================
# 4. TOKENIZATION VALIDATION
# ============================================================

def validate_tokenization():

    tokens = tokenize_text(
        "I don't feel happy today!"
    )

    expected = [
        "i",
        "don't",
        "feel",
        "happy",
        "today"
    ]

    record_check(

        "Contraction-preserving tokenization",

        tokens == expected,

        f"Tokens: {tokens}"
    )


# ============================================================
# 5. ZERO-VECTOR VALIDATION
# ============================================================

def validate_zero_vector_policy(lookup):

    zero_term = lookup.get("and")

    if zero_term is None:

        record_check(
            "Zero-vector classification",
            False,
            "Expected term 'and' not found."
        )

        return

    is_zero = is_zero_vector_term(zero_term)

    record_check(

        "Zero-vector classification",

        is_zero,

        "Zero-valued terms are recognized."
    )

    synthetic_matches = [

        {
            "scores": lookup["happy"],
            "token_length": 1
        },

        {
            "scores": lookup["and"],
            "token_length": 1
        }
    ]

    scores = calculate_vad_scores(
        synthetic_matches
    )

    expected_valence = round(
        lookup["happy"]["valence"],
        config.ROUNDING_PRECISION
    )

    record_check(

        "Zero-vector exclusion from averaging",

        scores["valence"] == expected_valence,

        f"Calculated valence: {scores['valence']}"
    )


# ============================================================
# 6. MULTIWORD MATCHING VALIDATION
# ============================================================

def validate_multiword_matching():

    synthetic_lookup = {

        "fear": {

            "term": "fear",

            "term_type": "unigram",

            "valence": -0.8,

            "arousal": 0.7,

            "dominance": -0.6
        },

        "fear of failure": {

            "term": "fear of failure",

            "term_type": "multi_word",

            "valence": -0.9,

            "arousal": 0.8,

            "dominance": -0.7
        },

        "failure": {

            "term": "failure",

            "term_type": "unigram",

            "valence": -0.7,

            "arousal": 0.6,

            "dominance": -0.5
        }
    }

    phrase_index = build_phrase_index(
        synthetic_lookup
    )

    result = analyze_vad(

        "fear of failure",

        lookup=synthetic_lookup,

        phrase_index=phrase_index
    )

    matched = result["matched_terms"]

    passed = (

        len(matched) == 1

        and matched[0]["term"] == "fear of failure"

        and result["coverage"]["coverage"] == 1.0
    )

    record_check(

        "Longest-match-first phrase handling",

        passed,

        f"Matched terms: {[item['term'] for item in matched]}"
    )


# ============================================================
# 7. REAL TEXT ANALYSIS VALIDATION
# ============================================================

def validate_real_text_analysis(lookup, phrase_index):

    sample = (
        "I feel happy and confident, "
        "but sometimes I experience fear of failure."
    )

    result = analyze_vad(

        sample,

        lookup=lookup,

        phrase_index=phrase_index
    )

    record_check(

        "Real-text analysis execution",

        result["matched_term_count"] > 0,

        f"Matched terms: {result['matched_term_count']}"
    )

    record_check(

        "Three VAD output dimensions",

        set(result["vad_scores"].keys()) == {

            "valence",
            "arousal",
            "dominance"
        },

        str(result["vad_scores"])
    )

    record_check(

        "Coverage range",

        0.0 <= result["coverage"]["coverage"] <= 1.0,

        f"Coverage: {result['coverage']['coverage']:.2%}"
    )

    record_check(

        "Zero-vector accounting",

        (
            result["matched_term_count"]
            ==
            result["scoring_term_count"]
            + result["zero_vector_term_count"]
        )
    )

    record_check(

        "Matched terms retained in output",

        len(result["matched_terms"])
        == result["matched_term_count"]
    )

    record_check(

        "Unmatched tokens retained",

        isinstance(result["unmatched_tokens"], list)
    )

    return result


# ============================================================
# 8. NO-MATCH VALIDATION
# ============================================================

def validate_no_match_behavior(lookup, phrase_index):

    result = analyze_vad(

        "qzxv plmok",

        lookup=lookup,

        phrase_index=phrase_index
    )

    passed = (

        result["matched_term_count"] == 0

        and result["vad_scores"]["valence"] is None

        and result["vad_scores"]["arousal"] is None

        and result["vad_scores"]["dominance"] is None
    )

    record_check(

        "No-match handling",

        passed,

        "Unavailable scores are returned as None."
    )


# ============================================================
# 9. SAFETY VALIDATION
# ============================================================

def validate_safety_output(result):

    record_check(

        "No clinical diagnosis claim",

        result["clinical_diagnosis"] is False
    )

    record_check(

        "No confirmed psychological-state claim",

        result["actual_psychological_state_confirmed"] is False
    )

    record_check(

        "Disclaimer included",

        isinstance(result["disclaimer"], str)
        and len(result["disclaimer"]) > 0
    )


# ============================================================
# 10. INPUT VALIDATION
# ============================================================

def validate_invalid_input(lookup, phrase_index):

    try:

        analyze_vad(

            "",

            lookup=lookup,

            phrase_index=phrase_index
        )

        passed = False

    except ValueError:

        passed = True

    record_check(
        "Empty text rejected",
        passed
    )

    try:

        analyze_vad(

            123,

            lookup=lookup,

            phrase_index=phrase_index
        )

        passed = False

    except TypeError:

        passed = True

    record_check(
        "Non-string input rejected",
        passed
    )


# ============================================================
# 11. SAVE VALIDATION REPORT
# ============================================================

def save_validation_report():

    output_directory = (
        config.OUTPUT_DIR
        / "module_validation"
    )

    output_directory.mkdir(
        parents=True,
        exist_ok=True
    )

    total = len(validation_results)

    passed = sum(
        item["passed"]
        for item in validation_results
    )

    failed = total - passed

    report = {

        "module_name": config.MODULE_NAME,

        "module_version": config.MODULE_VERSION,

        "total_checks": total,

        "passed_checks": passed,

        "failed_checks": failed,

        "validation_status": (
            "PASSED" if failed == 0 else "FAILED"
        ),

        "checks": validation_results,

        "clinical_validation": False,

        "real_world_accuracy_established": False
    }

    report_path = (
        output_directory
        / "vad_module_validation_summary.json"
    )

    with open(
        report_path,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            report,
            file,
            indent=4,
            ensure_ascii=False
        )

    return report_path, report


# ============================================================
# 12. MAIN VALIDATION PIPELINE
# ============================================================

def main():

    print("=" * 75)

    print("VAD MODULE END-TO-END VALIDATION")

    print("=" * 75)

    # Configuration.
    validate_configuration()

    # Load lexicon.
    lookup, summary = load_vad_lexicon()

    phrase_index = build_phrase_index(
        lookup
    )

    # Lexicon checks.
    validate_lexicon(
        lookup,
        summary
    )

    # Tokenization.
    validate_tokenization()

    # Scoring policy.
    validate_zero_vector_policy(
        lookup
    )

    # Phrase matching.
    validate_multiword_matching()

    # Real input.
    result = validate_real_text_analysis(
        lookup,
        phrase_index
    )

    # No-match handling.
    validate_no_match_behavior(
        lookup,
        phrase_index
    )

    # Safety outputs.
    validate_safety_output(
        result
    )

    # Input validation.
    validate_invalid_input(
        lookup,
        phrase_index
    )

    # Summary.
    report_path, report = save_validation_report()

    print("\n" + "=" * 75)

    print("VALIDATION SUMMARY")

    print("=" * 75)

    print(
        f"Total checks : {report['total_checks']}"
    )

    print(
        f"Passed       : {report['passed_checks']}"
    )

    print(
        f"Failed       : {report['failed_checks']}"
    )

    print(
        f"Status       : {report['validation_status']}"
    )

    print("\nValidation report:")

    print(report_path)

    print("=" * 75)

    if report["failed_checks"] > 0:

        raise SystemExit(
            "VAD MODULE VALIDATION FAILED."
        )

    print("\nVAD MODULE VALIDATION: PASSED")


if __name__ == "__main__":

    main()
