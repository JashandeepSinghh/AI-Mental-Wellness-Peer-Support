
"""
VAD Text Analyzer
=================

NRC VAD Lexicon-based text analysis.

Features:
1. Text validation and preprocessing.
2. Contraction-preserving tokenization.
3. Unigram and multiword expression matching.
4. Longest-match-first strategy.
5. Non-overlapping lexical matching.
6. Mean VAD score calculation.
7. Zero-vector term handling.
8. Lexical coverage and uncertainty reporting.

Scoring policy:
- All matched terms contribute to lexical coverage.
- Zero-vector terms are retained but excluded from averages.
- Non-zero terms determine mean VAD scores.
- No scoring terms means VAD scores are None.

IMPORTANT:
These are linguistic indicators, not clinical measurements.
"""

import re
from datetime import datetime, timezone

from scripts.model_development.vad import vad_config as config

from scripts.model_development.vad.vad_lexicon import (
    load_vad_lexicon,
    normalize_term
)


# ============================================================
# 1. TOKENIZER
# ============================================================

def tokenize_text(text):

    normalized_text = text.casefold()

    return re.findall(
        r"[^\W_]+(?:['’][^\W_]+)*",
        normalized_text,
        flags=re.UNICODE
    )


# ============================================================
# 2. TEXT VALIDATION
# ============================================================

def validate_text(text):

    if not isinstance(text, str):

        raise TypeError(
            "Input text must be a string."
        )

    text = text.strip()

    if len(text) < config.MIN_TEXT_LENGTH:

        raise ValueError(
            "Input text is too short."
        )

    if len(text) > config.MAX_TEXT_LENGTH:

        raise ValueError(
            f"Input text exceeds {config.MAX_TEXT_LENGTH} characters."
        )

    return text


# ============================================================
# 3. BUILD PHRASE INDEX
# ============================================================

def build_phrase_index(lookup):

    phrase_index = {}

    for term in lookup:

        tokens = tuple(
            normalize_term(term).split()
        )

        if len(tokens) <= 1:
            continue

        phrase_index.setdefault(
            tokens[0],
            []
        ).append(tokens)

    for first_token in phrase_index:

        phrase_index[first_token].sort(
            key=lambda phrase: (
                -len(phrase),
                phrase
            )
        )

    return phrase_index


# ============================================================
# 4. FIND LONGEST MATCH
# ============================================================

def find_longest_match(
    tokens,
    start_index,
    lookup,
    phrase_index
):

    current_token = tokens[start_index]

    # Multiword expressions first.
    for phrase in phrase_index.get(current_token, []):

        phrase_length = len(phrase)

        end_index = start_index + phrase_length

        if end_index > len(tokens):
            continue

        if tuple(tokens[start_index:end_index]) == phrase:

            term = " ".join(phrase)

            if term in lookup:

                return {

                    "term": term,

                    "start_index": start_index,

                    "end_index": end_index,

                    "token_length": phrase_length,

                    "scores": lookup[term]
                }

    # Unigram fallback.
    if current_token in lookup:

        entry = lookup[current_token]

        return {

            "term": current_token,

            "start_index": start_index,

            "end_index": start_index + 1,

            "token_length": 1,

            "scores": entry
        }

    return None


# ============================================================
# 5. MATCH LEXICON TERMS
# ============================================================

def match_lexicon_terms(
    tokens,
    lookup,
    phrase_index
):

    matched_terms = []

    unmatched_tokens = []

    index = 0

    while index < len(tokens):

        match = find_longest_match(
            tokens,
            index,
            lookup,
            phrase_index
        )

        if match is not None:

            matched_terms.append(match)

            index = match["end_index"]

        else:

            unmatched_tokens.append(tokens[index])

            index += 1

    return matched_terms, unmatched_tokens


# ============================================================
# 6. ZERO-VECTOR CLASSIFICATION
# ============================================================

def is_zero_vector_term(scores):

    return all(

        scores[dimension] == 0.0

        for dimension in (
            "valence",
            "arousal",
            "dominance"
        )
    )


# ============================================================
# 7. CALCULATE VAD SCORES
# ============================================================

def calculate_vad_scores(matched_terms):

    scoring_terms = [

        match

        for match in matched_terms

        if not is_zero_vector_term(match["scores"])
    ]

    if not scoring_terms:

        return {

            "valence": None,

            "arousal": None,

            "dominance": None
        }

    dimensions = {

        "valence": [],

        "arousal": [],

        "dominance": []
    }

    for match in scoring_terms:

        for dimension in dimensions:

            dimensions[dimension].append(
                match["scores"][dimension]
            )

    return {

        dimension: round(
            sum(values) / len(values),
            config.ROUNDING_PRECISION
        )

        for dimension, values in dimensions.items()
    }


# ============================================================
# 8. CALCULATE LEXICAL COVERAGE
# ============================================================

def calculate_lexical_coverage(
    total_tokens,
    matched_terms
):

    if total_tokens == 0:

        return {

            "matched_token_count": 0,

            "unmatched_token_count": 0,

            "coverage": 0.0
        }

    matched_count = sum(

        match["token_length"]

        for match in matched_terms
    )

    unmatched_count = total_tokens - matched_count

    return {

        "matched_token_count": matched_count,

        "unmatched_token_count": unmatched_count,

        "coverage": round(
            matched_count / total_tokens,
            4
        )
    }


# ============================================================
# 9. FORMAT MATCHED TERMS
# ============================================================

def format_matched_terms(matched_terms):

    formatted = []

    for match in matched_terms:

        scores = match["scores"]

        zero_vector = is_zero_vector_term(scores)

        formatted.append({

            "term": scores["term"],

            "term_type": scores["term_type"],

            "matched_token_count": match["token_length"],

            "valence": scores["valence"],

            "arousal": scores["arousal"],

            "dominance": scores["dominance"],

            "included_in_vad_average": not zero_vector
        })

    return formatted


# ============================================================
# 10. GENERATE WARNINGS
# ============================================================

def generate_warnings(
    matched_terms,
    coverage,
    total_tokens,
    scoring_term_count=None
):
    """
    Generate interpretation warnings.

    Backward compatibility:
    Older calls can omit scoring_term_count.
    It will be calculated automatically.
    """

    warnings = []

    # --------------------------------------------------------
    # A. Automatically calculate scoring term count
    # --------------------------------------------------------

    if scoring_term_count is None:

        scoring_term_count = sum(

            not is_zero_vector_term(match["scores"])

            for match in matched_terms
        )

    # --------------------------------------------------------
    # B. No lexical matches
    # --------------------------------------------------------

    if not matched_terms:

        warnings.append(
            "No lexicon terms matched. VAD scores are unavailable; "
            "this must not be interpreted as emotional neutrality."
        )

        return warnings

    # --------------------------------------------------------
    # C. No scoring terms
    # --------------------------------------------------------

    if scoring_term_count == 0:

        warnings.append(
            "All matched lexicon entries have zero scores across "
            "the three VAD dimensions. VAD scores are unavailable."
        )

    # --------------------------------------------------------
    # D. Low coverage
    # --------------------------------------------------------

    if coverage < config.LOW_COVERAGE_WARNING_THRESHOLD:

        warnings.append(
            "Low lexical coverage: most input tokens were not "
            "represented by matched lexicon entries."
        )

    # --------------------------------------------------------
    # E. Short input
    # --------------------------------------------------------

    if total_tokens < 3:

        warnings.append(
            "Very short text provides limited linguistic context."
        )

    # --------------------------------------------------------
    # F. Limited scoring terms
    # --------------------------------------------------------

    if 0 < scoring_term_count < config.MINIMUM_MATCHED_TERMS:

        warnings.append(
            "Insufficient non-zero matched terms for reliable "
            "lexical interpretation."
        )

    # --------------------------------------------------------
    # G. Context limitation
    # --------------------------------------------------------

    if scoring_term_count > 0:

        warnings.append(
            "Negation and contextual meaning are not used to modify "
            "lexicon scores in this version."
        )

    return warnings


# ============================================================
# 11. COMPLETE VAD ANALYSIS
# ============================================================

def analyze_vad(
    text,
    lookup=None,
    phrase_index=None
):

    # Validate input.
    text = validate_text(text)

    # Load lexicon if not supplied.
    if lookup is None:

        lookup, _ = load_vad_lexicon()

    # Build phrase index if not supplied.
    if phrase_index is None:

        phrase_index = build_phrase_index(lookup)

    # Tokenize input.
    tokens = tokenize_text(text)

    # Match terms.
    matched_terms, unmatched_tokens = match_lexicon_terms(
        tokens,
        lookup,
        phrase_index
    )

    # Separate scoring and zero-vector terms.
    scoring_terms = [

        match

        for match in matched_terms

        if not is_zero_vector_term(match["scores"])
    ]

    zero_vector_terms = [

        match["term"]

        for match in matched_terms

        if is_zero_vector_term(match["scores"])
    ]

    # Calculate VAD scores.
    vad_scores = calculate_vad_scores(matched_terms)

    # Calculate coverage.
    coverage_statistics = calculate_lexical_coverage(
        len(tokens),
        matched_terms
    )

    # Generate warnings.
    warnings = generate_warnings(

        matched_terms,

        coverage_statistics["coverage"],

        len(tokens),

        scoring_term_count=len(scoring_terms)
    )

    # Format terms.
    formatted_matches = format_matched_terms(
        matched_terms
    )

    return {

        "module_name": config.MODULE_NAME,

        "module_version": config.MODULE_VERSION,

        "timestamp_utc": datetime.now(
            timezone.utc
        ).isoformat(),

        "input_text": text,

        "token_count": len(tokens),

        "matched_term_count": len(matched_terms),

        "scoring_term_count": len(scoring_terms),

        "zero_vector_term_count": len(zero_vector_terms),

        "zero_vector_terms": zero_vector_terms,

        "matched_terms": formatted_matches,

        "unmatched_tokens": unmatched_tokens,

        "vad_scores": vad_scores,

        "coverage": coverage_statistics,

        "warnings": warnings,

        "scoring_policy": (
            "All matched terms contribute to lexical coverage. "
            "Terms with all three VAD scores equal to zero are "
            "excluded from VAD score averaging."
        ),

        "clinical_diagnosis": False,

        "actual_psychological_state_confirmed": False,

        "disclaimer": config.DISCLAIMER
    }


# ============================================================
# 12. MAIN
# ============================================================

if __name__ == "__main__":

    print("=" * 70)

    print("VAD TEXT ANALYSIS ENGINE")

    print("=" * 70)

    lexicon, _ = load_vad_lexicon()

    phrase_index = build_phrase_index(lexicon)

    sample_text = (
        "I feel happy and confident, "
        "but sometimes I experience fear of failure."
    )

    result = analyze_vad(
        sample_text,
        lookup=lexicon,
        phrase_index=phrase_index
    )

    print("\nInput text:")
    print(result["input_text"])

    print("\nToken count:")
    print(result["token_count"])

    print("\nMatched terms:")

    for term in result["matched_terms"]:

        print(
            f"  {term['term']} "
            f"({term['term_type']}) | "
            f"Included in average: "
            f"{term['included_in_vad_average']}"
        )

    print("\nVAD scores:")

    for dimension, score in result["vad_scores"].items():

        print(f"  {dimension}: {score}")

    print("\nScoring statistics:")

    print(
        "  Total matched terms:",
        result["matched_term_count"]
    )

    print(
        "  Non-zero scoring terms:",
        result["scoring_term_count"]
    )

    print(
        "  Zero-vector terms:",
        result["zero_vector_term_count"]
    )

    print("\nLexical coverage:")

    print(
        f"  Matched tokens: "
        f"{result['coverage']['matched_token_count']}"
    )

    print(
        f"  Unmatched tokens: "
        f"{result['coverage']['unmatched_token_count']}"
    )

    print(
        f"  Coverage: "
        f"{result['coverage']['coverage']:.2%}"
    )

    print("\nWarnings:")

    for warning in result["warnings"]:

        print(f"  - {warning}")

    if not result["warnings"]:

        print("  None")

    print("\nVAD text analysis: COMPLETED")

    print("=" * 70)
