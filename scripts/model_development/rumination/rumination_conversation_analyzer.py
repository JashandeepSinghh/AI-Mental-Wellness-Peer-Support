
"""
Rumination Conversation Analyzer
---------------------------------

Purpose:
    Analyze possible rumination-related linguistic indicators
    across multiple conversation messages.

Components:
    1. Rule-based single-message detection
    2. Lexical topic recurrence
    3. Semantic similarity using Sentence Transformers
    4. Indicator-to-topic alignment
    5. Evidence-based conversation-level signal generation

Important:
    Semantic similarity is contextual evidence only.
    It is not a rumination probability or clinical diagnosis.
"""

import re

from collections import Counter
from functools import lru_cache
from typing import Any, Optional

from .rumination_config import (
    ENABLE_CROSS_TURN_ANALYSIS,
    MIN_MESSAGES_FOR_CROSS_TURN_ANALYSIS,
    ENABLE_SEMANTIC_ANALYSIS,
)

from .rumination_detector import detect_rumination

from .rumination_semantic_similarity import (
    RuminationSemanticSimilarity,
)


# ============================================================
# 1. TEXT TOKENIZATION
# ============================================================

STOPWORDS = {
    "a", "an", "the", "and", "or", "but",
    "i", "me", "my", "myself", "we", "our",
    "you", "your", "he", "she", "they", "them",
    "is", "am", "are", "was", "were", "be",
    "been", "being", "do", "does", "did",
    "have", "has", "had", "can", "could",
    "will", "would", "should", "might",
    "to", "of", "in", "on", "at", "for",
    "from", "with", "about", "as", "by",
    "it", "this", "that", "these", "those",
    "what", "when", "where", "who", "how",
    "there", "here", "then", "than",
    "just", "very", "really", "also",
    "keep", "keeps", "kept", "think",
    "thinking", "thought", "thoughts",
    "happen", "happened", "happening",
    "stop", "cannot", "cant", "couldnt",
    "dont", "doesnt", "didnt",
}


TOKEN_PATTERN = re.compile(
    r"\b[a-z]{3,}\b",
    re.IGNORECASE,
)


def extract_keywords(text: str) -> set[str]:
    """Extract meaningful keywords from a message."""

    tokens = TOKEN_PATTERN.findall(
        text.lower()
    )

    return {
        token
        for token in tokens
        if token not in STOPWORDS
    }


# ============================================================
# 2. MESSAGE SIMILARITY
# ============================================================

def calculate_jaccard_similarity(
    first_tokens: set[str],
    second_tokens: set[str],
) -> float:
    """Calculate lexical Jaccard similarity."""

    if not first_tokens or not second_tokens:
        return 0.0

    intersection = first_tokens.intersection(
        second_tokens
    )

    union = first_tokens.union(
        second_tokens
    )

    return round(
        len(intersection) / len(union),
        4,
    )


# ============================================================
# 3. CROSS-TURN TOPIC RECURRENCE
# ============================================================

def analyze_repeated_concerns(
    messages: list[dict[str, Any]],
) -> dict[str, Any]:
    """
    Identify recurring keywords and lexical message relationships.
    """

    message_keywords = []

    for message in messages:

        message_keywords.append(
            extract_keywords(
                message["text"]
            )
        )

    keyword_frequency = Counter()

    for keywords in message_keywords:

        # Count each keyword once per message.
        keyword_frequency.update(
            keywords
        )

    repeated_keywords = {
        word: count
        for word, count in keyword_frequency.items()
        if count >= 2
    }

    repeated_pairs = []

    for first_index in range(len(messages)):

        for second_index in range(
            first_index + 1,
            len(messages),
        ):

            first_tokens = message_keywords[
                first_index
            ]

            second_tokens = message_keywords[
                second_index
            ]

            shared_keywords = (
                first_tokens.intersection(
                    second_tokens
                )
            )

            similarity = calculate_jaccard_similarity(
                first_tokens,
                second_tokens,
            )

            if shared_keywords:

                repeated_pairs.append({

                    "first_message_id": messages[
                        first_index
                    ]["message_id"],

                    "second_message_id": messages[
                        second_index
                    ]["message_id"],

                    "shared_keywords": sorted(
                        shared_keywords
                    ),

                    "jaccard_similarity": similarity,

                })

    return {

        "repeated_keywords": repeated_keywords,

        "repeated_keyword_count": len(
            repeated_keywords
        ),

        "repeated_message_pairs": repeated_pairs,

        "repeated_pair_count": len(
            repeated_pairs
        ),

        "message_keywords": message_keywords,

    }


# ============================================================
# 4. VALIDATE CONVERSATION
# ============================================================

def validate_messages(
    messages: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Validate conversation message structure and identifiers."""

    if not isinstance(messages, list):
        raise TypeError(
            "Messages must be provided as a list."
        )

    if len(messages) < 2:
        raise ValueError(
            "At least two messages are required."
        )

    validated_messages = []

    for index, message in enumerate(messages):

        if not isinstance(message, dict):
            raise TypeError(
                f"Message {index} must be a dictionary."
            )

        if "text" not in message:
            raise ValueError(
                f"Message {index} is missing its text field."
            )

        text = message["text"]

        if not isinstance(text, str):
            raise TypeError(
                f"Message {index} text must be a string."
            )

        if not text.strip():
            raise ValueError(
                f"Message {index} cannot be empty."
            )

        message_id = message.get(
            "message_id",
            f"message_{index + 1}",
        )

        validated_messages.append({

            "message_id": str(message_id),

            "text": text,

        })

    message_ids = [
        message["message_id"]
        for message in validated_messages
    ]

    if len(message_ids) != len(set(message_ids)):
        raise ValueError(
            "Message IDs must be unique."
        )

    return validated_messages


# ============================================================
# 5. INDICATOR-TO-TOPIC LEXICAL ALIGNMENT
# ============================================================

def analyze_indicator_topic_alignment(
    messages: list[dict[str, Any]],
    message_results: list[dict[str, Any]],
    message_keywords: list[set[str]],
) -> dict[str, Any]:
    """
    Check whether an indicator-containing message shares
    meaningful keywords with another conversation message.
    """

    aligned_evidence = []

    for index, result in enumerate(message_results):

        if not result["indicator_detected"]:
            continue

        indicator_keywords = message_keywords[index]

        for other_index in range(len(messages)):

            if index == other_index:
                continue

            other_keywords = message_keywords[
                other_index
            ]

            shared_keywords = (
                indicator_keywords.intersection(
                    other_keywords
                )
            )

            if not shared_keywords:
                continue

            aligned_evidence.append({

                "indicator_message_id": messages[
                    index
                ]["message_id"],

                "related_message_id": messages[
                    other_index
                ]["message_id"],

                "indicator_categories": result[
                    "detected_categories"
                ],

                "shared_topic_keywords": sorted(
                    shared_keywords
                ),

                "jaccard_similarity": (
                    calculate_jaccard_similarity(
                        indicator_keywords,
                        other_keywords,
                    )
                ),

            })

    return {

        "topic_alignment_detected": (
            len(aligned_evidence) > 0
        ),

        "aligned_evidence_count": len(
            aligned_evidence
        ),

        "aligned_evidence": aligned_evidence,

    }


# ============================================================
# 6. SEMANTIC ENGINE MANAGEMENT
# ============================================================

@lru_cache(maxsize=1)
def get_semantic_engine() -> RuminationSemanticSimilarity:
    """
    Load the semantic engine once and reuse it.

    Lazy initialization avoids loading the transformer model
    when the analyzer is imported but not executed.
    """

    return RuminationSemanticSimilarity()


# ============================================================
# 7. SEMANTIC CONVERSATION ANALYSIS
# ============================================================

def analyze_semantic_relationships(
    messages: list[dict[str, Any]],
    message_results: list[dict[str, Any]],
    semantic_engine: Optional[
        RuminationSemanticSimilarity
    ] = None,
) -> dict[str, Any]:
    """
    Calculate semantic relationships between conversation messages.

    Semantic similarity is used to rank potentially related
    messages for contextual review.

    It does not independently trigger the final rumination signal.
    """

    if not ENABLE_SEMANTIC_ANALYSIS:

        return {

            "semantic_analysis_enabled": False,

            "semantic_analysis_successful": False,

            "semantic_comparisons": [],

            "indicator_semantic_candidates": [],

            "semantic_candidate_count": 0,

            "error": None,

        }

    try:

        if semantic_engine is None:

            semantic_engine = get_semantic_engine()

        message_texts = [
            message["text"]
            for message in messages
        ]

        comparisons = semantic_engine.compare_messages(
            message_texts
        )

        # --------------------------------------------------------
        # FIND SEMANTIC PAIRS CONTAINING AN INDICATOR
        # --------------------------------------------------------

        indicator_candidates = []

        for comparison in comparisons:

            first_index = comparison[
                "message_1_index"
            ]

            second_index = comparison[
                "message_2_index"
            ]

            first_result = message_results[
                first_index
            ]

            second_result = message_results[
                second_index
            ]

            first_has_indicator = (
                first_result["indicator_detected"]
            )

            second_has_indicator = (
                second_result["indicator_detected"]
            )

            if not (
                first_has_indicator
                or second_has_indicator
            ):
                continue

            if first_has_indicator and second_has_indicator:

                indicator_categories = sorted(
                    set(first_result["detected_categories"])
                    | set(second_result["detected_categories"])
                )

            elif first_has_indicator:

                indicator_categories = (
                    first_result["detected_categories"]
                )

            else:

                indicator_categories = (
                    second_result["detected_categories"]
                )

            indicator_candidates.append({

                "first_message_id": messages[
                    first_index
                ]["message_id"],

                "second_message_id": messages[
                    second_index
                ]["message_id"],

                "first_message_has_indicator": (
                    first_has_indicator
                ),

                "second_message_has_indicator": (
                    second_has_indicator
                ),

                "indicator_categories": (
                    indicator_categories
                ),

                "similarity_score": comparison[
                    "similarity_score"
                ],

                "evidence_role": (
                    "Ranked semantic candidate for contextual review"
                ),

            })

        return {

            "semantic_analysis_enabled": True,

            "semantic_analysis_successful": True,

            "semantic_comparisons": comparisons,

            "indicator_semantic_candidates": (
                indicator_candidates
            ),

            "semantic_candidate_count": len(
                indicator_candidates
            ),

            "error": None,

        }

    except Exception as error:

        # Semantic analysis must not prevent the existing
        # rule-based and lexical analyzer from running.

        return {

            "semantic_analysis_enabled": True,

            "semantic_analysis_successful": False,

            "semantic_comparisons": [],

            "indicator_semantic_candidates": [],

            "semantic_candidate_count": 0,

            "error": (
                f"{type(error).__name__}: {error}"
            ),

        }


# ============================================================
# 8. CONVERSATION-LEVEL RUMINATION ANALYZER
# ============================================================

def analyze_conversation(
    messages: list[dict[str, Any]],
    semantic_engine: Optional[
        RuminationSemanticSimilarity
    ] = None,
    enable_semantic_analysis: Optional[bool] = None,
) -> dict[str, Any]:
    """
    Analyze rumination-related linguistic indicators across messages.

    Parameters:
        messages:
            List of dictionaries containing text and optional message_id.

        semantic_engine:
            Optional preloaded semantic engine. Supplying an engine
            allows reuse across multiple conversation analyses.

        enable_semantic_analysis:
            Optional per-call override of semantic analysis.
    """

    messages = validate_messages(
        messages
    )

    if not ENABLE_CROSS_TURN_ANALYSIS:

        raise RuntimeError(
            "Cross-turn analysis is disabled in rumination_config.py."
        )

    if len(messages) < MIN_MESSAGES_FOR_CROSS_TURN_ANALYSIS:

        raise ValueError(
            "Conversation does not meet the configured "
            "minimum message count."
        )

    # --------------------------------------------------------
    # 1. SINGLE-MESSAGE DETECTION
    # --------------------------------------------------------

    message_results = []

    for message in messages:

        result = detect_rumination(
            message["text"]
        )

        message_results.append({

            "message_id": message["message_id"],

            "indicator_detected": result[
                "any_indicator_detected"
            ],

            "detected_categories": result[
                "detected_categories"
            ],

            "total_pattern_matches": result[
                "total_pattern_matches"
            ],

        })

    # --------------------------------------------------------
    # 2. REPEATED CONCERN ANALYSIS
    # --------------------------------------------------------

    recurrence = analyze_repeated_concerns(
        messages
    )

    # --------------------------------------------------------
    # 3. COUNT INDICATOR MESSAGES
    # --------------------------------------------------------

    messages_with_indicators = [

        item

        for item in message_results

        if item["indicator_detected"]

    ]

    indicator_message_count = len(
        messages_with_indicators
    )

    # --------------------------------------------------------
    # 4. LEXICAL INDICATOR-TO-TOPIC ALIGNMENT
    # --------------------------------------------------------

    lexical_alignment = analyze_indicator_topic_alignment(
        messages,
        message_results,
        recurrence["message_keywords"],
    )

    # --------------------------------------------------------
    # 5. SEMANTIC ANALYSIS
    # --------------------------------------------------------

    semantic_enabled = (
        ENABLE_SEMANTIC_ANALYSIS
        if enable_semantic_analysis is None
        else enable_semantic_analysis
    )

    if not isinstance(semantic_enabled, bool):
        raise TypeError(
            "enable_semantic_analysis must be True, False, or None."
        )

    if semantic_enabled:

        semantic_analysis = analyze_semantic_relationships(
            messages,
            message_results,
            semantic_engine=semantic_engine,
        )

    else:

        semantic_analysis = {

            "semantic_analysis_enabled": False,

            "semantic_analysis_successful": False,

            "semantic_comparisons": [],

            "indicator_semantic_candidates": [],

            "semantic_candidate_count": 0,

            "error": None,

        }

    # --------------------------------------------------------
    # 6. FINAL SIGNAL DECISION
    # --------------------------------------------------------

    has_rumination_language = (
        indicator_message_count > 0
    )

    has_lexically_aligned_recurrence = (
        lexical_alignment["topic_alignment_detected"]
    )

    # IMPORTANT:
    # Semantic similarity alone does not activate this signal.
    #
    # We retain the validated lexical safeguard until semantic
    # alignment has been independently validated on a suitable
    # labelled conversation dataset.

    recurring_rumination_signal = (
        has_rumination_language
        and has_lexically_aligned_recurrence
    )

    # --------------------------------------------------------
    # 7. CATEGORY FREQUENCY
    # --------------------------------------------------------

    category_frequency = Counter()

    for result in message_results:

        category_frequency.update(
            result["detected_categories"]
        )

    # --------------------------------------------------------
    # 8. INTERPRETATION
    # --------------------------------------------------------

    if recurring_rumination_signal:

        interpretation = (
            "A rumination-related linguistic indicator "
            "shares topic keywords with another conversation "
            "message. Semantic comparisons are provided as "
            "additional contextual evidence. Human contextual "
            "review is recommended."
        )

    elif has_rumination_language:

        interpretation = (
            "Rumination-related linguistic patterns were "
            "detected, but they could not be aligned with "
            "a repeated topic using the current lexical method. "
            "Semantic candidates, when available, may assist "
            "contextual review."
        )

    elif recurrence["repeated_keyword_count"] > 0:

        interpretation = (
            "Repeated topic keywords were identified, but "
            "no configured rumination-related linguistic "
            "indicator was detected."
        )

    else:

        interpretation = (
            "No combined recurring-concern and rumination "
            "language signal was identified."
        )

    # --------------------------------------------------------
    # 9. LIMITATIONS
    # --------------------------------------------------------

    limitations = [

        "Keyword overlap does not establish semantic equivalence.",

        "A shared keyword does not prove two messages discuss the same concern.",

        "Different wording may describe the same underlying concern.",

        "Semantic similarity can be high during ordinary topic repetition.",

        "Semantic similarity is not a rumination probability.",

        "Semantic similarity alone does not trigger the final signal.",

        "Message order and time intervals are not analyzed.",

        "The system does not diagnose rumination or mental illness.",

        "Human contextual review is necessary before interpreting concerning patterns.",

    ]

    if not semantic_analysis["semantic_analysis_successful"]:

        limitations.append(
            "Semantic analysis was unavailable or unsuccessful; "
            "the lexical analysis remains operational."
        )

    # --------------------------------------------------------
    # 10. FINAL RESULT
    # --------------------------------------------------------

    return {

        "module": "Rumination Detection",

        "version": "1.2.0",

        "analysis_type": "conversation_level",

        "message_count": len(messages),

        "messages_with_indicators": (
            indicator_message_count
        ),

        "message_results": message_results,

        "repeated_concerns": {

            key: value

            for key, value in recurrence.items()

            if key != "message_keywords"

        },

        "topic_alignment": lexical_alignment,

        "semantic_analysis": semantic_analysis,

        "category_frequency": dict(
            category_frequency
        ),

        "recurring_rumination_signal": (
            recurring_rumination_signal
        ),

        "interpretation": interpretation,

        "limitations": limitations,

    }


# ============================================================
# 11. CONSOLE RESULT DISPLAY
# ============================================================

def print_conversation_result(
    result: dict[str, Any],
) -> None:

    print("\n" + "=" * 75)
    print("CONVERSATION-LEVEL RUMINATION ANALYSIS")
    print("=" * 75)

    print(
        f"Messages analyzed: {result['message_count']}"
    )

    print(
        f"Messages with indicators: "
        f"{result['messages_with_indicators']}"
    )

    print(
        f"Recurring concern signal: "
        f"{result['recurring_rumination_signal']}"
    )

    print("\nMESSAGE-LEVEL RESULTS")

    for item in result["message_results"]:

        print(f"\n{item['message_id']}")

        print(
            f"Indicator detected: "
            f"{item['indicator_detected']}"
        )

        print(
            f"Categories: "
            f"{item['detected_categories']}"
        )

    print("\nREPEATED CONCERNS")

    print(
        result["repeated_concerns"][
            "repeated_keywords"
        ]
    )

    print("\nLEXICAL TOPIC ALIGNMENT")

    print(
        "Alignment detected:",
        result["topic_alignment"][
            "topic_alignment_detected"
        ],
    )

    for item in result["topic_alignment"][
        "aligned_evidence"
    ]:

        print(
            f"{item['indicator_message_id']} <-> "
            f"{item['related_message_id']}"
        )

        print(
            "Shared topic keywords:",
            item["shared_topic_keywords"],
        )

    # --------------------------------------------------------
    # SEMANTIC RESULTS
    # --------------------------------------------------------

    print("\nSEMANTIC ANALYSIS")

    semantic = result["semantic_analysis"]

    print(
        "Analysis enabled:",
        semantic["semantic_analysis_enabled"]
    )

    print(
        "Analysis successful:",
        semantic["semantic_analysis_successful"]
    )

    if semantic["error"]:

        print(
            "Semantic error:",
            semantic["error"]
        )

    print(
        "Indicator-related semantic candidates:",
        semantic["semantic_candidate_count"]
    )

    for candidate in semantic[
        "indicator_semantic_candidates"
    ]:

        print("-" * 65)

        print(
            f"{candidate['first_message_id']} <-> "
            f"{candidate['second_message_id']}"
        )

        print(
            "Similarity:",
            candidate["similarity_score"]
        )

        print(
            "Indicator categories:",
            candidate["indicator_categories"]
        )

    # --------------------------------------------------------
    # CATEGORY FREQUENCY
    # --------------------------------------------------------

    print("\nCATEGORY FREQUENCY")

    print(
        result["category_frequency"]
    )

    # --------------------------------------------------------
    # INTERPRETATION
    # --------------------------------------------------------

    print("\nINTERPRETATION")

    print(
        result["interpretation"]
    )

    print("\nLIMITATIONS")

    for limitation in result["limitations"]:

        print(f"- {limitation}")

    print("\n" + "=" * 75)


# ============================================================
# 12. MANUAL EXECUTION
# ============================================================

if __name__ == "__main__":

    sample_conversation = [

        {
            "message_id": "message_1",

            "text": (
                "I made a mistake during my examination."
            ),
        },

        {
            "message_id": "message_2",

            "text": (
                "I keep remembering what happened "
                "in that examination."
            ),
        },

        {
            "message_id": "message_3",

            "text": (
                "I cannot stop thinking about that mistake."
            ),
        },

    ]

    result = analyze_conversation(
        sample_conversation
    )

    print_conversation_result(
        result
    )
