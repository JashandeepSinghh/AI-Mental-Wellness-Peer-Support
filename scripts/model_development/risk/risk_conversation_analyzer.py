
"""
Risk Conversation Analyzer
==========================

Analyzes multiple student messages using the existing
message-level risk-language detection engine.

Responsibilities:
1. Analyze individual messages.
2. Aggregate message-level risk categories.
3. Identify recurring linguistic indicators.
4. Track category progression across messages.
5. Determine whether human review is recommended.

IMPORTANT:
- This module does not diagnose.
- Repeated potential concerns do not automatically become urgent.
- Conversation patterns do not establish actual risk or intent.
- No explicit risk signal does not establish that a student is safe.
"""

from collections import Counter

from scripts.model_development.risk import risk_config as config

from scripts.model_development.risk.risk_detector import (
    detect_risk_language
)


# ============================================================
# 1. MODULE CONSTANTS
# ============================================================

CATEGORY = config.RISK_CATEGORIES

CONVERSATION_CATEGORY_PRIORITY = {

    CATEGORY["NO_EXPLICIT_RISK_SIGNAL"]: 0,

    CATEGORY["UNCERTAIN_OR_AMBIGUOUS"]: 1,

    CATEGORY["POTENTIAL_CONCERN"]: 2,

    CATEGORY["URGENT_LANGUAGE_SIGNAL"]: 3
}

MIN_MESSAGES = 1
MAX_MESSAGES = 100


# ============================================================
# 2. VALIDATE CONVERSATION INPUT
# ============================================================

def validate_conversation(messages):
    """
    Validate and normalize conversation messages.

    Accepted formats:

    A. List of strings:
       [
           "I feel stressed.",
           "Nothing seems to improve."
       ]

    B. List of dictionaries:
       [
           {"role": "user", "content": "I feel stressed."},
           {"role": "assistant", "content": "I am listening."}
       ]

    Only user/student messages are analyzed.
    Assistant and system messages are excluded.
    """

    if not isinstance(messages, list):

        raise TypeError(
            "Conversation must be provided as a list."
        )

    if not MIN_MESSAGES <= len(messages) <= MAX_MESSAGES:

        raise ValueError(
            f"Conversation must contain between "
            f"{MIN_MESSAGES} and {MAX_MESSAGES} messages."
        )

    normalized_messages = []

    for index, message in enumerate(messages):

        # ----------------------------------------------------
        # A. Plain-text message
        # ----------------------------------------------------

        if isinstance(message, str):

            role = "user"
            content = message

        # ----------------------------------------------------
        # B. Structured message
        # ----------------------------------------------------

        elif isinstance(message, dict):

            role = message.get("role", "user")
            content = message.get("content")

            if not isinstance(role, str):

                raise TypeError(
                    f"Message {index + 1}: role must be a string."
                )

            role = role.strip().lower()

            if role not in {
                "user",
                "student",
                "assistant",
                "system"
            }:

                raise ValueError(
                    f"Message {index + 1}: unsupported role '{role}'."
                )

            # Do not analyze assistant/system content.
            if role in {"assistant", "system"}:
                continue

        else:

            raise TypeError(
                f"Message {index + 1} must be a string or dictionary."
            )

        # ----------------------------------------------------
        # C. Validate student content
        # ----------------------------------------------------

        if not isinstance(content, str):

            raise TypeError(
                f"Message {index + 1}: content must be a string."
            )

        content = content.strip()

        if not content:

            raise ValueError(
                f"Message {index + 1}: content cannot be empty."
            )

        normalized_messages.append({

            "message_index": index + 1,

            "role": "user",

            "content": content
        })

    if not normalized_messages:

        raise ValueError(
            "No student/user messages are available for analysis."
        )

    return normalized_messages


# ============================================================
# 3. ANALYZE INDIVIDUAL MESSAGES
# ============================================================

def analyze_individual_messages(messages):
    """
    Run the existing risk detector on every student message.
    """

    message_results = []

    for message in messages:

        detection = detect_risk_language(
            message["content"]
        )

        message_results.append({

            "message_index": message["message_index"],

            "risk_category": detection["risk_category"],

            "escalation_level": detection["escalation_level"],

            "requires_human_review": (
                detection["requires_human_review"]
            ),

            "detected_indicators": (
                detection["detected_indicators"]
            ),

            "uncertainty_notes": (
                detection["uncertainty_notes"]
            ),

            "explanation": detection["explanation"]["summary"],

            "clinical_diagnosis": False,

            "actual_risk_confirmed": False
        })

    return message_results


# ============================================================
# 4. COUNT RISK CATEGORIES
# ============================================================

def calculate_category_counts(message_results):
    """
    Count the category assigned to each analyzed message.
    """

    counts = Counter(
        result["risk_category"]
        for result in message_results
    )

    return {

        category: counts.get(category, 0)

        for category in CONVERSATION_CATEGORY_PRIORITY
    }


# ============================================================
# 5. IDENTIFY RECURRING INDICATORS
# ============================================================

def identify_recurring_indicators(message_results):
    """
    Identify linguistic indicator categories appearing
    in two or more separate messages.

    Recurrence is not treated as proof of increased severity.
    """

    category_message_indexes = {}

    for result in message_results:

        message_index = result["message_index"]

        for indicator_category in result["detected_indicators"]:

            if indicator_category not in category_message_indexes:

                category_message_indexes[
                    indicator_category
                ] = []

            category_message_indexes[
                indicator_category
            ].append(message_index)

    recurring_indicators = {}

    for category, message_indexes in category_message_indexes.items():

        if len(message_indexes) >= 2:

            recurring_indicators[category] = {

                "message_count": len(message_indexes),

                "message_indexes": message_indexes,

                "interpretation": (
                    "This linguistic indicator appeared in multiple "
                    "messages. Recurrence alone does not establish "
                    "increased risk or actual intent."
                )
            }

    return recurring_indicators


# ============================================================
# 6. ANALYZE CONVERSATION PROGRESSION
# ============================================================

def analyze_conversation_progression(message_results):
    """
    Track the sequence of message-level classifications.

    This reports observed category changes without interpreting
    them as a confirmed psychological progression.
    """

    progression = [

        {
            "message_index": result["message_index"],

            "risk_category": result["risk_category"],

            "escalation_level": result["escalation_level"]

        }

        for result in message_results
    ]

    category_changes = []

    for previous, current in zip(
        message_results,
        message_results[1:]
    ):

        if previous["risk_category"] != current["risk_category"]:

            previous_priority = CONVERSATION_CATEGORY_PRIORITY[
                previous["risk_category"]
            ]

            current_priority = CONVERSATION_CATEGORY_PRIORITY[
                current["risk_category"]
            ]

            category_changes.append({

                "from_message": previous["message_index"],

                "to_message": current["message_index"],

                "previous_category": previous["risk_category"],

                "current_category": current["risk_category"],

                "direction": (
                    "higher_category_priority"
                    if current_priority > previous_priority
                    else (
                        "lower_category_priority"
                        if current_priority < previous_priority
                        else "unchanged_priority"
                    )
                )
            })

    return {

        "message_progression": progression,

        "category_changes": category_changes,

        "interpretation": (
            "This sequence records changes in message-level "
            "linguistic classifications. It does not establish "
            "a psychological or clinical progression."
        )
    }


# ============================================================
# 7. DETERMINE CONVERSATION CATEGORY
# ============================================================

def determine_conversation_category(message_results):
    """
    Return the highest-priority category observed in any
    individual message.

    Recurrence alone never upgrades severity.
    """

    highest_category = max(

        (
            result["risk_category"]
            for result in message_results
        ),

        key=lambda category: (
            CONVERSATION_CATEGORY_PRIORITY[category]
        )
    )

    return highest_category


# ============================================================
# 8. DETERMINE CONVERSATION HUMAN REVIEW
# ============================================================

def requires_conversation_human_review(conversation_category):

    return conversation_category in {

        CATEGORY["POTENTIAL_CONCERN"],

        CATEGORY["URGENT_LANGUAGE_SIGNAL"],

        CATEGORY["UNCERTAIN_OR_AMBIGUOUS"]
    }


# ============================================================
# 9. GENERATE CONVERSATION SUMMARY
# ============================================================

def generate_conversation_summary(
    conversation_category,
    category_counts,
    recurring_indicators,
    message_results
):

    analyzed_count = len(message_results)

    review_count = sum(

        result["requires_human_review"]

        for result in message_results
    )

    if conversation_category == CATEGORY["URGENT_LANGUAGE_SIGNAL"]:

        summary = (
            "At least one message matched an urgent-language "
            "pattern. Prompt human review is recommended. "
            "This does not confirm actual intent or danger."
        )

    elif conversation_category == CATEGORY["POTENTIAL_CONCERN"]:

        summary = (
            "At least one message matched a potential-concern "
            "pattern. Supportive follow-up and human review "
            "are recommended."
        )

    elif conversation_category == CATEGORY["UNCERTAIN_OR_AMBIGUOUS"]:

        summary = (
            "The conversation contains ambiguous or context-dependent "
            "risk-related language. Human review is recommended."
        )

    else:

        summary = (
            "No explicit configured risk-language pattern was "
            "detected in the analyzed messages. This does not "
            "establish that the student is safe."
        )

    return {

        "summary": summary,

        "total_messages_analyzed": analyzed_count,

        "messages_requiring_review": review_count,

        "category_counts": category_counts,

        "recurring_indicator_count": len(
            recurring_indicators
        )
    }


# ============================================================
# 10. COMPLETE CONVERSATION ANALYSIS
# ============================================================

def analyze_conversation(messages):
    """
    Execute the complete conversation-level risk analysis.

    Parameters
    ----------
    messages : list
        A list of student messages or role/content dictionaries.

    Returns
    -------
    dict
        Structured conversation-level linguistic analysis.
    """

    # --------------------------------------------------------
    # A. Validate input
    # --------------------------------------------------------

    normalized_messages = validate_conversation(messages)

    # --------------------------------------------------------
    # B. Analyze each message independently
    # --------------------------------------------------------

    message_results = analyze_individual_messages(
        normalized_messages
    )

    # --------------------------------------------------------
    # C. Calculate category counts
    # --------------------------------------------------------

    category_counts = calculate_category_counts(
        message_results
    )

    # --------------------------------------------------------
    # D. Identify recurring indicators
    # --------------------------------------------------------

    recurring_indicators = identify_recurring_indicators(
        message_results
    )

    # --------------------------------------------------------
    # E. Analyze category progression
    # --------------------------------------------------------

    progression = analyze_conversation_progression(
        message_results
    )

    # --------------------------------------------------------
    # F. Determine strongest conversation category
    # --------------------------------------------------------

    conversation_category = determine_conversation_category(
        message_results
    )

    # --------------------------------------------------------
    # G. Determine human review requirement
    # --------------------------------------------------------

    human_review = requires_conversation_human_review(
        conversation_category
    )

    # --------------------------------------------------------
    # H. Generate summary
    # --------------------------------------------------------

    summary = generate_conversation_summary(
        conversation_category,
        category_counts,
        recurring_indicators,
        message_results
    )

    # --------------------------------------------------------
    # I. Build structured output
    # --------------------------------------------------------

    return {

        "module_name": "Risk Conversation Analyzer",

        "module_version": "1.0.0",

        "timestamp_utc": __import__("datetime").datetime.now(
            __import__("datetime").timezone.utc
        ).isoformat(),

        "conversation_category": conversation_category,

        "requires_human_review": human_review,

        "escalation_level": config.ESCALATION_MAPPING[
            conversation_category
        ],

        "summary": summary,

        "message_results": message_results,

        "category_counts": category_counts,

        "recurring_indicators": recurring_indicators,

        "progression": progression,

        "clinical_diagnosis": False,

        "actual_risk_confirmed": False,

        "disclaimer": config.DEFAULT_DISCLAIMER
    }


# ============================================================
# 11. BASIC VALIDATION
# ============================================================

def validate_conversation_analyzer():

    # Test 1: Ordinary conversation.
    result = analyze_conversation([

        "I enjoyed studying today.",

        "I completed my assignment."
    ])

    assert result["conversation_category"] == (
        CATEGORY["NO_EXPLICIT_RISK_SIGNAL"]
    )

    assert result["requires_human_review"] is False

    # Test 2: Potential concern across messages.
    result = analyze_conversation([

        "I feel completely hopeless.",

        "Nothing seems to get better."
    ])

    assert result["conversation_category"] == (
        CATEGORY["POTENTIAL_CONCERN"]
    )

    assert result["requires_human_review"] is True

    # Test 3: Repeated potential concern must not
    # automatically become an urgent signal.
    assert result["conversation_category"] != (
        CATEGORY["URGENT_LANGUAGE_SIGNAL"]
    )

    # Test 4: Direct urgent message.
    result = analyze_conversation([

        "I am feeling stressed.",

        "I am going to kill myself."
    ])

    assert result["conversation_category"] == (
        CATEGORY["URGENT_LANGUAGE_SIGNAL"]
    )

    assert result["requires_human_review"] is True

    # Test 5: Hypothetical risk language.
    result = analyze_conversation([

        "What should someone do if they want to kill themselves?"
    ])

    assert result["conversation_category"] == (
        CATEGORY["UNCERTAIN_OR_AMBIGUOUS"]
    )

    assert result["requires_human_review"] is True

    # Test 6: Recurring indicators.
    result = analyze_conversation([

        "I feel completely hopeless.",

        "I feel hopeless again."
    ])

    assert "hopelessness_language" in (
        result["recurring_indicators"]
    )

    # Test 7: Output must never claim clinical diagnosis.
    assert result["clinical_diagnosis"] is False

    assert result["actual_risk_confirmed"] is False

    # Test 8: Role-based filtering.
    result = analyze_conversation([

        {
            "role": "user",
            "content": "I feel completely hopeless."
        },

        {
            "role": "assistant",
            "content": "I am here to listen."
        }
    ])

    assert result["summary"]["total_messages_analyzed"] == 1

    assert result["conversation_category"] == (
        CATEGORY["POTENTIAL_CONCERN"]
    )

    return True


# ============================================================
# 12. MAIN
# ============================================================

if __name__ == "__main__":

    validate_conversation_analyzer()

    print("=" * 75)
    print("RISK CONVERSATION ANALYZER")
    print("=" * 75)

    sample_conversation = [

        "I am feeling stressed about university.",

        "Nothing seems to be getting better.",

        "I don't know how much longer I can handle this."
    ]

    result = analyze_conversation(sample_conversation)

    print("\nCONVERSATION MESSAGES")

    for index, message in enumerate(
        sample_conversation,
        start=1
    ):

        print(f"\nMessage {index}:")
        print(message)

    print("\n" + "-" * 75)

    print(
        "Conversation category:",
        result["conversation_category"]
    )

    print(
        "Escalation:",
        result["escalation_level"]
    )

    print(
        "Human review:",
        result["requires_human_review"]
    )

    print("\nCategory counts:")

    for category, count in result["category_counts"].items():

        print(f"  {category}: {count}")

    print("\nRecurring indicators:")

    for category, details in result["recurring_indicators"].items():

        print(
            f"  {category}: "
            f"{details['message_count']} messages"
        )

    print("\nProgression:")

    for item in result["progression"]["message_progression"]:

        print(
            f"  Message {item['message_index']}: "
            f"{item['risk_category']}"
        )

    print("\nConversation summary:")
    print(result["summary"]["summary"])

    print("\nConversation analyzer validation: PASSED")

    print("=" * 75)
