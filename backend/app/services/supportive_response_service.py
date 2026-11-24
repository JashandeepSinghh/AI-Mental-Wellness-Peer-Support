"""
Supportive Response and Safety Escalation Service
==================================================

Generates responsible, empathetic, context-aware responses and recommends
wellness resources based on multi-dimensional NLP inference outputs.

Safety and Ethical Requirements:
- Does not claim to be a licensed therapist, psychiatrist, or medical provider.
- Does not diagnose mental health illnesses.
- Does not prescribe medication or guarantee emotional outcomes.
- Transparently distinguishes linguistic indicators from clinical certainty.
- Provides immediate, verified crisis helplines (988 Lifeline, Crisis Text Line)
  when concerning or urgent risk language is detected.
- Never claims human counsellors or emergency response are monitoring unless
  genuinely configured.
"""

from __future__ import annotations

from typing import Any


TRUSTED_CRISIS_CONTACTS = [
    {
        "name": "Tele-MANAS (Govt. of India)",
        "contact": "14416 or 1800-891-4416",
        "availability": "24/7, Toll-Free, Multi-lingual (India)",
        "description": "National Tele Mental Health Programme of India (MoHFW & NIMHANS) providing round-the-clock free mental health support and crisis counseling.",
        "region": "India",
    },
    {
        "name": "KIRAN Mental Health Helpline (Govt. of India)",
        "contact": "1800-599-0019",
        "availability": "24/7, Toll-Free (India)",
        "description": "National 24/7 mental health rehabilitation and psychological first-aid helpline by the Ministry of Social Justice & Empowerment.",
        "region": "India",
    },
    {
        "name": "Vandrevala Foundation Helpline",
        "contact": "+91 9999 666 555",
        "availability": "24/7, Free & Confidential (India)",
        "description": "24/7 tele-counseling and crisis intervention across India in multiple regional languages.",
        "region": "India",
    },
    {
        "name": "988 Suicide & Crisis Lifeline",
        "contact": "Call or text 988",
        "availability": "24/7, Free & Confidential (US & Canada)",
        "description": (
            "Immediate support for anyone experiencing mental health distress, "
            "substance use concerns, or thoughts of self-harm."
        ),
        "region": "US & Canada",
    },
    {
        "name": "Crisis Text Line",
        "contact": "Text HOME to 741741",
        "availability": "24/7, Free via SMS (US, UK, Canada)",
        "description": "Connect with a trained crisis counselor via confidential text messaging.",
        "region": "US & Canada",
    },
    {
        "name": "The Trevor Project",
        "contact": "1-866-488-7386 or Text START to 678-678",
        "availability": "24/7, Free & Confidential (LGBTQ Youth)",
        "description": "Specialized crisis intervention and suicide prevention for LGBTQ youth.",
        "region": "US & Canada",
    },
    {
        "name": "Befrienders Worldwide",
        "contact": "https://www.befrienders.org",
        "availability": "Global Directory (32+ countries)",
        "description": "International directory of verified confidential emotional support helplines in 32+ countries.",
        "region": "International",
    },
    {
        "name": "Find A Helpline",
        "contact": "https://findahelpline.com",
        "availability": "Global Directory (130+ countries)",
        "description": "Free, confidential emotional support and local crisis lines worldwide sorted by country.",
        "region": "International",
    },
]


WELLNESS_RESOURCES = [
    {
        "id": "box-breathing",
        "title": "Box Breathing Rhythmic Technique",
        "category": "stress_management",
        "summary": "A 4-part breathing exercise to calm the sympathetic nervous system.",
        "steps": [
            "Inhale slowly through your nose for 4 seconds.",
            "Hold your breath gently for 4 seconds.",
            "Exhale smoothly through your mouth for 4 seconds.",
            "Pause and hold empty for 4 seconds.",
            "Repeat for 3 to 5 minutes.",
        ],
        "target_emotions": ["stress", "nervousness", "fear", "anxiety", "overwhelm"],
    },
    {
        "id": "sensory-grounding",
        "title": "5-4-3-2-1 Sensory Grounding",
        "category": "anxiety_coping",
        "summary": "Engage your five senses to interrupt overwhelming thought spirals.",
        "steps": [
            "Notice 5 things you can see.",
            "Notice 4 things you can physically feel.",
            "Notice 3 things you can hear.",
            "Notice 2 things you can smell.",
            "Notice 1 thing you can taste.",
        ],
        "target_emotions": ["fear", "nervousness", "confusion", "anxiety"],
    },
    {
        "id": "cognitive-defusion",
        "title": "Cognitive Defusion & Thought Labeling",
        "category": "emotional_regulation",
        "summary": "Observe distressing thoughts as transient mental events rather than facts.",
        "steps": [
            "Identify the distressing thought (e.g., 'I will fail').",
            "Rephrase: 'I am noticing the thought that I will fail.'",
            "Observe the physical sensation in your body without pushing it away.",
            "Let the thought pass like clouds moving across the sky.",
        ],
        "target_emotions": ["disappointment", "remorse", "sadness", "annoyance"],
    },
    {
        "id": "progressive-relaxation",
        "title": "Progressive Muscle Relaxation",
        "category": "mindfulness",
        "summary": "Systematically release physical muscle tension from head to toe.",
        "steps": [
            "Sit or lie comfortably with arms resting naturally.",
            "Tense the muscles in your feet for 5 seconds, then completely release.",
            "Move progressively up through calves, thighs, abdomen, shoulders, and jaw.",
            "Notice the warm sensation of relaxation washing over your body.",
        ],
        "target_emotions": ["stress", "anger", "nervousness"],
    },
    {
        "id": "sleep-hygiene",
        "title": "Sleep Sanctuary & Circadian Rhythm Support",
        "category": "sleep_hygiene",
        "summary": "Evidence-based habits to support restorative sleep architecture.",
        "steps": [
            "Maintain a consistent wake-up time 7 days a week.",
            "Turn off bright screens and blue light 45 minutes before sleep.",
            "Keep the bedroom dark, quiet, and cool (around 65°F / 18°C).",
            "If awake after 20 minutes, engage in a low-stimulus activity in dim light.",
        ],
        "target_emotions": ["exhaustion", "stress", "sadness"],
    },
    {
        "id": "self-compassion",
        "title": "Self-Compassion Pause",
        "category": "self_care",
        "summary": "Treat yourself with the same warmth you would offer a struggling close friend.",
        "steps": [
            "Place a soothing hand over your heart or arm.",
            "Acknowledge: 'This is a moment of difficulty. Difficulty is part of being human.'",
            "Ask: 'What kind, supportive words would I say to a friend in this exact situation?'",
            "Offer those gentle, encouraging words to yourself.",
        ],
        "target_emotions": ["grief", "sadness", "embarrassment", "remorse", "disappointment"],
    },
    {
        "id": "pomodoro-decompression",
        "title": "Academic Overwhelm Decompression",
        "category": "academic_stress",
        "summary": "Break heavy coursework into 15-minute low-pressure focus intervals.",
        "steps": [
            "Choose one very small micro-task.",
            "Focus on that task for just 15 minutes with no distractions.",
            "Take a mandatory 5-minute break away from screens.",
            "Acknowledge your effort without demanding perfection.",
        ],
        "target_emotions": ["stress", "confusion", "annoyance"],
    },
    {
        "id": "social-reconnection",
        "title": "Gentle Steps Toward Connection",
        "category": "social_support",
        "summary": "Low-stakes ways to alleviate feelings of isolation and loneliness.",
        "steps": [
            "Recognize that loneliness is a universal signal that your connection needs care.",
            "Send a simple, low-pressure message to one acquaintance or family member.",
            "Visit a shared community space (library or café) to enjoy ambient belonging.",
            "Consider joining a student interest club or peer discussion group.",
        ],
        "target_emotions": ["lonely", "sadness", "grief"],
    },
]


def select_wellness_resources(
    emotions: list[str],
    stress_detected: bool,
    loneliness_detected: bool,
) -> list[dict[str, Any]]:
    """Select the most relevant wellness resources based on detected signals."""
    matched_ids: set[str] = set()
    selected: list[dict[str, Any]] = []

    if stress_detected:
        for res in WELLNESS_RESOURCES:
            if res["category"] in ("stress_management", "academic_stress"):
                if res["id"] not in matched_ids:
                    matched_ids.add(res["id"])
                    selected.append(res)

    if loneliness_detected:
        for res in WELLNESS_RESOURCES:
            if res["category"] == "social_support":
                if res["id"] not in matched_ids:
                    matched_ids.add(res["id"])
                    selected.append(res)

    for emotion in emotions:
        emo_lower = emotion.lower()
        for res in WELLNESS_RESOURCES:
            if emo_lower in res["target_emotions"]:
                if res["id"] not in matched_ids:
                    matched_ids.add(res["id"])
                    selected.append(res)

    if not selected:
        # Fallback to general grounding tools
        selected = [WELLNESS_RESOURCES[0], WELLNESS_RESOURCES[1], WELLNESS_RESOURCES[5]]

    return selected[:3]


def generate_supportive_guidance(
    analysis: dict[str, Any],
) -> dict[str, Any]:
    """
    Synthesize model inference signals into an empathetic supportive response,
    wellness resource recommendations, and safety escalation guidance.
    """
    modules = analysis.get("modules", {})
    emotion_mod = modules.get("emotion", {})
    sentiment_mod = modules.get("sentiment", {})
    stress_mod = modules.get("stress", {})
    lonely_mod = modules.get("loneliness", {})
    risk_mod = modules.get("risk", {})

    # Extract detected emotions
    detected_emotions: list[str] = []
    confidence_analysis = emotion_mod.get("confidence_analysis", {})
    if confidence_analysis and "selected_labels" in confidence_analysis:
        detected_emotions = confidence_analysis.get("selected_labels", [])
    elif "predictions" in emotion_mod:
        preds = emotion_mod["predictions"]
        if isinstance(preds, dict):
            sorted_preds = sorted(
                [(k, v) for k, v in preds.items() if k != "neutral"],
                key=lambda x: x[1],
                reverse=True,
            )
            detected_emotions = [k for k, v in sorted_preds if v > 0.15][:3]

    sentiment = sentiment_mod.get("predicted_class", "neutral")
    stress_detected = stress_mod.get("predicted_class") == "stress"
    loneliness_detected = lonely_mod.get("predicted_class") == "lonely"

    risk_category = risk_mod.get("risk_category", "no_explicit_risk_signal")
    escalation_level = risk_mod.get("escalation_level", "none")

    is_urgent = (
        risk_category == "urgent_language_signal"
        or escalation_level == "prompt_human_review"
    )
    is_concerning = (
        risk_category == "potential_concern"
        or escalation_level in ("human_review", "supportive_follow_up")
    )

    # 1. Safety escalation metadata
    safety_guidance: dict[str, Any] | None = None
    if is_urgent:
        safety_guidance = {
            "is_urgent": True,
            "is_concerning": True,
            "risk_category": risk_category,
            "escalation_level": escalation_level,
            "message": (
                "Your well-being is deeply important. Because you may be experiencing intense pain "
                "or acute distress right now, please know that you do not have to carry this alone. "
                "Free, confidential support from compassionate professionals is available 24/7."
            ),
            "crisis_contacts": TRUSTED_CRISIS_CONTACTS,
        }
    elif is_concerning:
        safety_guidance = {
            "is_urgent": False,
            "is_concerning": True,
            "risk_category": risk_category,
            "escalation_level": escalation_level,
            "message": (
                "It sounds like you are carrying a heavy burden right now. While I am an AI peer "
                "support companion and cannot provide therapy or clinical diagnosis, supportive "
                "resources and professional care are always here for you."
            ),
            "crisis_contacts": TRUSTED_CRISIS_CONTACTS[:2],
        }

    # 2. Empathetic supportive response text
    if is_urgent:
        supportive_text = (
            "I hear how much pain and weight you are experiencing in this moment. Please remember "
            "that what you are feeling is valid, but you don't have to face this by yourself. "
            "Connecting with someone trained to support you through moments like this can help make "
            "things clearer and safer. Please reach out to one of the crisis resources below."
        )
    elif stress_detected and loneliness_detected:
        supportive_text = (
            "It sounds like you are navigating both intense pressure and feeling isolated right now. "
            "Carrying a lot of stress while feeling like there is no one around to share it with is "
            "exhausting. I want to acknowledge how much courage it takes to put that into words. "
            "Let's take things one moment at a time."
        )
    elif stress_detected:
        supportive_text = (
            "I hear how much tension and stress you're experiencing right now. When demands feel high, "
            "our minds and bodies often tense up in response. It is completely understandable to feel "
            "overwhelmed. Would it help to take a slow breath together or explore a small coping step?"
        )
    elif loneliness_detected:
        supportive_text = (
            "Feeling disconnected or lonely can be such a quiet, aching experience. Thank you for "
            "sharing that with me. Please remember that feeling lonely doesn't mean you are unworthy "
            "of connection—it's simply a sign that your social and emotional needs deserve gentle care."
        )
    elif any(e in detected_emotions for e in ("sadness", "grief")):
        supportive_text = (
            "I notice feelings of sadness in what you shared. It is completely okay to feel down, "
            "and you don't need to force yourself to feel positive right away. Giving yourself "
            "permission to simply feel what you are feeling is often the first gentle step toward healing."
        )
    elif any(e in detected_emotions for e in ("fear", "nervousness")):
        supportive_text = (
            "It sounds like anxiety or uncertainty is weighing on your mind. When apprehension takes "
            "over, grounding yourself in the present moment can bring some stability back. Take a deep, "
            "gentle breath with me."
        )
    elif any(e in detected_emotions for e in ("joy", "gratitude", "optimism")):
        supportive_text = (
            "It's really heartening to hear positive reflections and moments of gratitude in your message. "
            "Savoring these moments of warmth builds lasting emotional resilience."
        )
    else:
        supportive_text = (
            "Thank you for sharing your thoughts with me. I'm here to listen, support, and help you "
            "reflect on whatever you are experiencing right now."
        )

    # 3. Recommended wellness resources
    recommended_resources = select_wellness_resources(
        detected_emotions,
        stress_detected,
        loneliness_detected,
    )

    return {
        "supportive_response": supportive_text,
        "detected_emotions": detected_emotions,
        "sentiment": sentiment,
        "stress_detected": stress_detected,
        "loneliness_detected": loneliness_detected,
        "safety_guidance": safety_guidance,
        "wellness_resources": recommended_resources,
    }
