

from scripts.model_development.rumination.rumination_conversation_analyzer import analyze_conversation


TEST_CASES = [

    {
        "id": "EDGE001",
        "description": "Repeated examination topic, unrelated relationship rumination",
        "expected": False,
        "messages": [
            {"message_id": "m1", "text": "My examination is scheduled for Monday."},
            {"message_id": "m2", "text": "I need to prepare for my examination."},
            {"message_id": "m3", "text": "I cannot stop thinking about my relationship."},
        ],
    },

    {
        "id": "EDGE002",
        "description": "Repeated football topic, unrelated past-focused thinking",
        "expected": False,
        "messages": [
            {"message_id": "m1", "text": "I played football yesterday."},
            {"message_id": "m2", "text": "Football practice starts tomorrow."},
            {"message_id": "m3", "text": "I keep thinking about what happened at home."},
        ],
    },

    {
        "id": "EDGE003",
        "description": "Same concern repeated with rumination language",
        "expected": True,
        "messages": [
            {"message_id": "m1", "text": "I made a mistake during my examination."},
            {"message_id": "m2", "text": "I keep thinking about my examination mistake."},
            {"message_id": "m3", "text": "I cannot stop thinking about that mistake."},
        ],
    },

    {
        "id": "EDGE004",
        "description": "Repeated ordinary topic without rumination language",
        "expected": False,
        "messages": [
            {"message_id": "m1", "text": "I have a programming assignment."},
            {"message_id": "m2", "text": "My programming assignment is due tomorrow."},
        ],
    },

    {
        "id": "EDGE005",
        "description": "Rumination expression without repeated concern",
        "expected": False,
        "messages": [
            {"message_id": "m1", "text": "I cannot stop thinking about my interview."},
            {"message_id": "m2", "text": "I attended my lecture today."},
        ],
    },

]


def run_tests():

    passed = 0
    failed = 0

    print("\n" + "=" * 90)
    print("CONVERSATION EDGE-CASE TESTS")
    print("=" * 90)

    for case in TEST_CASES:

        result = analyze_conversation(
            case["messages"]
        )

        predicted = result["recurring_rumination_signal"]

        status = (
            "PASS"
            if predicted == case["expected"]
            else "FAIL"
        )

        if status == "PASS":
            passed += 1
        else:
            failed += 1

        print(f"\n{case['id']} - {case['description']}")
        print(f"Expected: {case['expected']}")
        print(f"Predicted: {predicted}")
        print(f"Status: {status}")

        if status == "FAIL":
            print(
                "Repeated keywords:",
                result["repeated_concerns"]["repeated_keywords"]
            )
            print(
                "Messages with indicators:",
                result["messages_with_indicators"]
            )

    print("\n" + "=" * 90)
    print(f"Total: {len(TEST_CASES)}")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    print("=" * 90)
    return failed


def test_rumination_conversation_edge_cases():
    failed = run_tests()
    assert failed == 0, f"{failed} edge case tests failed"


if __name__ == "__main__":
    run_tests()
