
"""
Semantic Similarity Validation Suite
------------------------------------
Tests:
1. Paraphrased concerns
2. Unrelated topics
3. Different concerns within the same topic
4. Rumination versus ordinary repetition
5. Ambiguous messages

Important:
    This suite evaluates semantic similarity scores.
    It does not diagnose rumination or establish a
    clinically validated similarity threshold.
"""

import math
from collections import defaultdict

import torch

from sentence_transformers import util

from scripts.model_development.rumination.rumination_semantic_similarity import (
    RuminationSemanticSimilarity
)


# =============================================================
# 1. VALIDATION DATASET
# =============================================================

TEST_PAIRS = [

    # ---------------------------------------------------------
    # CATEGORY 1: PARAPHRASED CONCERNS
    # Expected: Semantically related
    # ---------------------------------------------------------

    {
        "id": "PARA001",
        "category": "paraphrases",
        "expected": "related",
        "message_1": (
            "I keep thinking about the mistake I made in my examination."
        ),
        "message_2": (
            "I cannot forget the error I made during my test."
        )
    },

    {
        "id": "PARA002",
        "category": "paraphrases",
        "expected": "related",
        "message_1": (
            "I am constantly worried about failing my upcoming exams."
        ),
        "message_2": (
            "The thought of not passing my examinations keeps bothering me."
        )
    },

    {
        "id": "PARA003",
        "category": "paraphrases",
        "expected": "related",
        "message_1": (
            "I feel that nobody understands what I am going through."
        ),
        "message_2": (
            "I feel completely alone because no one understands my situation."
        )
    },

    {
        "id": "PARA004",
        "category": "paraphrases",
        "expected": "related",
        "message_1": (
            "I regret the way I behaved during that argument."
        ),
        "message_2": (
            "I wish I had handled our disagreement differently."
        )
    },

    # ---------------------------------------------------------
    # CATEGORY 2: UNRELATED TOPICS
    # Expected: Semantically unrelated
    # ---------------------------------------------------------

    {
        "id": "UNREL001",
        "category": "unrelated_topics",
        "expected": "unrelated",
        "message_1": (
            "I enjoy playing football with my friends."
        ),
        "message_2": (
            "I am unable to sleep because I keep worrying about my future."
        )
    },

    {
        "id": "UNREL002",
        "category": "unrelated_topics",
        "expected": "unrelated",
        "message_1": (
            "The weather is beautiful today."
        ),
        "message_2": (
            "I regret the decision I made in my relationship."
        )
    },

    {
        "id": "UNREL003",
        "category": "unrelated_topics",
        "expected": "unrelated",
        "message_1": (
            "I bought a new laptop for programming."
        ),
        "message_2": (
            "I feel lonely even when I am surrounded by people."
        )
    },

    {
        "id": "UNREL004",
        "category": "unrelated_topics",
        "expected": "unrelated",
        "message_1": (
            "I watched an interesting movie last night."
        ),
        "message_2": (
            "I keep blaming myself for my past decisions."
        )
    },

    # ---------------------------------------------------------
    # CATEGORY 3: DIFFERENT CONCERNS, SAME GENERAL TOPIC
    # Expected: Different concerns
    # ---------------------------------------------------------

    {
        "id": "SAME001",
        "category": "different_same_topic",
        "expected": "different",
        "message_1": (
            "I keep regretting the wrong answer I wrote in my examination."
        ),
        "message_2": (
            "I need to know the examination timetable for next semester."
        )
    },

    {
        "id": "SAME002",
        "category": "different_same_topic",
        "expected": "different",
        "message_1": (
            "I am worried that my relationship is falling apart."
        ),
        "message_2": (
            "I need to buy a birthday present for my partner."
        )
    },

    {
        "id": "SAME003",
        "category": "different_same_topic",
        "expected": "different",
        "message_1": (
            "I cannot stop worrying about my financial problems."
        ),
        "message_2": (
            "I need to open a new savings account at the bank."
        )
    },

    {
        "id": "SAME004",
        "category": "different_same_topic",
        "expected": "different",
        "message_1": (
            "I feel lonely because my friends have stopped talking to me."
        ),
        "message_2": (
            "I am organizing a football match with my friends."
        )
    },

    # ---------------------------------------------------------
    # CATEGORY 4: RUMINATION VS ORDINARY REPETITION
    # Expected: Different semantic concern
    # ---------------------------------------------------------

    {
        "id": "ORD001",
        "category": "ordinary_repetition",
        "expected": "different",
        "message_1": (
            "I played football with my friends yesterday."
        ),
        "message_2": (
            "We are planning another football match this weekend."
        )
    },

    {
        "id": "ORD002",
        "category": "ordinary_repetition",
        "expected": "different",
        "message_1": (
            "I attended my mathematics lecture today."
        ),
        "message_2": (
            "Tomorrow I have another mathematics lecture."
        )
    },

    {
        "id": "ORD003",
        "category": "ordinary_repetition",
        "expected": "different",
        "message_1": (
            "I cooked dinner with my family."
        ),
        "message_2": (
            "We are planning to cook dinner together tomorrow."
        )
    },

    # ---------------------------------------------------------
    # CATEGORY 5: AMBIGUOUS MESSAGES
    # Expected: Ambiguous; no binary judgment
    # ---------------------------------------------------------

    {
        "id": "AMB001",
        "category": "ambiguous",
        "expected": "ambiguous",
        "message_1": (
            "I keep thinking about everything."
        ),
        "message_2": (
            "There are so many things on my mind."
        )
    },

    {
        "id": "AMB002",
        "category": "ambiguous",
        "expected": "ambiguous",
        "message_1": (
            "Things have been difficult recently."
        ),
        "message_2": (
            "I am not sure what to do anymore."
        )
    },

    {
        "id": "AMB003",
        "category": "ambiguous",
        "expected": "ambiguous",
        "message_1": (
            "I keep remembering what happened."
        ),
        "message_2": (
            "That situation is still on my mind."
        )
    }
]


# =============================================================
# 2. VALIDATION ENGINE
# =============================================================

class SemanticSimilarityValidator:

    def __init__(self):

        print("\nInitializing semantic similarity engine...")

        self.engine = RuminationSemanticSimilarity()

        self.results = []

    # ---------------------------------------------------------
    # 3. PREPARE UNIQUE MESSAGES
    # ---------------------------------------------------------

    def prepare_messages(self):

        unique_messages = []

        message_to_index = {}

        for pair in TEST_PAIRS:

            for message_key in ["message_1", "message_2"]:

                message = pair[message_key]

                if message not in message_to_index:

                    message_to_index[message] = len(unique_messages)

                    unique_messages.append(message)

        return unique_messages, message_to_index

    # ---------------------------------------------------------
    # 4. GENERATE EMBEDDINGS ONCE
    # ---------------------------------------------------------

    def calculate_all_scores(self):

        messages, message_to_index = self.prepare_messages()

        print(f"\nUnique messages: {len(messages)}")

        print("Generating embeddings...")

        embeddings = self.engine.generate_embeddings(messages)

        similarity_matrix = util.cos_sim(
            embeddings,
            embeddings
        )

        for pair in TEST_PAIRS:

            index_1 = message_to_index[pair["message_1"]]

            index_2 = message_to_index[pair["message_2"]]

            score = similarity_matrix[index_1][index_2].item()

            if not math.isfinite(score):
                raise ValueError(
                    f"Invalid similarity score in {pair['id']}"
                )

            if not -1.0001 <= score <= 1.0001:
                raise ValueError(
                    f"Similarity outside cosine range in {pair['id']}"
                )

            self.results.append({
                **pair,
                "similarity_score": round(float(score), 4)
            })

    # ---------------------------------------------------------
    # 5. DISPLAY INDIVIDUAL RESULTS
    # ---------------------------------------------------------

    def display_results(self):

        print("\n" + "=" * 85)

        print("SEMANTIC SIMILARITY VALIDATION RESULTS")

        print("=" * 85)

        for result in self.results:

            print("\n" + "-" * 85)

            print(f"Test ID: {result['id']}")

            print(f"Category: {result['category']}")

            print(f"Expected: {result['expected']}")

            print(f"Message 1: {result['message_1']}")

            print(f"Message 2: {result['message_2']}")

            print(
                f"Cosine Similarity: "
                f"{result['similarity_score']:.4f}"
            )

    # ---------------------------------------------------------
    # 6. CATEGORY-LEVEL SCORE ANALYSIS
    # ---------------------------------------------------------

    def analyze_categories(self):

        grouped_scores = defaultdict(list)

        for result in self.results:

            grouped_scores[result["category"]].append(
                result["similarity_score"]
            )

        print("\n" + "=" * 85)

        print("CATEGORY-LEVEL ANALYSIS")

        print("=" * 85)

        for category, scores in grouped_scores.items():

            average = sum(scores) / len(scores)

            minimum = min(scores)

            maximum = max(scores)

            print(f"\nCategory: {category}")

            print(f"Number of pairs: {len(scores)}")

            print(f"Average similarity: {average:.4f}")

            print(f"Minimum similarity: {minimum:.4f}")

            print(f"Maximum similarity: {maximum:.4f}")

    # ---------------------------------------------------------
    # 7. EXECUTION SUMMARY
    # ---------------------------------------------------------

    def display_summary(self):

        print("\n" + "=" * 85)

        print("VALIDATION SUMMARY")

        print("=" * 85)

        print(f"Total test pairs: {len(self.results)}")

        print("Embedding generation: Completed")

        print("Cosine similarity calculation: Completed")

        print("Score range validation: Passed")

        print(
            "Classification threshold: NOT SET"
        )

        print(
            "Clinical interpretation: NOT PERFORMED"
        )

        print("\nDevice information:")

        print(self.engine.get_device_information())

    # ---------------------------------------------------------
    # 8. RUN COMPLETE VALIDATION
    # ---------------------------------------------------------

    def run(self):

        self.calculate_all_scores()

        self.display_results()

        self.analyze_categories()

        self.display_summary()


# =============================================================
# 9. MAIN EXECUTION
# =============================================================

def test_rumination_semantic_similarity():
    validator = SemanticSimilarityValidator()
    validator.calculate_all_scores()
    assert len(validator.results) > 0
    for res in validator.results:
        assert -1.0 <= res["similarity_score"] <= 1.0


if __name__ == "__main__":

    validator = SemanticSimilarityValidator()

    validator.run()
