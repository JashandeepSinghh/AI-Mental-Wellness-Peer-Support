
"""
Expanded Rumination Topic-Similarity Validation
------------------------------------------------
Compares:
1. Lexical similarity (Jaccard)
2. Semantic similarity (Sentence Transformer)

Categories:
- Same concern
- Different concern
- Ordinary repetition
- Unrelated topics
- Ambiguous relationship

This is an engineering validation experiment.
It does NOT diagnose rumination.
"""

import csv
import re
from collections import defaultdict
from pathlib import Path

from sentence_transformers import util

from rumination_semantic_similarity import (
    RuminationSemanticSimilarity
)


# ============================================================
# 1. VALIDATION DATASET
# ============================================================

VALIDATION_PAIRS = [

    # --------------------------------------------------------
    # A. SAME CONCERN, DIFFERENT WORDING
    # --------------------------------------------------------

    {
        "id": "SC001",
        "category": "same_concern",
        "expected": "same_concern",
        "message_1": "I keep thinking about the mistake I made in my examination.",
        "message_2": "I cannot forget the error I made during my test."
    },
    {
        "id": "SC002",
        "category": "same_concern",
        "expected": "same_concern",
        "message_1": "I am constantly worried about failing my upcoming exams.",
        "message_2": "The thought of not passing my examinations keeps bothering me."
    },
    {
        "id": "SC003",
        "category": "same_concern",
        "expected": "same_concern",
        "message_1": "I feel that nobody understands what I am going through.",
        "message_2": "I feel completely alone because no one understands my situation."
    },
    {
        "id": "SC004",
        "category": "same_concern",
        "expected": "same_concern",
        "message_1": "I regret the way I behaved during that argument.",
        "message_2": "I wish I had handled our disagreement differently."
    },
    {
        "id": "SC005",
        "category": "same_concern",
        "expected": "same_concern",
        "message_1": "I cannot stop blaming myself for what happened yesterday.",
        "message_2": "I keep holding myself responsible for yesterday's incident."
    },
    {
        "id": "SC006",
        "category": "same_concern",
        "expected": "same_concern",
        "message_1": "I am scared that my friends will eventually abandon me.",
        "message_2": "I keep worrying that the people close to me will leave."
    },

    # --------------------------------------------------------
    # B. DIFFERENT CONCERNS WITHIN A SIMILAR GENERAL TOPIC
    # --------------------------------------------------------

    {
        "id": "DC001",
        "category": "different_concern",
        "expected": "different_concern",
        "message_1": "I keep regretting the wrong answer I wrote in my examination.",
        "message_2": "I need to know the examination timetable for next semester."
    },
    {
        "id": "DC002",
        "category": "different_concern",
        "expected": "different_concern",
        "message_1": "I am worried that my relationship is falling apart.",
        "message_2": "I need to buy a birthday present for my partner."
    },
    {
        "id": "DC003",
        "category": "different_concern",
        "expected": "different_concern",
        "message_1": "I cannot stop worrying about my financial problems.",
        "message_2": "I need to open a new savings account at the bank."
    },
    {
        "id": "DC004",
        "category": "different_concern",
        "expected": "different_concern",
        "message_1": "I feel lonely because my friends have stopped talking to me.",
        "message_2": "I am organizing a football match with my friends."
    },
    {
        "id": "DC005",
        "category": "different_concern",
        "expected": "different_concern",
        "message_1": "I am afraid that I will lose my current job.",
        "message_2": "I need to prepare a presentation for my workplace."
    },
    {
        "id": "DC006",
        "category": "different_concern",
        "expected": "different_concern",
        "message_1": "I keep worrying about my health and future illness.",
        "message_2": "I need to schedule my routine medical checkup."
    },

    # --------------------------------------------------------
    # C. ORDINARY REPETITION
    # Same topic, but no necessary recurring negative concern
    # --------------------------------------------------------

    {
        "id": "OR001",
        "category": "ordinary_repetition",
        "expected": "ordinary_repetition",
        "message_1": "I played football with my friends yesterday.",
        "message_2": "We are planning another football match this weekend."
    },
    {
        "id": "OR002",
        "category": "ordinary_repetition",
        "expected": "ordinary_repetition",
        "message_1": "I attended my mathematics lecture today.",
        "message_2": "Tomorrow I have another mathematics lecture."
    },
    {
        "id": "OR003",
        "category": "ordinary_repetition",
        "expected": "ordinary_repetition",
        "message_1": "I cooked dinner with my family.",
        "message_2": "We are planning to cook dinner together tomorrow."
    },
    {
        "id": "OR004",
        "category": "ordinary_repetition",
        "expected": "ordinary_repetition",
        "message_1": "I watched a documentary about wildlife.",
        "message_2": "I want to watch another wildlife documentary tonight."
    },
    {
        "id": "OR005",
        "category": "ordinary_repetition",
        "expected": "ordinary_repetition",
        "message_1": "I went shopping for some new clothes.",
        "message_2": "I might go shopping again this weekend."
    },
    {
        "id": "OR006",
        "category": "ordinary_repetition",
        "expected": "ordinary_repetition",
        "message_1": "I practiced Python programming this morning.",
        "message_2": "I will continue practicing Python programming tomorrow."
    },

    # --------------------------------------------------------
    # D. UNRELATED TOPICS
    # --------------------------------------------------------

    {
        "id": "UT001",
        "category": "unrelated_topics",
        "expected": "unrelated",
        "message_1": "I enjoy playing football with my friends.",
        "message_2": "I am unable to sleep because I keep worrying about my future."
    },
    {
        "id": "UT002",
        "category": "unrelated_topics",
        "expected": "unrelated",
        "message_1": "The weather is beautiful today.",
        "message_2": "I regret the decision I made in my relationship."
    },
    {
        "id": "UT003",
        "category": "unrelated_topics",
        "expected": "unrelated",
        "message_1": "I bought a new laptop for programming.",
        "message_2": "I feel lonely even when I am surrounded by people."
    },
    {
        "id": "UT004",
        "category": "unrelated_topics",
        "expected": "unrelated",
        "message_1": "I watched an interesting movie last night.",
        "message_2": "I keep blaming myself for my past decisions."
    },
    {
        "id": "UT005",
        "category": "unrelated_topics",
        "expected": "unrelated",
        "message_1": "My favorite restaurant serves excellent food.",
        "message_2": "I am anxious about my upcoming job interview."
    },
    {
        "id": "UT006",
        "category": "unrelated_topics",
        "expected": "unrelated",
        "message_1": "I enjoy listening to classical music.",
        "message_2": "I cannot stop thinking about my previous failure."
    },

    # --------------------------------------------------------
    # E. AMBIGUOUS RELATIONSHIPS
    # --------------------------------------------------------

    {
        "id": "AM001",
        "category": "ambiguous",
        "expected": "ambiguous",
        "message_1": "I keep thinking about everything.",
        "message_2": "There are so many things on my mind."
    },
    {
        "id": "AM002",
        "category": "ambiguous",
        "expected": "ambiguous",
        "message_1": "Things have been difficult recently.",
        "message_2": "I am not sure what to do anymore."
    },
    {
        "id": "AM003",
        "category": "ambiguous",
        "expected": "ambiguous",
        "message_1": "I keep remembering what happened.",
        "message_2": "That situation is still on my mind."
    },
    {
        "id": "AM004",
        "category": "ambiguous",
        "expected": "ambiguous",
        "message_1": "I have been feeling strange these days.",
        "message_2": "Something has been bothering me lately."
    },
    {
        "id": "AM005",
        "category": "ambiguous",
        "expected": "ambiguous",
        "message_1": "I cannot get it out of my head.",
        "message_2": "It keeps coming back whenever I am alone."
    },
    {
        "id": "AM006",
        "category": "ambiguous",
        "expected": "ambiguous",
        "message_1": "I have been thinking a lot lately.",
        "message_2": "My mind has been occupied with many things."
    }
]


# ============================================================
# 2. LEXICAL SIMILARITY CONFIGURATION
# ============================================================

STOPWORDS = {
    "i", "me", "my", "myself", "we", "our", "ours",
    "you", "your", "yours", "he", "she", "it", "they",
    "them", "their", "this", "that", "these", "those",
    "a", "an", "the", "and", "or", "but", "if",
    "to", "of", "in", "on", "at", "for", "with",
    "from", "by", "about", "as", "is", "am", "are",
    "was", "were", "be", "been", "being", "do",
    "does", "did", "have", "has", "had", "can",
    "could", "will", "would", "should", "not",
    "no", "so", "very", "just", "there", "here",
    "what", "when", "where", "who", "how", "why",
    "keep", "keeps", "kept", "feel", "feeling",
    "think", "thinking", "thought", "thing", "things"
}


def tokenize_for_lexical_similarity(text):

    tokens = re.findall(r"\b[a-zA-Z]+\b", text.lower())

    return {
        token
        for token in tokens
        if token not in STOPWORDS and len(token) > 1
    }


def calculate_jaccard_similarity(text_1, text_2):

    tokens_1 = tokenize_for_lexical_similarity(text_1)
    tokens_2 = tokenize_for_lexical_similarity(text_2)

    union = tokens_1 | tokens_2
    intersection = tokens_1 & tokens_2

    if not union:
        return 0.0

    return len(intersection) / len(union)


# ============================================================
# 3. VALIDATION ENGINE
# ============================================================

class ExpandedSimilarityValidator:

    def __init__(self):

        print("\nInitializing semantic similarity engine...")

        self.engine = RuminationSemanticSimilarity()

        self.results = []

    # --------------------------------------------------------
    # 4. COLLECT UNIQUE MESSAGES
    # --------------------------------------------------------

    def collect_unique_messages(self):

        messages = []
        seen = set()

        for pair in VALIDATION_PAIRS:

            for key in ("message_1", "message_2"):

                message = pair[key]

                if message not in seen:

                    seen.add(message)
                    messages.append(message)

        return messages

    # --------------------------------------------------------
    # 5. GENERATE SEMANTIC EMBEDDINGS
    # --------------------------------------------------------

    def generate_similarity_matrix(self, messages):

        print(f"\nUnique messages: {len(messages)}")

        print("Generating semantic embeddings...")

        embeddings = self.engine.generate_embeddings(messages)

        print("Calculating semantic similarity matrix...")

        matrix = util.cos_sim(embeddings, embeddings)

        return matrix

    # --------------------------------------------------------
    # 6. RUN ALL PAIR COMPARISONS
    # --------------------------------------------------------

    def evaluate_pairs(self):

        messages = self.collect_unique_messages()

        message_indices = {
            message: index
            for index, message in enumerate(messages)
        }

        similarity_matrix = self.generate_similarity_matrix(messages)

        for pair in VALIDATION_PAIRS:

            index_1 = message_indices[pair["message_1"]]
            index_2 = message_indices[pair["message_2"]]

            semantic_score = similarity_matrix[index_1][index_2].item()

            lexical_score = calculate_jaccard_similarity(
                pair["message_1"],
                pair["message_2"]
            )

            if not (-1.0001 <= semantic_score <= 1.0001):
                raise ValueError(
                    f"Invalid semantic score: {pair['id']}"
                )

            if not (0.0 <= lexical_score <= 1.0):
                raise ValueError(
                    f"Invalid lexical score: {pair['id']}"
                )

            self.results.append({
                **pair,
                "lexical_jaccard": round(float(lexical_score), 4),
                "semantic_cosine": round(float(semantic_score), 4)
            })

    # --------------------------------------------------------
    # 7. DISPLAY PAIRWISE RESULTS
    # --------------------------------------------------------

    def display_results(self):

        print("\n" + "=" * 95)
        print("EXPANDED SIMILARITY VALIDATION")
        print("=" * 95)

        for result in self.results:

            print("\n" + "-" * 95)

            print(f"Test ID: {result['id']}")
            print(f"Category: {result['category']}")
            print(f"Expected relationship: {result['expected']}")

            print(f"Message 1: {result['message_1']}")
            print(f"Message 2: {result['message_2']}")

            print(
                f"Lexical Jaccard: "
                f"{result['lexical_jaccard']:.4f}"
            )

            print(
                f"Semantic Cosine: "
                f"{result['semantic_cosine']:.4f}"
            )

    # --------------------------------------------------------
    # 8. CATEGORY-LEVEL ANALYSIS
    # --------------------------------------------------------

    def analyze_categories(self):

        grouped = defaultdict(list)

        for result in self.results:

            grouped[result["category"]].append(result)

        print("\n" + "=" * 95)
        print("CATEGORY-LEVEL COMPARISON")
        print("=" * 95)

        for category, records in grouped.items():

            lexical_scores = [
                record["lexical_jaccard"]
                for record in records
            ]

            semantic_scores = [
                record["semantic_cosine"]
                for record in records
            ]

            lexical_average = sum(lexical_scores) / len(lexical_scores)

            semantic_average = sum(semantic_scores) / len(semantic_scores)

            print(f"\nCategory: {category}")
            print(f"Number of pairs: {len(records)}")

            print(
                f"Average lexical similarity: "
                f"{lexical_average:.4f}"
            )

            print(
                f"Average semantic similarity: "
                f"{semantic_average:.4f}"
            )

            print(
                f"Lexical range: "
                f"{min(lexical_scores):.4f} - "
                f"{max(lexical_scores):.4f}"
            )

            print(
                f"Semantic range: "
                f"{min(semantic_scores):.4f} - "
                f"{max(semantic_scores):.4f}"
            )

    # --------------------------------------------------------
    # 9. SAVE RESULTS TO CSV
    # --------------------------------------------------------

    def save_results(self):

        project_root = Path(__file__).resolve().parents[3]

        output_directory = (
            project_root
            / "training"
            / "rumination"
            / "semantic_validation"
        )

        output_directory.mkdir(
            parents=True,
            exist_ok=True
        )

        output_file = (
            output_directory
            / "expanded_similarity_validation.csv"
        )

        fieldnames = [
            "id",
            "category",
            "expected",
            "message_1",
            "message_2",
            "lexical_jaccard",
            "semantic_cosine"
        ]

        with output_file.open(
            "w",
            newline="",
            encoding="utf-8-sig"
        ) as file:

            writer = csv.DictWriter(
                file,
                fieldnames=fieldnames
            )

            writer.writeheader()

            for result in self.results:

                writer.writerow({
                    field: result[field]
                    for field in fieldnames
                })

        print("\nValidation results saved:")
        print(output_file)

    # --------------------------------------------------------
    # 10. DISPLAY FINAL SUMMARY
    # --------------------------------------------------------

    def display_summary(self):

        print("\n" + "=" * 95)
        print("FINAL VALIDATION SUMMARY")
        print("=" * 95)

        print(f"Total validation pairs: {len(self.results)}")

        print(
            f"Same concern: "
            f"{sum(r['category'] == 'same_concern' for r in self.results)}"
        )

        print(
            f"Different concern: "
            f"{sum(r['category'] == 'different_concern' for r in self.results)}"
        )

        print(
            f"Ordinary repetition: "
            f"{sum(r['category'] == 'ordinary_repetition' for r in self.results)}"
        )

        print(
            f"Unrelated topics: "
            f"{sum(r['category'] == 'unrelated_topics' for r in self.results)}"
        )

        print(
            f"Ambiguous: "
            f"{sum(r['category'] == 'ambiguous' for r in self.results)}"
        )

        print("\nLexical similarity: Evaluated")
        print("Semantic similarity: Evaluated")
        print("Rumination classification: NOT PERFORMED")
        print("Similarity threshold: NOT SET")
        print("Clinical interpretation: NOT PERFORMED")

        print("\nDevice information:")

        print(self.engine.get_device_information())

    # --------------------------------------------------------
    # 11. COMPLETE EXECUTION
    # --------------------------------------------------------

    def run(self):

        self.evaluate_pairs()

        self.display_results()

        self.analyze_categories()

        self.save_results()

        self.display_summary()


# ============================================================
# 12. MAIN
# ============================================================

if __name__ == "__main__":

    validator = ExpandedSimilarityValidator()

    validator.run()
