
"""
Rumination Semantic Similarity Engine
-------------------------------------

Purpose:
    Compare the semantic meaning of conversation messages.

Technology:
    Sentence Transformers
    all-MiniLM-L6-v2
    Cosine Similarity
    PyTorch / CUDA

Important:
    Semantic similarity identifies potentially related messages.
    It is not a rumination probability, diagnosis, or independent
    indicator of a person's mental state.
"""

from typing import Any, Dict, List, Optional, Union

import torch
from sentence_transformers import SentenceTransformer, util

from .rumination_config import (
    ENABLE_SEMANTIC_ANALYSIS,
    SEMANTIC_MODEL_NAME,
    SEMANTIC_EMBEDDING_DIMENSION,
    SEMANTIC_MAX_SEQUENCE_LENGTH,
    SEMANTIC_BATCH_SIZE,
    SEMANTIC_DEVICE,
    MAX_TEXT_LENGTH,
)


class RuminationSemanticSimilarity:

    # ============================================================
    # 1. MODEL CONFIGURATION
    # ============================================================

    MODEL_NAME = SEMANTIC_MODEL_NAME

    EMBEDDING_DIMENSION = SEMANTIC_EMBEDDING_DIMENSION

    MAX_SEQUENCE_LENGTH = SEMANTIC_MAX_SEQUENCE_LENGTH

    BATCH_SIZE = SEMANTIC_BATCH_SIZE

    # ============================================================
    # 2. INITIALIZATION
    # ============================================================

    def __init__(
        self,
        device: Optional[str] = SEMANTIC_DEVICE
    ) -> None:

        if not ENABLE_SEMANTIC_ANALYSIS:
            raise RuntimeError(
                "Semantic analysis is disabled in rumination_config.py."
            )

        # --------------------------------------------------------
        # DEVICE SELECTION
        # --------------------------------------------------------

        if device is None:
            device = (
                "cuda"
                if torch.cuda.is_available()
                else "cpu"
            )

        if device not in ("cuda", "cpu"):
            raise ValueError(
                "Device must be either 'cuda' or 'cpu'."
            )

        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA was requested but is not available."
            )

        self.device = device

        # --------------------------------------------------------
        # LOAD PRETRAINED SENTENCE TRANSFORMER
        # --------------------------------------------------------

        print("\nLoading semantic similarity model...")

        self.model = SentenceTransformer(
            self.MODEL_NAME,
            device=self.device
        )

        # --------------------------------------------------------
        # CONFIGURE MAXIMUM SEQUENCE LENGTH
        # --------------------------------------------------------

        self.model.max_seq_length = self.MAX_SEQUENCE_LENGTH

        # --------------------------------------------------------
        # VERIFY EMBEDDING DIMENSION
        # --------------------------------------------------------

        self.embedding_dimension = (
            self.model.get_embedding_dimension()
        )

        if self.embedding_dimension != self.EMBEDDING_DIMENSION:
            raise RuntimeError(
                "Unexpected embedding dimension. "
                f"Expected {self.EMBEDDING_DIMENSION}, "
                f"received {self.embedding_dimension}."
            )

        print("Model loaded successfully.")
        print(f"Model: {self.MODEL_NAME}")
        print(f"Device: {self.device}")
        print(
            f"Embedding dimensions: "
            f"{self.embedding_dimension}"
        )
        print(
            f"Maximum sequence length: "
            f"{self.MAX_SEQUENCE_LENGTH}"
        )
        print(f"Batch size: {self.BATCH_SIZE}")

    # ============================================================
    # 3. INPUT VALIDATION
    # ============================================================

    @staticmethod
    def _validate_text(text: str) -> str:
        """
        Validate and normalize one input message.
        """

        if not isinstance(text, str):
            raise TypeError(
                "Each message must be a string."
            )

        text = text.strip()

        if not text:
            raise ValueError(
                "Message cannot be empty."
            )

        if len(text) < 3:
            raise ValueError(
                "Message must contain at least 3 characters."
            )

        if len(text) > MAX_TEXT_LENGTH:
            raise ValueError(
                f"Message exceeds maximum length "
                f"of {MAX_TEXT_LENGTH} characters."
            )

        return text

    # ============================================================
    # 4. VALIDATE MESSAGE COLLECTION
    # ============================================================

    @classmethod
    def _validate_messages(
        cls,
        messages: List[str],
        minimum_messages: int = 1
    ) -> List[str]:
        """
        Validate a collection of messages.
        """

        if not isinstance(messages, list):
            raise TypeError(
                "Messages must be provided as a list."
            )

        if len(messages) < minimum_messages:
            raise ValueError(
                f"At least {minimum_messages} message(s) "
                "are required."
            )

        return [
            cls._validate_text(message)
            for message in messages
        ]

    # ============================================================
    # 5. GENERATE SENTENCE EMBEDDINGS
    # ============================================================

    def generate_embeddings(
        self,
        texts: Union[str, List[str]]
    ) -> torch.Tensor:
        """
        Convert messages into normalized sentence embeddings.

        Returns:
            torch.Tensor:
                Single string input:
                    Shape (1, embedding_dimension)

                List input:
                    Shape (number_of_messages, embedding_dimension)
        """

        if isinstance(texts, str):
            validated_texts = [
                self._validate_text(texts)
            ]

        else:
            validated_texts = self._validate_messages(
                texts,
                minimum_messages=1
            )

        # --------------------------------------------------------
        # GENERATE NORMALIZED EMBEDDINGS
        # --------------------------------------------------------

        with torch.inference_mode():

            embeddings = self.model.encode(
                validated_texts,
                batch_size=self.BATCH_SIZE,
                convert_to_tensor=True,
                normalize_embeddings=True,
                show_progress_bar=False
            )

        # --------------------------------------------------------
        # OUTPUT VALIDATION
        # --------------------------------------------------------

        if embeddings.ndim != 2:
            raise RuntimeError(
                "Unexpected embedding tensor dimensions."
            )

        if embeddings.shape[1] != self.EMBEDDING_DIMENSION:
            raise RuntimeError(
                "Generated embedding dimension does not match "
                "the configured dimension."
            )

        return embeddings

    # ============================================================
    # 6. CALCULATE COSINE SIMILARITY
    # ============================================================

    def calculate_similarity(
        self,
        message_1: str,
        message_2: str
    ) -> float:
        """
        Calculate semantic similarity between two messages.

        Returns:
            float:
                Cosine similarity score.
        """

        message_1 = self._validate_text(message_1)

        message_2 = self._validate_text(message_2)

        embeddings = self.generate_embeddings(
            [message_1, message_2]
        )

        # Normalized embeddings make dot product equivalent
        # to cosine similarity.

        similarity = torch.dot(
            embeddings[0],
            embeddings[1]
        ).item()

        return float(similarity)

    # ============================================================
    # 7. GENERATE PAIRWISE SIMILARITY MATRIX
    # ============================================================

    def calculate_similarity_matrix(
        self,
        messages: List[str]
    ) -> torch.Tensor:
        """
        Calculate cosine similarity between every message pair.

        Returns:
            torch.Tensor:
                Square similarity matrix of shape (N, N).

        Note:
            The diagonal contains self-similarity values.
        """

        validated_messages = self._validate_messages(
            messages,
            minimum_messages=1
        )

        embeddings = self.generate_embeddings(
            validated_messages
        )

        with torch.inference_mode():

            similarity_matrix = util.cos_sim(
                embeddings,
                embeddings
            )

        return similarity_matrix

    # ============================================================
    # 8. COMPARE MULTIPLE CONVERSATION MESSAGES
    # ============================================================

    def compare_messages(
        self,
        messages: List[str]
    ) -> List[Dict[str, Any]]:
        """
        Compare all unique pairs of conversation messages.

        Results are sorted from highest to lowest similarity.
        """

        validated_messages = self._validate_messages(
            messages,
            minimum_messages=2
        )

        # --------------------------------------------------------
        # GENERATE PAIRWISE SIMILARITY MATRIX
        # --------------------------------------------------------

        similarity_matrix = self.calculate_similarity_matrix(
            validated_messages
        )

        # --------------------------------------------------------
        # EXTRACT UNIQUE MESSAGE PAIRS
        # --------------------------------------------------------

        comparisons = []

        message_count = len(validated_messages)

        for i in range(message_count):

            for j in range(i + 1, message_count):

                score = similarity_matrix[i][j].item()

                comparisons.append({

                    "message_1_index": i,

                    "message_2_index": j,

                    "message_1": validated_messages[i],

                    "message_2": validated_messages[j],

                    "similarity_score": round(
                        float(score),
                        4
                    )

                })

        # --------------------------------------------------------
        # SORT BY SIMILARITY
        # --------------------------------------------------------

        comparisons.sort(
            key=lambda item: item["similarity_score"],
            reverse=True
        )

        return comparisons

    # ============================================================
    # 9. GET DEVICE INFORMATION
    # ============================================================

    def get_device_information(self) -> Dict[str, Any]:
        """
        Return model execution device and GPU information.
        """

        information = {

            "device": self.device,

            "cuda_available": torch.cuda.is_available(),

            "model_name": self.MODEL_NAME,

            "embedding_dimension": self.embedding_dimension,

            "maximum_sequence_length": self.MAX_SEQUENCE_LENGTH,

            "batch_size": self.BATCH_SIZE,

        }

        if self.device == "cuda":

            gpu_index = torch.device(
                self.device
            ).index

            if gpu_index is None:
                gpu_index = torch.cuda.current_device()

            information["gpu_name"] = (
                torch.cuda.get_device_name(gpu_index)
            )

            information["gpu_memory_allocated_gb"] = round(
                torch.cuda.memory_allocated(gpu_index)
                / (1024 ** 3),
                3
            )

            information["gpu_memory_reserved_gb"] = round(
                torch.cuda.memory_reserved(gpu_index)
                / (1024 ** 3),
                3
            )

        return information


# ================================================================
# 10. STANDALONE EXECUTION
# ================================================================

if __name__ == "__main__":

    semantic_engine = RuminationSemanticSimilarity()

    messages = [

        "I keep thinking about the mistake I made in my examination.",

        "I cannot forget the error I made during my test.",

        "I enjoy playing football with my friends.",

        "I am worried about what will happen in my future."

    ]

    print("\nComparing conversation messages...\n")

    results = semantic_engine.compare_messages(
        messages
    )

    for result in results:

        print("-" * 75)

        print(
            f"Message {result['message_1_index'] + 1}: "
            f"{result['message_1']}"
        )

        print(
            f"Message {result['message_2_index'] + 1}: "
            f"{result['message_2']}"
        )

        print(
            f"Cosine Similarity: "
            f"{result['similarity_score']}"
        )

    print("\nGPU / Device Information:")

    print(
        semantic_engine.get_device_information()
    )
