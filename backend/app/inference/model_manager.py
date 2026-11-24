
"""
Centralized Inference Model Manager
===================================

Responsibilities:
- Resolve the backend inference device.
- Load transformer models lazily.
- Load tokenizers from local checkpoint directories.
- Reconstruct custom PyTorch checkpoints.
- Keep loaded models in memory for reuse.
- Prevent duplicate concurrent model initialization.
- Provide model-loading status.
- Avoid loading training datasets or training configurations.

This manager currently handles:
    1. Emotion
    2. Sentiment
    3. Stress
    4. Loneliness

Rule-based and lexicon-based modules will be integrated
through the inference service in subsequent steps.
"""

import logging
import threading

from pathlib import Path

import torch

from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DistilBertConfig,
    DistilBertForSequenceClassification,
)

from backend.app.core.config import (
    Settings,
    get_settings,
)


logger = logging.getLogger(__name__)


class ModelManagerError(RuntimeError):
    """Raised when model initialization or loading fails."""


class ModelManager:

    # ========================================================
    # 1. INITIALIZATION
    # ========================================================

    def __init__(self, settings: Settings | None = None):

        self.settings = settings or get_settings()

        self.device = self._resolve_device()

        self.models = {}
        self.tokenizers = {}
        self.vocabularies = {}

        self._loaded = set()
        self._failed = {}

        # Protect model initialization from concurrent requests.
        self._locks = {
            "emotion": threading.Lock(),
            "sentiment": threading.Lock(),
            "sentiment_bilstm": threading.Lock(),
            "stress": threading.Lock(),
            "loneliness": threading.Lock(),
        }

        self._shutdown = False

        logger.info(
            "Model manager initialized | device=%s",
            self.device,
        )

    # ========================================================
    # 2. DEVICE RESOLUTION
    # ========================================================

    def _resolve_device(self) -> torch.device:

        requested_device = self.settings.INFERENCE_DEVICE

        if requested_device == "cpu":
            return torch.device("cpu")

        if requested_device == "cuda":

            if not torch.cuda.is_available():
                raise ModelManagerError(
                    "CUDA was explicitly requested, "
                    "but CUDA is unavailable."
                )

            return torch.device("cuda")

        # Automatic device selection.
        if torch.cuda.is_available():
            return torch.device("cuda")

        return torch.device("cpu")

    # ========================================================
    # 3. PATH VALIDATION
    # ========================================================

    @staticmethod
    def _validate_directory(path: Path, module: str):

        if not path.exists():
            raise ModelManagerError(
                f"{module} checkpoint directory does not exist: "
                f"{path}"
            )

        if not path.is_dir():
            raise ModelManagerError(
                f"{module} checkpoint path is not a directory: "
                f"{path}"
            )

    @staticmethod
    def _validate_checkpoint(path: Path, module: str):

        if not path.exists():
            raise ModelManagerError(
                f"{module} checkpoint file not found: {path}"
            )

        if not path.is_file():
            raise ModelManagerError(
                f"{module} checkpoint is not a file: {path}"
            )

    # ========================================================
    # 4. TOKENIZER LOADING
    # ========================================================

    def _load_tokenizer(self, path: Path, module: str):

        self._validate_directory(path, module)

        try:

            tokenizer = AutoTokenizer.from_pretrained(
                str(path),
                local_files_only=True,
            )

            logger.info(
                "%s tokenizer loaded from local checkpoint.",
                module,
            )

            return tokenizer

        except Exception as error:

            raise ModelManagerError(
                f"Unable to load {module} tokenizer: "
                f"{type(error).__name__}"
            ) from error

    # ========================================================
    # 5. HUGGING FACE CHECKPOINT LOADING
    # ========================================================

    def _load_huggingface_model(
        self,
        path: Path,
        module: str,
    ):

        self._validate_directory(path, module)

        try:

            model = (
                AutoModelForSequenceClassification
                .from_pretrained(
                    str(path),
                    local_files_only=True,
                )
            )

            return model

        except Exception as error:

            raise ModelManagerError(
                f"Unable to load {module} model checkpoint: "
                f"{type(error).__name__}"
            ) from error

    # ========================================================
    # 6. CUSTOM PYTORCH CHECKPOINT LOADING
    # ========================================================

    def _load_custom_distilbert(
        self,
        checkpoint_path: Path,
        module: str,
        num_labels: int,
        id2label: dict,
        label2id: dict,
    ):

        """
        Reconstruct the DistilBERT architecture without loading
        pretrained weights from the internet.

        The custom training checkpoints contain state_dict keys
        prefixed with 'model.' because the original training
        classes wrapped the Hugging Face model inside self.model.
        """

        self._validate_checkpoint(
            checkpoint_path,
            module,
        )

        try:

            checkpoint = torch.load(
                checkpoint_path,
                map_location="cpu",
                weights_only=True,
            )

            if not isinstance(checkpoint, dict):
                raise ModelManagerError(
                    f"{module} checkpoint has an invalid structure."
                )

            state_dict = checkpoint.get("model_state_dict")

            if not isinstance(state_dict, dict):
                raise ModelManagerError(
                    f"{module} checkpoint does not contain "
                    "model_state_dict."
                )

            # Reconstruct the original DistilBERT configuration.
            model_config = DistilBertConfig(
                num_labels=num_labels,
                id2label=id2label,
                label2id=label2id,
                problem_type="single_label_classification",
            )

            model = DistilBertForSequenceClassification(
                model_config
            )

            # Remove the original wrapper's "model." prefix.
            cleaned_state_dict = {}

            for key, value in state_dict.items():

                if key.startswith("model."):
                    new_key = key[len("model."):]
                else:
                    new_key = key

                cleaned_state_dict[new_key] = value

            # Strict loading verifies architecture compatibility.
            model.load_state_dict(
                cleaned_state_dict,
                strict=True,
            )

            return model

        except ModelManagerError:
            raise

        except Exception as error:

            raise ModelManagerError(
                f"Unable to reconstruct {module} checkpoint: "
                f"{type(error).__name__}"
            ) from error

    # ========================================================
    # 7. COMMON MODEL REGISTRATION
    # ========================================================

    def _register_model(
        self,
        module: str,
        model: torch.nn.Module,
        tokenizer,
    ):

        model = model.to(self.device)

        model.eval()

        self.models[module] = model
        self.tokenizers[module] = tokenizer

        self._loaded.add(module)
        self._failed.pop(module, None)

        logger.info(
            "%s model loaded successfully | device=%s",
            module,
            self.device,
        )

        return model

    # ========================================================
    # 8. EMOTION MODEL
    # ========================================================

    def load_emotion(self):

        module = "emotion"

        if module in self._loaded:
            return self.models[module]

        with self._locks[module]:

            if module in self._loaded:
                return self.models[module]

            try:

                path = self.settings.EMOTION_MODEL_PATH

                tokenizer = self._load_tokenizer(
                    path,
                    module,
                )

                model = self._load_huggingface_model(
                    path,
                    module,
                )

                return self._register_model(
                    module,
                    model,
                    tokenizer,
                )

            except Exception as error:

                self._failed[module] = str(error)

                logger.error(
                    "Emotion model loading failed: %s",
                    type(error).__name__,
                )

                raise

    # ========================================================
    # 9. SENTIMENT MODEL
    # ========================================================

    def load_sentiment(self):

        module = "sentiment"

        if module in self._loaded:
            return self.models[module]

        with self._locks[module]:

            if module in self._loaded:
                return self.models[module]

            try:

                path = self.settings.SENTIMENT_MODEL_PATH

                tokenizer = self._load_tokenizer(
                    path,
                    module,
                )

                model = self._load_huggingface_model(
                    path,
                    module,
                )

                return self._register_model(
                    module,
                    model,
                    tokenizer,
                )

            except Exception as error:

                self._failed[module] = str(error)

                logger.error(
                    "Sentiment model loading failed: %s",
                    type(error).__name__,
                )

                raise

    # ========================================================
    # 10. STRESS MODEL
    # ========================================================

    def load_stress(self):

        module = "stress"

        if module in self._loaded:
            return self.models[module]

        with self._locks[module]:

            if module in self._loaded:
                return self.models[module]

            try:

                path = self.settings.STRESS_MODEL_PATH

                tokenizer = self._load_tokenizer(
                    path,
                    module,
                )

                checkpoint_path = (
                    path / "best_checkpoint.pt"
                )

                model = self._load_custom_distilbert(
                    checkpoint_path=checkpoint_path,
                    module=module,
                    num_labels=2,
                    id2label={
                        0: "non_stress",
                        1: "stress",
                    },
                    label2id={
                        "non_stress": 0,
                        "stress": 1,
                    },
                )

                return self._register_model(
                    module,
                    model,
                    tokenizer,
                )

            except Exception as error:

                self._failed[module] = str(error)

                logger.error(
                    "Stress model loading failed: %s",
                    type(error).__name__,
                )

                raise

    # ========================================================
    # 11. LONELINESS MODEL
    # ========================================================

    def load_loneliness(self):

        module = "loneliness"

        if module in self._loaded:
            return self.models[module]

        with self._locks[module]:

            if module in self._loaded:
                return self.models[module]

            try:

                path = self.settings.LONELINESS_MODEL_PATH

                tokenizer = self._load_tokenizer(
                    path,
                    module,
                )

                checkpoint_path = (
                    path / "best_checkpoint.pt"
                )

                model = self._load_custom_distilbert(
                    checkpoint_path=checkpoint_path,
                    module=module,
                    num_labels=2,
                    id2label={
                        0: "non_lonely",
                        1: "lonely",
                    },
                    label2id={
                        "non_lonely": 0,
                        "lonely": 1,
                    },
                )

                return self._register_model(
                    module,
                    model,
                    tokenizer,
                )

            except Exception as error:

                self._failed[module] = str(error)

                logger.error(
                    "Loneliness model loading failed: %s",
                    type(error).__name__,
                )

                raise

    # ========================================================
    # 11b. SENTIMENT BILSTM MODEL
    # ========================================================

    def load_sentiment_bilstm(self):

        module = "sentiment_bilstm"

        if module in self._loaded:
            return self.models[module]

        with self._locks[module]:

            if module in self._loaded:
                return self.models[module]

            try:

                checkpoint_path = self.settings.SENTIMENT_BILSTM_MODEL_PATH
                vocab_path = self.settings.SENTIMENT_BILSTM_VOCAB_PATH

                self._validate_checkpoint(checkpoint_path, module)

                checkpoint = torch.load(
                    checkpoint_path,
                    map_location="cpu",
                    weights_only=True,
                )

                if not isinstance(checkpoint, dict):
                    raise ModelManagerError(
                        f"{module} checkpoint has an invalid structure."
                    )

                state_dict = checkpoint.get("model_state_dict")

                if not isinstance(state_dict, dict):
                    raise ModelManagerError(
                        f"{module} checkpoint does not contain model_state_dict."
                    )

                vocabulary = checkpoint.get("vocabulary")
                if vocabulary is None and vocab_path.exists():
                    import json
                    with open(vocab_path, "r", encoding="utf-8") as file:
                        raw_vocab = json.load(file)
                        vocabulary = raw_vocab.get("vocabulary", raw_vocab)

                if not isinstance(vocabulary, dict):
                    raise ModelManagerError(
                        f"{module} vocabulary could not be resolved."
                    )

                from scripts.model_development.sentiment.tweeteval_bilstm_model import (
                    TweetEvalBiLSTM,
                )

                model = TweetEvalBiLSTM(
                    vocabulary_size=len(vocabulary),
                )

                model.load_state_dict(
                    state_dict,
                    strict=True,
                )

                model = model.to(self.device)
                model.eval()

                self.models[module] = model
                self.vocabularies[module] = vocabulary

                self._loaded.add(module)
                self._failed.pop(module, None)

                logger.info(
                    "%s model loaded successfully | device=%s",
                    module,
                    self.device,
                )

                return model

            except Exception as error:

                self._failed[module] = str(error)

                logger.error(
                    "Sentiment BiLSTM model loading failed: %s",
                    type(error).__name__,
                )

                raise

    def get_vocabulary(self, module: str):

        if module not in self.vocabularies:
            loader = getattr(self, f"load_{module}", None)
            if loader:
                loader()

        return self.vocabularies.get(module)

    # ========================================================
    # 12. GENERIC MODEL ACCESS
    # ========================================================

    def get_model(self, module: str):

        loaders = {
            "emotion": self.load_emotion,
            "sentiment": self.load_sentiment,
            "sentiment_bilstm": self.load_sentiment_bilstm,
            "stress": self.load_stress,
            "loneliness": self.load_loneliness,
        }

        if module not in loaders:
            raise ValueError(
                f"Unknown transformer module: {module}"
            )

        return loaders[module]()

    def get_tokenizer(self, module: str):

        # Calling get_model guarantees that its tokenizer
        # has also been initialized.
        self.get_model(module)

        return self.tokenizers[module]

    # ========================================================
    # 13. OPTIONAL EAGER LOADING
    # ========================================================

    def load_all_transformers(self):

        """
        Load all four transformer models.

        This is optional. Lazy loading is recommended by default.
        """

        results = {}

        loaders = {
            "emotion": self.load_emotion,
            "sentiment": self.load_sentiment,
            "stress": self.load_stress,
            "loneliness": self.load_loneliness,
        }

        for module, loader in loaders.items():

            results[module] = loader()

        return results

    # ========================================================
    # 14. MODEL STATUS
    # ========================================================

    def get_status(self):

        modules = [
            "emotion",
            "sentiment",
            "sentiment_bilstm",
            "stress",
            "loneliness",
        ]

        module_status = {}

        for module in modules:

            if module in self._loaded:
                status = "loaded"

            elif module in self._failed:
                status = "failed"

            else:
                status = "not_loaded"

            module_status[module] = {
                "status": status,
                "error": self._failed.get(module),
            }

        return {
            "device": str(self.device),
            "cuda_available": torch.cuda.is_available(),
            "shutdown": self._shutdown,
            "modules": module_status,
        }

    # ========================================================
    # 15. RESOURCE SHUTDOWN
    # ========================================================

    def shutdown(self):

        """
        Release model references during application shutdown.

        This does not run automatically. The FastAPI lifespan
        integration will call it in a later step.
        """

        if self._shutdown:
            return

        self.models.clear()
        self.tokenizers.clear()
        self.vocabularies.clear()
        self._loaded.clear()

        if self.device.type == "cuda":
            torch.cuda.empty_cache()

        self._shutdown = True

        logger.info("Inference model manager shut down.")
