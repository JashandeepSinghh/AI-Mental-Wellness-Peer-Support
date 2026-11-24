
"""
Unified Inference Pipeline
==========================

Connects four transformer models and three analytical modules.

Transformer modules:
    1. Emotion
    2. Sentiment
    3. Stress
    4. Loneliness

Analytical modules:
    5. Risk
    6. Rumination
    7. VAD

Confidence aggregation:
    - Aggregates all eight registered modules.
    - Sentiment BiLSTM remains not_evaluated until integrated.
    - No artificial overall confidence.
    - No safety probability.
    - Missing risk evaluation remains unknown.
    - Raw user text is not included in analytical module outputs.
"""

from __future__ import annotations

from typing import Any

import torch

from backend.app.inference.model_manager import ModelManager

from backend.app.inference.emotion_thresholds import (
    load_emotion_thresholds,
)

from scripts.model_development.confidence.confidence_metrics import (
    analyze_multiclass_probabilities,
    analyze_multilabel_probabilities,
)

from scripts.model_development.confidence.confidence_aggregator import (
    aggregate_all_modules,
)

from scripts.model_development.risk.risk_detector import (
    detect_risk_language,
)

from scripts.model_development.rumination.rumination_detector import (
    detect_rumination,
)

from scripts.model_development.sentiment.tweeteval_bilstm_dataset import (
    encode_text,
)

from scripts.model_development.vad.vad_analyzer import (
    analyze_vad as run_vad_analysis,
)


class InferencePipelineError(RuntimeError):
    """Raised when unified inference cannot be completed."""


class InferencePipeline:

    # ==========================================================
    # 1. MODULE REGISTRY
    # ==========================================================

    REGISTERED_MODULES = (
        "emotion",
        "sentiment",
        "stress",
        "loneliness",
        "risk",
        "rumination",
        "vad",
        "sentiment_bilstm",
    )

    # ==========================================================
    # 2. MODEL LABEL CONFIGURATION
    # ==========================================================

    EMOTION_LABELS = [
        "admiration", "amusement", "anger", "annoyance",
        "approval", "caring", "confusion", "curiosity",
        "desire", "disappointment", "disapproval", "disgust",
        "embarrassment", "excitement", "fear", "gratitude",
        "grief", "joy", "love", "nervousness", "optimism",
        "pride", "realization", "relief", "remorse",
        "sadness", "surprise", "neutral",
    ]

    SENTIMENT_LABELS = [
        "negative",
        "neutral",
        "positive",
    ]

    STRESS_LABELS = [
        "non_stress",
        "stress",
    ]

    LONELINESS_LABELS = [
        "non_lonely",
        "lonely",
    ]

    # ==========================================================
    # 3. TOKENIZATION CONFIGURATION
    # ==========================================================

    SEQUENCE_LENGTHS = {
        "emotion": 256,
        "sentiment": 160,
        "stress": 256,
        "loneliness": 384,
    }

    # ==========================================================
    # 4. INITIALIZATION
    # ==========================================================

    def __init__(
        self,
        model_manager: ModelManager | None = None,
    ):

        self.model_manager = (
            model_manager
            if model_manager is not None
            else ModelManager()
        )

        self.emotion_thresholds = load_emotion_thresholds(
            self.EMOTION_LABELS
        )

    # ==========================================================
    # 5. INPUT VALIDATION
    # ==========================================================

    @staticmethod
    def _validate_text(text: str) -> str:

        if not isinstance(text, str):
            raise ValueError(
                "Input text must be a string."
            )

        normalized_text = text.strip()

        if not normalized_text:
            raise ValueError(
                "Input text cannot be empty."
            )

        if len(normalized_text) > 20000:
            raise ValueError(
                "Input text exceeds the maximum length."
            )

        return normalized_text

    # ==========================================================
    # 6. PRIVACY SANITIZATION
    # ==========================================================

    @staticmethod
    def _remove_raw_text_fields(
        value: Any,
    ) -> Any:
        """
        Recursively remove raw user-text fields from analytical
        module outputs before they become API response data.

        This preserves numerical results, indicators, counts,
        warnings, and other non-text analytical metadata.
        """

        sensitive_keys = {
            "input_text",
            "original_text",
            "raw_text",
            "source_text",
        }

        if isinstance(value, dict):

            sanitized = {}

            for key, item in value.items():

                if (
                    isinstance(key, str)
                    and key.lower() in sensitive_keys
                ):
                    continue

                sanitized[key] = (
                    InferencePipeline._remove_raw_text_fields(
                        item
                    )
                )

            return sanitized

        if isinstance(value, list):

            return [
                InferencePipeline._remove_raw_text_fields(
                    item
                )
                for item in value
            ]

        if isinstance(value, tuple):

            return tuple(
                InferencePipeline._remove_raw_text_fields(
                    item
                )
                for item in value
            )

        return value

    # ==========================================================
    # 7. CONFIDENCE METRICS
    # ==========================================================

    def _analyze_confidence(
        self,
        module_name: str,
        labels: list[str],
        probabilities: list[float],
        output_type: str,
    ) -> dict[str, Any]:

        if output_type == "multi_class":

            return analyze_multiclass_probabilities(
                probabilities=probabilities,
                class_names=labels,
            )

        if output_type == "multi_label":

            confidence_result = (
                analyze_multilabel_probabilities(
                    probabilities=probabilities,
                    label_names=labels,
                    threshold=0.5,
                )
            )

            # Emotion uses individual validation thresholds.
            if module_name == "emotion":

                selected_labels = []

                for item in confidence_result["label_results"]:

                    label = item["label_name"]

                    probability = item["probability"]

                    threshold = self.emotion_thresholds[label]

                    predicted_positive = (
                        probability >= threshold
                    )

                    item["threshold"] = threshold

                    item["predicted_positive"] = (
                        bool(predicted_positive)
                    )

                    if predicted_positive:
                        selected_labels.append(label)

                confidence_result["selected_labels"] = (
                    selected_labels
                )

                confidence_result["threshold"] = (
                    "individual_validation_selected"
                )

                confidence_result["thresholds"] = (
                    self.emotion_thresholds.copy()
                )

            return confidence_result

        raise ValueError(
            f"Unsupported confidence output type: {output_type}"
        )

    # ==========================================================
    # 8. TRANSFORMER INFERENCE ENGINE
    # ==========================================================

    def _run_transformer(
        self,
        module_name: str,
        text: str,
        labels: list[str],
        output_type: str,
    ) -> dict[str, Any]:

        # ------------------------------------------------------
        # A. Lazy model loading
        # ------------------------------------------------------

        loader = getattr(
            self.model_manager,
            f"load_{module_name}",
            None,
        )

        if loader is None:
            raise InferencePipelineError(
                f"No model loader registered for {module_name}."
            )

        try:

            model = loader()

            tokenizer = self.model_manager.get_tokenizer(
                module_name
            )

        except Exception as exc:

            raise InferencePipelineError(
                f"Unable to load {module_name} model."
            ) from exc

        if model is None or tokenizer is None:
            raise InferencePipelineError(
                f"{module_name} model or tokenizer is unavailable."
            )

        # ------------------------------------------------------
        # B. Tokenization
        # ------------------------------------------------------

        encoded_inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=self.SEQUENCE_LENGTHS[module_name],
        )

        encoded_inputs = {
            key: value.to(self.model_manager.device)
            for key, value in encoded_inputs.items()
        }

        # ------------------------------------------------------
        # C. Model forward pass
        # ------------------------------------------------------

        model.eval()

        try:

            with torch.inference_mode():
                output = model(**encoded_inputs)

        except Exception as exc:

            raise InferencePipelineError(
                f"{module_name} inference failed."
            ) from exc

        logits = output.logits

        # ------------------------------------------------------
        # D. Output validation
        # ------------------------------------------------------

        expected_classes = len(labels)

        if (
            logits.ndim != 2
            or logits.shape[-1] != expected_classes
        ):

            raise InferencePipelineError(
                f"Unexpected output shape for {module_name}: "
                f"{tuple(logits.shape)}"
            )

        if not torch.isfinite(logits).all():

            raise InferencePipelineError(
                f"{module_name} produced non-finite logits."
            )

        # ------------------------------------------------------
        # E. Probability calculation
        # ------------------------------------------------------

        if output_type == "multi_label":

            probabilities = torch.sigmoid(logits)[0]

        elif output_type == "multi_class":

            probabilities = torch.softmax(
                logits,
                dim=-1,
            )[0]

        else:

            raise ValueError(
                f"Unsupported output type: {output_type}"
            )

        # ------------------------------------------------------
        # F. Preserve raw model outputs
        # ------------------------------------------------------

        raw_logits = (
            logits[0]
            .detach()
            .cpu()
            .tolist()
        )

        probability_values = (
            probabilities
            .detach()
            .cpu()
            .tolist()
        )

        label_results = {
            label: float(probability)
            for label, probability in zip(
                labels,
                probability_values,
            )
        }

        # ------------------------------------------------------
        # G. Confidence analysis
        # ------------------------------------------------------

        confidence_analysis = self._analyze_confidence(
            module_name=module_name,
            labels=labels,
            probabilities=probability_values,
            output_type=output_type,
        )

        # ------------------------------------------------------
        # H. Emotion threshold information
        # ------------------------------------------------------

        selected_labels = None

        thresholds_applied = False

        threshold_source = None

        if module_name == "emotion":

            selected_labels = (
                confidence_analysis["selected_labels"]
            )

            thresholds_applied = True

            threshold_source = (
                "validation_summary.json"
            )

        # ------------------------------------------------------
        # I. Standardized model output
        # ------------------------------------------------------

        result = {

            "module_name": module_name,

            "status": "evaluated",

            "output_type": output_type,

            "predictions": label_results,

            "raw_logits": raw_logits,

            "probabilities": probability_values,

            "confidence_analysis": confidence_analysis,

            "selected_labels": selected_labels,

            "thresholds_applied": thresholds_applied,

            "threshold_source": threshold_source,

            "calibration_status": "not_calibrated",

            "is_calibrated": False,

            "clinical_diagnosis": False,

        }

        if output_type == "multi_class":

            result["predicted_class"] = (
                confidence_analysis["predicted_class"]
            )

        elif output_type == "multi_label":

            result["label_count"] = len(labels)

            result["selected_label_count"] = len(
                selected_labels
            )

            result["threshold_selection_status"] = (
                "validation_selected_thresholds_applied"
            )

        return result

    # ==========================================================
    # 9. TRANSFORMER MODULE INTERFACES
    # ==========================================================

    def analyze_emotion(
        self,
        text: str,
    ) -> dict[str, Any]:

        return self._run_transformer(
            module_name="emotion",
            text=text,
            labels=self.EMOTION_LABELS,
            output_type="multi_label",
        )

    def analyze_sentiment(
        self,
        text: str,
    ) -> dict[str, Any]:

        return self._run_transformer(
            module_name="sentiment",
            text=text,
            labels=self.SENTIMENT_LABELS,
            output_type="multi_class",
        )

    def analyze_sentiment_bilstm(
        self,
        text: str,
    ) -> dict[str, Any]:

        module_name = "sentiment_bilstm"
        labels = self.SENTIMENT_LABELS
        output_type = "multi_class"

        try:
            model = self.model_manager.load_sentiment_bilstm()
            vocabulary = self.model_manager.get_vocabulary(module_name)
        except Exception as exc:
            raise InferencePipelineError(
                f"Unable to load {module_name} model."
            ) from exc

        if model is None or vocabulary is None:
            raise InferencePipelineError(
                f"{module_name} model or vocabulary is unavailable."
            )

        token_ids = encode_text(text, vocabulary)
        input_ids = torch.tensor(
            [token_ids],
            dtype=torch.long,
            device=self.model_manager.device,
        )
        lengths = torch.tensor(
            [len(token_ids)],
            dtype=torch.long,
            device=self.model_manager.device,
        )

        model.eval()
        try:
            with torch.inference_mode():
                logits = model(input_ids=input_ids, lengths=lengths)
        except Exception as exc:
            raise InferencePipelineError(
                f"{module_name} inference failed."
            ) from exc

        expected_classes = len(labels)
        if (
            logits.ndim != 2
            or logits.shape[-1] != expected_classes
        ):
            raise InferencePipelineError(
                f"Unexpected output shape for {module_name}: "
                f"{tuple(logits.shape)}"
            )

        if not torch.isfinite(logits).all():
            raise InferencePipelineError(
                f"{module_name} produced non-finite logits."
            )

        probabilities = torch.softmax(
            logits,
            dim=-1,
        )[0]

        raw_logits = (
            logits[0]
            .detach()
            .cpu()
            .tolist()
        )

        probability_values = (
            probabilities
            .detach()
            .cpu()
            .tolist()
        )

        label_results = {
            label: float(probability)
            for label, probability in zip(
                labels,
                probability_values,
            )
        }

        confidence_analysis = self._analyze_confidence(
            module_name=module_name,
            labels=labels,
            probabilities=probability_values,
            output_type=output_type,
        )

        result = {
            "module_name": module_name,
            "status": "evaluated",
            "output_type": output_type,
            "predictions": label_results,
            "raw_logits": raw_logits,
            "probabilities": probability_values,
            "confidence_analysis": confidence_analysis,
            "selected_labels": None,
            "thresholds_applied": False,
            "threshold_source": None,
            "calibration_status": "not_calibrated",
            "is_calibrated": False,
            "clinical_diagnosis": False,
            "predicted_class": confidence_analysis["predicted_class"],
        }

        return result

    def analyze_stress(
        self,
        text: str,
    ) -> dict[str, Any]:

        return self._run_transformer(
            module_name="stress",
            text=text,
            labels=self.STRESS_LABELS,
            output_type="multi_class",
        )

    def analyze_loneliness(
        self,
        text: str,
    ) -> dict[str, Any]:

        return self._run_transformer(
            module_name="loneliness",
            text=text,
            labels=self.LONELINESS_LABELS,
            output_type="multi_class",
        )

    # ==========================================================
    # 10. RISK MODULE
    # ==========================================================

    @staticmethod
    def analyze_risk(
        text: str,
    ) -> dict[str, Any]:

        output = detect_risk_language(text)

        if not isinstance(output, dict):
            raise InferencePipelineError(
                "Risk module returned an invalid output."
            )

        return {
            **output,
            "module_name": "risk",
            "status": "evaluated",
            "category": output.get("risk_category"),
            "requires_human_review": output.get(
                "requires_human_review"
            ),
            "detected_indicators": output.get(
                "detected_indicators",
                {},
            ),
            "safety_probability": None,
        }

    # ==========================================================
    # 11. RUMINATION MODULE
    # ==========================================================

    @staticmethod
    def analyze_rumination(
        text: str,
    ) -> dict[str, Any]:

        output = detect_rumination(text)

        if not isinstance(output, dict):
            raise InferencePipelineError(
                "Rumination module returned an invalid output."
            )

        detected_categories = output.get(
            "detected_categories",
            [],
        )

        sanitized_output = (
            InferencePipeline._remove_raw_text_fields(
                output
            )
        )

        return {
            **sanitized_output,
            "module_name": "rumination",
            "status": "evaluated",
            "detected_indicators": detected_categories,
            "indicators": detected_categories,
            "ambiguity_flags": [],
            "analysis_type": "single_text",
        }

    # ==========================================================
    # 12. VAD MODULE
    # ==========================================================

    @staticmethod
    def analyze_vad(
        text: str,
    ) -> dict[str, Any]:

        output = run_vad_analysis(text)

        if not isinstance(output, dict):
            raise InferencePipelineError(
                "VAD module returned an invalid output."
            )

        sanitized_output = (
            InferencePipeline._remove_raw_text_fields(
                output
            )
        )

        coverage = sanitized_output.get(
            "coverage",
            {},
        )

        return {
            **sanitized_output,
            "module_name": "vad",
            "status": "evaluated",
            "coverage": coverage.get(
                "coverage",
                0.0,
            ) if isinstance(coverage, dict) else coverage,
            "scoring_term_count": sanitized_output.get(
                "scoring_term_count",
                0,
            ),
        }

    # ==========================================================
    # 13. CONFIDENCE AGGREGATION ADAPTER
    # ==========================================================

    @staticmethod
    def _build_confidence_report(
        modules: dict[str, Any],
    ) -> dict[str, Any]:

        aggregation_inputs = {}

        # ------------------------------------------------------
        # A. Classification modules
        # ------------------------------------------------------

        for module_name in (
            "emotion",
            "sentiment",
            "sentiment_bilstm",
            "stress",
            "loneliness",
        ):

            module_output = modules.get(module_name)

            if (
                isinstance(module_output, dict)
                and module_output.get("status") == "evaluated"
            ):

                aggregation_inputs[module_name] = (
                    module_output["confidence_analysis"]
                )

        # ------------------------------------------------------
        # B. Rule-based modules
        # ------------------------------------------------------

        for module_name in (
            "risk",
            "rumination",
            "vad",
        ):

            module_output = modules.get(module_name)

            if (
                isinstance(module_output, dict)
                and module_output.get("status") == "evaluated"
            ):

                aggregation_inputs[module_name] = module_output

        return aggregate_all_modules(
            aggregation_inputs
        )

    # ==========================================================
    # 14. UNIFIED TEXT ANALYSIS
    # ==========================================================

    def analyze_text(
        self,
        text: str,
        include_transformers: bool = True,
    ) -> dict[str, Any]:

        text = self._validate_text(text)

        results = {
            "input_length": len(text),
            "status": "completed",
            "modules": {},
        }

        modules = results["modules"]

        try:

            # --------------------------------------------------
            # A. Transformer modules
            # --------------------------------------------------

            if include_transformers:

                modules["emotion"] = (
                    self.analyze_emotion(text)
                )

                modules["sentiment"] = (
                    self.analyze_sentiment(text)
                )

                modules["sentiment_bilstm"] = (
                    self.analyze_sentiment_bilstm(text)
                )

                modules["stress"] = (
                    self.analyze_stress(text)
                )

                modules["loneliness"] = (
                    self.analyze_loneliness(text)
                )

            # --------------------------------------------------
            # B. Rule-based and lexicon modules
            # --------------------------------------------------

            modules["risk"] = (
                self.analyze_risk(text)
            )

            modules["rumination"] = (
                self.analyze_rumination(text)
            )

            modules["vad"] = (
                self.analyze_vad(text)
            )

            # --------------------------------------------------
            # C. Eight-module confidence aggregation
            # --------------------------------------------------

            results["confidence_report"] = (
                self._build_confidence_report(modules)
            )

        except Exception as exc:

            raise InferencePipelineError(
                "Unified inference pipeline failed."
            ) from exc

        # ------------------------------------------------------
        # D. Explicit module accounting
        # ------------------------------------------------------

        confidence_report = results["confidence_report"]

        registered_module_count = len(
            self.REGISTERED_MODULES
        )

        evaluated_module_count = sum(
            1
            for module_output in modules.values()
            if (
                isinstance(module_output, dict)
                and module_output.get("status") == "evaluated"
            )
        )

        not_evaluated_module_count = (
            registered_module_count - evaluated_module_count
        )

        results["registered_module_count"] = (
            registered_module_count
        )

        results["evaluated_module_count"] = (
            evaluated_module_count
        )

        results["not_evaluated_module_count"] = (
            not_evaluated_module_count
        )

        # Backward-compatible field:
        # module_count now consistently means registered modules.
        results["module_count"] = registered_module_count

        # Keep the aggregator's own count visible for consistency
        # checking rather than silently overwriting its report.
        results["confidence_report_module_count"] = (
            confidence_report.get("module_count")
        )

        results["module_count_consistent"] = (
            confidence_report.get("module_count")
            == registered_module_count
            and confidence_report.get("evaluated_module_count")
            == evaluated_module_count
        )

        # ------------------------------------------------------
        # E. Pipeline disclaimers
        # ------------------------------------------------------

        results["disclaimers"] = [

            "Model probabilities are not calibrated.",

            "Confidence metrics describe model output "
            "behavior, not clinical certainty.",

            "Emotion thresholds are validation-selected "
            "for label selection and do not calibrate probabilities.",

            "Outputs are linguistic and statistical signals, "
            "not clinical diagnoses.",

            "Absence of detected risk indicators does not "
            "establish safety.",

            "No overall confidence or safety probability "
            "is calculated.",

            "Raw user text is excluded from analytical "
            "module response fields.",

        ]

        return results
