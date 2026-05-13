from dcase2026_task1.models.audio_flamingo3 import (
    AudioFlamingo3AudioCaptioningSkill,
    AudioFlamingo3ClassificationSkill,
    AudioFlamingo3Model,
)
from dcase2026_task1.models.base import (
    AudioLanguageModel,
    ModelInput,
    ModelSkill,
)
from dcase2026_task1.models.qwen_text import (
    QwenTextClassificationSkill,
    QwenTextModel,
)
from dcase2026_task1.models.qwen3_6_35b_a3b import (
    Qwen3_6_35BA3BMetadataSummarizationSkill,
    Qwen3_6_35BA3BModel,
)
from dcase2026_task1.tasks import (
    AudioCaptioningResponse,
    AudioCaptioningTask,
    ClassificationResponse,
    ClassificationTask,
    MetadataSummarizationResponse,
    MetadataSummarizationTask,
)

AudioFlamingo3ClassificationTask = AudioFlamingo3ClassificationSkill
AudioFlamingo3Classifier = AudioFlamingo3Model
QwenTextClassificationTask = QwenTextClassificationSkill
QwenTextClassifier = QwenTextModel

__all__ = [
    "AudioCaptioningResponse",
    "AudioCaptioningTask",
    "AudioFlamingo3AudioCaptioningSkill",
    "AudioFlamingo3ClassificationSkill",
    "AudioFlamingo3ClassificationTask",
    "AudioFlamingo3Classifier",
    "AudioFlamingo3Model",
    "AudioLanguageModel",
    "ClassificationResponse",
    "ClassificationTask",
    "ModelInput",
    "ModelSkill",
    "MetadataSummarizationResponse",
    "MetadataSummarizationTask",
    "Qwen3_6_35BA3BMetadataSummarizationSkill",
    "Qwen3_6_35BA3BModel",
    "QwenTextClassificationSkill",
    "QwenTextClassificationTask",
    "QwenTextClassifier",
    "QwenTextModel",
]
