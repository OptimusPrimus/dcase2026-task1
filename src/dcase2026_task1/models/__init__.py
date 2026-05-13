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
from dcase2026_task1.models.qwen import (
    QwenClassificationSkill,
    QwenMetadataSummarizationSkill,
    QwenModel,
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
QwenClassificationTask = QwenClassificationSkill
QwenClassifier = QwenModel

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
    "QwenClassificationSkill",
    "QwenClassificationTask",
    "QwenClassifier",
    "QwenMetadataSummarizationSkill",
    "QwenModel",
]
