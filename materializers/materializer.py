from zenml.artifacts import ModelArtifact, DataArtifact, BaseArtifact
from typing import Dict, Any, Tuple, Annotated
import torch.nn

# Definieren von benutzerdefinierten Artefakten
class SVDArtifact(ModelArtifact):
    TYPE_NAME = "SVDArtifact"

class KNNArtifact(ModelArtifact):
    TYPE_NAME = "KNNArtifact"

class BaselineArtifact(ModelArtifact):
    TYPE_NAME = "BaselineArtifact"

class ContentBasedArtifact(DataArtifact):
    TYPE_NAME = "ContentBasedArtifact"


