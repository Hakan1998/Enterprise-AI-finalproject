import os
import pickle
from typing import Type
from surprise import Trainset
from zenml.materializers.base_materializer import BaseMaterializer
from zenml.enums import ArtifactType

class TrainsetMaterializer(BaseMaterializer):
    ASSOCIATED_TYPES = (Trainset,)
    ASSOCIATED_ARTIFACT_TYPE = ArtifactType.DATA

    def handle_input(self, data_type: Type[Trainset]) -> Trainset:
        super().handle_input(data_type)
        with open(os.path.join(self.artifact.uri, "trainset.pkl"), 'rb') as f:
            return pickle.load(f)

    def handle_return(self, trainset: Trainset) -> None:
        super().handle_return(trainset)
        with open(os.path.join(self.artifact.uri, "trainset.pkl"), 'wb') as f:
            pickle.dump(trainset, f)