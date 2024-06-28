# materializers/trainset_materializer.py
from zenml.materializers.base_materializer import BaseMaterializer
from surprise import Trainset
import os
import pickle

class TrainsetMaterializer(BaseMaterializer):
    ASSOCIATED_TYPES = (Trainset,)
    ASSOCIATED_ARTIFACT_TYPE = "surprise_trainset"

    def handle_input(self, data_type):
        super().handle_input(data_type)
        with open(self.artifact.uri, 'rb') as f:
            return pickle.load(f)

    def handle_return(self, data):
        super().handle_return(data)
        with open(self.artifact.uri, 'wb') as f:
            pickle.dump(data, f)
