import pickle
from zenml.materializers.base_materializer import BaseMaterializer
from surprise import Dataset, Trainset
from zenml.enums import ArtifactType

class DatasetMaterializer(BaseMaterializer):
    ASSOCIATED_TYPES = (Dataset,)
    ASSOCIATED_ARTIFACT_TYPE = ArtifactType.DATA

    def handle_input(self, data_type):
        super().handle_input(data_type)
        with open(self.artifact.uri, 'rb') as f:
            return pickle.load(f)

    def handle_return(self, data):
        super().handle_return(data)
        with open(self.artifact.uri, 'wb') as f:
            pickle.dump(data, f)

class TrainsetMaterializer(BaseMaterializer):
    ASSOCIATED_TYPES = (Trainset,)
    ASSOCIATED_ARTIFACT_TYPE = ArtifactType.DATA

    def handle_input(self, data_type):
        super().handle_input(data_type)
        with open(self.artifact.uri, 'rb') as f:
            return pickle.load(f)

    def handle_return(self, data):
        super().handle_return(data)
        with open(self.artifact.uri, 'wb') as f:
            pickle.dump(data, f)

# Register the materializers
from zenml.materializers.materializer_registry import materializer_registry

materializer_registry.register_materializer(Dataset, DatasetMaterializer)
materializer_registry.register_materializer(Trainset, TrainsetMaterializer)
