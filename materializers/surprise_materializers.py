import pickle
from zenml.materializers.base_materializer import BaseMaterializer
from surprise import Dataset, Trainset
from zenml.enums import ArtifactType

class DatasetMaterializer(BaseMaterializer):
    """
    Materializer to handle the serialization and deserialization of Surprise Dataset objects.

    This materializer handles the input and output operations for Dataset objects
    by serializing them with pickle for storage in ZenML artifacts.
    """
    ASSOCIATED_TYPES = (Dataset,)
    ASSOCIATED_ARTIFACT_TYPE = ArtifactType.DATA

    def handle_input(self, data_type):
        """
        Deserialize a Dataset object from a pickle file.

        Args:
            data_type: The expected data type for the deserialized object.

        Returns:
            Dataset: The deserialized Dataset object.
        """
        super().handle_input(data_type)
        with open(self.artifact.uri, 'rb') as f:
            return pickle.load(f)

    def handle_return(self, data):
        """
        Serialize a Dataset object to a pickle file.

        Args:
            data: The Dataset object to be serialized.
        """
        super().handle_return(data)
        with open(self.artifact.uri, 'wb') as f:
            pickle.dump(data, f)

class TrainsetMaterializer(BaseMaterializer):
    """
    Materializer to handle the serialization and deserialization of Surprise Trainset objects.

    This materializer handles the input and output operations for Trainset objects
    by serializing them with pickle for storage in ZenML artifacts.
    """
    ASSOCIATED_TYPES = (Trainset,)
    ASSOCIATED_ARTIFACT_TYPE = ArtifactType.DATA

    def handle_input(self, data_type):
        """
        Deserialize a Trainset object from a pickle file.

        Args:
            data_type: The expected data type for the deserialized object.

        Returns:
            Trainset: The deserialized Trainset object.
        """
        super().handle_input(data_type)
        with open(self.artifact.uri, 'rb') as f:
            return pickle.load(f)

    def handle_return(self, data):
        """
        Serialize a Trainset object to a pickle file.

        Args:
            data: The Trainset object to be serialized.
        """
        super().handle_return(data)
        with open(self.artifact.uri, 'wb') as f:
            pickle.dump(data, f)

# Register the materializers
from zenml.materializers.materializer_registry import materializer_registry

materializer_registry.register_materializer(Dataset, DatasetMaterializer)
materializer_registry.register_materializer(Trainset, TrainsetMaterializer)
