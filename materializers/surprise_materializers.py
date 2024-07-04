import pickle
import numpy as np
from surprise import Dataset, Trainset
from zenml import pipeline
from zenml.client import Client
from zenml.materializers.base_materializer import BaseMaterializer
from zenml.materializers.materializer_registry import materializer_registry
from zenml.enums import ArtifactType

# Materializer for Surprise Dataset objects
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

# Materializer for Surprise Trainset objects
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

# Materializer for numpy int64 objects
class NumpyInt64Materializer(BaseMaterializer):
    """
    Materializer to handle the serialization and deserialization of numpy int64 objects.

    This materializer handles the input and output operations for numpy int64 objects
    by serializing them with numpy's save and load functions for storage in ZenML artifacts.
    """
    ASSOCIATED_TYPES = (np.int64,)
    ASSOCIATED_ARTIFACT_TYPES = (np.int64,)

    def handle_input(self, data_type: type[np.int64]) -> np.int64:
        """
        Read the artifact as a numpy int64.

        Args:
            data_type: The expected data type for the deserialized object.

        Returns:
            np.int64: The deserialized numpy int64 object.
        """
        super().handle_input(data_type)
        with open(self.artifact.uri, 'rb') as f:
            return np.load(f)

    def handle_return(self, data: np.int64) -> None:
        """
        Write a numpy int64 to the artifact store.

        Args:
            data: The numpy int64 object to be serialized.
        """
        super().handle_return(data)
        with open(self.artifact.uri, 'wb') as f:
            np.save(f, data)

# Register the custom materializers
# Register the custom materializers
materializer_registry.register_materializer(Dataset, DatasetMaterializer)
materializer_registry.register_materializer(Trainset, TrainsetMaterializer)
materializer_registry.register_materializer(np.int64, NumpyInt64Materializer)
