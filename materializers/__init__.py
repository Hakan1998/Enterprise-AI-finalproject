# __init__.py or any other initialization file


from zenml.materializers.materializer_registry import materializer_registry
from materializers.trainset_materializer import TrainsetMaterializer
from surprise import BaselineOnly, KNNBasic, Trainset, SVD

materializer_registry.register_materializer(Trainset, TrainsetMaterializer)


# Register the materializer
from zenml.materializers.materializer_registry import materializer_registry
from materializers.dataframe_materializer import DatasetMaterializer
from surprise import Dataset
materializer_registry.register_materializer(Dataset, DatasetMaterializer)

from zenml.materializers.materializer_registry import materializer_registry
from materializers.surprise_materializers import DatasetMaterializer, TrainsetMaterializer, SVDMaterializer, KNNBasicMaterializer, BaselineOnlyMaterializer
from materializers.list_materializer import ListMaterializer  # Import the new materializer

materializer_registry.register_materializer(Dataset, DatasetMaterializer)
materializer_registry.register_materializer(Trainset, TrainsetMaterializer)
materializer_registry.register_materializer(SVD, SVDMaterializer)
materializer_registry.register_materializer(KNNBasic, KNNBasicMaterializer)
materializer_registry.register_materializer(BaselineOnly, BaselineOnlyMaterializer)
materializer_registry.register_materializer(list, ListMaterializer)  # Register the new materializer


