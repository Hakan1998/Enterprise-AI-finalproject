from zenml import step
from zenml.integrations.mlflow.model_deployers import MLFlowModelDeployer
from zenml.integrations.mlflow.services import MLFlowDeploymentService

@step(enable_cache=False)
def prediction_service_loader(pipeline_name: str, step_name: str) -> MLFlowDeploymentService:
    """
    Finds and returns the MLflow prediction service deployed by the specified pipeline step.
    """
    model_deployer = MLFlowModelDeployer.get_active_model_deployer()
    existing_services = model_deployer.find_model_server(
        pipeline_name=pipeline_name,
        pipeline_step_name=step_name,
    )
    #CODE HERE
    # To predict the target variable of the inference data, we need to load the prediction service deployed by the training pipeline.
    # Therefore, we need to find the prediction service deployed by the training pipeline.
    # We can do this by using the find_model_server function of the model_deployer object.
    # The find_model_server function takes the pipeline_name and the pipeline_step_name as arguments.
    # Please save the result of the find_model_server function in the variable existing_services.
    # DOC https://sdkdocs.zenml.io/0.13.1/api_docs/model_deployers/#zenml.model_deployers.base_model_deployer.BaseModelDeployer.find_model_server
    if not existing_services:
        raise RuntimeError(
            f"No MLflow prediction service deployed by the "
            f"{step_name} step in the {pipeline_name} "
        )

    return existing_services[0]   