from kfp.v2 import compiler,dsl
from kfp.v2.dsl import Artifact,Dataset,Input,Model,Output,Metrics,ClassificationMetrics,component
from kfp.v2.google.client import AIPlatformClient
import kfp
from google.cloud import aiplatform
from google_cloud_pipeline_components import aiplatform as gcc_aip
from kfp.v2.google import experimental
import sys
PROJECT_ID = 'peak-catbird-324206'
REGION = 'us-central1'
DEPLOY_IMAGE = "us-docker.pkg.dev/vertex-ai/prediction/sklearn-cpu.0-23:latest"
ARTIFACT_DIR = 'gs://aivertex-bucket/custom_model_bucket/'
DATASET_PATH = ARTIFACT_DIR+"LoanApplyData-bank.csv"
BUCKET_NAME = 'aivertex-bucket'
SUB_FOLDER = 'custom_model_bucket/'
PIPELINE_NAME = "custom-pipeline-kfp-test"
MODEL_DISPLAY_NAME = "custom-model-kfp-test"
CUSTOM_JOB_NAME = "custom-runjob-kfp-test"
ENDPOINT_NAME = "custom-endpoint-kfp-test"
TEMPLATE_JSON = "train_upload_deploy-kfp-test.json"

sys.path.append("..")
from config import constants as cfg
from src import gcp_connection as gc

gc.aiplatform_init()



@component(packages_to_install = ["pandas","fsspec","gcsfs","sklearn"])
def Prepare_dataset(dataset_train: Output[Dataset],dataset_path:str):
    """
    input: Dataset_path (GCS location of dataset)
    Output: Dataset artifact object
    """
    from sklearn.preprocessing import LabelEncoder
    import pandas as pd
    
    data = pd.read_csv(dataset_path)
    for col in data.drop(columns=["target"]).select_dtypes(include='object').columns:
        le = LabelEncoder()
        data[col] = le.fit_transform(data[col])
    
    data.to_csv(dataset_train.path)
    
@component(packages_to_install = ["pandas","fsspec","gcsfs","sklearn","joblib","google-cloud"])
def Training_model(dataset: Input[Dataset],
                   project_id:str,
                   bucket_name:str,
                   sub_folder:str
                  ):
    from sklearn.ensemble import RandomForestClassifier
    from google.cloud import storage
    import pandas as pd
    import joblib

    data = pd.read_csv(dataset.path)
    model = RandomForestClassifier()
    model.fit(data.drop(columns=["target"]),data.target)
    
    joblib.dump(model,"model.joblib")
    client = storage.Client(project=project_id)
    bucket = client.get_bucket(bucket_name)
    blob = bucket.blob(sub_folder+'model.joblib')
    blob.upload_from_filename("model.joblib")    


@kfp.dsl.pipeline(name=PIPELINE_NAME, pipeline_root=ARTIFACT_DIR)
def pipeline(
    project: str = PROJECT_ID,
    model_display_name: str = MODEL_DISPLAY_NAME,
    serving_container_image_uri: str = DEPLOY_IMAGE,
):
    dataset_op = Prepare_dataset(dataset_path=DATASET_PATH)

    train_task = Training_model(dataset_op.outputs["dataset_train"],project,BUCKET_NAME,SUB_FOLDER)

    experimental.run_as_aiplatform_custom_job(
        train_task,
        display_name=CUSTOM_JOB_NAME)

    model_upload_op = gcc_aip.ModelUploadOp(
        project=project,
        display_name=model_display_name,
        artifact_uri=ARTIFACT_DIR,
        serving_container_image_uri=serving_container_image_uri,
#         serving_container_environment_variables={"NOT_USED": "NO_VALUE"},
    )

    model_upload_op.after(train_task)

    endpoint_create_op = gcc_aip.EndpointCreateOp(
        project=project,
        display_name=ENDPOINT_NAME,
    )

    model_deploy_op = gcc_aip.ModelDeployOp(  
        project=project,
        endpoint=endpoint_create_op.outputs["endpoint"],
        model=model_upload_op.outputs["model"],
        deployed_model_display_name=model_display_name,
        machine_type="n1-standard-4",
    )



from kfp.v2 import compiler  

compiler.Compiler().compile(
    pipeline_func=pipeline, package_path=TEMPLATE_JSON
)


api_client = AIPlatformClient(
                project_id=PROJECT_ID,
                region=REGION
                )


response = api_client.create_run_from_job_spec(
    TEMPLATE_JSON,
)