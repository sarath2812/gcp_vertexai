# /**
#  * @author Sarath Chandra Asapu
#  * @email sarath.chandra.asapu@accenture.com
#  * @create date 2021-09-17 13:07:01
#  * @modify date 2021-09-17 13:07:01
#  * @desc [description]
#  */

from kfp.v2 import compiler,dsl
from kfp.v2.dsl import Artifact,Dataset,Input,Model,Output,Metrics,ClassificationMetrics,component
from kfp.v2.google.client import AIPlatformClient
import kfp
from kfp.v2 import compiler  
from google.cloud import aiplatform
from google_cloud_pipeline_components import aiplatform as gcc_aip
from kfp.v2.google import experimental
import sys
import os
sys.path.append("..")
from config import constants as cfg
from src import gcp_connection as gc

gc.aiplatform_init()

PROJECT_ID = cfg.PROJECT_ID
REGION = cfg.REGION
DEPLOY_IMAGE = cfg.DEPLOY_IMAGE
ARTIFACT_DIR = cfg.ARTIFACT_DIR
DATASET_PATH = cfg.DATASET_PATH
BUCKET_NAME = cfg.BUCKET_NAME
SUB_FOLDER = cfg.SUB_FOLDER
PIPELINE_NAME = cfg.PIPELINE_NAME
MODEL_DISPLAY_NAME = cfg.MODEL_DISPLAY_NAME
CUSTOM_JOB_NAME = cfg.CUSTOM_JOB_NAME
ENDPOINT_NAME = cfg.ENDPOINT_NAME
TEMPLATE_JSON = cfg.TEMPLATE_JSON
JOB_NAME = cfg.JOB_NAME


print("="*100)
print(PROJECT_ID) 
print(REGION) 
print(DEPLOY_IMAGE)
print(ARTIFACT_DIR)
print(DATASET_PATH)
print(BUCKET_NAME) 
print(SUB_FOLDER) 
print(PIPELINE_NAME)
print(MODEL_DISPLAY_NAME)
print(CUSTOM_JOB_NAME)
print(ENDPOINT_NAME) 
print(TEMPLATE_JSON) 
print("="*100)


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


compiler.Compiler().compile(
    pipeline_func=pipeline, package_path=TEMPLATE_JSON
)

api_client = AIPlatformClient(
                project_id=PROJECT_ID,
                region=REGION
                )

response = api_client.create_run_from_job_spec(job_spec_path=TEMPLATE_JSON, job_id=JOB_NAME)