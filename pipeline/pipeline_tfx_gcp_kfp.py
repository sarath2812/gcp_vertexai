# /**
#  * @author Sarath Chandra Asapu
#  * @email sarath.chandra.asapu@accenture.com
#  * @create date 2021-09-17 13:02:11
#  * @modify date 2021-09-17 13:02:11
#  * @desc [description]
#  */

import pandas as pd
import os
from tfx import v1 as tfx
import tensorflow as tf
import tensorflow_data_validation as tfdv
from kfp.v2.google import client
from tensorflow_transform.tf_metadata import schema_utils
from tfx.orchestration.experimental.interactive.interactive_context import InteractiveContext
from sklearn import preprocessing
import sys
sys.path.append("..")
from config import config as cfg
from src import trainer_v2 as MODULE_FILE

# Global Variables
PIPELINE_NAME_TFX = cfg.PIPELINE_NAME_TFX
DATASET = cfg.DATASET
DATA_PATH = cfg.DATA_PATH
SERVING_MODEL_DIR = cfg.SERVING_MODEL_DIR
PIPELINE_ROOT = cfg.PIPELINE_ROOT
METADATA = cfg.METADATA
TEMPLATE_JSON = cfg.TEMPLATE_JSON
PROJECT_ID = cfg.PROJECT_ID
REGION = cfg.REGION
MODULE_PATH = cfg.MODULE_PATH
PIPELINE_TEMPLATE = cfg.PIPELINE_TEMPLATE

def pipeline():
    # Prepare dataset
    data = tfx.components.CsvExampleGen(input_base=DATA_PATH)


    vertex_job_spec = {
        'project': PROJECT_ID,
        'worker_pool_specs': [{
            'machine_spec': {
                'machine_type': 'n1-standard-4',
            },
            'replica_count': 1,
            'container_spec': {
                'image_uri': 'gcr.io/tfx-oss-public/tfx:{}'.format(tfx.__version__),
            },
        }],
    }

    #Trainer object
    trainer = tfx.extensions.google_cloud_ai_platform.Trainer(
        module_file=MODULE_PATH,
        examples=data.outputs['examples'],
        train_args=tfx.proto.TrainArgs(num_steps=100),
        eval_args=tfx.proto.EvalArgs(num_steps=5),
        custom_config={
            tfx.extensions.google_cloud_ai_platform.ENABLE_UCAIP_KEY:
                True,
            tfx.extensions.google_cloud_ai_platform.UCAIP_REGION_KEY:
                REGION,
            tfx.extensions.google_cloud_ai_platform.TRAINING_ARGS_KEY:
                vertex_job_spec,
        }
    )   

    # Pusher object
    pusher = tfx.components.Pusher(
      model=trainer.outputs['model'],
      push_destination=tfx.proto.PushDestination(
          filesystem=tfx.proto.PushDestination.Filesystem(
              base_directory=SERVING_MODEL_DIR)))

    # components sequence
    components = [
        data,
        trainer,
        pusher,
    ]
    
    pipeline = tfx.dsl.Pipeline(
      pipeline_name=PIPELINE_NAME_TFX,
      pipeline_root=PIPELINE_ROOT,
      components=components)
    
    runner = tfx.orchestration.experimental.KubeflowV2DagRunner(
    config=tfx.orchestration.experimental.KubeflowV2DagRunnerConfig(),
    output_filename=PIPELINE_TEMPLATE)
    
    runner.run(pipeline)

    # pipeline creation
    # pipeline = tfx.dsl.Pipeline(
    #     PIPELINE_NAME_TFX=PIPELINE_NAME_TFX,
    #     pipeline_root=PIPELINE_ROOT,
    #     metadata_connection_config=tfx.orchestration.metadata.sqlite_metadata_connection_config(METADATA),
    #     components=components)
    
    runner = tfx.orchestration.experimental.KubeflowV2DagRunner(
    config=tfx.orchestration.experimental.KubeflowV2DagRunnerConfig(),
    output_filename=TEMPLATE_JSON)
    
    # return "SUCCESS"


if __name__ == "__main__":
    pipeline()
    # pipeline run
    pipelines_client = client.AIPlatformClient(
    project_id=PROJECT_ID,
    region=REGION,
    )
    response = pipelines_client.create_run_from_job_spec(TEMPLATE_JSON)