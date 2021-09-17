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
from tensorflow_transform.tf_metadata import schema_utils
from tfx.orchestration.experimental.interactive.interactive_context import InteractiveContext
from sklearn import preprocessing
import sys
sys.path.append("..")
from config import config as cfg
from src import trainer_v2 as MODULE_FILE

# Global Variables
PIPELINE_NAME = cfg.PIPELINE_NAME
DATASET = cfg.DATASET
DATA_PATH = cfg.DATA_PATH
SERVING_MODEL_DIR = cfg.SERVING_MODEL_DIR
PIPELINE_ROOT = cfg.PIPELINE_ROOT
METADATA = cfg.METADATA

def pipeline():
    # Prepare dataset
    data = tfx.components.CsvExampleGen(input_base=DATA_PATH)

    #Trainer object
    trainer = tfx.components.Trainer(
                    module_file=MODULE_FILE,
                    examples=data.outputs['examples'],
                    train_args=tfx.proto.TrainArgs(num_steps=100),
                    eval_args=tfx.proto.EvalArgs(num_steps=5)
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

    # pipeline creation
    pipeline = tfx.dsl.Pipeline(
        pipeline_name=PIPELINE_NAME,
        pipeline_root=PIPELINE_ROOT,
        metadata_connection_config=tfx.orchestration.metadata.sqlite_metadata_connection_config(METADATA),
        components=components)
    
    return pipeline


if __name__ == "__main__":
    pass
    # pipeline = pipeline()
    # pipeline run
    # tfx.orchestration.LocalDagRunner().run(pipeline)