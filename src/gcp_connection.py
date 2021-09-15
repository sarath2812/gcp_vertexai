import os
from google.cloud import aiplatform
import sys
sys.path.append("..")
from config import config as cfg


CREDENTIAL_PATH = r"../config/key.json"

os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = CREDENTIAL_PATH


def aiplatform_init():
    aiplatform.init(
        project=cfg.PROJECT_ID,
        location=cfg.REGION
    )


if __name__ == "__main__":
    # vertexai_init()
    pass