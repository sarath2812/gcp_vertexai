# /**
#  * @author Sarath Chandra Asapu
#  * @email sarath.chandra.asapu@accenture.com
#  * @create date 2021-09-17 13:07:17
#  * @modify date 2021-09-17 13:07:17
#  * @desc [description]
#  */

import os
from google.cloud import aiplatform
import sys
sys.path.append("..")
from config import constants as cfg


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