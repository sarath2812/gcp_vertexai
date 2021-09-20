# /**
#  * @author Sarath Chandra Asapu
#  * @email sarath.chandra.asapu@accenture.com
#  * @create date 2021-09-17 13:08:06
#  * @modify date 2021-09-17 13:08:06
#  * @desc [description]
#  */

PROJECT_ID = 'peak-catbird-324206'
REGION = 'us-central1'

PIPELINE_NAME_TFX = 'pipelinetfxvertexai2-cicd'
DATASET = "LoanApplyData-bank.csv"
DATA_PATH = 'gs://aivertex-bucket/TFX_Pipeline/data'
SERVING_MODEL_DIR = 'gs://aivertex-bucket/TFX_Pipeline/'+PIPELINE_NAME_TFX+'/serving_model_tfx'
PIPELINE_ROOT = 'gs://aivertex-bucket/TFX_Pipeline/pipeline_root'
PIPELINE_TEMPLATE = PIPELINE_NAME_TFX + 'pipelinetfx-cicd.json'
MODULE_PATH = 'gs://aivertex-bucket/TFX_Pipeline/trainer_v2.py'


# DEPLOY_IMAGE = "us-docker.pkg.dev/vertex-ai/prediction/sklearn-cpu.0-23:latest"
# ARTIFACT_DIR = 'gs://aivertex-bucket/custom_model_bucket/'
# DATASET_PATH = ARTIFACT_DIR+"LoanApplyData-bank.csv"
# BUCKET_NAME = 'aivertex-bucket'
# SUB_FOLDER = 'custom_model_bucket/'
# PIPELINE_NAME = "custom-pipeline-kfp-local-cicd"
# MODEL_DISPLAY_NAME = "custom-model-kfp-local-cicd"
# CUSTOM_JOB_NAME = "custom-runjob-kfp-local-cicd"
# ENDPOINT_NAME = "custom-endpoint-kfp-local-cicd"
# TEMPLATE_JSON = "pipeline-kfp-local-cicd.json"




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