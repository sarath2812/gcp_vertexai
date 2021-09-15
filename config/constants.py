
# PIPELINE_NAME = 'pipeline_tfx'
# DATASET = "LoanApplyData-bank.csv"
# DATA_PATH = '../data'
# SERVING_MODEL_DIR = '../artifacts/'+PIPELINE_NAME+'/serving_model'
# PIPELINE_ROOT = '../artifacts/'+PIPELINE_NAME
# METADATA = '../artifacts/'+PIPELINE_NAME+'/metadata'
# PROJECT_ID = 'peak-catbird-324206'
# REGION = 'us-central1'


PROJECT_ID = 'peak-catbird-324206'
REGION = 'us-central1'
DEPLOY_IMAGE = "us-docker.pkg.dev/vertex-ai/prediction/sklearn-cpu.0-23:latest"
ARTIFACT_DIR = 'gs://aivertex-bucket/custom_model_bucket/'
DATASET_PATH = ARTIFACT_DIR+"LoanApplyData-bank.csv"
BUCKET_NAME = 'aivertex-bucket'
SUB_FOLDER = 'custom_model_bucket/'
PIPELINE_NAME = "custom-pipeline-kfp-local"
MODEL_DISPLAY_NAME = "custom-model-kfp-local"
CUSTOM_JOB_NAME = "custom-runjob-kfp-local"
ENDPOINT_NAME = "custom-endpoint-kfp-local"
TEMPLATE_JSON = "pipeline-kfp-local.json"