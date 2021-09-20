FROM python:3.9-slim

ENV PYTHONUNBUFFERED True

COPY . ./

RUN pip install -r requirements.txt

WORKDIR ./pipeline

ENTRYPOINT ["python","pipeline_kfp_components.py"]