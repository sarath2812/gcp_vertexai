FROM python:3.9-slim

ENV PYTHONUNBUFFERED True

COPY . ./

RUN pip install -r requirements.txt

ENTRYPOINT ["python","./pipeline/pipeline_kfp.py"]