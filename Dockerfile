# torch image
FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

COPY requirements.txt /app/requirements.txt

# install requirements
RUN pip install -r /app/requirements.txt
