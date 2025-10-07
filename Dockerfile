FROM python:3.10.1-buster
USER root
RUN apt-get update
RUN apt-get install -y vim

WORKDIR app
COPY ./ ./

COPY pyproject.toml pdm.lock README.md ./
RUN apt-get update && apt-get install -y poppler-utils

RUN pip install -U pip setuptools wheel
RUN pip install pdm
RUN pdm install --prod --no-lock --no-editable
RUN pdm build
RUN pdm install


EXPOSE 8501

ENTRYPOINT ["pdm", "run", "src/server.py"]

