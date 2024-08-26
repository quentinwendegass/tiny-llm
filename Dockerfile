FROM python:3.12-slim-bookworm

WORKDIR /app

COPY src src
COPY configurations/tiny-stories configurations/tiny-stories
COPY requirements.txt .
COPY setup.py .

RUN apt-get update
RUN apt-get install pkg-config libhdf5-dev build-essential -y
RUN pip install -e .

EXPOSE 8000

ENTRYPOINT ["fastapi", "run", "src/api.py"]
