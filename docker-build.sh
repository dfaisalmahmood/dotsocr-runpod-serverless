#!/bin/bash

sudo docker build --platform linux/amd64 -t dotsocr-runpod-serverless:v1.0.0 --build-arg MODEL_NAME="rednote-hilab/dots.ocr" --build-arg BASE_PATH="/models" .

# Tag the image for Docker Hub
# sudo docker tag dotsocr-runpod-serverless:v1.0.0 dfaisalmahmood/dotsocr-runpod-serverless:latest

# Push the image to Docker Hub
# sudo docker push dfaisalmahmood/dotsocr-runpod-serverless:latest

# To run API server for testing, run the following:
# sudo docker run --runtime nvidia -it -p 8000:8000 dotsocr-runpod-serverless:v1.0.0

# To test the API, you can use curl or any HTTP client.
# curl -X POST http://localhost:8000/runsync \
#      -H "Content-Type: application/json" \
#      -d '{"input": {"prompt": "The quick brown fox jumps"}}'
