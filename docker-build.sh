#!/bin/bash

sudo docker build --platform linux/amd64 -t dfaisalmahmood/dotsocr-runpod-serverless:torch-2.3 --build-arg MODEL_NAME="rednote-hilab/dots.ocr" --build-arg BASE_PATH="/models" .

# To test, run the following:
# sudo docker run --runtime nvidia -it -p 8000:8000 dfaisalmahmood/dotsocr-runpod-serverless:torch-2.3