#!/bin/bash

sudo docker build -t dfaisalmahmood/dotsocr-runpod-serverless:torch-2.3 --build-arg MODEL_NAME="rednote-hilab/dots.ocr" --build-arg BASE_PATH="/models" .