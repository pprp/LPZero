#!/bin/bash

# HF_ENDPOINT=https://hf-mirror.com \
huggingface-cli download --resume-download --local-dir-use-symlinks True lm1b --local-dir ~/workspace/LPZero/data/lm1b