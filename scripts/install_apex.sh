#!/bin/bash

conda_path="$1"
conda_env_name="$2"
project_path="$PWD"

echo  "Activating conda $conda_env_name environment ... "
cd $conda_path
source \conda.sh
conda activate $conda_env_name
cd $project_path && cd apex

git clone https://github.com/NVIDIA/apex

pip install -v --disable-pip-version-check --no-build-isolation --no-cache-dir ./

