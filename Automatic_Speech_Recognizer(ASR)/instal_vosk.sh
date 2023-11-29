#!/bin/bash

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "Python is not installed."
    exit 1
fi

# Get the Python version
python_version=$(python3 --version 2>&1)

# Extract the Python version number
python_version_number=$(echo "$python_version" | awk '{print $2}')

# Check if the Python version is above 3.6
if [ "$(python3 -c "from distutils.version import LooseVersion; print(LooseVersion('$python_version_number') >= LooseVersion('3.6'))")" = "True" ]; then
    echo "Python is installed and version is above 3.6."
else
    echo "Python is installed but version is below 3.6."
fi


echo "Install Vosk"
pip install vosk

echo "Download English standard model"

# Specify the folder path you want to check
models_path="./vosk_models"

# Check if the folder already exists
if [ ! -d "$models_path" ]; then
    # If it doesn't exist, create the folder
    mkdir -p "$models_path"
fi

cd $models_path

# Define models to download
models=("vosk-model-small-en-us-0.15" "vosk-model-en-us-0.22-lgraph")
model_url_template="https://alphacephei.com/vosk/models/"

# Loop through the array and perform actions for each string
for model in "${models[@]}"; do
    echo $model
    # Check if the folder already exists
    if [ ! -d "$model" ]; then
        # If it doesn't exist, create the folder
        wget "$model_url_template$model.zip"
        unzip "$model.zip"
        rm "$model.zip"
    fi
done