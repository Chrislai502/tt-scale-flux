#!/bin/bash
module load python
conda activate scaling_diffusion

# Load the .env file if it exists
if [ -f .env ]; then
    # Export variables from .env file
    export $(grep -v '^#' .env | xargs)
else
    echo "Warning: .env file not found"
fi

# Verify the token is loaded (optional)
if [ -z "$HF_TOKEN" ]; then
    echo "Error: HF_TOKEN not set. Please create a .env file with HF_TOKEN=your_token"
    exit 1
fi