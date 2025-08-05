#!/bin/bash
# DVC Pipeline Runner Script

echo "Starting DVC Pipeline..."

# Ensure we're in the right directory
cd /app

# Pull any remote data if configured
echo "Pulling data from remote (if configured)..."
dvc pull || echo "No remote configured or no data to pull"

# Run the full pipeline
echo "Running DVC pipeline..."
dvc repro

# Show pipeline status
echo "Pipeline status:"
dvc status

# Show metrics
echo "Model metrics:"
dvc metrics show

echo "DVC Pipeline completed!"
