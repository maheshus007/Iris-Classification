# DVC Integration Guide

This project now includes Data Version Control (DVC) for managing ML pipelines, data versioning, and experiment tracking.

## DVC Setup

### Prerequisites
- DVC is installed via `requirements.txt`
- Project is initialized with DVC (`dvc init --no-scm`)

### Pipeline Structure

The DVC pipeline consists of three stages:

1. **data_preparation**: Validates the iris.csv dataset
2. **train_model**: Trains ML models using `src/generate_model.py`
3. **evaluate_model**: Evaluates model performance and generates metrics

### Pipeline Files

- `dvc.yaml`: Pipeline definition
- `params.yaml`: Parameters for the pipeline
- `metrics.json`: Generated metrics (accuracy, precision, recall, f1-score)
- `.dvc/config`: DVC configuration
- `data/iris.csv.dvc`: Data tracking file

### Usage

#### Running the Pipeline
```bash
# Run the complete pipeline
dvc repro

# Check pipeline status
dvc status

# View metrics
dvc metrics show
```

#### Using the Python Script
```bash
# Run pipeline
python scripts/dvc_manager.py run

# Check status
python scripts/dvc_manager.py status

# View metrics
python scripts/dvc_manager.py metrics
```

#### API Endpoints

The FastAPI application includes DVC endpoints:

- `POST /dvc/run-pipeline`: Run the DVC pipeline
- `GET /dvc/status`: Get pipeline status
- `GET /dvc/metrics`: Get current metrics

### Remote Storage (Optional)

To set up remote storage for data versioning:

1. **AWS S3**:
   ```bash
   dvc remote add -d s3-remote s3://your-bucket-name/dvc-store
   dvc remote modify s3-remote access_key_id YOUR_ACCESS_KEY
   dvc remote modify s3-remote secret_access_key YOUR_SECRET_KEY
   ```

2. **Google Cloud Storage**:
   ```bash
   dvc remote add -d gcs-remote gs://your-bucket-name/dvc-store
   ```

3. **Push/Pull Data**:
   ```bash
   dvc push  # Upload data to remote
   dvc pull  # Download data from remote
   ```

### Docker Integration

DVC is fully integrated with Docker:

- DVC dependencies are installed in the Docker image
- DVC configuration and data are mounted as volumes
- Environment variables for cloud storage are supported

### Files Created

- `dvc.yaml`: Pipeline definition
- `params.yaml`: Parameter configuration
- `scripts/dvc_manager.py`: Python management script
- `scripts/evaluate_model.py`: Model evaluation script
- `scripts/run_dvc_pipeline.sh`: Shell script for pipeline execution
- `.dvcignore`: Files to ignore in DVC tracking
- Updated `requirements.txt`: Added DVC dependency
- Updated `Dockerfile`: Added DVC initialization
- Updated `docker-compose.yml`: Added DVC volumes and environment variables
- Updated `src/api.py`: Added DVC API endpoints

### Benefits

1. **Pipeline Automation**: Automated ML workflow execution
2. **Data Versioning**: Track changes to datasets
3. **Experiment Tracking**: Compare different model versions
4. **Reproducibility**: Ensure consistent results across environments
5. **Collaboration**: Share data and models efficiently
6. **Integration**: Works with MLflow for comprehensive ML lifecycle management
