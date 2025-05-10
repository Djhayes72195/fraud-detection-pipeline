from pathlib import Path

OUTPUT_DIR = Path(__file__).resolve().parents[1] / "raw_data"
S3_BUCKET = "fraud-pipeline-daily-data"
S3_PREFIX = "raw/fraud"