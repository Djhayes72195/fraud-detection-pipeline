import argparse
from simulation.s3_reader import stream_transactions

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stream transactions to prediction API")
    parser.add_argument("--day", type=int, required=True, help="Which day's data to stream")
    parser.add_argument("--rate", type=float, default=0.1, help="Seconds to wait between requests")
    parser.add_argument("--model_day", type=int, default=5, help="Day of model being served")
    args = parser.parse_args()

    s3_path = f"s3://your-bucket/path/model_day_{args.day}.parquet"

    stream_transactions(
        api_url="http://localhost:8000/predict",
        delay=args.rate
    )
