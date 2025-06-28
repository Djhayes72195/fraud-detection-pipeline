import argparse
from simulation.constants import LOCAL_ENDPOINT, EC2_ENDPOINT
from simulation.s3_reader import stream_transactions

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stream transactions to prediction API")
    parser.add_argument("--day", type=int, required=True, help="Which day's data to stream")
    parser.add_argument("--rate", type=float, default=0.1, help="Seconds to wait between requests")
    parser.add_argument("--model_day", type=int, default=5, help="Day of model being served")
    parser.add_argument("--destination", type=str, default="local", help="Endpoint location (ec2/local)")
    args = parser.parse_args()

    api_url = EC2_ENDPOINT if args.destination == "ec2" else LOCAL_ENDPOINT

    stream_transactions(
        args.day,
        api_url=api_url,
        delay=args.rate,
    )
