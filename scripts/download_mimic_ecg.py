#!/usr/bin/env python3
"""
Download a subset of MIMIC-IV-ECG records via PhysioNet S3.

Setup (one-time):
  1. Go to https://physionet.org/settings/cloud/ and generate AWS credentials.
  2. Add them to ~/.aws/credentials:
       [physionet]
       aws_access_key_id = <key>
       aws_secret_access_key = <secret>
  3. Run this script — it uses the [physionet] AWS profile by default.

Usage:
    uv run python scripts/download_mimic_ecg.py --n 50
    uv run python scripts/download_mimic_ecg.py --n 50 --out dataset/mimic-iv-ecg --profile physionet

Output layout mirrors S3 prefix structure under --out (default: dataset/mimic-iv-ecg/).
"""

import argparse
import concurrent.futures
import sys
from pathlib import Path

import boto3
from botocore.config import Config
from botocore.exceptions import ClientError, NoCredentialsError, ProfileNotFound

S3_BUCKET = "physionet-open"
S3_PREFIX = "mimic-iv-ecg/1.0"
METADATA_KEYS = [
    f"{S3_PREFIX}/record_list.csv",
    f"{S3_PREFIX}/machine_measurements.csv",
]


def make_s3_client(profile: str):
    session = boto3.Session(profile_name=profile)
    return session.client(
        "s3",
        config=Config(max_pool_connections=20),
    )


def s3_download(s3, key: str, dest: Path) -> bool:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        return True
    try:
        s3.download_file(S3_BUCKET, key, str(dest), ExtraArgs={"RequestPayer": "requester"})
        return True
    except Exception as e:
        print(f"  FAIL  {key} — {e}")
        return False


def load_record_list(csv_path: Path) -> list[str]:
    """Parse record_list.csv → list of S3 keys (e.g. mimic-iv-ecg/1.0/files/p10/.../s10000032)."""
    records = []
    with open(csv_path) as f:
        for i, line in enumerate(f):
            line = line.strip()
            if i == 0 or not line:
                continue
            rel_path = line.split(",")[0]  # e.g. files/p10/p10000032/s10000032
            records.append(f"{S3_PREFIX}/{rel_path}")
    return records


def download_record(s3, record_key: str, out_dir: Path) -> tuple[str, bool]:
    """Download .hea and .dat for one record."""
    ok = True
    for ext in (".hea", ".dat"):
        key = f"{record_key}{ext}"
        dest = out_dir / key.removeprefix(f"{S3_PREFIX}/")
        if not s3_download(s3, key, dest):
            ok = False
    return record_key, ok


def main():
    parser = argparse.ArgumentParser(description="Download MIMIC-IV-ECG records via PhysioNet S3")
    parser.add_argument("--n", type=int, default=50, help="Number of records to download (default: 50)")
    parser.add_argument("--out", type=str, default="dataset/mimic-iv-ecg", help="Output directory")
    parser.add_argument("--workers", type=int, default=10, help="Parallel S3 download workers (default: 10)")
    parser.add_argument("--profile", type=str, default="physionet", help="AWS profile name (default: physionet)")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- Step 1: Connect to S3 using AWS profile ---
    print(f"Connecting to S3 with AWS profile '{args.profile}'...")
    try:
        s3 = make_s3_client(args.profile)
        # Probe access with a cheap head request
        s3.head_object(Bucket=S3_BUCKET, Key=f"{S3_PREFIX}/record_list.csv", RequestPayer="requester")
        print("  ok  — S3 access confirmed\n")
    except Exception as e:
        print(f"ERROR: Could not connect — {e}", file=sys.stderr)
        print(
            "\nSetup: go to https://physionet.org/settings/cloud/ to get AWS credentials,\n"
            "then add them to ~/.aws/credentials under [physionet].",
            file=sys.stderr,
        )
        sys.exit(1)

    # --- Step 2: Download metadata files ---
    print("Downloading metadata files...")
    for key in METADATA_KEYS:
        dest = out_dir / Path(key).name
        ok = s3_download(s3, key, dest)
        status = "ok" if ok else "FAIL"
        print(f"  {status}  {Path(key).name}")

    # --- Step 3: Parse record list and select first N ---
    record_list_path = out_dir / "record_list.csv"
    all_records = load_record_list(record_list_path)
    records_to_fetch = all_records[: args.n]
    print(f"\nDownloading {len(records_to_fetch)} records ({args.workers} parallel workers)...\n")

    # --- Step 4: Parallel S3 download ---
    succeeded, failed = 0, 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(download_record, s3, rec, out_dir): rec
            for rec in records_to_fetch
        }
        for i, future in enumerate(concurrent.futures.as_completed(futures), 1):
            rec_key, ok = future.result()
            record_name = rec_key.split("/")[-1]
            status = "ok" if ok else "FAIL"
            print(f"  [{i:>3}/{len(records_to_fetch)}] {status}  {record_name}")
            if ok:
                succeeded += 1
            else:
                failed += 1

    print(f"\nDone. {succeeded} records downloaded, {failed} failed.")
    print(f"Output: {out_dir.resolve()}")

    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
