#!/usr/bin/env python
import argparse, json, os, math
from typing import List, Dict
import boto3
import numpy as np

# S3 Vectors limits: PutVectors <= 500 vectors/req, payload <= 20 MiB
# and dimension 1..4096, TopK <= 30, etc.
# See: AWS "Limitations and restrictions"
MAX_PER_PUT = 500

def flush(client, bucket: str, index: str, batch: List[Dict]):
    if not batch:
        return
    client.put_vectors(
        vectorBucketName=bucket,
        indexName=index,
        vectors=batch
    )
    batch.clear()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--region", default=os.getenv("AWS_REGION", "us-east-1"))
    ap.add_argument("--bucket", required=True, help="S3 Vector bucket name")
    ap.add_argument("--index", required=True, help="S3 Vector index name")
    ap.add_argument("--jsonl", required=True, help="input JSONL from script A")
    ap.add_argument("--batch", type=int, default=400, help="<=500 recommended")
    ap.add_argument("--expect-dim", type=int, default=0, help="optional dimension check")
    args = ap.parse_args()

    s3v = boto3.client("s3vectors", region_name=args.region)

    batch = []
    with open(args.jsonl, "r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            emb = rec["embedding"]
            if args.expect-dim and len(emb) != args.expect-dim:
                raise ValueError(f"dimension mismatch: got {len(emb)}, expect {args.expect-dim}")

            # IMPORTANT: cast to float32 as required by S3 Vectors
            v = np.asarray(emb, dtype=np.float32).tolist()

            batch.append({
                "key": rec["id"],
                "data": {"float32": v},
                "metadata": rec.get("metadata", {})
            })

            if len(batch) >= min(args.batch, MAX_PER_PUT):
                flush(s3v, args.bucket, args.index, batch)

    flush(s3v, args.bucket, args.index, batch)
    print("[done] ingest completed")

if __name__ == "__main__":
    main()
