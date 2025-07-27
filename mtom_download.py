#!/usr/bin/env python
import argparse, json, sys
from typing import Optional, Tuple, Set
from datasets import load_dataset
from tqdm import tqdm

def parse_bbox(bbox_str: Optional[str]) -> Optional[Tuple[float, float, float, float]]:
    if not bbox_str:
        return None
    a = [float(x) for x in bbox_str.split(",")]
    if len(a) != 4:
        raise ValueError("bbox must be 'minLon,minLat,maxLon,maxLat'")
    return (a[0], a[1], a[2], a[3])

def in_bbox(lat: float, lon: float, bbox: Tuple[float, float, float, float]) -> bool:
    minLon, minLat, maxLon, maxLat = bbox
    return (minLat <= lat <= maxLat) and (minLon <= lon <= maxLon)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="output JSONL path")
    ap.add_argument("--bbox", help="minLon,minLat,maxLon,maxLat (WGS84)")
    ap.add_argument("--grid-cells", nargs="*", help="filter by grid_cell IDs (e.g., R111C222)")
    ap.add_argument("--limit", type=int, default=0, help="max rows (0 = no limit)")
    args = ap.parse_args()

    bbox = parse_bbox(args.bbox)
    grid_cells: Set[str] = set(args.grid_cells or [])

    # Stream the HF dataset (no full download)
    ds = load_dataset("Major-TOM/Core-S2L1C-SSL4EO", streaming=True)  # IterableDataset
    it = ds["train"] if "train" in ds else ds

    n_written = 0
    detected_dim = None

    with open(args.out, "w", encoding="utf-8") as f:
        for row in tqdm(it, desc="streaming"):
            if grid_cells and row.get("grid_cell") not in grid_cells:
                continue
            lat, lon = float(row["centre_lat"]), float(row["centre_lon"])
            if bbox and not in_bbox(lat, lon, bbox):
                continue

            emb = row["embedding"]
            if detected_dim is None:
                detected_dim = len(emb)
                print(f"[info] detected embedding dim = {detected_dim}", file=sys.stderr)

            rec = {
                "id": row["unique_id"],
                "embedding": emb,
                "metadata": {
                    "grid_cell": row.get("grid_cell"),
                    "product_id": row.get("product_id"),
                    "timestamp": row.get("timestamp"),
                    "centre_lat": lat,
                    "centre_lon": lon,
                },
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n_written += 1
            if args.limit and n_written >= args.limit:
                break

    print(f"[done] wrote {n_written} vectors to {args.out} (dim={detected_dim})")

if __name__ == "__main__":
    main()
