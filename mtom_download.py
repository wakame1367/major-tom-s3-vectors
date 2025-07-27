# mtom_download_geojson.py
#!/usr/bin/env python
import argparse, json, sys
from typing import List
from datasets import load_dataset
from shapely.geometry import shape, Point
from shapely.ops import unary_union
from shapely.prepared import prep
from tqdm import tqdm

def load_aoi(geojson_path: str):
    gj = json.load(open(geojson_path, "r", encoding="utf-8"))
    # FeatureCollection / Feature / Geometry どれでも受け付け
    if gj.get("type") == "FeatureCollection":
        geoms = [shape(feat["geometry"]) for feat in gj["features"]]
    elif gj.get("type") == "Feature":
        geoms = [shape(gj["geometry"])]
    else:
        geoms = [shape(gj)]
    union = unary_union(geoms)
    return prep(union)  # 高速化

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--geojson", required=True, help="AOI GeoJSON path")
    ap.add_argument("--out", required=True, help="output JSONL")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    aoi = load_aoi(args.geojson)
    ds = load_dataset("Major-TOM/Core-S2L1C-SSL4EO", streaming=True)  # HF からストリーミング
    it = ds["train"] if "train" in ds else ds

    n = 0
    dim = None
    with open(args.out, "w", encoding="utf-8") as f:
        for row in tqdm(it, desc="streaming"):
            lat, lon = float(row["centre_lat"]), float(row["centre_lon"])
            if not aoi.covers(Point(lon, lat)):  # 辺上も含めるなら covers が便利
                continue

            emb = row["embedding"]
            if dim is None:
                dim = len(emb)
                print(f"[info] detected embedding dim={dim}", file=sys.stderr)

            rec = {
                "id": row["unique_id"],
                "embedding": emb,  # 後段で float32 にキャスト
                "metadata": {
                    "grid_cell": row.get("grid_cell"),
                    "product_id": row.get("product_id"),
                    "timestamp": row.get("timestamp"),
                    "centre_lat": lat,
                    "centre_lon": lon
                }
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n += 1
            if args.limit and n >= args.limit:
                break
    print(f"[done] wrote {n} rows to {args.out} (dim={dim})")

if __name__ == "__main__":
    main()
