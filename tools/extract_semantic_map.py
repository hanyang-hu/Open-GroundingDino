import argparse
import json
from pathlib import Path
from collections import defaultdict

import numpy as np
from tqdm import tqdm


NAME_CANDIDATE_KEYS = [
    "semanticLabel",
    "semanticName",
    "label",
    "class",
    "className",
    "name",
]


def _normalize_name(v):
    if isinstance(v, bytes):
        v = v.decode("utf-8", errors="ignore")
    return str(v).strip()


def extract_semantic_map(bbox_dir: Path):
    npy_files = sorted(bbox_dir.glob("*.npy"))
    if not npy_files:
        raise RuntimeError(f"No .npy files found in: {bbox_dir}")

    id_to_names = defaultdict(set)
    id_counts = defaultdict(int)
    used_name_key = None

    for npy_path in tqdm(npy_files, desc="Scanning npy files"):
        arr = np.load(npy_path, allow_pickle=False)
        if arr.dtype.names is None:
            continue

        if "semanticId" not in arr.dtype.names:
            continue

        # Detect which name key exists in this file (first available key wins globally).
        if used_name_key is None:
            for k in NAME_CANDIDATE_KEYS:
                if k in arr.dtype.names:
                    used_name_key = k
                    break

        for rec in arr:
            sid = int(rec["semanticId"])
            id_counts[sid] += 1
            if used_name_key is not None and used_name_key in arr.dtype.names:
                name = _normalize_name(rec[used_name_key])
                if name:
                    id_to_names[sid].add(name)

    # Build stable output map.
    out_map = {}
    for sid in sorted(id_counts.keys()):
        names = sorted(id_to_names.get(sid, []))
        if len(names) == 0:
            out_map[str(sid)] = None
        elif len(names) == 1:
            out_map[str(sid)] = names[0]
        else:
            # If one id maps to multiple names in raw data, keep all for inspection.
            out_map[str(sid)] = names

    summary = {
        "bbox_dir": str(bbox_dir),
        "files_scanned": len(npy_files),
        "name_key_used": used_name_key,
        "semantic_id_counts": {str(k): int(v) for k, v in sorted(id_counts.items())},
        "semantic_id_to_name": out_map,
    }
    return summary


def main():
    parser = argparse.ArgumentParser("Extract semanticId -> semantic name from IsaacSim bbox npy files.")
    parser.add_argument(
        "--bbox-dir",
        type=Path,
        default=Path("data/surgical_instrument/bounding_box_2d_tight"),
        help="Directory containing IsaacSim bbox .npy files",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/surgical_instrument/semantic_id_map.json"),
        help="Output JSON path",
    )
    args = parser.parse_args()

    summary = extract_semantic_map(args.bbox_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"Wrote: {args.output}")
    print(f"name_key_used: {summary['name_key_used']}")
    print(f"semantic ids: {list(summary['semantic_id_to_name'].keys())}")


if __name__ == "__main__":
    main()
