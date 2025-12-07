#!/usr/bin/env python3
"""
Convert a JSON tuning run file (with top-level `observations` list) into a CSV
matching the project's `all_history_data.csv` column layout.

Behaviour:
- For each observation in `observations`, take `config` -> fill KNOBS columns.
- Take `objectives[0]` as `duration`.
- Set `app_id` = "not-important", `task_id` = "q1", `status` = 1.
- Generate task embeddings by calling `gen_task_embedding('q1')` from `util`.

Usage:
  python scripts/json_to_history_csv.py path/to/input.json path/to/output.csv

Assumes the repository environment is available on PYTHONPATH so imports work.
"""
import json
import os
import sys
import pandas as pd
import numpy as np
import torch
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
from util import gen_task_embedding, embedding_columns


def convert(json_path: str, out_csv: str):
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    observations = data.get('observations', [])
    if not observations:
        print("No observations found in JSON.")
        return

    rows = []
    # cache embedding for task_id 'q1' since user requested same task_id for all rows
    sql_embedding = np.ones(128)
    norm = np.linalg.norm(sql_embedding)
    sql_embedding = list(map(lambda x: x / norm, sql_embedding))
    emb_dict = {f'task_embedding_{i}': sql_embedding[i] for i in range(0, 128)}

    for obs in observations:
        cfg = obs.get('config', {}) or {}
        objectives = obs.get('objectives') or []
        duration = objectives[0] if len(objectives) > 0 else None

        row = {}
        row['app_id'] = 'not-important'
        row['task_id'] = 'all'
        # duration to int when possible
        try:
            row['duration'] = duration if duration is not None else None
        except Exception:
            row['duration'] = duration
        
        # if duration is not Infinity, then set status = 1
        if duration == float('inf'):
            row['status'] = 0
        else:
            row['status'] = 1

        row.update(cfg)

        # fill embeddings (gen_task_embedding returns dict task_embedding_i -> value)
        if emb_dict:
            for k, v in emb_dict.items():
                row[k] = v
        else:
            # if embedding generation failed, fill with None for expected embedding columns
            for k in embedding_columns:
                row[k] = None

        rows.append(row)

    df = pd.DataFrame(rows)

    # Write out
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(f"Wrote {len(df)} rows to {out_csv}")


def main():
    if len(sys.argv) < 3:
        print("Usage: python ./scripts/json_to_history_csv.py input.json output.csv")
        sys.exit(1)
    json_path = sys.argv[1]
    out_csv = sys.argv[2]
    convert(json_path, out_csv)

if __name__ == '__main__':
    main()
