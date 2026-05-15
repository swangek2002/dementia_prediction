"""
inference_test_ablation.py
==========================
Run SINGLE-backbone (GP-only) model inference on the TEST set to extract
per-patient CIF curves at multiple time points.

Used for the V2 ablation model (single GP backbone + 22-dim HES static, no HES backbone).

Outputs CSV with: patient_id, label, event_time_scaled, event_time_from_index_years,
cif_dementia_at_event, cif_death_at_event, cif_{risk}_{t}y for t in [1,2,3,5].

Usage:
    cd /Data0/swangek_data/991/CPRD/examples/modelling/SurvivEHR
    CUDA_VISIBLE_DEVICES=0 \
    PYTHONPATH=/Data0/swangek_data/991/FastEHR:/Data0/swangek_data/991/CPRD \
    /Data0/swangek_data/conda_envs/survivehr/bin/python inference_test_ablation.py
"""

import os
import sys
import torch
import numpy as np
import pandas as pd
import pickle
import logging
from pathlib import Path
from tqdm import tqdm
from torch.utils.data import DataLoader

from FastEHR.dataloader.foundational_loader import FoundationalDataModule
from SurvivEHR.examples.modelling.SurvivEHR.setup_finetune_experiment import (
    setup_finetune_experiment, FineTuneExperiment
)

logging.basicConfig(level=logging.WARNING)

# ---- Config ----
CKPT_PATH = "/Data0/swangek_data/991/CPRD/output/checkpoints/crPreTrain_small_1337_FineTune_Dementia_CR_hes_static_v2_ablation.ckpt"
GP_DS_PATH = "/Data0/swangek_data/991/CPRD/data/FoundationalModel/FineTune_Dementia_CR_hes_static_v2/"
GP_DB_PATH = "/Data0/swangek_data/991/CPRD/data/example_exercise_database.db"
GP_META_PATH = "/Data0/swangek_data/991/CPRD/data/FoundationalModel/PreTrain/meta_information_custom.pickle"
SUPERVISED_TIME_SCALE = 5.0
INDEX_AGE = 72.0
BATCH_SIZE = 32
NUM_WORKERS = 12
OUTPUT_CSV = "/Data0/swangek_data/991/CPRD/data/test_cif_v2_ablation.csv"

DEMENTIA_READ_CODES_SET = set([
    "F110.", "Eu00.", "Eu01.", "Eu02z", "Eu002", "E00..",
    "Eu023", "Eu00z", "Eu025", "Eu01z", "E001.", "F1100",
    "Eu001", "E004.", "Eu000", "Eu02.", "Eu013", "E000.",
    "Eu01y", "E001z", "F1101", "Eu020", "E004z", "E0021",
    "Eu02y", "Eu012", "Eu011", "E00z.", "E0040", "E003.",
    "E0020",
])


def build_patient_index(ds):
    """Map dataset index -> (patient_id, label, last_age_years) using preloaded parquet."""
    import pyarrow.parquet as pq

    index_map = {}
    cumsum = ds._cumsum
    file_keys = ds._file_keys

    for file_idx, file_key in enumerate(tqdm(file_keys, desc="Building patient index")):
        df = ds._preloaded_data[file_key]
        start_idx = cumsum[file_idx - 1] if file_idx > 0 else 0

        for row_idx in range(len(df)):
            global_idx = start_idx + row_idx
            row = df.iloc[row_idx]
            pid = int(row["PATIENT_ID"])
            events = row["EVENT"]
            days = row["DAYS_SINCE_BIRTH"]

            if len(events) == 0:
                label = "empty"
                last_age = 0.0
            else:
                last_event = events[-1]
                last_age = days[-1] / 365.25
                if last_event == "DEATH":
                    label = "death"
                elif last_event in DEMENTIA_READ_CODES_SET:
                    label = "dementia"
                else:
                    label = "censored"

            index_map[global_idx] = {
                "patient_id": pid,
                "label": label,
                "last_age": last_age,
            }

    return index_map


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Step 1: Load DataModule (supervised mode is required for convert_to_supervised)
    print("Loading DataModule...")
    dm = FoundationalDataModule(
        path_to_db=GP_DB_PATH,
        path_to_ds=GP_DS_PATH,
        load=True,
        tokenizer="tabular",
        overwrite_meta_information=GP_META_PATH,
        min_workers=NUM_WORKERS,
        seed=1337,
        batch_size=BATCH_SIZE,
        max_seq_length=512,
        global_diagnoses=True,
        repeating_events=False,
        supervised=True,
        supervised_time_scale=SUPERVISED_TIME_SCALE,
    )

    # Step 2: Build patient index for test set
    print("Building patient index from preloaded test data...")
    patient_index = build_patient_index(dm.test_set)
    print(f"  Indexed {len(patient_index)} patients")

    # Step 3: Load model
    print(f"Loading single-backbone model from {CKPT_PATH}")
    experiment = FineTuneExperiment.load_from_checkpoint(CKPT_PATH)
    experiment.eval()
    experiment.to(device)

    t_eval = experiment.surv_layer.t_eval  # np array, shape (1000,)

    # Indices for 1y / 2y / 3y / 5y in model time (t = years / SUPERVISED_TIME_SCALE)
    year_to_idx = {
        yr: int(np.argmin(np.abs(t_eval - yr / SUPERVISED_TIME_SCALE)))
        for yr in [1, 2, 3, 5]
    }
    print(f"Year → t_eval index mapping: {year_to_idx}")

    # Step 4: Non-shuffled test loader
    test_loader = DataLoader(
        dataset=dm.test_set,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        collate_fn=dm.collate_fn,
        shuffle=False,
    )

    # Step 5: Inference
    print("Running inference on test set...")
    results = []
    global_idx = 0

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Inference"):
            batch_device = {
                k: (v.to(device) if isinstance(v, torch.Tensor) else v)
                for k, v in batch.items()
            }
            bsz = batch_device["tokens"].shape[0]

            all_outputs, _, _ = experiment(
                batch_device, is_generation=True, return_loss=False, return_generation=True
            )

            pred_cdfs = all_outputs["surv"]["surv_CDF"]
            dementia_cdf = pred_cdfs[0]   # (bsz, 1000) — risk 0 = dementia
            death_cdf    = pred_cdfs[1]   # (bsz, 1000) — risk 1 = death

            target_age_deltas = batch["target_age_delta"].cpu().numpy()

            for i in range(bsz):
                info = patient_index.get(global_idx + i, {})
                pid = info.get("patient_id", -1)
                label = info.get("label", "unknown")
                last_age = info.get("last_age", 0.0)

                age_delta_scaled = float(target_age_deltas[i]) if i < len(target_age_deltas) else 0.0

                # CIF at event time (model time = age_delta_scaled)
                t_idx_event = int(np.argmin(np.abs(t_eval - age_delta_scaled)))
                t_idx_event = min(max(t_idx_event, 0), len(t_eval) - 1)
                cif_dem_at_event = float(dementia_cdf[i, t_idx_event])
                cif_death_at_event = float(death_cdf[i, t_idx_event])

                row = {
                    "patient_id": pid,
                    "label": label,
                    "event_time_scaled": age_delta_scaled,
                    "event_time_from_index_years": last_age - INDEX_AGE,
                    "cif_dementia_at_event": cif_dem_at_event,
                    "cif_death_at_event": cif_death_at_event,
                }
                for yr, idx in year_to_idx.items():
                    row[f"cif_dementia_{yr}y"] = float(dementia_cdf[i, idx])
                    row[f"cif_death_{yr}y"]    = float(death_cdf[i, idx])
                results.append(row)

            global_idx += bsz

    # Save
    df = pd.DataFrame(results)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\nSaved {len(df)} patient predictions to {OUTPUT_CSV}")

    # Quick summary
    print(f"\n{'='*60}")
    print(f"Summary by label:")
    for lbl in ["dementia", "death", "censored"]:
        sub = df[df["label"] == lbl]
        if len(sub) == 0:
            continue
        print(f"  {lbl} (n={len(sub)}):")
        print(f"    CIF_dementia@5y: mean={sub['cif_dementia_5y'].mean():.4f}, median={sub['cif_dementia_5y'].median():.4f}")
        print(f"    CIF_death@5y:    mean={sub['cif_death_5y'].mean():.4f}, median={sub['cif_death_5y'].median():.4f}")


if __name__ == "__main__":
    main()
