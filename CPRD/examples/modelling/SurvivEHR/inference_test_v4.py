"""
inference_test_metrics.py
=========================
Run V3 model inference on the TEST set to extract per-patient CIF curves
for both dementia and death risks.

Outputs a CSV with: patient_id, label, event times, CIF values at 1/2/3/5 years.

Usage:
    cd /Data0/swangek_data/991/CPRD/examples/modelling/SurvivEHR
    CUDA_VISIBLE_DEVICES=0 \
    PYTHONPATH=/Data0/swangek_data/991/FastEHR:/Data0/swangek_data/991/CPRD \
    /Data0/swangek_data/conda_envs/survivehr/bin/python inference_test_metrics.py
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
from SurvivEHR.examples.modelling.SurvivEHR.setup_dual_finetune_experiment import (
    setup_dual_finetune_experiment, DualFineTuneExperiment
)
from SurvivEHR.examples.modelling.SurvivEHR.dual_data_module import (
    build_hes_sequence_cache, HESTokenizer, DualCollateWrapper, load_yob_lookup
)

logging.basicConfig(level=logging.WARNING)

# ---- Config ----
CKPT_PATH = "/Data0/swangek_data/991/CPRD/output/checkpoints/crPreTrain_small_1337_FineTune_Dementia_CR_dual_v4.ckpt"
GP_DS_PATH = "/Data0/swangek_data/991/CPRD/data/FoundationalModel/FineTune_Dementia_CR_hes_static_v4/"
GP_DB_PATH = "/Data0/swangek_data/991/CPRD/data/example_exercise_database.db"
GP_META_PATH = "/Data0/swangek_data/991/CPRD/data/FoundationalModel/PreTrain/meta_information_custom.pickle"
HES_DB_PATH = "/Data0/swangek_data/991/CPRD/data/hes_pretrain_database.db"
HES_META_PATH = "/Data0/swangek_data/991/CPRD/data/FoundationalModel/PreTrain_HES/meta_information.pickle"
HES_BLOCK_SIZE = 256
SUPERVISED_TIME_SCALE = 5.0
INDEX_AGE = 72.0
BATCH_SIZE = 16
NUM_WORKERS = 12
OUTPUT_CSV = "/Data0/swangek_data/991/CPRD/data/test_cif_v4.csv"

DEMENTIA_READ_CODES_SET = set([
    "F110.", "Eu00.", "Eu01.", "Eu02z", "Eu002", "E00..",
    "Eu023", "Eu00z", "Eu025", "Eu01z", "E001.", "F1100",
    "Eu001", "E004.", "Eu000", "Eu02.", "Eu013", "E000.",
    "Eu01y", "E001z", "F1101", "Eu020", "E004z", "E0021",
    "Eu02y", "Eu012", "Eu011", "E00z.", "E0040", "E003.",
    "E0020",
])


def build_patient_index(ds):
    """Build mapping from dataset index -> (patient_id, label, event_time_from_index_years).
    Reads the preloaded parquet data that the dataset uses."""

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
            days_since_birth = row["DAYS_SINCE_BIRTH"]

            if len(events) == 0:
                label = "empty"
                event_time_from_index = 0.0
            else:
                last_event = events[-1]
                if last_event == "DEATH":
                    label = "death"
                elif last_event in DEMENTIA_READ_CODES_SET:
                    label = "dementia"
                else:
                    label = "censored"

                # Event time from index date (age 72)
                last_age = days_since_birth[-1] / 365.25
                event_time_from_index = last_age - INDEX_AGE

            index_map[global_idx] = {
                "patient_id": pid,
                "label": label,
                "event_time_from_index_years": event_time_from_index,
            }

    return index_map


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Step 1: Load DataModule
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

    # Step 2: Build patient index from TEST set
    print("Building patient index from test set...")
    patient_index = build_patient_index(dm.test_set)
    print(f"  Indexed {len(patient_index)} patients")

    # Step 3: Build HES components
    print("Building HES components...")
    with open(HES_META_PATH, "rb") as f:
        hes_meta = pickle.load(f)
    hes_tokenizer = HESTokenizer(hes_meta)
    yob_lookup = load_yob_lookup(GP_DB_PATH)
    hes_cache, _ = build_hes_sequence_cache(HES_DB_PATH, HES_META_PATH, yob_lookup)

    original_collate = dm.collate_fn
    dm.collate_fn = DualCollateWrapper(
        original_collate, hes_cache, hes_tokenizer,
        hes_block_size=HES_BLOCK_SIZE, time_scale=SUPERVISED_TIME_SCALE,
    )

    # Step 4: Load model
    print(f"Loading model from {CKPT_PATH}")
    experiment = DualFineTuneExperiment.load_from_checkpoint(CKPT_PATH)
    experiment.eval()
    experiment.to(device)

    t_eval = experiment.surv_layer.t_eval  # np array, shape (1000,)

    # Precompute time indices for 1, 2, 3, 5 years
    year_indices = {}
    for year in [1, 2, 3, 5]:
        t_target = year / SUPERVISED_TIME_SCALE
        t_idx = min(int(np.argmin(np.abs(t_eval - t_target))), len(t_eval) - 1)
        year_indices[year] = t_idx
        print(f"  Year {year} -> t_model={t_target:.2f}, t_idx={t_idx}")

    # Step 5: Create NON-SHUFFLED test loader
    test_loader = DataLoader(
        dataset=dm.test_set,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        collate_fn=dm.collate_fn,
        shuffle=False,
    )

    # Step 6: Run inference
    print("Running inference on test set...")
    results = []
    global_idx = 0

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Inference"):
            batch_device = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                          for k, v in batch.items()}

            bsz = batch_device["tokens"].shape[0]

            # Forward with CDF prediction
            all_outputs, _, _ = experiment(batch_device, return_loss=False, return_generation=True)

            pred_cdfs = all_outputs["surv"]["surv_CDF"]
            dementia_cdf = pred_cdfs[0]  # shape (bsz, 1000) — risk 0 = dementia
            death_cdf = pred_cdfs[1]     # shape (bsz, 1000) — risk 1 = death

            target_age_deltas = batch["target_age_delta"].cpu().numpy()

            for i in range(bsz):
                info = patient_index.get(global_idx + i, {})
                pid = info.get("patient_id", -1)
                label = info.get("label", "unknown")
                event_time_from_index = info.get("event_time_from_index_years", 0.0)

                age_delta_scaled = float(target_age_deltas[i]) if i < len(target_age_deltas) else 0.0

                # CIF at event time
                t_idx_event = int(np.argmin(np.abs(t_eval - age_delta_scaled)))
                cif_dementia_at_event = float(dementia_cdf[i, t_idx_event])
                cif_death_at_event = float(death_cdf[i, t_idx_event])

                row = {
                    "patient_id": pid,
                    "label": label,
                    "event_time_from_index_years": event_time_from_index,
                    "event_time_scaled": age_delta_scaled,
                    "cif_dementia_at_event": cif_dementia_at_event,
                    "cif_death_at_event": cif_death_at_event,
                }

                # CIF at specific years
                for year in [1, 2, 3, 5]:
                    tidx = year_indices[year]
                    row[f"cif_dementia_{year}y"] = float(dementia_cdf[i, tidx])
                    row[f"cif_death_{year}y"] = float(death_cdf[i, tidx])

                results.append(row)

            global_idx += bsz

    # Save
    df = pd.DataFrame(results)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\nSaved {len(df)} patient predictions to {OUTPUT_CSV}")

    # Summary
    print(f"\n{'='*60}")
    print(f"Summary by label:")
    for label in ["dementia", "death", "censored"]:
        sub = df[df["label"] == label]
        if len(sub) == 0:
            continue
        print(f"  {label} (n={len(sub)}):")
        print(f"    event_time_from_index: mean={sub['event_time_from_index_years'].mean():.2f}y, "
              f"median={sub['event_time_from_index_years'].median():.2f}y")
        print(f"    CIF_dementia@5y: mean={sub['cif_dementia_5y'].mean():.4f}, "
              f"median={sub['cif_dementia_5y'].median():.4f}")
        print(f"    CIF_death@5y: mean={sub['cif_death_5y'].mean():.4f}, "
              f"median={sub['cif_death_5y'].median():.4f}")

    # Quick check: 5y stats
    dem_5y = ((df['label'] == 'dementia') & (df['event_time_from_index_years'] <= 5.0)).sum()
    death_5y = ((df['label'] == 'death') & (df['event_time_from_index_years'] <= 5.0)).sum()
    followup_5y = (df['event_time_from_index_years'] > 5.0).sum()
    print(f"\n5-year stats:")
    print(f"  Dementia within 5y: {dem_5y}")
    print(f"  Death within 5y: {death_5y}")
    print(f"  Follow-up >= 5y: {followup_5y}")


if __name__ == "__main__":
    main()
