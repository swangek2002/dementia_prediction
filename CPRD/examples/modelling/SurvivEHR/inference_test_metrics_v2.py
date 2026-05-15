"""
inference_test_metrics_v2.py
============================
V3 model TEST set inference with FINE-GRAINED CIF grid + asymptotic pi values,
for time-aligned AUROC computation (Step 3 of PLAN_ADDITIONAL_METRICS.md).

Differences from v1 (inference_test_metrics.py):
- Saves CIF on a ~26-point grid (every 0.2 years) instead of just 4 fixed points
- Saves pi (asymptotic mixing weight) for each risk
- Outputs both CSV (basic info + pi) and NPZ (fine CIF grid)

Usage:
    cd /Data0/swangek_data/991/CPRD/examples/modelling/SurvivEHR
    CUDA_VISIBLE_DEVICES=0 \
    PYTHONPATH=/Data0/swangek_data/991/FastEHR:/Data0/swangek_data/991/CPRD \
    /Data0/swangek_data/conda_envs/survivehr/bin/python inference_test_metrics_v2.py
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

CKPT_PATH = "/Data0/swangek_data/991/CPRD/output/checkpoints/crPreTrain_small_1337_FineTune_Dementia_CR_dual_v3.ckpt"
GP_DS_PATH = "/Data0/swangek_data/991/CPRD/data/FoundationalModel/FineTune_Dementia_CR_hes_static_v3/"
GP_DB_PATH = "/Data0/swangek_data/991/CPRD/data/example_exercise_database.db"
GP_META_PATH = "/Data0/swangek_data/991/CPRD/data/FoundationalModel/PreTrain/meta_information_custom.pickle"
HES_DB_PATH = "/Data0/swangek_data/991/CPRD/data/hes_pretrain_database.db"
HES_META_PATH = "/Data0/swangek_data/991/CPRD/data/FoundationalModel/PreTrain_HES/meta_information.pickle"
HES_BLOCK_SIZE = 256
SUPERVISED_TIME_SCALE = 5.0
INDEX_AGE = 72.0
BATCH_SIZE = 16
NUM_WORKERS = 12
OUTPUT_CSV = "/Data0/swangek_data/991/CPRD/data/test_cif_v3_aligned.csv"
OUTPUT_NPZ = "/Data0/swangek_data/991/CPRD/data/test_cif_v3_fine.npz"

DEMENTIA_READ_CODES_SET = set([
    "F110.", "Eu00.", "Eu01.", "Eu02z", "Eu002", "E00..",
    "Eu023", "Eu00z", "Eu025", "Eu01z", "E001.", "F1100",
    "Eu001", "E004.", "Eu000", "Eu02.", "Eu013", "E000.",
    "Eu01y", "E001z", "F1101", "Eu020", "E004z", "E0021",
    "Eu02y", "Eu012", "Eu011", "E00z.", "E0040", "E003.",
    "E0020",
])


def build_patient_index(ds):
    """Build mapping from dataset index -> (patient_id, label, event_time_from_index_years)."""
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

    print("Building patient index from test set...")
    patient_index = build_patient_index(dm.test_set)
    n_patients = len(patient_index)
    print(f"  Indexed {n_patients} patients")

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

    print(f"Loading model from {CKPT_PATH}")
    experiment = DualFineTuneExperiment.load_from_checkpoint(CKPT_PATH)
    experiment.eval()
    experiment.to(device)

    t_eval = experiment.surv_layer.t_eval  # (1000,)

    # Fine grid: every 40 indices = every 0.2y in real time. Plus the last index 999.
    fine_grid_indices = list(range(0, 1000, 40)) + [999]
    fine_grid_indices = sorted(set(fine_grid_indices))
    fine_grid_indices_np = np.array(fine_grid_indices)
    fine_grid_years = fine_grid_indices_np / 999.0 * 5.0
    print(f"Fine grid: {len(fine_grid_indices)} points at years {fine_grid_years.round(2).tolist()}")

    # Also keep year indices for backward compatibility (1, 2, 3, 5)
    year_indices = {}
    for year in [1, 2, 3, 5]:
        t_target = year / SUPERVISED_TIME_SCALE
        t_idx = min(int(np.argmin(np.abs(t_eval - t_target))), len(t_eval) - 1)
        year_indices[year] = t_idx

    test_loader = DataLoader(
        dataset=dm.test_set,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        collate_fn=dm.collate_fn,
        shuffle=False,
    )

    print("Running inference on test set...")
    results = []
    all_cif_dem = np.zeros((n_patients, len(fine_grid_indices)), dtype=np.float32)
    all_cif_death = np.zeros((n_patients, len(fine_grid_indices)), dtype=np.float32)
    all_pids = np.zeros(n_patients, dtype=np.int64)

    global_idx = 0
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Inference"):
            batch_device = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                          for k, v in batch.items()}
            bsz = batch_device["tokens"].shape[0]

            all_outputs, _, _ = experiment(batch_device, return_loss=False, return_generation=True)

            pred_cdfs = all_outputs["surv"]["surv_CDF"]
            pred_pis = all_outputs["surv"]["surv_pi"]
            dementia_cdf = pred_cdfs[0]   # (bsz, 1000)
            death_cdf = pred_cdfs[1]
            dementia_pi = pred_pis[0]     # (bsz, 1000) — pi is constant across t
            death_pi = pred_pis[1]

            target_age_deltas = batch["target_age_delta"].cpu().numpy()

            for i in range(bsz):
                info = patient_index.get(global_idx + i, {})
                pid = info.get("patient_id", -1)
                label = info.get("label", "unknown")
                event_time_from_index = info.get("event_time_from_index_years", 0.0)

                age_delta_scaled = float(target_age_deltas[i]) if i < len(target_age_deltas) else 0.0
                t_idx_event = int(np.argmin(np.abs(t_eval - age_delta_scaled)))

                all_pids[global_idx + i] = pid
                all_cif_dem[global_idx + i] = dementia_cdf[i, fine_grid_indices_np]
                all_cif_death[global_idx + i] = death_cdf[i, fine_grid_indices_np]

                row = {
                    "patient_id": pid,
                    "label": label,
                    "event_time_from_index_years": event_time_from_index,
                    "event_time_scaled": age_delta_scaled,
                    "cif_dementia_at_event": float(dementia_cdf[i, t_idx_event]),
                    "cif_death_at_event": float(death_cdf[i, t_idx_event]),
                    "pi_dementia": float(dementia_pi[i, -1]),
                    "pi_death": float(death_pi[i, -1]),
                }
                for year in [1, 2, 3, 5]:
                    tidx = year_indices[year]
                    row[f"cif_dementia_{year}y"] = float(dementia_cdf[i, tidx])
                    row[f"cif_death_{year}y"] = float(death_cdf[i, tidx])

                results.append(row)
            global_idx += bsz

    df = pd.DataFrame(results)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\nSaved CSV: {OUTPUT_CSV} ({len(df)} rows)")

    np.savez(OUTPUT_NPZ,
             patient_ids=all_pids,
             cif_dementia=all_cif_dem,
             cif_death=all_cif_death,
             t_grid_years=fine_grid_years)
    print(f"Saved NPZ: {OUTPUT_NPZ} (shape cif_dementia={all_cif_dem.shape})")

    # Quick sanity print
    print(f"\nQuick stats:")
    print(f"  pi_dementia: mean={df.pi_dementia.mean():.4f}, median={df.pi_dementia.median():.4f}")
    print(f"  pi_death:    mean={df.pi_death.mean():.4f}, median={df.pi_death.median():.4f}")
    print(f"\nBy label, pi_dementia (asymptotic dementia probability):")
    for label in ["dementia", "death", "censored"]:
        sub = df[df["label"] == label]
        if len(sub) == 0:
            continue
        print(f"  {label} (n={len(sub)}): mean={sub.pi_dementia.mean():.4f}, median={sub.pi_dementia.median():.4f}")


if __name__ == "__main__":
    main()
