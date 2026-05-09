"""
inference_train_cif.py
======================
Run V2 model inference on the TRAIN set to extract per-patient CIF_dementia.
Outputs a CSV: PATIENT_ID, label, event_time_years, cif_dementia_at_event

Usage:
    CUDA_VISIBLE_DEVICES=0 \
    PYTHONPATH=/Data0/swangek_data/991/FastEHR:/Data0/swangek_data/991/CPRD \
    python inference_train_cif.py
"""

import os
import sys
import torch
import numpy as np
import pandas as pd
import pickle
import logging
import bisect
from pathlib import Path
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence

from FastEHR.dataloader.foundational_loader import FoundationalDataModule
from SurvivEHR.examples.modelling.SurvivEHR.setup_dual_finetune_experiment import (
    setup_dual_finetune_experiment, DualFineTuneExperiment
)
from SurvivEHR.examples.modelling.SurvivEHR.dual_data_module import (
    build_hes_sequence_cache, HESTokenizer, DualCollateWrapper, load_yob_lookup
)

logging.basicConfig(level=logging.WARNING)

# ---- Config ----
CKPT_PATH = "/Data0/swangek_data/991/CPRD/output/checkpoints/crPreTrain_small_1337_FineTune_Dementia_CR_dual_v2.ckpt"
GP_DS_PATH = "/Data0/swangek_data/991/CPRD/data/FoundationalModel/FineTune_Dementia_CR_hes_static_v2/"
GP_DB_PATH = "/Data0/swangek_data/991/CPRD/data/example_exercise_database.db"
GP_META_PATH = "/Data0/swangek_data/991/CPRD/data/FoundationalModel/PreTrain/meta_information_custom.pickle"
HES_DB_PATH = "/Data0/swangek_data/991/CPRD/data/hes_pretrain_database.db"
HES_META_PATH = "/Data0/swangek_data/991/CPRD/data/FoundationalModel/PreTrain_HES/meta_information.pickle"
HES_BLOCK_SIZE = 256
SUPERVISED_TIME_SCALE = 5.0
BATCH_SIZE = 16
NUM_WORKERS = 12
OUTPUT_CSV = "/Data0/swangek_data/991/CPRD/data/train_cif_dementia_v2.csv"

DEMENTIA_READ_CODES_SET = set([
    "F110.", "Eu00.", "Eu01.", "Eu02z", "Eu002", "E00..",
    "Eu023", "Eu00z", "Eu025", "Eu01z", "E001.", "F1100",
    "Eu001", "E004.", "Eu000", "Eu02.", "Eu013", "E000.",
    "Eu01y", "E001z", "F1101", "Eu020", "E004z", "E0021",
    "Eu02y", "Eu012", "Eu011", "E00z.", "E0040", "E003.",
    "E0020",
])


class DualCollateWrapperWithPID(DualCollateWrapper):
    """Extends DualCollateWrapper to pass through PATIENT_ID."""

    def __call__(self, batch_items):
        gp_batch = super().__call__(batch_items)
        # Extract patient IDs from raw parquet data via dataset index
        # batch_items are dicts returned by __getitem__, which DON'T have PATIENT_ID.
        # We'll add PIDs in the main loop by a different mechanism.
        return gp_batch


def build_patient_index(ds):
    """Build mapping from dataset index -> (patient_id, label, event_time_years).
    Reads the parquet files that the dataset uses."""
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
            dates = row["DATE"]

            if len(events) == 0:
                label = "empty"
                event_time = 0
            else:
                last_event = events[-1]
                if last_event == "DEATH":
                    label = "death"
                elif last_event in DEMENTIA_READ_CODES_SET:
                    label = "dementia"
                else:
                    label = "censored"

            index_map[global_idx] = {
                "patient_id": pid,
                "label": label,
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

    # Step 2: Build patient index from parquets
    print("Building patient index from preloaded data...")
    patient_index = build_patient_index(dm.train_set)
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

    # Step 5: Create NON-SHUFFLED train loader
    train_loader = DataLoader(
        dataset=dm.train_set,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        collate_fn=dm.collate_fn,
        shuffle=False,  # CRITICAL: no shuffle so we can track by index
    )

    # Step 6: Run inference
    print("Running inference on train set...")
    results = []
    global_idx = 0

    with torch.no_grad():
        for batch in tqdm(train_loader, desc="Inference"):
            # Move to device
            batch_device = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                          for k, v in batch.items()}

            bsz = batch_device["tokens"].shape[0]

            # Forward with CDF prediction
            all_outputs, _, _ = experiment(batch_device, return_loss=False, return_generation=True)

            pred_cdfs = all_outputs["surv"]["surv_CDF"]
            # pred_cdfs[0] = dementia CIF, shape (bsz, 1000)
            dementia_cdf = pred_cdfs[0]

            # target_age_delta was set by convert_to_supervised in collate
            # It's the scaled time-to-event (actual years / SUPERVISED_TIME_SCALE)
            target_age_deltas = batch["target_age_delta"].cpu().numpy()

            for i in range(bsz):
                info = patient_index.get(global_idx + i, {})
                pid = info.get("patient_id", -1)
                label = info.get("label", "unknown")

                age_delta_scaled = float(target_age_deltas[i]) if i < len(target_age_deltas) else 0.0

                # CIF at event time
                t_idx = np.argmin(np.abs(t_eval - age_delta_scaled))
                cif_at_event = float(dementia_cdf[i, t_idx])
                cif_at_max = float(dementia_cdf[i, -1])

                results.append({
                    "patient_id": pid,
                    "label": label,
                    "event_time_scaled": age_delta_scaled,
                    "event_time_years": age_delta_scaled * SUPERVISED_TIME_SCALE,
                    "cif_dementia_at_event": cif_at_event,
                    "cif_dementia_at_max": cif_at_max,
                })

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
        print(f"    CIF at event: mean={sub['cif_dementia_at_event'].mean():.4f}, "
              f"median={sub['cif_dementia_at_event'].median():.4f}, "
              f"p95={sub['cif_dementia_at_event'].quantile(0.95):.4f}")

    # Top 1% of DEATH+censored
    non_dem = df[df["label"].isin(["death", "censored"])]
    threshold_1pct = non_dem["cif_dementia_at_event"].quantile(0.99)
    top1 = non_dem[non_dem["cif_dementia_at_event"] >= threshold_1pct]
    # Also filter censored patients with < 2 years observation
    top1_filtered = top1[~((top1["label"] == "censored") & (top1["event_time_years"] < 2.0))]

    print(f"\n{'='*60}")
    print(f"Top 1% of DEATH+censored (CIF >= {threshold_1pct:.4f}):")
    print(f"  Total: {len(top1)} (after filtering short censored: {len(top1_filtered)})")
    print(f"  DEATH: {len(top1[top1['label']=='death'])}")
    print(f"  Censored: {len(top1[top1['label']=='censored'])}")
    print(f"  Mean CIF: {top1['cif_dementia_at_event'].mean():.4f}")


if __name__ == "__main__":
    main()
