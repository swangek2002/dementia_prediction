"""
Generic dual-backbone inference + cohort-level metrics for all post-leakage-fix models.

Models covered:
  - dual_v5         (V5 second-round-plus self-training, top 5%)
  - dual_v4         (V4 second-round self-training)
  - dual_v3         (V3 self-training)
  - dual_v2         (V2 label corrections only, no self-training)
  - dual            (dual v2 clean baseline, no V2 corrections)
  - dual_crossattn  (cross-attention fusion variant)

Each: run inference, save full CIF NPZ, compute cause-specific + true CR C_td on cleaned cohort.

Usage:
    python inference_dual_cohort_ctd.py <model_tag>
where model_tag in: v4, v3, v2, dual_baseline, crossattn
"""
import os
import sys
import torch
import numpy as np
import pandas as pd
import logging
from pathlib import Path
from tqdm import tqdm
from torch.utils.data import DataLoader

from FastEHR.dataloader.foundational_loader import FoundationalDataModule
from SurvivEHR.examples.modelling.SurvivEHR.setup_dual_finetune_experiment import (
    DualFineTuneExperiment,
)
from SurvivEHR.examples.modelling.SurvivEHR.dual_data_module import (
    build_hes_sequence_cache, HESTokenizer, DualCollateWrapper, load_yob_lookup,
)
import pickle

logging.basicConfig(level=logging.WARNING)

# Common paths
GP_DB_PATH = "/Data0/swangek_data/991/CPRD/data/example_exercise_database.db"
GP_META_PATH = "/Data0/swangek_data/991/CPRD/data/FoundationalModel/PreTrain/meta_information_custom.pickle"
HES_DB_PATH = "/Data0/swangek_data/991/CPRD/data/hes_pretrain_database.db"
HES_META_PATH = "/Data0/swangek_data/991/CPRD/data/FoundationalModel/PreTrain_HES/meta_information.pickle"
HES_BLOCK_SIZE = 256
LEAKY_TXT = "/Data0/swangek_data/991/CPRD/data/leaky_patients_test.txt"
SUPERVISED_TIME_SCALE = 5.0
INDEX_AGE = 72.0
BATCH_SIZE = 16
NUM_WORKERS = 8

DEMENTIA_READ_CODES_SET = set([
    "F110.", "Eu00.", "Eu01.", "Eu02z", "Eu002", "E00..",
    "Eu023", "Eu00z", "Eu025", "Eu01z", "E001.", "F1100",
    "Eu001", "E004.", "Eu000", "Eu02.", "Eu013", "E000.",
    "Eu01y", "E001z", "F1101", "Eu020", "E004z", "E0021",
    "Eu02y", "Eu012", "Eu011", "E00z.", "E0040", "E003.",
    "E0020",
])

MODEL_CONFIGS = {
    "v5": {
        "ckpt": "/Data0/swangek_data/991/CPRD/output/checkpoints/crPreTrain_small_1337_FineTune_Dementia_CR_dual_v5.ckpt",
        "ds": "/Data0/swangek_data/991/CPRD/data/FoundationalModel/FineTune_Dementia_CR_hes_static_v5/",
        "npz": "/Data0/swangek_data/991/CPRD/data/test_cif_v5_full.npz",
    },
    "v4": {
        "ckpt": "/Data0/swangek_data/991/CPRD/output/checkpoints/crPreTrain_small_1337_FineTune_Dementia_CR_dual_v4.ckpt",
        "ds": "/Data0/swangek_data/991/CPRD/data/FoundationalModel/FineTune_Dementia_CR_hes_static_v4/",
        "npz": "/Data0/swangek_data/991/CPRD/data/test_cif_v4_full.npz",
    },
    "v3": {
        "ckpt": "/Data0/swangek_data/991/CPRD/output/checkpoints/crPreTrain_small_1337_FineTune_Dementia_CR_dual_v3.ckpt",
        "ds": "/Data0/swangek_data/991/CPRD/data/FoundationalModel/FineTune_Dementia_CR_hes_static_v3/",
        "npz": "/Data0/swangek_data/991/CPRD/data/test_cif_v3_full.npz",
    },
    "v2": {
        "ckpt": "/Data0/swangek_data/991/CPRD/output/checkpoints/crPreTrain_small_1337_FineTune_Dementia_CR_dual_v2.ckpt",
        "ds": "/Data0/swangek_data/991/CPRD/data/FoundationalModel/FineTune_Dementia_CR_hes_static_v2/",
        "npz": "/Data0/swangek_data/991/CPRD/data/test_cif_v2_full.npz",
    },
    "dual_baseline": {
        "ckpt": "/Data0/swangek_data/991/CPRD/output/checkpoints/crPreTrain_small_1337_FineTune_Dementia_CR_dual.ckpt",
        "ds": "/Data0/swangek_data/991/CPRD/data/FoundationalModel/FineTune_Dementia_CR_hes_static/",
        "npz": "/Data0/swangek_data/991/CPRD/data/test_cif_dual_baseline_full.npz",
    },
    "crossattn": {
        "ckpt": "/Data0/swangek_data/991/CPRD/output/checkpoints/crPreTrain_small_1337_FineTune_Dementia_CR_dual_crossattn.ckpt",
        "ds": "/Data0/swangek_data/991/CPRD/data/FoundationalModel/FineTune_Dementia_CR_hes_static/",
        "npz": "/Data0/swangek_data/991/CPRD/data/test_cif_crossattn_full.npz",
    },
}


def build_patient_index(ds):
    index_map = {}
    cumsum = ds._cumsum
    file_keys = ds._file_keys
    for file_idx, file_key in enumerate(tqdm(file_keys, desc="Building patient index", leave=False)):
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
            index_map[global_idx] = {"patient_id": pid, "label": label}
    return index_map


def run_inference(model_tag):
    cfg = MODEL_CONFIGS[model_tag]
    ckpt_path = cfg["ckpt"]
    ds_path = cfg["ds"]
    output_npz = cfg["npz"]

    if os.path.exists(output_npz):
        print(f"[{model_tag}] NPZ already exists at {output_npz}, skipping inference")
        return output_npz

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[{model_tag}] Using device: {device}")

    print(f"[{model_tag}] Loading DataModule from {ds_path}")
    dm = FoundationalDataModule(
        path_to_db=GP_DB_PATH,
        path_to_ds=ds_path,
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

    print(f"[{model_tag}] Building patient index from preloaded test data...")
    patient_index = build_patient_index(dm.test_set)
    print(f"[{model_tag}] Indexed {len(patient_index)} patients")

    print(f"[{model_tag}] Building HES components...")
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

    print(f"[{model_tag}] Loading dual model from {ckpt_path}")
    experiment = DualFineTuneExperiment.load_from_checkpoint(ckpt_path)
    experiment.eval()
    experiment.to(device)

    t_eval = experiment.surv_layer.t_eval

    test_loader = DataLoader(
        dataset=dm.test_set,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        collate_fn=dm.collate_fn,
        shuffle=False,
    )

    print(f"[{model_tag}] Running inference...")
    all_dem_cdf = []
    all_dth_cdf = []
    all_pids = []
    all_labels = []
    all_event_t_scaled = []
    global_idx = 0
    with torch.no_grad():
        for batch in tqdm(test_loader, desc=f"[{model_tag}] Inference"):
            batch_device = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                          for k, v in batch.items()}
            bsz = batch_device["tokens"].shape[0]
            all_outputs, _, _ = experiment(batch_device, return_loss=False, return_generation=True)
            pred_cdfs = all_outputs["surv"]["surv_CDF"]
            d0 = pred_cdfs[0]
            d1 = pred_cdfs[1]
            dem_cdf = d0.cpu().numpy() if isinstance(d0, torch.Tensor) else np.asarray(d0)
            dth_cdf = d1.cpu().numpy() if isinstance(d1, torch.Tensor) else np.asarray(d1)
            target_age_deltas = batch["target_age_delta"].cpu().numpy()

            for i in range(bsz):
                info = patient_index.get(global_idx + i, {})
                all_pids.append(info.get("patient_id", -1))
                all_labels.append(info.get("label", "unknown"))
                all_event_t_scaled.append(float(target_age_deltas[i]) if i < len(target_age_deltas) else 0.0)
                all_dem_cdf.append(dem_cdf[i])
                all_dth_cdf.append(dth_cdf[i])
            global_idx += bsz

    all_dem_cdf = np.array(all_dem_cdf)
    all_dth_cdf = np.array(all_dth_cdf)
    all_pids = np.array(all_pids)
    all_labels = np.array(all_labels)
    all_event_t_scaled = np.array(all_event_t_scaled)

    np.savez(output_npz,
             patient_ids=all_pids, labels=all_labels,
             event_time_scaled=all_event_t_scaled,
             cif_dementia=all_dem_cdf, cif_death=all_dth_cdf,
             t_eval=t_eval)
    print(f"[{model_tag}] Saved to {output_npz} ({os.path.getsize(output_npz)/1024/1024:.1f} MB)")

    # Cleanup model + GPU memory before returning
    del experiment, dm
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return output_npz


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python inference_dual_cohort_ctd.py <model_tag>")
        print(f"Available tags: {list(MODEL_CONFIGS.keys())}")
        sys.exit(1)
    tag = sys.argv[1]
    if tag not in MODEL_CONFIGS:
        print(f"Unknown tag '{tag}'. Available: {list(MODEL_CONFIGS.keys())}")
        sys.exit(1)
    run_inference(tag)
