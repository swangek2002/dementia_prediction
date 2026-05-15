"""
Generic single-backbone (FineTuneExperiment) inference for ANY earlier single-backbone model.
Saves full CIF NPZ for downstream cohort C_td computation.

Models covered by this script:
  - hes_aug  (GP + HES dementia label only, V1 baseline, 27-dim static)
  - hes_fusion  (GP sequence fused with HES events, expanded test set ~22K patients)
  - idx68_cv_fold0...4  (5-fold CV, single GP, no HES, idx age 68)
  - idx60 / idx70 / idx74 / idx75  (single GP, different index ages)

Usage:
    python inference_single_generic.py <tag>
where <tag> in: hes_aug, hes_fusion, idx68_cv_fold0..4, idx60, idx70, idx74, idx75
"""
import os
import sys
import torch
import numpy as np
import pandas as pd
import logging
from tqdm import tqdm
from torch.utils.data import DataLoader

from FastEHR.dataloader.foundational_loader import FoundationalDataModule
from SurvivEHR.examples.modelling.SurvivEHR.setup_finetune_experiment import FineTuneExperiment

logging.basicConfig(level=logging.WARNING)

GP_META_PATH = "/Data0/swangek_data/991/CPRD/data/FoundationalModel/PreTrain/meta_information_custom.pickle"
SUPERVISED_TIME_SCALE = 5.0
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

CKPT_DIR = "/Data0/swangek_data/991/CPRD/output/checkpoints"
DS_DIR = "/Data0/swangek_data/991/CPRD/data/FoundationalModel"
NPZ_DIR = "/Data0/swangek_data/991/CPRD/data"
DEFAULT_DB = "/Data0/swangek_data/991/CPRD/data/example_exercise_database.db"
FUSION_DB = "/Data0/swangek_data/991/CPRD/data/example_exercise_database_hes_fusion.db"

MODEL_CONFIGS = {
    "hes_aug": {
        "ckpt": f"{CKPT_DIR}/crPreTrain_small_1337_FineTune_Dementia_CR_hes_aug.ckpt",
        "ds":   f"{DS_DIR}/FineTune_Dementia_CR_hes_aug/",
        "db":   DEFAULT_DB,
        "npz":  f"{NPZ_DIR}/test_cif_hes_aug_full.npz",
    },
    "hes_fusion": {
        "ckpt": f"{CKPT_DIR}/crPreTrain_small_1337_FineTune_Dementia_CR_hes_fusion.ckpt",
        "ds":   f"{DS_DIR}/FineTune_Dementia_CR_hes_fusion/",
        "db":   FUSION_DB,
        "npz":  f"{NPZ_DIR}/test_cif_hes_fusion_full.npz",
    },
    "idx60": {
        "ckpt": f"{CKPT_DIR}/crPreTrain_small_1337_FineTune_Dementia_CR_idx60.ckpt",
        "ds":   f"{DS_DIR}/FineTune_Dementia_CR_idx60/",
        "db":   DEFAULT_DB,
        "npz":  f"{NPZ_DIR}/test_cif_idx60_full.npz",
    },
    "idx70": {
        "ckpt": f"{CKPT_DIR}/crPreTrain_small_1337_FineTune_Dementia_CR_idx70.ckpt",
        "ds":   f"{DS_DIR}/FineTune_Dementia_CR_idx70/",
        "db":   DEFAULT_DB,
        "npz":  f"{NPZ_DIR}/test_cif_idx70_full.npz",
    },
    "idx74": {
        "ckpt": f"{CKPT_DIR}/crPreTrain_small_1337_FineTune_Dementia_CR_idx74.ckpt",
        "ds":   f"{DS_DIR}/FineTune_Dementia_CR_idx74/",
        "db":   DEFAULT_DB,
        "npz":  f"{NPZ_DIR}/test_cif_idx74_full.npz",
    },
    "idx75": {
        "ckpt": f"{CKPT_DIR}/crPreTrain_small_1337_FineTune_Dementia_CR_idx75.ckpt",
        "ds":   f"{DS_DIR}/FineTune_Dementia_CR_idx75/",
        "db":   DEFAULT_DB,
        "npz":  f"{NPZ_DIR}/test_cif_idx75_full.npz",
    },
}
# Add 5 CV folds
for k in range(5):
    MODEL_CONFIGS[f"idx68_cv_fold{k}"] = {
        "ckpt": f"{CKPT_DIR}/crPreTrain_small_1337_FineTune_Dementia_CR_idx68_cv_fold{k}.ckpt",
        "ds":   f"{DS_DIR}/FineTune_Dementia_CR_idx68_cv/fold{k}/",
        "db":   DEFAULT_DB,
        "npz":  f"{NPZ_DIR}/test_cif_idx68_cv_fold{k}_full.npz",
    }


def build_patient_index(ds):
    """Build idx → (PATIENT_ID, label) mapping from preloaded test parquet."""
    index_map = {}
    cumsum = ds._cumsum
    file_keys = ds._file_keys
    for file_idx, file_key in enumerate(tqdm(file_keys, desc="PatientIndex", leave=False)):
        df = ds._preloaded_data[file_key]
        start_idx = cumsum[file_idx - 1] if file_idx > 0 else 0
        for row_idx in range(len(df)):
            global_idx = start_idx + row_idx
            row = df.iloc[row_idx]
            pid = int(row["PATIENT_ID"])
            events = row["EVENT"]
            if len(events) == 0:
                label = "empty"
            else:
                last_event = events[-1]
                if last_event == "DEATH":
                    label = "death"
                elif last_event in DEMENTIA_READ_CODES_SET:
                    label = "dementia"
                else:
                    label = "censored"
            index_map[global_idx] = {"patient_id": pid, "label": label}
    return index_map


def run_inference(tag):
    cfg = MODEL_CONFIGS[tag]
    if os.path.exists(cfg["npz"]):
        print(f"[{tag}] NPZ exists at {cfg['npz']}, skipping")
        return cfg["npz"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[{tag}] device={device}, ckpt={cfg['ckpt']}, ds={cfg['ds']}, db={cfg['db']}")

    dm = FoundationalDataModule(
        path_to_db=cfg["db"],
        path_to_ds=cfg["ds"],
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
    print(f"[{tag}] DataModule loaded")

    patient_index = build_patient_index(dm.test_set)
    print(f"[{tag}] Patient index: {len(patient_index)} patients")

    experiment = FineTuneExperiment.load_from_checkpoint(cfg["ckpt"])
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

    all_dem_cdf, all_dth_cdf = [], []
    all_pids, all_labels, all_event_t = [], [], []
    global_idx = 0
    with torch.no_grad():
        for batch in tqdm(test_loader, desc=f"[{tag}]"):
            batch_device = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            bsz = batch_device["tokens"].shape[0]
            all_outputs, _, _ = experiment(batch_device, return_loss=False, return_generation=True)
            pred_cdfs = all_outputs["surv"]["surv_CDF"]
            dem_cdf = pred_cdfs[0].cpu().numpy() if isinstance(pred_cdfs[0], torch.Tensor) else np.asarray(pred_cdfs[0])
            dth_cdf = pred_cdfs[1].cpu().numpy() if isinstance(pred_cdfs[1], torch.Tensor) else np.asarray(pred_cdfs[1])
            tads = batch["target_age_delta"].cpu().numpy()

            for i in range(bsz):
                info = patient_index.get(global_idx + i, {})
                all_pids.append(info.get("patient_id", -1))
                all_labels.append(info.get("label", "unknown"))
                all_event_t.append(float(tads[i]))
                all_dem_cdf.append(dem_cdf[i])
                all_dth_cdf.append(dth_cdf[i])
            global_idx += bsz

    np.savez(cfg["npz"],
        patient_ids=np.array(all_pids),
        labels=np.array(all_labels),
        event_time_scaled=np.array(all_event_t),
        cif_dementia=np.array(all_dem_cdf),
        cif_death=np.array(all_dth_cdf),
        t_eval=t_eval,
    )
    print(f"[{tag}] Saved {cfg['npz']} ({os.path.getsize(cfg['npz'])/1024/1024:.1f} MB)")

    del experiment, dm
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return cfg["npz"]


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(f"Available tags: {list(MODEL_CONFIGS.keys())}")
        sys.exit(1)
    tag = sys.argv[1]
    if tag not in MODEL_CONFIGS:
        print(f"Unknown tag '{tag}'")
        sys.exit(1)
    run_inference(tag)
