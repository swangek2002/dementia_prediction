"""
inference_v5_full_ctd_clean.py
==============================
Run V5 inference saving FULL CIF curves, then compute Antolini's C_td on:
  (a) full test set (sanity check — should reproduce 0.7810 from eval pipeline)
  (b) cleaned test set (16 prevalent-leaky patients removed)

Outputs:
  /Data0/swangek_data/991/CPRD/data/test_cif_v5_full.npz       (full curves per patient)
  /Data0/swangek_data/991/CPRD/v5_ctd_clean.log                (C_td comparison)
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
    setup_dual_finetune_experiment, DualFineTuneExperiment,
)
from SurvivEHR.examples.modelling.SurvivEHR.dual_data_module import (
    build_hes_sequence_cache, HESTokenizer, DualCollateWrapper, load_yob_lookup,
)
import pickle

logging.basicConfig(level=logging.WARNING)

CKPT_PATH = "/Data0/swangek_data/991/CPRD/output/checkpoints/crPreTrain_small_1337_FineTune_Dementia_CR_dual_v5.ckpt"
GP_DS_PATH = "/Data0/swangek_data/991/CPRD/data/FoundationalModel/FineTune_Dementia_CR_hes_static_v5/"
GP_DB_PATH = "/Data0/swangek_data/991/CPRD/data/example_exercise_database.db"
GP_META_PATH = "/Data0/swangek_data/991/CPRD/data/FoundationalModel/PreTrain/meta_information_custom.pickle"
HES_DB_PATH = "/Data0/swangek_data/991/CPRD/data/hes_pretrain_database.db"
HES_META_PATH = "/Data0/swangek_data/991/CPRD/data/FoundationalModel/PreTrain_HES/meta_information.pickle"
HES_BLOCK_SIZE = 256
LEAKY_TXT = "/Data0/swangek_data/991/CPRD/data/leaky_patients_test.txt"
OUTPUT_NPZ = "/Data0/swangek_data/991/CPRD/data/test_cif_v5_full.npz"
SUPERVISED_TIME_SCALE = 5.0
INDEX_AGE = 72.0
BATCH_SIZE = 16
NUM_WORKERS = 12

DEMENTIA_READ_CODES_SET = set([
    "F110.", "Eu00.", "Eu01.", "Eu02z", "Eu002", "E00..",
    "Eu023", "Eu00z", "Eu025", "Eu01z", "E001.", "F1100",
    "Eu001", "E004.", "Eu000", "Eu02.", "Eu013", "E000.",
    "Eu01y", "E001z", "F1101", "Eu020", "E004z", "E0021",
    "Eu02y", "Eu012", "Eu011", "E00z.", "E0040", "E003.",
    "E0020",
])


def build_patient_index(ds):
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
                "event_time_from_index_years": last_age - INDEX_AGE,
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

    print("Building patient index from preloaded test data...")
    patient_index = build_patient_index(dm.test_set)
    print(f"  Indexed {len(patient_index)} patients")

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

    print(f"Loading dual model from {CKPT_PATH}")
    experiment = DualFineTuneExperiment.load_from_checkpoint(CKPT_PATH)
    experiment.eval()
    experiment.to(device)

    t_eval = experiment.surv_layer.t_eval  # np array, shape (1000,) — model time [0,1]
    print(f"t_eval: shape={t_eval.shape}, range=[{t_eval[0]:.4f}, {t_eval[-1]:.4f}]")

    test_loader = DataLoader(
        dataset=dm.test_set,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        collate_fn=dm.collate_fn,
        shuffle=False,
    )

    print("Running inference on test set (saving full curves)...")
    all_dem_cdf = []
    all_dth_cdf = []
    all_pids = []
    all_labels = []
    all_event_t_scaled = []
    global_idx = 0
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Inference"):
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

    all_dem_cdf = np.array(all_dem_cdf)  # (N, 1000)
    all_dth_cdf = np.array(all_dth_cdf)
    all_pids = np.array(all_pids)
    all_labels = np.array(all_labels)
    all_event_t_scaled = np.array(all_event_t_scaled)

    np.savez(OUTPUT_NPZ,
             patient_ids=all_pids,
             labels=all_labels,
             event_time_scaled=all_event_t_scaled,
             cif_dementia=all_dem_cdf,
             cif_death=all_dth_cdf,
             t_eval=t_eval)
    print(f"\nSaved full curves to {OUTPUT_NPZ}")

    # ---- C_td computation ----
    from pycox.evaluation import EvalSurv

    def compute_ctd(idx_mask, tag):
        sub_dem = all_dem_cdf[idx_mask]   # (n, 1000)
        sub_dth = all_dth_cdf[idx_mask]
        sub_lbl = all_labels[idx_mask]
        sub_t   = all_event_t_scaled[idx_mask]

        # Per-cause C_td using survival curves S_k(t) = 1 - F_k(t)
        # Pycox EvalSurv expects survival DataFrame (time x patient)
        # Durations in same time units as t_eval (model time)
        n = len(sub_dem)
        print(f"\n[{tag}] n={n}, label dist: {pd.Series(sub_lbl).value_counts().to_dict()}")

        # Dementia C_td: event = 1 if dementia, else 0
        evt_dem = (sub_lbl == "dementia").astype(int)
        S_dem = pd.DataFrame(1 - sub_dem.T, index=t_eval)  # (1000, n)
        ev_dem = EvalSurv(S_dem, sub_t.astype(float), evt_dem, censor_surv="km")
        ctd_dem = ev_dem.concordance_td('antolini')

        # Death C_td
        evt_dth = (sub_lbl == "death").astype(int)
        S_dth = pd.DataFrame(1 - sub_dth.T, index=t_eval)
        ev_dth = EvalSurv(S_dth, sub_t.astype(float), evt_dth, censor_surv="km")
        ctd_dth = ev_dth.concordance_td('antolini')

        # Overall C_td: combined event indicator + combined survival = S_dem * S_dth
        # (this is one common definition; the eval pipeline uses its own — match best we can)
        evt_any = ((sub_lbl == "dementia") | (sub_lbl == "death")).astype(int)
        S_any = pd.DataFrame(1 - (sub_dem + sub_dth).clip(0, 1).T, index=t_eval)
        ev_any = EvalSurv(S_any, sub_t.astype(float), evt_any, censor_surv="km")
        ctd_any = ev_any.concordance_td('antolini')

        # Brier scores at horizon t=1.0 (model time) = 5y
        ibs_dem = ev_dem.integrated_brier_score(np.linspace(0.01, 0.99, 100))
        ibs_dth = ev_dth.integrated_brier_score(np.linspace(0.01, 0.99, 100))

        print(f"  C_td dementia = {ctd_dem:.4f}")
        print(f"  C_td death    = {ctd_dth:.4f}")
        print(f"  C_td overall  = {ctd_any:.4f}")
        print(f"  IBS dementia  = {ibs_dem:.4f}")
        print(f"  IBS death     = {ibs_dth:.4f}")
        return ctd_dem, ctd_dth, ctd_any, ibs_dem, ibs_dth

    leaky_ids = set(int(x.strip()) for x in open(LEAKY_TXT) if x.strip())
    mask_full = np.ones(len(all_pids), dtype=bool)
    mask_clean = ~np.isin(all_pids, list(leaky_ids))
    print(f"\nLeaky to exclude: {len(leaky_ids)} (found in CSV: {(~mask_clean).sum()})")

    print("\n" + "=" * 60)
    print("V5 metrics — FULL test set (sanity check vs. eval pipeline)")
    print("=" * 60)
    full = compute_ctd(mask_full, "FULL")

    print("\n" + "=" * 60)
    print("V5 metrics — CLEANED test set (16 prevalent-leaky removed)")
    print("=" * 60)
    clean = compute_ctd(mask_clean, "CLEAN")

    print("\n" + "=" * 60)
    print("DELTA (clean - full)")
    print("=" * 60)
    print(f"  C_td dementia : {clean[0] - full[0]:+.4f}")
    print(f"  C_td death    : {clean[1] - full[1]:+.4f}")
    print(f"  C_td overall  : {clean[2] - full[2]:+.4f}")
    print(f"  IBS dementia  : {clean[3] - full[3]:+.4f}")
    print(f"  IBS death     : {clean[4] - full[4]:+.4f}")


if __name__ == "__main__":
    main()
