"""
V2 Ablation (Single GP backbone, no HES backbone, no fusion) — inference + save CIF NPZ.
This is the FineTuneExperiment class (not Dual).
"""
import os
import torch
import numpy as np
import pandas as pd
import logging
from tqdm import tqdm
from torch.utils.data import DataLoader

from FastEHR.dataloader.foundational_loader import FoundationalDataModule
from SurvivEHR.examples.modelling.SurvivEHR.setup_finetune_experiment import FineTuneExperiment
import pickle

logging.basicConfig(level=logging.WARNING)

CKPT_PATH = "/Data0/swangek_data/991/CPRD/output/checkpoints/crPreTrain_small_1337_FineTune_Dementia_CR_hes_static_v2_ablation.ckpt"
GP_DS_PATH = "/Data0/swangek_data/991/CPRD/data/FoundationalModel/FineTune_Dementia_CR_hes_static_v2/"
GP_DB_PATH = "/Data0/swangek_data/991/CPRD/data/example_exercise_database.db"
GP_META_PATH = "/Data0/swangek_data/991/CPRD/data/FoundationalModel/PreTrain/meta_information_custom.pickle"
OUTPUT_NPZ = "/Data0/swangek_data/991/CPRD/data/test_cif_v2_ablation_full.npz"
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


def build_patient_index(ds):
    index_map = {}
    cumsum = ds._cumsum
    file_keys = ds._file_keys
    for file_idx, file_key in enumerate(tqdm(file_keys, desc="Index", leave=False)):
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


def main():
    if os.path.exists(OUTPUT_NPZ):
        print(f"NPZ exists at {OUTPUT_NPZ}, skipping inference")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print(f"Loading DataModule from {GP_DS_PATH}")
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

    print("Building patient index...")
    patient_index = build_patient_index(dm.test_set)
    print(f"  Indexed {len(patient_index)} patients")

    print(f"Loading single-backbone model from {CKPT_PATH}")
    experiment = FineTuneExperiment.load_from_checkpoint(CKPT_PATH)
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

    print("Inference...")
    all_dem_cdf = []
    all_dth_cdf = []
    all_pids = []
    all_labels = []
    all_event_t_scaled = []
    global_idx = 0
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="V2_ablation"):
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
                all_event_t_scaled.append(float(target_age_deltas[i]))
                all_dem_cdf.append(dem_cdf[i])
                all_dth_cdf.append(dth_cdf[i])
            global_idx += bsz

    all_dem_cdf = np.array(all_dem_cdf)
    all_dth_cdf = np.array(all_dth_cdf)
    all_pids = np.array(all_pids)
    all_labels = np.array(all_labels)
    all_event_t_scaled = np.array(all_event_t_scaled)

    np.savez(OUTPUT_NPZ,
             patient_ids=all_pids, labels=all_labels,
             event_time_scaled=all_event_t_scaled,
             cif_dementia=all_dem_cdf, cif_death=all_dth_cdf,
             t_eval=t_eval)
    print(f"Saved to {OUTPUT_NPZ}")


if __name__ == "__main__":
    main()
