"""
dual_data_module.py
===================
DataModule wrapper that yields batches containing both GP and HES inputs.

Implementation:
  - Uses GP FoundationalDataModule as the primary DataModule
  - Maintains an in-memory HES sequence cache (patient_id -> tokens, ages)
  - In collate_fn, appends HES sequences for each patient in the batch
"""

import os
import pickle
import sqlite3
import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
import logging
from datetime import datetime


def build_hes_sequence_cache(hes_db_path, hes_meta_path, yob_lookup=None):
    """
    Build an in-memory cache of HES sequences for all patients.

    Returns:
        hes_cache: dict {patient_id: {"tokens": list[str], "ages": list[float]}}
        hes_tokenizer: the HES tokenizer from meta_information
    """
    logging.info(f"Building HES sequence cache from {hes_db_path}")

    # Load HES meta information (contains tokenizer)
    with open(hes_meta_path, "rb") as f:
        hes_meta = pickle.load(f)

    # Get the tokenizer from the HES meta info
    # The meta_information stores event frequency table; we need the tokenizer
    # We'll build a simple tokenizer mapping from the meta info
    hes_event_table = hes_meta.get("event_table", hes_meta.get("diagnosis_table", None))
    if hes_event_table is None:
        # Try to find the event frequency info
        for key in hes_meta:
            if hasattr(hes_meta[key], 'columns') and 'event' in [c.lower() for c in hes_meta[key].columns]:
                hes_event_table = hes_meta[key]
                break

    # Read all HES events from database
    conn = sqlite3.connect(hes_db_path)
    cursor = conn.cursor()
    cursor.execute("""
        SELECT PATIENT_ID, EVENT, DATE
        FROM diagnosis_table
        ORDER BY PATIENT_ID, DATE
    """)

    hes_cache = {}
    current_pid = None
    current_tokens = []
    current_dates = []

    for row in cursor:
        pid, event, date_str = int(row[0]), row[1], row[2]
        if pid != current_pid:
            if current_pid is not None and len(current_tokens) > 0:
                hes_cache[current_pid] = {
                    "tokens": current_tokens,
                    "dates": current_dates,
                }
            current_pid = pid
            current_tokens = []
            current_dates = []
        current_tokens.append(event)
        current_dates.append(date_str)

    # Don't forget the last patient
    if current_pid is not None and len(current_tokens) > 0:
        hes_cache[current_pid] = {
            "tokens": current_tokens,
            "dates": current_dates,
        }

    conn.close()

    # Convert dates to ages (years since birth) if yob_lookup provided
    if yob_lookup is not None:
        for pid, data in hes_cache.items():
            yob = yob_lookup.get(pid)
            if yob is None:
                # Use dates as relative values
                ages = []
                for d in data["dates"]:
                    try:
                        dt = datetime.strptime(str(d)[:10], "%Y-%m-%d")
                        ages.append(dt.year + dt.month / 12.0)
                    except:
                        ages.append(0.0)
                data["ages"] = ages
            else:
                ages = []
                if hasattr(yob, 'year'):
                    yob_year = yob.year
                else:
                    try:
                        yob_year = int(str(yob)[:4])
                    except:
                        yob_year = 1960
                for d in data["dates"]:
                    try:
                        dt = datetime.strptime(str(d)[:10], "%Y-%m-%d")
                        age = (dt - datetime(yob_year, 1, 1)).days / 365.25
                        ages.append(age)
                    except:
                        ages.append(0.0)
                data["ages"] = ages
    else:
        # Convert dates to approximate years
        for pid, data in hes_cache.items():
            ages = []
            for d in data["dates"]:
                try:
                    dt = datetime.strptime(str(d)[:10], "%Y-%m-%d")
                    ages.append(dt.year + dt.month / 12.0)
                except:
                    ages.append(0.0)
            data["ages"] = ages

    logging.info(f"HES cache built: {len(hes_cache)} patients with HES records")

    return hes_cache, hes_meta


class HESTokenizer:
    """Simple tokenizer for HES ICD-10 codes based on meta_information vocabulary."""

    def __init__(self, hes_meta):
        # Build vocabulary from meta info
        self._token_to_id = {"<PAD>": 0, "<UNK>": 1}
        idx = 2

        # Try to extract event list from meta_information
        for key in ["event_table", "diagnosis_table"]:
            if key in hes_meta and hasattr(hes_meta[key], 'iterrows'):
                for _, row in hes_meta[key].iterrows():
                    event = row.get("event", row.get("EVENT", None))
                    if event is not None and event not in self._token_to_id:
                        self._token_to_id[str(event)] = idx
                        idx += 1
                break

        # Also try polars-style tables
        if idx == 2:
            for key in hes_meta:
                tbl = hes_meta[key]
                if hasattr(tbl, 'columns') and 'event' in [c.lower() for c in tbl.columns]:
                    col_name = 'event' if 'event' in tbl.columns else 'EVENT'
                    if hasattr(tbl, 'to_list'):
                        events = tbl[col_name].to_list()
                    elif hasattr(tbl, 'tolist'):
                        events = tbl[col_name].tolist()
                    else:
                        events = list(tbl[col_name])
                    for event in events:
                        if str(event) not in self._token_to_id:
                            self._token_to_id[str(event)] = idx
                            idx += 1
                    break

        self._id_to_token = {v: k for k, v in self._token_to_id.items()}
        self.vocab_size = len(self._token_to_id)
        logging.info(f"HES tokenizer built with vocab_size={self.vocab_size}")

    def encode(self, tokens):
        """Encode a list of token strings to integer IDs."""
        return [self._token_to_id.get(str(t), 1) for t in tokens]  # 1 = <UNK>

    def decode(self, ids):
        """Decode integer IDs back to token strings."""
        return [self._id_to_token.get(i, "<UNK>") for i in ids]


class DualCollateWrapper:
    """
    Wraps the existing GP collate_fn to add HES inputs to each batch.

    Usage:
        original_collate = dm.collate_fn
        dm.collate_fn = DualCollateWrapper(original_collate, hes_cache, hes_tokenizer, hes_block_size=256)
    """

    def __init__(self, original_collate, hes_cache, hes_tokenizer, hes_block_size=256,
                 time_scale=1.0):
        self.original_collate = original_collate
        self.hes_cache = hes_cache
        self.hes_tokenizer = hes_tokenizer
        self.hes_block_size = hes_block_size
        self.time_scale = time_scale
        # Copy attributes from original collate for compatibility
        if hasattr(original_collate, 'supervised'):
            self.supervised = original_collate.supervised
        if hasattr(original_collate, 'supervised_time_scale'):
            self.supervised_time_scale = original_collate.supervised_time_scale

    def __call__(self, batch_items):
        # 1. Process GP data with original collate
        gp_batch = self.original_collate(batch_items)

        bsz = gp_batch['tokens'].shape[0]
        hes_tokens_list = []
        hes_ages_list = []

        # 2. Build HES inputs for each patient
        for item in batch_items:
            pid = int(item.get("PATIENT_ID", -1))
            hes_data = self.hes_cache.get(pid, None)

            if hes_data is not None and len(hes_data["tokens"]) > 0:
                raw_tokens = hes_data["tokens"]
                raw_ages = hes_data["ages"]

                # Truncate to hes_block_size (keep most recent)
                if len(raw_tokens) > self.hes_block_size:
                    raw_tokens = raw_tokens[-self.hes_block_size:]
                    raw_ages = raw_ages[-self.hes_block_size:]

                encoded = self.hes_tokenizer.encode(raw_tokens)
                hes_tokens_list.append(torch.tensor(encoded, dtype=torch.long))
                hes_ages_list.append(torch.tensor(raw_ages, dtype=torch.float) / self.time_scale)
            else:
                # No HES records — empty sequence
                hes_tokens_list.append(torch.tensor([], dtype=torch.long))
                hes_ages_list.append(torch.tensor([], dtype=torch.float))

        # 3. Pad HES sequences
        if all(len(t) == 0 for t in hes_tokens_list):
            hes_tokens_padded = torch.zeros((bsz, 1), dtype=torch.long)
            hes_ages_padded = torch.zeros((bsz, 1), dtype=torch.float)
            hes_attention_mask = torch.zeros((bsz, 1), dtype=torch.float)
        else:
            hes_tokens_padded = pad_sequence(
                [t if len(t) > 0 else torch.tensor([0], dtype=torch.long) for t in hes_tokens_list],
                batch_first=True, padding_value=0
            )
            hes_ages_padded = pad_sequence(
                [a if len(a) > 0 else torch.tensor([0.0]) for a in hes_ages_list],
                batch_first=True, padding_value=0.0
            )
            hes_attention_mask = (hes_tokens_padded != 0).float()
            # Fix: truly empty sequences should have all-zero mask
            for i, t in enumerate(hes_tokens_list):
                if len(t) == 0:
                    hes_attention_mask[i] = 0

        hes_values_padded = torch.full_like(hes_ages_padded, float('nan'))

        # HES static covariates: use first 27 dims from GP static covariates
        hes_static = gp_batch['static_covariates'][:, :27].clone()

        # 4. Add to batch
        gp_batch['hes_tokens'] = hes_tokens_padded
        gp_batch['hes_ages'] = hes_ages_padded
        gp_batch['hes_values'] = hes_values_padded
        gp_batch['hes_static_covariates'] = hes_static
        gp_batch['hes_attention_mask'] = hes_attention_mask

        return gp_batch


def load_yob_lookup(db_path):
    """Load year-of-birth lookup from static_table."""
    import pandas as pd
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT PATIENT_ID, YEAR_OF_BIRTH FROM static_table")
    rows = cursor.fetchall()
    conn.close()
    yob_lookup = {}
    for pid, yob_str in rows:
        try:
            yob_lookup[int(pid)] = datetime.strptime(str(yob_str)[:10], "%Y-%m-%d")
        except:
            try:
                yob_lookup[int(pid)] = datetime(int(str(yob_str)[:4]), 1, 1)
            except:
                pass
    return yob_lookup
