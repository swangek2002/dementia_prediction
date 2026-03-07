import pickle
from pathlib import Path

base_dir = "/Data0/swangek_data/991/CPRD/data/FoundationalModel/PreTrain"
old_prefix = "/mnt/d/991/CPRD/data/FoundationalModel/PreTrain/"

for split in ["train", "test", "val"]:
    pkl_path = f"{base_dir}/file_row_count_dict_{split}.pickle"
    with open(pkl_path, "rb") as f:
        d = pickle.load(f)
    
    new_d = {}
    for k, v in d.items():
        k_str = str(k)
        # Extract relative path after split=xxx/
        idx = k_str.find(f"split={split}/")
        if idx != -1:
            rel = k_str[idx + len(f"split={split}/"):]
            new_d[rel] = v
        else:
            new_d[k_str] = v
    
    with open(pkl_path, "wb") as f:
        pickle.dump(new_d, f)
    
    print(f"Fixed {split}: {len(new_d)} entries")
    print(f"  Sample key: {list(new_d.keys())[0]}")

print("Done!")
