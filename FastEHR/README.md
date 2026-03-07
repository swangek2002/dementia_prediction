# FastEHR

`FastEHR` provides a **scalable workflow** for transforming raw **EHR event data** into a format optimized for **Machine Learning**.  
Leverages **SQLite, Polars, and parallel processing** to handle large-scale EHR data efficiently.  

---

## 📌 **Overview**  
`FastEHR` is a **high-performance data pipeline** designed for **extracting, transforming, and storing EHR event data** in a format optimized for **deep learning and analytics**.  

It supports:  
✔️ **Efficient SQLite database creation from CSV files**  
✔️ **Fast SQL queries** via **SQLite**  
✔️ **Memory-efficient processing** using **Polars' LazyFrames**  
✔️ **Parallelized data extraction** for large-scale EHR data  
✔️ **Exporting structured data to Parquet for ML frameworks**  


---

## 🏗 **Installation**   

**1️⃣ Clone the Repository**  
```
bash
git clone https://github.com/cwlgadd/FastEHR.git  
cd FastEHR
```

Ensure the directory FastEHR is added to is on your PythonPath.

**2️⃣ Install Dependencies**
Ensure you have Python >=3.8 and install required packages:
```
pip install -r requirements.txt  
```
---

## 🎯 Usage

`FastEHR` dataloaders produce ragged lists of patient historical events, with a variety of pre-processing features. These can be used for any ML task, including self-supervised learning.

Additionally, `FastEHR` allows you to produce **Clinical Prediction Model** cohorts, indexing upon different criteria and linking to different outcomes.

Splits across datasets can be linked by shared origins (for exmaple General Practice, or Hospital ID) to avoid data leakage.

---
## 📂 Examples

The `examples/` folder contains:

1️⃣ Building the SQLite database 

Convert your .CSV files into an indexed SQLite database for fast querying

2️⃣ Building ML-ready dataloaders 

Using the SQLite database, construct a linked dataset ready for GPU processing.

    📌 Extracts data from SQLite & generates a deep learning dataset.
     
3️⃣ Building ML-ready dataloaders for Clinical Prediction Models

    📌 Indexed on different criteria
    📌 With processing for various target outcomes

---
## 🔡 EHR Event Tokenization

Tokenization converts raw EHR events into structured representations for deep learning models.
This allows EHR sequences to be used in transformer models, RNNs, or sequence-based ML models.




