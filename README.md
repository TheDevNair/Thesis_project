# MSc Thesis: Deep Neural Network H->bb Tagger

This repo holds the code and resources for my Master's thesis: **"Development of a Deep Neural Network Double b-tagger for Boosted Topologies using CMS Open Data."**

The goal was to build a classifier that can distinguish Higgs boson decays ($H \rightarrow b\bar{b}$) from the QCD background in boosted topologies. I used CMS Open Data (Run 2).

**Thesis PDF:** [DevNair_MSc_Thesis.pdf](./DevNair_MSc_Thesis.pdf)

### 📂 What's in here
This is a flat directory containing the full pipeline I ran on my laptop—from raw ROOT files to final plots.

* **`root_to_parquet.py`**
  The first step. I wrote this to convert the raw ROOT NTuples from CMS Open Data into Parquet format. It handles the type casting and compression.

* **`remove_columns6.py`**
  This handles the data selection. It applies the physics cuts (mass windows, pT range) and defines the labels (ensuring Signal/Background exclusivity).

* **`hyperparameter.py`**
  My Keras Tuner script. I used this to scan for the optimal architecture (dropout rates, layer units, learning rate).

* **`Hbb_tagger.py`**
  **The main analysis script.** This builds the model (using Focal Loss to handle the class imbalance), runs the training loop, and calculates the metrics.

* **`overlaid_feature_plots_final.py`**
  Validation plots. It overlays the Signal vs. Background distributions for the input features (like tau21 and flight distance).

### ⚠️ Heads up on Paths
Since I developed this locally for my degree, **the file paths are hardcoded to my local machine** (you'll see a lot of `C:/Users/Dev/...`).

If you want to run this:
1. Clone the repo.
2. Open the script you want to use.
3. Ctrl+F for "C:/" and swap it with the path to your own data folder.

### 📊 Results
* **ROC AUC:** 0.9441
* **Average Precision:** 0.9004

---
*Dev Nair*
