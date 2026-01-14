# InSpecLearn4SDL  
*Interpretable Spectral Learning for Self-Driving Labs*  


## 📘 Overview  

**InSpecLearn4SDL** is the official implementation of the methods described in  
📄 *"Interpretable Spectral Features Predict Conductivity in Self-Driving Doped Conjugated Polymer Labs"* ([arXiv:2509.21330](https://arxiv.org/abs/2509.21330)).  

The repository provides an **interpretable QSPR pipeline** that predicts the electrical conductivity of doped conjugated polymers using **optical spectra** and **processing parameters**.  
It is designed for integration into **Self-Driving Labs (SDLs)**, enabling **data-efficient, automated, and interpretable** property prediction workflows.

---

## 🧠 Key Highlights  

- 🔍 **Spectral Featurization:**  
  **Genetic Algorithm** is used to adaptively select important spectral regions and use the area under the curve (AUC) as features  

- 🧩 **Interpretability:**  
  Domain-knowledge-driven feature expansion and **SHAP-based feature selection** retain physically meaningful descriptors.  


- 📈 **Performance:**  
  The hybrid model (expert + data-driven features) achieves high predictive accuracy while reducing experimental effort by ~33%.  

- 🔬 **Generalizable:**  
  Extendable to other spectroscopy–property relationships (e.g. Raman, FTIR, XANES).  

---

### 🧪 **Data Set — Experimental Data from the Amassian Group (North Carolina State University)**  

- 🧫 **Processing Conditions:**  
  Variation in *solvent concentration* and *annealing temperature* across experiments.  

- 🌈 **Spectroscopic Measurements:**  
  Includes three spectral types —  
  **Pre-anneal UV–Vis**, **Post-anneal UV–Vis**, and **Post-dope UV–Vis–NIR** spectra.  

- 📊 **Dataset Size:**  
  A total of **128 doped conjugated polymer samples**, each with paired spectral and conductivity measurements.  

---

### 🗂️ **Project Structure**

- **Code/**
  - `main_final_QSPR_models.ipynb` — 💡 Main Jupyter notebook containing the complete QSPR modeling workflow  
  - `helper.py` — 🧰 Utility functions for data preprocessing and analysis  
  - `generate_adaptive_boundaries_optimization.py` — 🧬 Genetic Algorithm for adaptive spectral boundary optimization  
  - `find_correct_clusters_and_do_ks_test.py` — 📊 Train/test data clustering, KS-tests, and statistical validation  

- **Data/**
  - `spring24_solvtemp_jpm.csv` — 📁 Main experimental dataset (solvent concentration and annealing temperature)  
  - `[experiment folders]/` — 🧪 Individual experiment spectral data files  


---

