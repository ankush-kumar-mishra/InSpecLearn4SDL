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
  **Genetic Algorithm** is used to adaptively select important spectral regions and use Area Under the Curve as Features  

- 🧩 **Interpretability:**  
  Domain-knowledge-driven feature expansion and **SHAP-based feature selection** retain physically meaningful descriptors.  


- 📈 **Performance:**  
  The hybrid model (expert + data-driven features) achieves high predictive accuracy while reducing experimental effort by ~33%.  

- 🔬 **Generalizable:**  
  Extendable to other spectroscopy–property relationships (e.g. Raman, FTIR, XANES).  

---

**Data Set - Experimental Data from Aram Amassian Group from North Carolina State University**

  Processing conditions: solvent concentration and annealing temperature

  Spectra types: pre-anneal UV–Vis, post-anneal UV–Vis, post-dope UV–Vis–NIR

  Dataset size: 128 samples

## 📂 Repository Structure  


**Project Structure**

├── Code/ │ ├── main_final_QSPR_models.ipynb # Main notebook with QSPR models │ ├── helper.py # Helper functions for data processing │ ├── generate_adaptive_boundaries_optimization.py # Genetic algorithm for boundary optimization │ └── find_correct_clusters_and_do_ks_test.py # Train/test splitting and statistical tests ├── Data/ │ ├── spring24_solvtemp_jpm.csv # Main dataset │ └── [experiment folders]/ # Individual experiment spectral data └── 

