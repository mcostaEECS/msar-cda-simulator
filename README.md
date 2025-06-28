# MS-AR(k) Change Detection Simulator

[![IEEE](https://img.shields.io/badge/IEEE-GRSL-8A2BE2)](https://ieeexplore.ieee.org/document/XXXXXXX)
[![Python 3.8+](https://img.shields.io/badge/Python-3.8%2B-blue)](https://python.org)
[![License](https://img.shields.io/badge/License-BSD--3--Clause-green)](LICENSE)

Official implementation for the paper:  
**"Generalized Multi-temporal Change Detection with Markov-switching Autoregressive (AR) Model"** (To be submitted to IEEE GRSL/TGRS)

> This repository provides the Python implementation of the **Markov-Switching Autoregressive (MS-AR)** model for multi-temporal change detection, as proposed in the article:

The simulator supports different MS-AR model variants, including:
- `MSAR`  →  (k = 11, ℓ = 1, h = 0)  
- `MSARk` →  (k = 23, ℓ = 2, h = 0)  
- `MSARkh` →  (k = 23, ℓ = 2, h = 1)


## 📁 Repository Structure

```
msar-cda-simulator/
├── main_simulation/        # Main simulation engine over full dataset
│   └── main.py
├── analysis/               # ROC curve generation and performance plots
│   └── mainROC.py
├── tools/                  # Analytical simulators (e.g., anomaly, recursion)
│   └── ...functions
├── data/                   # Sample data or loader scripts
├── results/                # Output figures and result .mat files
├── requirements.txt        # Python dependencies
├── LICENSE                 # MIT License
└── README.md               # Project overview and instructions


## 🚀 How to Run

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run Main Simulation

```bash
python main_simulation/main.py

### 3. Run Classifier and ROC Analysis

```bash
python analysis/mainROC.py
```

Output files are saved in the `results/` directory.


## 📊 Dataset

This simulator was developed for experiments using the **CARABAS-II UWB-SAR dataset**, available from AFRL:

🔗 https://www.sdms.afrl.af.mil/index.php?collection=vhf_change_detection

> ⚠️ **Note**: The dataset is not included in this repository due to licensing restrictions.  
> See `data/README.md` for instructions on how to prepare compatible `.mat` files and load target positions.


## 📦 Requirements

Core Python libraries used include:

- `numpy`, `scipy`, `matplotlib`
- `opencv-python`, `scikit-image`, `Pillow`
- `pyprind`, `statsmodels`, `psutil`
- `torch` (for parallel processing)

Install all via:

```bash
pip install -r requirements.txt
```

## 📜 License

This project is licensed under the [MIT License](LICENSE).


## 📌 Citation

The code used to generate the results in this paper **will be made publicly available upon publication** at:

🔗 https://github.com/mcostaEECS/msar-cda-simulator

```bibtex
@unpublished{Costa2025MSAR,
  author       = {Marcello G. Costa and others},
  title        = {Generalized Multi-temporal Change Detection with Markov-switching Autoregressive (AR) Model},
  note         = {Manuscript submitted for publication},
  year         = {2025},
  howpublished = {\url{https://github.com/mcostaEECS/msar-cda-simulator}}
}
```

## 🤝 Contributing

Feel free to open issues or pull requests for bug fixes, improvements, or feature additions.
