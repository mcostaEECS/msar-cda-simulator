# MS-AR(k) Change Detection Simulator

This repository provides the Python implementation of the **Markov-Switching Autoregressive (MS-AR)** model for multi-temporal change detection, as proposed in the article:

> **"Generalized Multi-temporal Change Detection with Markov-switching Autoregressive (AR) Model"**  
> _Marcello G. Costa et al., IEEE Journal Submission, 2025._

The simulator supports different MS-AR model variants, including:
- `MSAR`  →  (k = 11, ℓ = 1, h = 0)  
- `MSARk` →  (k = 23, ℓ = 2, h = 0)  
- `MSARkh` →  (k = 23, ℓ = 2, h = 1)
<<<<<<< HEAD

---

## 📁 Repository Structure
=======
>>>>>>> 18816c1 (README update)

---

## 📁 Repository Structure

```
msar-cda-simulator/
├── main_simulation/        # Main simulation engine over full dataset
│   └── run_main_simulation.py
├── analysis/               # ROC curve generation and performance plots
│   └── classifier_analysis.py
├── tools/                  # Analytical simulators (e.g., anomaly, recursion)
│   └── ...
├── data/                   # Sample data or loader scripts
├── results/                # Output figures and result .mat files
├── requirements.txt        # Python dependencies
├── LICENSE                 # MIT License
└── README.md               # Project overview and instructions
<<<<<<< HEAD

## 🚀 How to Run

=======
```

---

## 🚀 How to Run

>>>>>>> 18816c1 (README update)
### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run Main Simulation

```bash
<<<<<<< HEAD
python main_simulation/main.py
=======
python main_simulation/run_main_simulation.py
>>>>>>> 18816c1 (README update)
```

### 3. Run Classifier and ROC Analysis

```bash
python analysis/classifier_analysis.py
```

Output files are saved in the `results/` directory.

---

## 📊 Dataset

This simulator was developed for experiments using the **CARABAS-II UWB-SAR dataset**, available from AFRL:

🔗 https://www.sdms.afrl.af.mil/index.php?collection=vhf_change_detection

> ⚠️ **Note**: The dataset is not included in this repository due to licensing restrictions.  
> See `data/README.md` for instructions on how to prepare compatible `.mat` files and load target positions.

---

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

---

## 📜 License

This project is licensed under the [MIT License](LICENSE).

---

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

---

## 🤝 Contributing

Feel free to open issues or pull requests for bug fixes, improvements, or feature additions.
