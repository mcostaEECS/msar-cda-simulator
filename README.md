# MS-AR(k) Change Detection Simulator

This repository contains the Python implementation of the Markov-Switching Autoregressive (MS-AR) Change Detection Algorithm proposed in:

> **"Generalized Multi-temporal Change Detection with Markov-switching Autoregressive (AR) Model"**  
> _Marcello G. Costa et al., IEEE Journal Submission, 2025._

## 📦 Repository Structure

msar-cda-simulator/
├── main_simulation/ # Main simulation engine over full dataset
│ └── run_main_simulation.py
├── analysis/ # ROC curve generation and performance plots
│ └── classifier_analysis.py
├── tools/ # Analytical simulators (e.g., anomaly, recursion)
│ └── ...
├── data/ # Sample data or loader scripts
├── results/ # Output figures and result .mat files
├── requirements.txt # Python dependencies
├── LICENSE # MIT License
└── README.md # You are here

> **"Generalized Multi-temporal Change Detection with Markov-switching Autoregressive (AR) Model"**  
> M. G. Costa *et al.*, manuscript in preparation, 2025.
>
> # Install the requirements
> pip install -r requirements.txt


