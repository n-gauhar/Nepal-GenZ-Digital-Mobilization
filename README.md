# Nepal Gen Z Digital Mobilization

This repository contains the source code used for preprocessing multilingual YouTube metadata and implementing the Susceptible–Quarantined–Contacted–Infected–Recovered (SQCIR) epidemiological model for studying the dynamics of the September 2025 Nepal Gen Z digital mobilization.

The repository accompanies our research on modeling online collective action using compartmental epidemic models, parameter estimation, sensitivity analysis, and fractional-order extensions.

---

## Repository Structure

```text
Nepal-GenZ-Digital-Mobilization/
│
├── preprocessing/
│   ├── preprocess_translate_titles_colab.py
│   └── README.md
│
├── sample_data/
│   ├── youtube_metadata_sample.csv
│   ├── data_dictionary.md
│   └── README.md
│
├── sqcir/
│   ├── sensitivity_analysis.py
│   ├── r0_surface_analysis.py
│   ├── sqcir_data_fit.py
│   ├── mobility_snapshots.py
│   ├── caputo_fractional_examples.py
│   ├── caputo_phase_peak_analysis.py
│   └── README.md
│
├── figures/
├── docs/
├── requirements.txt
├── LICENSE
└── README.md
```

---

## Workflow

The analysis pipeline consists of the following stages:

1. Collect publicly available YouTube metadata related to the September 2025 Nepal Gen Z movement.
2. Clean multilingual video titles by removing emojis, hashtags, and unnecessary whitespace.
3. Detect the language of each title and translate non-English titles into English.
4. Construct the SQCIR epidemiological model.
5. Estimate model parameters using engagement data.
6. Perform sensitivity analysis and R₀ analysis.
7. Evaluate fractional-order Caputo extensions.
8. Generate publication-quality figures.

---

## Repository Contents

### Preprocessing

The preprocessing scripts perform:

* Video title cleaning
* Language detection
* Heuristic correction of romanized Hindi/Nepali text
* English translation

---

### SQCIR Model

The modeling scripts include:

* SQCIR numerical simulation
* Parameter estimation
* R₀ sensitivity analysis
* Three-dimensional R₀ parameter surfaces
* Mobility visualization
* Caputo fractional-order simulations
* Phase-plane and peak-shift analysis

---

## Data Availability

A representative sample of the YouTube metadata is included in the `sample_data` directory to demonstrate the expected input format for the preprocessing scripts.

The complete dataset used in the associated publications is not included in this repository because it supports multiple ongoing research projects. Researchers interested in collaboration or additional information are encouraged to contact the corresponding author.

---

## Requirements

The Python packages required to run the scripts are listed in `requirements.txt`.

Typical dependencies include:

* numpy
* pandas
* matplotlib
* scipy
* langdetect
* deep-translator
* emoji

---

## Running the Code

Example workflow:

```bash
# preprocess metadata
python preprocessing/preprocess_translate_titles_colab.py

# perform sensitivity analysis
python sqcir/sensitivity_analysis.py

# generate R₀ parameter surfaces
python sqcir/r0_surface_analysis.py

# fit the SQCIR model
python sqcir/sqcir_data_fit.py
```

---

## Citation

If you use this repository in your research, please cite the associated publication.

**Manuscript status:** Under review.

Citation information will be updated after publication.

---

## License

This project is distributed under the MIT License.

