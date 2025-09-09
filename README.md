# Cross Connection PUF Cracking Project

This project simulates **Arbiter PUFs** and a **Cross-Connection (COCO)-like PUF** and demonstrates how simple **linear machine learning models** can be used to attack and analyze them.

## Features
- Implements Arbiter PUF with a linear delay model.
- Provides a feature transform (phi-vector) for mapping challenges to linear space.
- Builds a COCO-like PUF using multiple APUFs combined (XOR, majority, or concatenation).
- Generates large Challenge-Response Pair (CRP) datasets.
- Trains and evaluates **LinearSVC** and **Logistic Regression** to crack the PUF.
- Reports accuracy, confusion matrices, and best hyperparameters.

## Requirements
Install the required Python packages:
```bash
pip install numpy scikit-learn
```

## Usage
Run the main script:
```bash
python arbiter_puf_coco_crack.py
```

By default, it:
- Simulates a 64-bit Arbiter PUF (4 combined in XOR mode).
- Generates 25,000 CRPs.
- Trains LinearSVC and LogisticRegression.
- Prints evaluation metrics.

## Example Output
```
Generating 25000 CRPs, n_bits=64, COCO mode=xor, k=4 ...
Training LinearSVC and LogisticRegression...
---- LinearSVC ----
Best params: {'C': 1}
Accuracy: 0.982
...
```

## Customization
You can modify parameters inside the script:
- `n_bits`: challenge length (default 64).
- `k_apufs`: number of Arbiter PUFs combined.
- `mode`: 'xor', 'majority', or 'concat'.
- `NUM_CRPS`: number of challenge-response pairs to generate.

## References
- PUF modeling and machine learning attacks: [pypuf documentation](https://pypi.org/project/pypuf/)

---
Author: **Sumit Kumar**
