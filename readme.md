
# Neural Network Feature Preference: A Falsification Study

## Overview

This project investigates whether a neural network preferentially
learns a simple predictive feature over a nonlinear XOR feature.

The study uses synthetic binary classification data, PyTorch
multilayer perceptrons, baseline comparisons, permutation ablation,
learning curves, and noise-level robustness experiments.

## Research Question

When simple and nonlinear features provide comparable predictive
information, does a neural network consistently learn or rely
more strongly on the simpler feature?

## Experiments

1. MLP versus logistic regression baseline.
2. Simple-only versus XOR-only MLP.
3. Feature-group permutation ablation.
4. Learning curves across training epochs.
5. Robustness across different noise levels.
6. Paired statistical comparison and confidence interval.

## Tech Stack

- Python
- PyTorch
- NumPy
- Pandas
- Matplotlib
- scikit-learn
- SciPy

## Installation

Create and activate a virtual environment, then install dependencies:

```bash
python -m pip install -r requirements.txt
```

## Run Experiments

```bash
python src/run_experiment.py
python src/analyze_results.py
python src/baseline_experiment.py
python src/learning_robustness.py
```

## Outputs

CSV files containing experimental results are saved in `results/`.
Figures are saved in `figures/`.

## Preliminary Findings

The initial baseline comparison did not establish a statistically
significant accuracy difference between simple-only and XOR-only
models.

The permutation-ablation experiment showed greater sensitivity to
the XOR feature group in the tested configuration.

These findings are exploratory and do not establish a universal
preference of neural networks for either feature type.

## Limitations

- Synthetic data only.
- Limited number of random seeds and noise levels.
- Architecture and optimization can influence learning speed.
- Permutation ablation can introduce distribution shifts.
- Further controls and replications are needed.

## Reproducibility

Experiments use fixed random seeds. Results may vary across hardware
and software versions.

## Experimental Results

### CSV Data
- [Initial experiment](results/experiment_results.csv)
- [Feature ablation](results/ablation_results.csv)
- [Baseline comparison](results/baseline_comparison.csv)
- [Learning curves](results/learning_curves.csv)
- [Noise robustness](results/noise_robustness.csv)

### Figures
- [Model comparison](figures/model_comparison.png)
- [Feature ablation](figures/feature_ablation.png)
- [Full model accuracy](figures/full_accuracy.png)
- [Baseline comparison](figures/baseline_comparison.png)
- [Learning curves](figures/learning_curves.png)
- [Noise robustness](figures/noise_robustness.png)
