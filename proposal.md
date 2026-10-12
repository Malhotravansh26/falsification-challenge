
# The Falsification Challenge
## Do Neural Networks Prefer Simpler Predictive Features?

### 1. Abstract

This project investigates whether a neural network preferentially relies on a simple predictive feature over a nonlinear XOR feature when both provide comparable predictive information.

A synthetic binary classification dataset was constructed with a noisy linear feature and a noisy nonlinear feature group. A multilayer perceptron (MLP) was evaluated using multiple random seeds, feature-group permutation ablation, and simple-only versus XOR-only baselines.

The preliminary results did not establish a statistically significant difference in predictive accuracy between the simple-only and XOR-only models. Further controlled experiments are required to determine whether neural networks exhibit a systematic preference for simpler features.

### 2. Research Question

When simple and nonlinear features provide comparable predictive information, does a neural network consistently learn or rely more strongly on the simpler feature?

### 3. Hypotheses

**Alternative hypothesis (H1):** Under controlled conditions, neural networks will demonstrate greater reliance on a simple predictive feature than on a nonlinear feature group of comparable predictive strength.

**Null hypothesis (H0):** The experiment will not demonstrate a consistent preference for the simple feature under the specified conditions.

### 4. Dataset Design

A synthetic binary classification dataset was generated using NumPy.

- Target: balanced binary labels.
- Simple feature: a noisy feature directly associated with the target.
- Nonlinear feature group: two binary variables whose XOR relationship predicts the target.
- Noise rate: 20%.
- Sample size: 8,000 examples per dataset.
- Evaluation: 75% training data and 25% test data.
- Randomization: ten random seeds.

The intended design gives both feature groups comparable predictive strength before model training.

### 5. Models and Experiments

The experiments include:

1. **Full MLP:** trained with both feature groups.
2. **Simple-only MLP:** trained using only the simple feature.
3. **XOR-only MLP:** trained using only the two nonlinear features.
4. **Permutation ablation:** feature groups are shuffled at evaluation time to estimate their contribution to model predictions.

The baseline comparison uses the same data-generation procedure and reproducible random seeds.

### 6. Preliminary Results

The simple-only versus XOR-only comparison produced the following results:

- Mean accuracy difference (simple minus XOR): 0.0018.
- 95% confidence interval: [-0.0055, 0.0092].
- Paired t-test p-value: 0.5848.
- Full-model mean accuracy: approximately 0.8054.

The permutation-ablation experiment produced these mean accuracy drops:

- Simple feature group: 0.12875.
- XOR feature group: 0.18300.

These results suggest that the trained model was more sensitive to permutation of the XOR feature group in this particular experiment. However, the difference was not statistically significant in the separate simple-only versus XOR-only accuracy comparison.

### Additional Experiments: Learning Curves and Noise Robustness

**Learning curves:** Simple-only and XOR-only neural networks were evaluated at multiple training checkpoints to compare their learning trajectories.

**Noise robustness:** Noise probabilities of 0.05, 0.15, 0.25, and 0.35 were tested.

At noise 0.25:
- Simple-only accuracy: 75.36%
- XOR-only accuracy: 75.28%

At noise 0.35:
- Simple-only accuracy: 64.48%
- XOR-only accuracy: 65.29%

The results suggest that performance decreases at higher noise levels in this synthetic setup. These findings are preliminary and do not establish a universal preference for simple or nonlinear features.

**Supporting data:** See `results/learning_curves.csv` and `results/noise_robustness.csv`.


### 7. Interpretation

The current evidence does not establish a general preference for simple features.

The baseline comparison indicates that the two feature groups achieved similar standalone predictive performance. The ablation results suggest that the trained model's predictions were sensitive to the nonlinear feature group.

These findings are exploratory. Standalone accuracy and permutation importance measure different properties and should not be interpreted as interchangeable measures of feature preference.

### 8. Limitations

- The experiments use synthetic data rather than real-world datasets.
- Permutation importance can be affected by changes in feature distributions.
- Multiple random seeds do not eliminate all experimental confounders.
- Training duration and architecture may affect learning speed.
- A single noise level cannot establish robustness.
- Accuracy differences alone do not directly measure the order in which features are learned.
- The current analysis does not establish a universal property of neural networks.


### Additional Experiments: Learning Curves and Noise Robustness

The learning-curve experiment compares the test accuracy of simple-only and XOR-only MLPs across training checkpoints.

The robustness experiment evaluates noise probabilities of 0.05, 0.15, 0.25, and 0.35.

At noise 0.25:
- Simple-only accuracy: 75.36%
- XOR-only accuracy: 75.28%

At noise 0.35:
- Simple-only accuracy: 64.48%
- XOR-only accuracy: 65.29%

Performance decreased at higher noise levels in this synthetic setup. These results are preliminary and do not establish a universal preference for simple or nonlinear features.

Supporting data:
- `results/learning_curves.csv`
- `results/noise_robustness.csv`


### 9. Future Work

Future experiments should include:

1. Learning curves to compare how quickly each feature group is learned.
2. Noise-level sensitivity analysis.
3. Multiple neural-network architectures.
4. Explicit controls for feature strength and optimization difficulty.
5. Confidence intervals and effect sizes for feature-reliance differences.
6. Replication on additional synthetic datasets.

### 10. Conclusion

This project implements a reproducible starting point for investigating neural-network feature preferences using synthetic data, baseline models, and feature ablation.

The preliminary evidence does not establish the hypothesis that neural networks consistently prefer simpler features. Additional controlled experiments are necessary to distinguish feature simplicity from predictive strength and optimization difficulty.
