
import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from scipy import stats

os.makedirs("results", exist_ok=True)
os.makedirs("figures", exist_ok=True)

EPOCHS = 100
NOISE = 0.20
SEEDS = [11, 22, 33, 44, 55, 66, 77, 88, 99, 111]


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def make_data(n=8000, seed=42, noise=0.20):
    rng = np.random.default_rng(seed)

    # Balanced binary target
    y = rng.integers(0, 2, n)

    # Simple feature: directly predicts y, with controlled noise
    simple = np.where(y == 1, 1.0, -1.0)
    simple[rng.random(n) < noise] *= -1

    # XOR features: their interaction predicts y
    x1 = rng.choice([-1.0, 1.0], n)
    x2 = np.where(
        (x1 != np.where(y == 1, 1.0, -1.0)),
        1.0, -1.0
    )
    x2[rng.random(n) < noise] *= -1

    X = np.column_stack([simple, x1, x2]).astype(np.float32)
    return X, y.astype(np.float32)


class MLP(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )

    def forward(self, x):
        return self.net(x).squeeze(1)


def train_model(X_train, y_train, seed):
    set_seed(seed)
    model = MLP(X_train.shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=0.003)
    loss_fn = nn.BCEWithLogitsLoss()

    X_t = torch.tensor(X_train, dtype=torch.float32)
    y_t = torch.tensor(y_train, dtype=torch.float32)

    for _ in range(EPOCHS):
        model.train()
        optimizer.zero_grad()
        loss = loss_fn(model(X_t), y_t)
        loss.backward()
        optimizer.step()

    return model


def evaluate(X, y, seed):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=seed, stratify=y
    )

    model = train_model(X_train, y_train, seed)
    model.eval()

    with torch.no_grad():
        logits = model(torch.tensor(X_test, dtype=torch.float32))
        predictions = (
            torch.sigmoid(logits).numpy() >= 0.5
        ).astype(int)

    return accuracy_score(y_test, predictions)


def main():
    records = []

    for seed in SEEDS:
        X, y = make_data(seed=seed)

        # Same dataset and split for each comparison
        idx_train, idx_test = train_test_split(
            np.arange(len(y)),
            test_size=0.25,
            random_state=seed,
            stratify=y
        )

        # Simple-only: feature column 0
        simple_acc = evaluate(
            X[:, [0]], y, seed
        )

        # XOR-only: feature columns 1 and 2
        xor_acc = evaluate(
            X[:, [1, 2]], y, seed
        )

        # Full model: all three features
        full_acc = evaluate(X, y, seed)

        records.append({
            "seed": seed,
            "simple_only_accuracy": simple_acc,
            "xor_only_accuracy": xor_acc,
            "full_model_accuracy": full_acc
        })

        print(
            f"Seed {seed}: "
            f"Simple={simple_acc:.3f}, "
            f"XOR={xor_acc:.3f}, "
            f"Full={full_acc:.3f}"
        )

    df = pd.DataFrame(records)
    df.to_csv("results/baseline_comparison.csv", index=False)

    print("\nMean test accuracies:")
    print(df.iloc[:, 1:].mean().round(4))

    # Paired comparison across seeds
    differences = (
        df["simple_only_accuracy"] - df["xor_only_accuracy"]
    )
    mean_diff = differences.mean()
    se = stats.sem(differences)
    critical = stats.t.ppf(0.975, len(differences) - 1)
    ci = (mean_diff - critical * se, mean_diff + critical * se)
    test = stats.ttest_rel(
        df["simple_only_accuracy"],
        df["xor_only_accuracy"]
    )

    print(f"\nMean difference (simple - XOR): {mean_diff:.4f}")
    print(f"95% CI: [{ci[0]:.4f}, {ci[1]:.4f}]")
    print(f"Paired t-test p-value: {test.pvalue:.4f}")

    plt.figure(figsize=(8, 5))
    plt.boxplot(
        [
            df["simple_only_accuracy"],
            df["xor_only_accuracy"],
            df["full_model_accuracy"]
        ],
        tick_labels=["Simple only", "XOR only", "Full model"]
    )
    plt.ylabel("Test accuracy")
    plt.title("Neural Network Baseline Comparison")
    plt.tight_layout()
    plt.savefig("figures/baseline_comparison.png", dpi=200)
    plt.close()

    print("\nSaved results/baseline_comparison.csv")
    print("Saved figures/baseline_comparison.png")


if __name__ == "__main__":
    main()
