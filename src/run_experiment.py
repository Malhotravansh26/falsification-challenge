
import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

os.makedirs("results", exist_ok=True)
os.makedirs("figures", exist_ok=True)

EPOCHS = 100
NOISE = 0.20

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def make_data(n=8000, seed=42, noise=NOISE):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, n)

    # Simple signal: one feature predicts the label
    simple = np.where(y == 1, 1.0, -1.0)
    flip_simple = rng.random(n) < noise
    simple[flip_simple] *= -1

    # Complex signal: XOR of two features predicts the label
    x1 = rng.choice([-1.0, 1.0], n)
    x2 = np.where(
        (x1 != np.where(y == 1, 1.0, -1.0)),
        1.0, -1.0
    )
    flip_xor = rng.random(n) < noise
    x2[flip_xor] *= -1

    X = np.column_stack([simple, x1, x2]).astype(np.float32)
    return X, y.astype(np.float32)

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )

    def forward(self, x):
        return self.net(x).squeeze(1)

def train_model(X, y, seed):
    set_seed(seed)
    model = MLP()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.003)
    loss_fn = nn.BCEWithLogitsLoss()

    xt = torch.tensor(X, dtype=torch.float32)
    yt = torch.tensor(y, dtype=torch.float32)

    for _ in range(EPOCHS):
        model.train()
        optimizer.zero_grad()
        loss = loss_fn(model(xt), yt)
        loss.backward()
        optimizer.step()

    return model

def predict(model, X):
    model.eval()
    with torch.no_grad():
        logits = model(torch.tensor(X, dtype=torch.float32))
        return (torch.sigmoid(logits).numpy() >= 0.5).astype(int)

def main():
    records = []

    for seed in [11, 22, 33, 44, 55, 66, 77, 88, 99, 111]:
        X, y = make_data(seed=seed)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.25, random_state=seed,
            stratify=y
        )

        model = train_model(X_train, y_train, seed)
        full_acc = accuracy_score(y_test, predict(model, X_test))

        # Permutation ablation: shuffle a feature group on test data
        rng = np.random.default_rng(seed + 1000)

        X_simple_removed = X_test.copy()
        X_simple_removed[:, 0] = rng.permutation(
            X_simple_removed[:, 0]
        )
        simple_ablated_acc = accuracy_score(
            y_test, predict(model, X_simple_removed)
        )

        X_xor_removed = X_test.copy()
        idx = rng.permutation(len(X_test))
        X_xor_removed[:, 1:3] = X_xor_removed[idx, 1:3]
        xor_ablated_acc = accuracy_score(
            y_test, predict(model, X_xor_removed)
        )

        records.append({
            "seed": seed,
            "full_accuracy": full_acc,
            "simple_ablation_accuracy": simple_ablated_acc,
            "xor_ablation_accuracy": xor_ablated_acc,
            "simple_importance": full_acc - simple_ablated_acc,
            "xor_importance": full_acc - xor_ablated_acc
        })

        print(
            f"Seed {seed}: full={full_acc:.3f}, "
            f"simple drop={full_acc-simple_ablated_acc:.3f}, "
            f"XOR drop={full_acc-xor_ablated_acc:.3f}"
        )

    df = pd.DataFrame(records)
    df.to_csv("results/ablation_results.csv", index=False)

    print("\nMean results:")
    print(df.mean(numeric_only=True).round(4))

    print("\nMean feature importance:")
    print(df[["simple_importance", "xor_importance"]].mean())

    plt.figure(figsize=(8, 5))
    plt.boxplot([
        df["simple_importance"],
        df["xor_importance"]
    ], tick_labels=["Simple feature", "XOR feature"])
    plt.ylabel("Accuracy drop after permutation")
    plt.title("Feature-group reliance across random seeds")
    plt.tight_layout()
    plt.savefig("figures/feature_ablation.png", dpi=200)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(df["seed"], df["full_accuracy"], marker="o")
    plt.xlabel("Random seed")
    plt.ylabel("Test accuracy")
    plt.title("Full model accuracy across random seeds")
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig("figures/full_accuracy.png", dpi=200)
    plt.close()

    print("\nSaved results/ablation_results.csv")
    print("Saved figures/feature_ablation.png")
    print("Saved figures/full_accuracy.png")

if __name__ == "__main__":
    main()
