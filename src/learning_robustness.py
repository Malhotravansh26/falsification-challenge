
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
SEEDS = [11, 22, 33, 44, 55]
NOISE_LEVELS = [0.05, 0.15, 0.25, 0.35]


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def make_data(n=6000, seed=42, noise=0.20):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, n)

    # Simple predictive feature with controlled label noise
    simple = np.where(y == 1, 1.0, -1.0)
    simple[rng.random(n) < noise] *= -1

    # XOR feature group with the same nominal noise rate
    x1 = rng.choice([-1.0, 1.0], n)
    target_sign = np.where(y == 1, 1.0, -1.0)
    x2 = np.where(x1 != target_sign, 1.0, -1.0)
    x2[rng.random(n) < noise] *= -1

    X_simple = simple.reshape(-1, 1).astype(np.float32)
    X_xor = np.column_stack([x1, x2]).astype(np.float32)

    return X_simple, X_xor, y.astype(np.float32)


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


def fit_and_evaluate(X_train, y_train, X_test, y_test, seed):
    set_seed(seed)

    model = MLP(X_train.shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=0.003)
    loss_fn = nn.BCEWithLogitsLoss()

    xt = torch.tensor(X_train, dtype=torch.float32)
    yt = torch.tensor(y_train, dtype=torch.float32)
    xv = torch.tensor(X_test, dtype=torch.float32)

    checkpoints = [1, 5, 10, 20, 40, 60, 80, 100]
    curve = []

    for epoch in range(1, EPOCHS + 1):
        model.train()
        optimizer.zero_grad()

        logits = model(xt)
        loss = loss_fn(logits, yt)
        loss.backward()
        optimizer.step()

        if epoch in checkpoints:
            model.eval()
            with torch.no_grad():
                probs = torch.sigmoid(model(xv)).numpy()
                predictions = (probs >= 0.5).astype(int)

            curve.append({
                "epoch": epoch,
                "test_accuracy": accuracy_score(
                    y_test, predictions
                )
            })

    return curve


def main():
    # Experiment A: learning curves at fixed noise
    curve_records = []

    for seed in SEEDS:
        X_simple, X_xor, y = make_data(seed=seed, noise=0.20)

        indices = np.arange(len(y))
        train_idx, test_idx = train_test_split(
            indices,
            test_size=0.25,
            random_state=seed,
            stratify=y
        )

        for name, X in [
            ("Simple", X_simple),
            ("XOR", X_xor)
        ]:
            curve = fit_and_evaluate(
                X[train_idx], y[train_idx],
                X[test_idx], y[test_idx], seed
            )

            for row in curve:
                curve_records.append({
                    "seed": seed,
                    "feature_group": name,
                    **row
                })

        print(f"Learning curves completed for seed {seed}")

    curve_df = pd.DataFrame(curve_records)
    curve_df.to_csv(
        "results/learning_curves.csv", index=False
    )

    summary = curve_df.groupby(
        ["feature_group", "epoch"]
    )["test_accuracy"].agg(["mean", "std"]).reset_index()

    plt.figure(figsize=(9, 5))
    for name in ["Simple", "XOR"]:
        subset = summary[summary["feature_group"] == name]
        plt.plot(
            subset["epoch"], subset["mean"],
            marker="o", label=name
        )

    plt.xlabel("Training epoch")
    plt.ylabel("Mean test accuracy")
    plt.title("Learning Curves: Simple vs XOR")
    plt.legend()
    plt.tight_layout()
    plt.savefig("figures/learning_curves.png", dpi=200)
    plt.close()

    # Experiment B: robustness across noise levels
    noise_records = []

    for noise in NOISE_LEVELS:
        for seed in SEEDS:
            X_simple, X_xor, y = make_data(
                seed=seed, noise=noise
            )

            indices = np.arange(len(y))
            train_idx, test_idx = train_test_split(
                indices,
                test_size=0.25,
                random_state=seed,
                stratify=y
            )

            for name, X in [
                ("Simple", X_simple),
                ("XOR", X_xor)
            ]:
                curve = fit_and_evaluate(
                    X[train_idx], y[train_idx],
                    X[test_idx], y[test_idx], seed
                )

                final_accuracy = curve[-1]["test_accuracy"]

                noise_records.append({
                    "noise": noise,
                    "seed": seed,
                    "feature_group": name,
                    "test_accuracy": final_accuracy
                })

        print(f"Noise level {noise:.2f} completed")

    noise_df = pd.DataFrame(noise_records)
    noise_df.to_csv(
        "results/noise_robustness.csv", index=False
    )

    noise_summary = noise_df.groupby(
        ["noise", "feature_group"]
    )["test_accuracy"].agg(["mean", "std"]).reset_index()

    plt.figure(figsize=(9, 5))
    for name in ["Simple", "XOR"]:
        subset = noise_summary[
            noise_summary["feature_group"] == name
        ]
        plt.plot(
            subset["noise"], subset["mean"],
            marker="o", label=name
        )

    plt.xlabel("Noise probability")
    plt.ylabel("Mean test accuracy")
    plt.title("Robustness to Increasing Noise")
    plt.legend()
    plt.tight_layout()
    plt.savefig("figures/noise_robustness.png", dpi=200)
    plt.close()

    print("\nLearning curve summary:")
    print(summary.to_string(index=False))

    print("\nNoise robustness summary:")
    print(noise_summary.to_string(index=False))

    print("\nAll experiments completed.")
    print("Results saved in results/")
    print("Plots saved in figures/")


if __name__ == "__main__":
    main()
