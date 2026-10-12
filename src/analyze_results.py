
import numpy as np
import pandas as pd
from scipy import stats

df = pd.read_csv("results/ablation_results.csv")

simple = df["simple_importance"].to_numpy()
xor = df["xor_importance"].to_numpy()

# Paired differences across the same random seeds
difference = xor - simple

mean_diff = difference.mean()
sd_diff = difference.std(ddof=1)
n = len(difference)

# 95% confidence interval for mean paired difference
se = sd_diff / np.sqrt(n)
critical = stats.t.ppf(0.975, df=n - 1)

ci_low = mean_diff - critical * se
ci_high = mean_diff + critical * se

# Two-sided paired t-test
test = stats.ttest_rel(xor, simple)

print("Number of paired seeds:", n)
print(f"Mean XOR importance: {xor.mean():.4f}")
print(f"Mean simple importance: {simple.mean():.4f}")
print(f"Mean paired difference (XOR - simple): {mean_diff:.4f}")
print(f"95% CI: [{ci_low:.4f}, {ci_high:.4f}]")
print(f"Paired t-test statistic: {test.statistic:.4f}")
print(f"Paired t-test p-value: {test.pvalue:.4f}")

if ci_low > 0:
    print("Interpretation: XOR importance was higher in this experiment.")
elif ci_high < 0:
    print("Interpretation: Simple feature importance was higher in this experiment.")
else:
    print("Interpretation: The confidence interval includes zero.")

print(
    "\nCaution: seeds share the same experimental design. "
    "This test does not establish a universal neural-network preference."
)
