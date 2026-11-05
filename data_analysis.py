import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns

# plotting individual histograms 
versions = 4
for version in range(1, versions+1):
    data, failure = np.loadtxt(f"data/version_{version}.csv", unpack=True, delimiter=',')
    if not np.array_equal(np.zeros(failure.shape), failure):
        plt.figure(figsize=(8, 6))
        sns.histplot(failure, stat="density", kde=True)
        plt.xlabel(f"Average failure rate", fontsize=12)
        plt.ylabel("Density", fontsize=12)
        plt.savefig(f"data/charts/version_{version}_failures.pdf")
        plt.close()

    plt.figure(figsize=(8, 6))
    sns.histplot(data, stat="density", kde=True)

    plt.title(f"Version {version} Average Excess Steps PDF", fontsize=15)
    plt.xlabel(f"Average Excess Steps", fontsize=12)
    plt.ylabel("Density", fontsize=12)

    plt.savefig(f"data/charts/version_{version}_histogram.pdf")
    plt.close()

# plot containing the specified versions for comparison 
versions = [(2, "red"), (3, "blue")]
file_name = "data/charts/2_vs_3.pdf"

plt.figure(figsize=(8, 6))
for i, color in versions:
    data, _ = np.loadtxt(f"data/version_{i}.csv", unpack=True, delimiter=",")
    sns.histplot(data, stat="density", kde=True, color=color, label=f"Version {i}")

plt.xlabel(f"Average Excess Steps", fontsize=12)
plt.ylabel("Density", fontsize=12)
plt.legend()

plt.savefig(file_name)
plt.close()