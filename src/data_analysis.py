import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns

# plotting individual histograms 
versions = 3
for version in range(1, versions+1):
    if version == 1:
        data, failure = np.loadtxt("data/version_1.csv", unpack=True, delimiter=',')
        failure /= 8100 # average the result
        plt.figure(figsize=(8, 6))
        sns.histplot(failure, stat="density", kde=True)
        plt.xlabel(f"Average failure rate", fontsize=12)
        plt.ylabel("Density", fontsize=12)
        plt.savefig("data/version_1_failures.pdf")
        plt.close()

    else:
        data = np.loadtxt(f"data/version_{version}.csv", unpack=True)

    plt.figure(figsize=(8, 6))
    sns.histplot(data, stat="density", kde=True)

    plt.xlabel(f"Average Excess Steps", fontsize=12)
    plt.ylabel("Density", fontsize=12)

    plt.savefig(f"data/version_{version}_histogram.pdf")
    plt.close()

# plot containing all for comparison 
colors = ["red", "blue"]
plt.figure(figsize=(8, 6))
for i, color in enumerate(colors):
    data = np.loadtxt(f"data/version_{i+2}.csv", unpack=True)

    sns.histplot(data, stat="density", kde=True, color=color, label=f"Version {i+2}")

plt.xlabel(f"Average Excess Steps", fontsize=12)
plt.ylabel("Density", fontsize=12)
plt.legend()

plt.savefig(f"data/histogram_all.pdf")
plt.close()