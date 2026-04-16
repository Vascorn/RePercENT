import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import os


def plot_cancer_type_distribution(test_loader, train_loader, script_dir, title="Distribution of Cancer Types in Train and Test Sets", train_color="tab:blue", test_color="tab:orange", savefig_name="cancer_type_distribution.pdf"):
    test_counts = {}
    for batch in test_loader:
        for ct in batch['cancer_type']:
            test_counts[ct] = test_counts.get(ct, 0) + 1
    
    
    print(f"Counts for each cancer type: {test_counts}")
    train_counts = {}
    for batch in train_loader:
        for ct in batch['cancer_type']:
            train_counts[ct] = train_counts.get(ct, 0) + 1

    print(f"Counts for each cancer type: {train_counts}")
    
    cancer_types = sorted(set(train_counts) | set(test_counts))
    train_values = [train_counts.get(cancer_type, 0) for cancer_type in cancer_types]
    test_values = [test_counts.get(cancer_type, 0) for cancer_type in cancer_types]

    x = np.arange(len(cancer_types))
    width = 0.4

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    ax.grid(axis="x", linestyle="--", alpha=0.5)
    ax.bar(x - width / 2, train_values, width, label="Train", color= train_color)
    ax.bar(x + width / 2, test_values, width, label="Test", color= test_color)
    ax.set_title(title)
    ax.set_xlabel("Cancer Type")
    ax.set_ylabel("Count")
    ax.set_xticks(x)
    ax.set_xticklabels(cancer_types, rotation=45, ha="right")
    
    ax.legend()
    plt.tight_layout()
    
    figures_dir = os.path.join(script_dir, "figures")
    os.makedirs(figures_dir, exist_ok=True)
    plt.savefig(os.path.join(figures_dir, savefig_name))

    plt.clf()
    return