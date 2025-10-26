# analyze.py
"""
Universal and final analysis script for all Vision-BDH experiments.

This script intelligently finds and loads CSV log files for different
training runs (CIFAR-10, CIFAR-100) and generates a complete
set of comparative visualizations.
"""

import matplotlib.pyplot as plt
import pandas as pd
import os

# ==============================================================================
# STEP 1: CONFIGURE PATHS TO ALL CSV LOG FILES
# ==============================================================================
# All output plots will be saved here
IMAGE_DIR = "./analysis_results"
os.makedirs(IMAGE_DIR, exist_ok=True)

# Define paths to the metrics files for all experiments
# This script will safely handle cases where a file doesn't exist.
paths = {
    # --- CIFAR-10 Experiment ---
    "bdh_v1_c10": "./checkpoints_v1_cifar10/metrics_v1_cifar10.csv",
    "bdh_v2_c10": "./checkpoints_v2_cifar10/metrics_v2_cifar10.csv",
    "vit_c10": "./checkpoints_vit_tiny_cifar10/metrics_vit_tiny_cifar10.csv",

    # --- CIFAR-100 Experiment ---
    "bdh_v2_c100": "./checkpoints_v2_cifar100/metrics_v2_cifar100.csv",
    "vit_c100": "./checkpoints_vit_tiny_cifar100/metrics_vit_tiny_cifar100.csv",
    "res_c100": "./checkpoints_resnet20_cifar100/metrics_resnet20_cifar100.csv",
    "mn_v2_c100": "./checkpoints_mobilenetv2_cifar100/metrics_mobilenetv2_cifar100.csv",
    "e_net": "./checkpoints_efficientnet_cifar100/metrics_efficientnet_cifar100.csv",
    "deit": "./checkpoints_deit_tiny_cifar100/metrics_deit_tiny_cifar100.csv",
}

def load_data(path, model_name):
    """Safely loads a CSV, returning an empty DataFrame if it's missing."""
    if not os.path.exists(path):
        print(f"INFO: Log file for '{model_name}' not found at '{path}'.")
        return pd.DataFrame()
    print(f"SUCCESS: Loaded logs for '{model_name}' from '{path}'")
    return pd.read_csv(path)

# Load all available data into a dictionary of DataFrames
dataframes = {key: load_data(path, key) for key, path in paths.items()}


# ==============================================================================
# STEP 2: DEFINE FINAL TEST ACCURACIES (Manual Entry)
# ==============================================================================
# This is the single source of truth for final test accuracies,
# as they are not always in the CSV log file.
# !!! UPDATE THESE VALUES WITH YOUR FINAL RESULTS !!!

final_results = {
    "cifar10": {
        "Vision-BDH v1": {
            "params_millions": 3.56,
            "final_test_accuracy": 80.43, # <-- UPDATE ok
            "dataframe_key": "bdh_v1_c10",
            "color": "#FF6347"
        },
        "Vision-BDH v2": {
            "params_millions": 3.2,
            "final_test_accuracy": 80.45, # <-- UPDATE ok
            "dataframe_key": "bdh_v2_c10",
            "color": "#D62728"
        },
        "ViT-Tiny": {
            "params_millions": 5.7,
            "final_test_accuracy": 76.05, # <-- UPDATE ok
            "dataframe_key": "vit_c10",
            "color": "#1F77B4"
        }
    },
    "cifar100": {
        "Vision-BDH v2": {
            "params_millions": 3.2,
            "final_test_accuracy": 51.44, # <-- UPDATE Ok
            "dataframe_key": "bdh_v2_c100",
            "color": "#D62728"
        },
        "ViT-Tiny": {
            "params_millions": 5.7,
            "final_test_accuracy": 46.53, # <-- UPDATE ok
            "dataframe_key": "vit_c100",
            "color": "#1F77B4"
        },
        "Resnet20": {
            "params_millions": 0.27,
            "final_test_accuracy": 45.62, # <-- UPDATE ok
            "dataframe_key": "res_c100",
            "color": "#9467BD"
        },
        "MobileNetV2": {
            "params_millions": 2.35,
            "final_test_accuracy": 33.83, # <-- UPDATE ok
            "dataframe_key": "mn_v2_c100",
            "color": "#2CA02C"
        },
        "EfficientNet-B0": {
            "params_millions": 4.14,
            "final_test_accuracy": 40.20, # <-- UPDATE ok
            "dataframe_key": "e_net",
            "color": "#FF7F0E"
        },
        "DeiT-Tiny": {
            "params_millions": 5.51,
            "final_test_accuracy": 35.31, # <-- UPDATE ok
            "dataframe_key": "deit",
            "color": "#17BECF"
        },
        
    }
}


# ==============================================================================
# STEP 3: VISUALIZATION FUNCTIONS (Now they use the CSV data!)
# ==============================================================================

def plot_learning_curves(results, dataframes, dataset_name, save_dir):
    """Plots validation accuracy curves for all models on a given dataset."""
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(12, 7))
    has_data_to_plot = False

    for name, data in results.items():
        df_key = data["dataframe_key"]
        df = dataframes.get(df_key)
        if df is not None and not df.empty and "epoch" in df and "val_accuracy" in df:
            ax.plot(df["epoch"], df["val_accuracy"], label=f'{name} (Val Acc.)',
                    color=data["color"], marker=".", markersize=8, linewidth=2.5)
            has_data_to_plot = True

    if not has_data_to_plot:
        print(f"INFO: No epoch data found for {dataset_name}. Skipping learning curve plot.")
        plt.close(fig)
        return

    ax.set_xlabel("Epoch", fontsize=14, fontweight="bold")
    ax.set_ylabel("Validation Accuracy (%)", fontsize=14, fontweight="bold")
    ax.set_title(f"Learning Curve Comparison on {dataset_name}", fontsize=18, fontweight="bold")
    ax.legend(fontsize=12)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    fig.tight_layout()

    save_path = os.path.join(save_dir, f"{dataset_name.lower()}_learning_curves.png")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"✓ Saved Learning Curve plot to {save_path}")
    plt.close(fig)

def plot_final_accuracy(results, dataset_name, save_dir):
    """Bar chart comparing final test accuracies."""
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(8, 6))
    
    model_names = list(results.keys())
    accuracies = [m["final_test_accuracy"] for m in results.values()]
    colors = [m["color"] for m in results.values()]

    bars = ax.bar(model_names, accuracies, color=colors, alpha=0.9, edgecolor="black")

    ax.set_title(f"Final Test Accuracy on {dataset_name}", fontsize=16, fontweight="bold")
    ax.set_ylabel("Test Accuracy (%)", fontsize=12)
    ax.set_ylim(0, 100)

    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, yval + 1, f"{yval:.2f}%", 
                ha="center", va="bottom", fontsize=12, fontweight="bold")

    fig.tight_layout()
    save_path = os.path.join(save_dir, f"{dataset_name.lower()}_final_accuracy.png")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"✓ Saved Final Accuracy plot to {save_path}")
    plt.close(fig)

# ==============================================================================
# STEP 4: EXECUTION LOGIC
# ==============================================================================

if __name__ == "__main__":
    print("\n" + "=" * 70)
    print(" GENERATING ALL PLOTS FOR VISION-BDH EXPERIMENTS")
    print("=" * 70 + "\n")

    # --- Generate plots for CIFAR-10 ---
    # POPRAWIONY WARUNEK: Sprawdza, czy załadowano jakiekolwiek dane dla CIFAR-10
    cifar10_data_loaded = any(
        not dataframes.get(data["dataframe_key"], pd.DataFrame()).empty 
        for data in final_results["cifar10"].values()
    )
    
    if cifar10_data_loaded:
        print("--- Analyzing CIFAR-10 Experiment ---")
        plot_learning_curves(final_results["cifar10"], dataframes, "CIFAR-10", IMAGE_DIR)
        plot_final_accuracy(final_results["cifar10"], "CIFAR-10", IMAGE_DIR)
        # You can add the parameter efficiency plot here as well if you like
        print("-" * 40 + "\n")
    else:
        print("--- Skipping CIFAR-10 Analysis (no data found) ---\n")

    # --- Generate plots for CIFAR-100 ---
    cifar100_data_loaded = any(
        not dataframes.get(data["dataframe_key"], pd.DataFrame()).empty
        for data in final_results["cifar100"].values()
    )

    if cifar100_data_loaded:
        print("--- Analyzing CIFAR-100 Experiment ---")
        plot_learning_curves(final_results["cifar100"], dataframes, "CIFAR-100", IMAGE_DIR)
        plot_final_accuracy(final_results["cifar100"], "CIFAR-100", IMAGE_DIR)
        print("-" * 40 + "\n")
    else:
        print("--- Skipping CIFAR-100 Analysis (no data found) ---\n")
        
    print("=" * 70)
    print(" ANALYSIS COMPLETE. Plots are in:", IMAGE_DIR)
    print("=" * 70)