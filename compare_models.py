"""
compare_models.py - Train and Compare All Models
==================================================
Trains all available model architectures sequentially on the specified
dataset and generates a comparison report.

Usage:
    python compare_models.py --dataset fer2013
    python compare_models.py --dataset fer2013 --epochs 30
"""

import os
import sys
import json
import time
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
from train import Trainer


# All models to compare
ALL_MODELS = ["mini_xception", "efficientnet", "resnet", "hsemotion"]


def train_single_model(model_name, dataset_name, epochs=None):
    """
    Train a single model and return its results.
    
    Returns:
        dict: {model_name, best_val_acc, train_time, epochs_trained, history_path}
    """
    print("\n" + "=" * 70)
    print(f"  TRAINING: {model_name.upper()} on {dataset_name.upper()}")
    print("=" * 70)

    # Override epochs if specified
    original_epochs = config.EPOCHS
    if epochs is not None:
        config.EPOCHS = epochs

    start_time = time.time()

    try:
        trainer = Trainer(dataset_name=dataset_name, model_name=model_name)
        history = trainer.train()

        train_time = time.time() - start_time
        best_val_acc = trainer.best_val_acc
        epochs_trained = len(history['val_acc'])

        result = {
            "model_name": model_name,
            "dataset": dataset_name,
            "best_val_acc": best_val_acc,
            "final_train_acc": history['train_acc'][-1] if history['train_acc'] else 0,
            "final_val_acc": history['val_acc'][-1] if history['val_acc'] else 0,
            "final_train_loss": history['train_loss'][-1] if history['train_loss'] else 0,
            "final_val_loss": history['val_loss'][-1] if history['val_loss'] else 0,
            "epochs_trained": epochs_trained,
            "train_time_seconds": train_time,
            "train_time_formatted": f"{int(train_time // 60)}m {int(train_time % 60)}s",
            "status": "success"
        }

        # Model-specific info
        model_cfg = config.MODEL_CONFIGS.get(model_name, {})
        result["img_size"] = model_cfg.get("img_size", 48)
        result["num_channels"] = model_cfg.get("num_channels", 1)
        result["lr"] = model_cfg.get("lr", 0.001)
        result["batch_size"] = model_cfg.get("batch_size", 64)

        # Parameter counts
        total_params = sum(p.numel() for p in trainer.model.parameters())
        trainable_params = sum(
            p.numel() for p in trainer.model.parameters() if p.requires_grad
        )
        result["total_params"] = total_params
        result["trainable_params"] = trainable_params

    except Exception as e:
        train_time = time.time() - start_time
        result = {
            "model_name": model_name,
            "dataset": dataset_name,
            "best_val_acc": 0,
            "status": "failed",
            "error": str(e),
            "train_time_seconds": train_time,
            "train_time_formatted": f"{int(train_time // 60)}m {int(train_time % 60)}s"
        }
        print(f"\n[ERROR] {model_name} training failed: {e}")

    # Restore epochs
    config.EPOCHS = original_epochs

    return result


def print_comparison_table(results):
    """Print a formatted comparison table."""
    print("\n" + "=" * 90)
    print("  MODEL COMPARISON RESULTS")
    print("=" * 90)

    # Header
    header = f"{'Model':<16} {'Val Acc':>10} {'Train Acc':>10} {'Params':>12} {'Time':>10} {'Status':>8}"
    print(f"\n{header}")
    print("-" * 90)

    # Sort by best_val_acc
    sorted_results = sorted(results, key=lambda x: x.get("best_val_acc", 0), reverse=True)

    for r in sorted_results:
        if r["status"] == "success":
            params_str = f"{r.get('trainable_params', 0):,}"
            print(f"  {r['model_name']:<14} {r['best_val_acc']:>9.2f}% "
                  f"{r.get('final_train_acc', 0):>9.2f}% "
                  f"{params_str:>12} "
                  f"{r['train_time_formatted']:>10} "
                  f"{'OK':>8}")
        else:
            print(f"  {r['model_name']:<14} {'N/A':>10} {'N/A':>10} "
                  f"{'N/A':>12} {r['train_time_formatted']:>10} {'FAIL':>8}")

    print("-" * 90)

    # Best model
    best = sorted_results[0] if sorted_results else None
    if best and best["status"] == "success":
        print(f"\n  >> BEST MODEL: {best['model_name']} "
              f"({best['best_val_acc']:.2f}% val accuracy)")

    print("=" * 90 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Train and Compare All Models")
    parser.add_argument("--dataset", type=str, default="fer2013",
                        choices=["fer2013", "ferplus", "rafdb", "ckplus"],
                        help="Dataset to train on (default: fer2013)")
    parser.add_argument("--epochs", type=int, default=None,
                        help="Override max epochs (default: use config)")
    parser.add_argument("--models", nargs="+", default=None,
                        choices=ALL_MODELS,
                        help="Specific models to train (default: all)")
    args = parser.parse_args()

    models_to_train = args.models or ALL_MODELS
    dataset_name = args.dataset

    print("\n" + "=" * 70)
    print("  MODEL COMPARISON PIPELINE")
    print(f"  Dataset: {dataset_name} | Models: {', '.join(models_to_train)}")
    print("=" * 70)
    print(f"\n  Device: {config.DEVICE}")
    print(f"  Max epochs: {args.epochs or config.EPOCHS}")

    # Train each model
    all_results = []
    total_start = time.time()

    for i, model_name in enumerate(models_to_train):
        print(f"\n\n{'#' * 70}")
        print(f"  [{i+1}/{len(models_to_train)}] Training {model_name.upper()}")
        print(f"{'#' * 70}")

        result = train_single_model(model_name, dataset_name, args.epochs)
        all_results.append(result)

        # Print interim result
        if result["status"] == "success":
            print(f"\n  >> {model_name}: {result['best_val_acc']:.2f}% "
                  f"in {result['train_time_formatted']}")
        else:
            print(f"\n  >> {model_name}: FAILED")

    # Total time
    total_time = time.time() - total_start
    total_minutes = int(total_time // 60)
    total_seconds = int(total_time % 60)

    # Print comparison
    print_comparison_table(all_results)
    print(f"  Total training time: {total_minutes}m {total_seconds}s\n")

    # Save results to JSON
    results_path = os.path.join(
        config.OUTPUT_DIR,
        f"model_comparison_{dataset_name}.json"
    )
    with open(results_path, 'w') as f:
        json.dump({
            "dataset": dataset_name,
            "total_time_seconds": total_time,
            "results": all_results
        }, f, indent=2)
    print(f"  Results saved: {results_path}")


if __name__ == "__main__":
    main()
