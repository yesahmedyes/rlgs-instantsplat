#!/usr/bin/env python3
"""
Script to collect metrics from hyperparameter experiments and generate CSV files.
"""

import os
import json
import glob
import csv
from pathlib import Path
import argparse


def parse_hyperparameter_string(hyp_string):
    """Parse hyperparameter string to extract individual values."""
    # Format: K{K}_Nlr{N_lr}
    parts = hyp_string.split("_")
    params = {}

    for part in parts:
        if part.startswith("K"):
            params["K"] = int(part[1:])
        elif part.startswith("Nlr"):
            params["N_lr"] = float(part[3:])

    return params


def find_metrics_files(outputs_dir):
    """Find all metrics_by_cam.json files in the outputs directory."""
    # Updated pattern: output_infer_hyp/{dataset}/{scene}/{n_views}_views/{hyp_string}/test/ours_*/renders/metrics_by_cam.json
    pattern = os.path.join(outputs_dir, "*", "*", "*", "*", "test", "ours_*", "renders", "metrics_by_cam.json")
    return glob.glob(pattern)


def collect_metrics(outputs_dir):
    """Collect all metrics from hyperparameter experiments."""
    metrics_files = find_metrics_files(outputs_dir)
    results = []

    for metrics_file in metrics_files:
        try:
            # Parse the path to extract experiment info
            path_parts = Path(metrics_file).parts

            # Find the output_infer_hyp directory index
            outputs_idx = None
            for i, part in enumerate(path_parts):
                if part == "output_infer_hyp":
                    outputs_idx = i
                    break

            if outputs_idx is None:
                continue

            # Extract components: output_infer_hyp/{dataset}/{scene}/{n_views}_views/{hyp_string}/...
            dataset = path_parts[outputs_idx + 1]
            scene = path_parts[outputs_idx + 2]
            n_views_dir = path_parts[outputs_idx + 3]
            hyp_string = path_parts[outputs_idx + 4]

            # Extract number of views
            n_views = n_views_dir.replace("_views", "")

            # Parse hyperparameters
            params = parse_hyperparameter_string(hyp_string)

            # Load metrics
            with open(metrics_file, "r") as f:
                metrics_data = json.load(f)

            if "averages" in metrics_data:
                averages = metrics_data["averages"]

                result = {
                    "experiment": f"{dataset}/{scene}",
                    "hyp_string": hyp_string,
                    "dataset": dataset,
                    "scene": scene,
                    "n_views": n_views,
                    "K": params.get("K", "N/A"),
                    "N_lr": params.get("N_lr", "N/A"),
                    "views": n_views,
                    "ssim": averages.get("ssim", 0.0),
                    "psnr": averages.get("psnr", 0.0),
                    "lpips": averages.get("lpips", 0.0),
                    "metrics_file": metrics_file,
                }
                results.append(result)

        except Exception as e:
            print(f"Error processing {metrics_file}: {e}")
            continue

    return results


def generate_csv_files(results, results_dir):
    """Generate CSV files for each scene and a total CSV file."""

    # Create results directory if it doesn't exist
    os.makedirs(results_dir, exist_ok=True)

    # Group results by scene
    scene_groups = {}
    for result in results:
        scene_key = f"{result['dataset']}/{result['scene']}"
        if scene_key not in scene_groups:
            scene_groups[scene_key] = []
        scene_groups[scene_key].append(result)

    # Group results by hyperparameter configuration for averaging
    hyp_groups = {}
    for result in results:
        hyp_key = result["hyp_string"]
        if hyp_key not in hyp_groups:
            hyp_groups[hyp_key] = []
        hyp_groups[hyp_key].append(result)

    # CSV headers for scene files
    scene_headers = ["K", "N_lr", "views", "ssim", "psnr", "lpips"]

    # Generate CSV files for each scene
    total_experiments = 0
    for scene_key in sorted(scene_groups.keys()):
        scene_results = scene_groups[scene_key]
        # Sort results within each scene by hyperparameters (K first, then others)
        scene_results.sort(key=lambda x: (x["K"], x["N_lr"]))

        total_experiments += len(scene_results)

        # Create safe filename from scene key
        safe_scene_name = scene_key.replace("/", "_")
        scene_csv_file = os.path.join(results_dir, f"{safe_scene_name}.csv")

        with open(scene_csv_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(scene_headers)

            for result in scene_results:
                writer.writerow(
                    [
                        result["K"],
                        result["N_lr"],
                        result["views"],
                        f"{result['ssim']:.4f}",
                        f"{result['psnr']:.2f}",
                        f"{result['lpips']:.4f}",
                    ]
                )

        print(f"Scene CSV written: {scene_csv_file}")

    # Generate total.csv with averaged results across all scenes
    total_csv_file = os.path.join(results_dir, "total.csv")
    total_headers = [
        "K",
        "N_lr",
        "avg_views",
        "avg_ssim",
        "avg_psnr",
        "avg_lpips",
        "scenes_count",
    ]

    with open(total_csv_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(total_headers)

        # Sort hyperparameter configurations by numeric values
        sorted_hyp_configs = sorted(
            hyp_groups.keys(),
            key=lambda hyp: (
                hyp_groups[hyp][0]["K"],
                hyp_groups[hyp][0]["N_lr"],
            ),
        )

        for hyp_config in sorted_hyp_configs:
            hyp_results = hyp_groups[hyp_config]

            # Calculate averages
            avg_ssim = sum(r["ssim"] for r in hyp_results) / len(hyp_results)
            avg_psnr = sum(r["psnr"] for r in hyp_results) / len(hyp_results)
            avg_lpips = sum(r["lpips"] for r in hyp_results) / len(hyp_results)
            avg_views = sum(int(r["views"]) for r in hyp_results) / len(hyp_results)
            scenes_count = len(hyp_results)

            # Get hyperparameter values from the first result
            first_result = hyp_results[0]

            writer.writerow(
                [
                    first_result["K"],
                    first_result["N_lr"],
                    f"{avg_views:.1f}",
                    f"{avg_ssim:.4f}",
                    f"{avg_psnr:.2f}",
                    f"{avg_lpips:.4f}",
                    scenes_count,
                ]
            )

    print(f"Total CSV written: {total_csv_file}")
    print(f"Processed {total_experiments} experiments across {len(scene_groups)} scenes")


def main():
    parser = argparse.ArgumentParser(description="Collect metrics from hyperparameter experiments")
    parser.add_argument("--outputs_dir", type=str, default="outputs", help="Directory containing experiment outputs (default: outputs)")
    parser.add_argument("--results_dir", type=str, default="results", help="Directory to save CSV results (default: results)")

    args = parser.parse_args()

    # Convert to absolute path
    outputs_dir = os.path.abspath(args.outputs_dir)
    results_dir = os.path.abspath(args.results_dir)

    print(f"Looking for metrics in: {outputs_dir}")
    print(f"Results directory: {results_dir}")

    if not os.path.exists(outputs_dir):
        print(f"Error: Outputs directory not found: {outputs_dir}")
        return

    # Collect metrics
    results = collect_metrics(outputs_dir)

    if not results:
        print("No metrics files found. Make sure experiments have been run and rendered.")
        return

    # Generate CSV files
    generate_csv_files(results, results_dir)

    # Print summary to console (organized by scene)
    print("\n" + "=" * 80)
    print("SUMMARY BY SCENE")
    print("=" * 80)

    # Group results by scene for console output
    scene_groups = {}
    for result in results:
        scene_key = f"{result['dataset']}/{result['scene']}"
        if scene_key not in scene_groups:
            scene_groups[scene_key] = []
        scene_groups[scene_key].append(result)

    for scene_key in sorted(scene_groups.keys()):
        print(f"\n{scene_key}:")
        print("-" * 60)
        print(f"{'Hyper Parameters':<40} {'Views':<6} {'SSIM':<8} {'PSNR':<8} {'LPIPS':<8}")
        print("-" * 60)

        # Sort by numeric values for console output too
        sorted_results = sorted(scene_groups[scene_key], key=lambda x: (x["K"], x["N_lr"]))
        for result in sorted_results:
            hyp_params = f"K={result['K']},N_lr={result['N_lr']}"
            print(f"{hyp_params:<40} {result['views']:<6} {result['ssim']:<8.4f} {result['psnr']:<8.2f} {result['lpips']:<8.4f}")


if __name__ == "__main__":
    main()
