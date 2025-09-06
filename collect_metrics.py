#!/usr/bin/env python3
"""
Script to collect metrics from hyperparameter experiments and generate a markdown table.
"""

import os
import json
import glob
from pathlib import Path
import argparse


def parse_hyperparameter_string(hyp_string):
    """Parse hyperparameter string to extract individual values."""
    # Format: K{K}_Nlr{N_lr}_ri{reshuffle_interval}_rsl{reward_set_len}_ec{entropy_coef}
    parts = hyp_string.split("_")
    params = {}

    for part in parts:
        if part.startswith("K"):
            params["K"] = part[1:]
        elif part.startswith("Nlr"):
            params["N_lr"] = part[3:]
        elif part.startswith("ri"):
            params["reshuffle_interval"] = part[2:]
        elif part.startswith("rsl"):
            params["reward_set_len"] = part[3:]
        elif part.startswith("ec"):
            params["entropy_coef"] = part[2:]

    return params


def find_metrics_files(outputs_dir):
    """Find all metrics_by_cam.json files in the outputs directory."""
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

            # Find the outputs directory index
            outputs_idx = None
            for i, part in enumerate(path_parts):
                if part == "outputs":
                    outputs_idx = i
                    break

            if outputs_idx is None:
                continue

            # Extract components: outputs/{hyp_string}/{dataset}/{scene}/{n_views}_views/...
            hyp_string = path_parts[outputs_idx + 1]
            dataset = path_parts[outputs_idx + 2]
            scene = path_parts[outputs_idx + 3]
            n_views_dir = path_parts[outputs_idx + 4]

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
                    "reshuffle_interval": params.get("reshuffle_interval", "N/A"),
                    "reward_set_len": params.get("reward_set_len", "N/A"),
                    "entropy_coef": params.get("entropy_coef", "N/A"),
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


def generate_markdown_table(results, output_file):
    """Generate a markdown table with the collected metrics, organized by scene."""

    # Group results by scene
    scene_groups = {}
    for result in results:
        scene_key = f"{result['dataset']}/{result['scene']}"
        if scene_key not in scene_groups:
            scene_groups[scene_key] = []
        scene_groups[scene_key].append(result)

    # Sort scenes alphabetically
    sorted_scenes = sorted(scene_groups.keys())

    markdown_content = """# Hyperparameter Experiment Results

## Results by Scene

"""

    total_experiments = 0

    # Create a table for each scene
    for scene_key in sorted_scenes:
        scene_results = scene_groups[scene_key]
        # Sort results within each scene by hyperparameter string
        scene_results.sort(key=lambda x: x["hyp_string"])

        total_experiments += len(scene_results)

        markdown_content += f"### {scene_key}\n\n"
        markdown_content += "| Hyper Parameters | Views | SSIM | PSNR | LPIPS |\n"
        markdown_content += "|------------------|-------|------|------|-------|\n"

        for result in scene_results:
            hyp_params = f"K={result['K']}, N_lr={result['N_lr']}, ri={result['reshuffle_interval']}, rsl={result['reward_set_len']}, ec={result['entropy_coef']}"

            markdown_content += (
                f"| {hyp_params} | {result['n_views']} | {result['ssim']:.4f} | {result['psnr']:.2f} | {result['lpips']:.4f} |\n"
            )

        markdown_content += "\n"

        # Add detailed breakdown for this scene
        markdown_content += "#### Detailed Results\n\n"
        for result in scene_results:
            markdown_content += f"**{result['hyp_string']}** ({result['n_views']} views)\n"
            markdown_content += f"- SSIM: {result['ssim']:.6f}\n"
            markdown_content += f"- PSNR: {result['psnr']:.4f}\n"
            markdown_content += f"- LPIPS: {result['lpips']:.6f}\n"
            markdown_content += f"- Metrics file: `{result['metrics_file']}`\n\n"

        markdown_content += "---\n\n"

    # Add summary at the end
    markdown_content += f"## Summary\n\n"
    markdown_content += f"Total scenes: {len(sorted_scenes)}\n"
    markdown_content += f"Total experiments processed: {total_experiments}\n\n"

    # Add overall best results
    if results:
        best_ssim = max(results, key=lambda x: x["ssim"])
        best_psnr = max(results, key=lambda x: x["psnr"])
        best_lpips = min(results, key=lambda x: x["lpips"])

        markdown_content += "### Best Results Overall\n\n"
        markdown_content += f"- **Best SSIM**: {best_ssim['ssim']:.6f} ({best_ssim['experiment']}, {best_ssim['hyp_string']})\n"
        markdown_content += f"- **Best PSNR**: {best_psnr['psnr']:.4f} ({best_psnr['experiment']}, {best_psnr['hyp_string']})\n"
        markdown_content += f"- **Best LPIPS**: {best_lpips['lpips']:.6f} ({best_lpips['experiment']}, {best_lpips['hyp_string']})\n"

    # Write to file
    with open(output_file, "w") as f:
        f.write(markdown_content)

    print(f"Markdown table written to: {output_file}")
    print(f"Processed {total_experiments} experiments across {len(sorted_scenes)} scenes")


def main():
    parser = argparse.ArgumentParser(description="Collect metrics from hyperparameter experiments")
    parser.add_argument("--outputs_dir", type=str, default="outputs", help="Directory containing experiment outputs (default: outputs)")
    parser.add_argument(
        "--output_file", type=str, default="experiment_results.md", help="Output markdown file (default: experiment_results.md)"
    )

    args = parser.parse_args()

    # Convert to absolute path
    outputs_dir = os.path.abspath(args.outputs_dir)
    output_file = os.path.abspath(args.output_file)

    print(f"Looking for metrics in: {outputs_dir}")
    print(f"Output file: {output_file}")

    if not os.path.exists(outputs_dir):
        print(f"Error: Outputs directory not found: {outputs_dir}")
        return

    # Collect metrics
    results = collect_metrics(outputs_dir)

    if not results:
        print("No metrics files found. Make sure experiments have been run and rendered.")
        return

    # Generate markdown table
    generate_markdown_table(results, output_file)

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

        scene_results = sorted(scene_groups[scene_key], key=lambda x: x["hyp_string"])
        for result in scene_results:
            hyp_params = f"K={result['K']},N_lr={result['N_lr']},ri={result['reshuffle_interval']},rsl={result['reward_set_len']},ec={result['entropy_coef']}"
            print(f"{hyp_params:<40} {result['n_views']:<6} {result['ssim']:<8.4f} {result['psnr']:<8.2f} {result['lpips']:<8.4f}")


if __name__ == "__main__":
    main()
