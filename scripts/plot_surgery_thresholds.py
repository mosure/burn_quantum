#!/usr/bin/env python3
"""Render the recorded fixed-shot scientific results (requires Matplotlib)."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def panel(axis, study, title, xlabel, basis=None, instrument=None):
    cells = study["cells"]
    if basis:
        cells = [cell for cell in cells if cell["basis"] == basis]
    if instrument:
        cells = [cell for cell in cells if cell["instrument"] == instrument]
    local = basis or instrument
    size_name = "distance" if local else "size"
    sizes = sorted({cell[size_name] for cell in cells})
    colors = ["#2563eb", "#0891b2", "#059669", "#d97706", "#be185d"]
    for color, size in zip(colors, sizes):
        selected = [cell for cell in cells if cell[size_name] == size]
        selected.sort(key=lambda cell: cell.get("probability", cell.get("noise")))
        x, y, lo, hi = [], [], [], []
        for cell in selected:
            counts = cell["study"]["counts"] if local else cell["counts"]
            failures = counts["logical_failures"] + counts["decoder_limits"] + counts["inconsistent"]
            rate = failures / counts["attempts"]
            x.append(100 * cell.get("probability", cell.get("noise")))
            y.append(rate)
            lo.append(rate - cell["failure_interval"]["lower"])
            hi.append(cell["failure_interval"]["upper"] - rate)
        axis.errorbar(x, y, yerr=[lo, hi], marker="o", markersize=3.5,
                      linewidth=1.5, capsize=2, color=color,
                      label=f'{"d" if local else "L"} = {size}')
    bracket = study["finite_size_brackets"][local] if local else study["finite_size_bracket"]
    if bracket:
        axis.axvspan(100 * bracket["lower"], 100 * bracket["upper"],
                     color="#64748b", alpha=0.10, zorder=0)
    axis.set_title(title, loc="left", fontsize=11, fontweight="bold")
    axis.set_xlabel(xlabel)
    axis.set_ylabel("Unconditional logical failure probability")
    axis.set_ylim(-0.025, 1.025)
    axis.grid(alpha=0.18)
    axis.spines[["top", "right"]].set_visible(False)
    axis.legend(frameon=False, fontsize=8, loc="upper left")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    data = json.loads(args.results.read_text())["studies"]
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9,
                         "svg.fonttype": "none", "axes.labelcolor": "#243247",
                         "text.color": "#182235"})
    fig, axes = plt.subplots(3, 2, figsize=(11.8, 11.6), constrained_layout=True)
    panel(axes[0, 0], data["six-ring-qualification"],
          "a  Periodic six-ring: independent outcome erasure", "Erased XX/ZZ outcome probability (%)")
    panel(axes[0, 1], data["optical-qualification"],
          "b  Encoded optical model: ideal Shor sources, Bell boost", "Independent per-photon loss probability (%)")
    panel(axes[1, 0], data["surgery-threshold"],
          "c  ZZ merge/split: noisy data and check records", "Binary data / readout fault probability p (%)", "ZZ")
    panel(axes[1, 1], data["surgery-threshold"],
          "d  XX merge/split: noisy data and check records", "Binary data / readout fault probability p (%)", "XX")
    panel(axes[2, 0], data["surface-confirmation"],
          "e  Actual six-ring memory: no accepted bracket", "Physical outcome flip probability p (%)", instrument="memory")
    panel(axes[2, 1], data["surface-threshold"],
          "f  Actual six-ring merge: perfect observed closure", "Physical outcome flip probability p (%)", instrument="merge")
    fig.suptitle("Local simulation: finite-size threshold qualification", fontsize=15, fontweight="bold")
    fig.supxlabel("Fixed trials per point: a–d 20,000 · e 100,000 · f 50,000. Simultaneous 99% intervals within each family.\n"
                  "Shading: resolved finite-size bracket. Lines guide the eye; no asymptotic fit or hardware claim.",
                  fontsize=9)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output)
    if args.output.suffix.lower() == ".svg":
        # Matplotlib adds trailing spaces to multiline path attributes.
        args.output.write_text("\n".join(line.rstrip() for line in args.output.read_text().splitlines()) + "\n")
    fig.savefig(args.output.with_suffix(".png"), dpi=150)


if __name__ == "__main__":
    main()
