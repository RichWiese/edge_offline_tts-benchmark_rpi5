#!/usr/bin/env python3
"""
===============================================================================
Script:         eval_mos_nisqa.py
Project:        Offline Text-to-Speech Benchmarking on ARM Edge Devices
Author:         Richard Wieser, BSc
Institution:    FH Technikum Wien, Master's Program in Embedded Systems
Year:           2026
Repository:     https://github.com/RichWiese/edge_offline_tts-benchmark_rpi5
License:        MIT License

Description:
    Automates acoustic speech quality evaluation (MOS) across generated WAV
    directories using NISQA (v2.0). Executes the official NISQA 'run_predict.py'
    script via subprocess, enriches raw predictions with sentence IDs and
    linguistic categories, and exports detailed and summary CSV files.

    Note: Must be executed from within a cloned NISQA repository containing
    'run_predict.py' and the pretrained 'nisqa_tts.tar' model weights.
===============================================================================
"""

import os
import re
import subprocess
import sys
from pathlib import Path
import pandas as pd

# =============================================================================
# CONFIGURATION
# =============================================================================
BASE_DIR = Path("../").resolve()

# Path to pretrained NISQA TTS weights
MODEL_PATH = Path("weights/nisqa_tts.tar")
if not MODEL_PATH.exists():
    MODEL_PATH = Path("nisqa_tts.tar")

# PyTorch DataLoader batch size and worker threads
BATCH_SIZE = 32
NUM_WORKERS = 6

# Mapping of corpus sentence IDs to their linguistic test categories
CATEGORY_MAP = {
    **{f"K{i}": "Kurzbefehl" for i in range(1, 11)},
    **{f"M{i}": "Standard" for i in range(1, 11)},
    **{f"H{i}": "Kompositum" for i in range(1, 6)},
    **{f"H{i}": "Zahlen_Einheiten" for i in range(6, 11)},
    "H11": "Homograph",
    "H12": "Homograph",
    "H13": "Abkuerzung",
    "H14": "Abkuerzung",
    "H15": "Abkuerzung",
    "H16": "Intonation",
    "H17": "Zahlen_Einheiten",
    "H18": "Anglizismen",
    **{f"L{i}": "Langtext" for i in range(1, 6)},
}
# =============================================================================


def find_all_wav_directories(base_dir: Path, exclude_dir: Path):
    """Recursively find all subdirectories that directly contain .wav files."""
    wav_dirs = []
    for root, _, files in os.walk(base_dir):
        root_path = Path(root).resolve()

        # Skip the NISQA directory itself and hidden folders (e.g., .git)
        if exclude_dir in root_path.parents or root_path == exclude_dir:
            continue
        if any(part.startswith(".") for part in root_path.parts):
            continue

        if any(f.lower().endswith(".wav") for f in files):
            wav_dirs.append(root_path)

    return sorted(wav_dirs)


def extract_prompt_id(filename: str) -> str:
    """Extract the sentence ID (e.g., K1, M4, H16, L2) from the WAV filename."""
    match = re.search(r"(?:^|[_.-])([KMHL]\d{1,2})(?:$|[_.-])", filename)
    return match.group(1) if match else "Unbekannt"


def enrich_csv_results(raw_csv_path: Path, target_csv_path: Path):
    """Read raw NISQA CSV output, add Prompt_ID and Kategorie, and save as CSV."""
    if not raw_csv_path.exists():
        return False

    df = pd.read_csv(raw_csv_path)

    # Add sentence ID and category metadata
    df["Prompt_ID"] = df["deg"].apply(extract_prompt_id)
    df["Kategorie"] = df["Prompt_ID"].map(CATEGORY_MAP).fillna("Sonstige")

    # Move metadata columns to the front
    other_cols = [c for c in df.columns if c not in ["Prompt_ID", "Kategorie"]]
    df = df[["Prompt_ID", "Kategorie"] + other_cols]

    # Save semicolon-separated CSV in the target directory
    df.to_csv(target_csv_path, sep=";", index=False)
    print(f"  -> Detail-Ergebnisse: {target_csv_path.name}")
    return True


def generate_summary(results_csv_path: Path, summary_csv_path: Path):
    """Create an aggregated summary CSV per category (mean and standard deviation)."""
    if not results_csv_path.exists():
        return

    df = pd.read_csv(results_csv_path, sep=";")

    metric_cols = ["mos_pred", "noi_pred", "dis_pred", "col_pred"]
    available_metrics = [m for m in metric_cols if m in df.columns]

    summary_rows = []

    # 1. Compute mean and std per linguistic category
    for cat, group in df.groupby("Kategorie"):
        row = {
            "Kategorie": cat,
            "Anzahl": len(group),
            "MOS_Mean": round(group["mos_pred"].mean(), 4),
            "MOS_Std": round(group["mos_pred"].std(ddof=0), 4)
            if len(group) > 1
            else 0.0,
        }
        for m in available_metrics:
            if m != "mos_pred":
                short_name = m.split("_")[0].capitalize()
                row[f"{short_name}_Mean"] = round(group[m].mean(), 4)
        summary_rows.append(row)

    df_summary = pd.DataFrame(summary_rows)

    # 2. Compute overall summary row across all evaluated samples
    gesamt_row = {
        "Kategorie": "GESAMT",
        "Anzahl": len(df),
        "MOS_Mean": round(df["mos_pred"].mean(), 4),
        "MOS_Std": round(df["mos_pred"].std(ddof=0), 4)
        if len(df) > 1
        else 0.0,
    }
    for m in available_metrics:
        if m != "mos_pred":
            short_name = m.split("_")[0].capitalize()
            gesamt_row[f"{short_name}_Mean"] = round(df[m].mean(), 4)

    df_summary = pd.concat(
        [df_summary, pd.DataFrame([gesamt_row])], ignore_index=True
    )

    df_summary.to_csv(summary_csv_path, sep=";", index=False)
    print(f"  -> Zusammenfassung:   {summary_csv_path.name}")


def main():
    if not MODEL_PATH.exists():
        print(
            f"[Fehler] Modellgewichte nicht gefunden unter: {MODEL_PATH.resolve()}"
        )
        sys.exit(1)

    current_dir = Path.cwd().resolve()
    print(f"Scanne Basisverzeichnis: {BASE_DIR}")

    wav_directories = find_all_wav_directories(BASE_DIR, exclude_dir=current_dir)

    if not wav_directories:
        print(f"Keine Ordner mit .wav-Dateien unter {BASE_DIR} gefunden.")
        return

    print(f"{len(wav_directories)} WAV-Verzeichnisse gefunden:\n")
    for d in wav_directories:
        print(f"  • {os.path.relpath(d, BASE_DIR)}")

    for audio_dir in wav_directories:
        parent_dir = audio_dir.parent
        subfolder_name = audio_dir.name

        # Define output file paths in the parent model directory
        dest_results_csv = parent_dir / f"{subfolder_name}_nisqa_results.csv"
        dest_summary_csv = parent_dir / f"{subfolder_name}_nisqa_summary.csv"

        print(f"\n{'='*70}")
        print(f"EVALUIERE: {os.path.relpath(audio_dir, BASE_DIR)}")
        print(f"Zielordner: {parent_dir}")
        print(f"{'='*70}")

        # Run NISQA prediction (writes temporary NISQA_results.csv into audio_dir)
        cmd = [
            sys.executable,
            "run_predict.py",
            "--mode",
            "predict_dir",
            "--pretrained_model",
            str(MODEL_PATH),
            "--data_dir",
            str(audio_dir),
            "--output_dir",
            str(audio_dir),
            "--bs",
            str(BATCH_SIZE),
            "--num_workers",
            str(NUM_WORKERS),
        ]

        result = subprocess.run(cmd)
        raw_csv = audio_dir / "NISQA_results.csv"

        if result.returncode == 0 and raw_csv.exists():
            # 1. Enrich raw results and save to parent folder
            success = enrich_csv_results(raw_csv, dest_results_csv)
            # 2. Generate category summary CSV
            if success:
                generate_summary(dest_results_csv, dest_summary_csv)
            # 3. Remove temporary raw file from WAV folder
            raw_csv.unlink()
        else:
            print(f"[Fehler] Inferenz fehlgeschlagen für: {audio_dir}")

    print(f"\n{'='*70}")
    print("Fertig! Alle CSV-Dateien liegen in den jeweiligen Modellordnern.")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
