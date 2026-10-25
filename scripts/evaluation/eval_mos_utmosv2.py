#!/usr/bin/env python3
"""
===============================================================================
Script:         eval_mos_utmosv2.py
Project:        Offline Text-to-Speech Benchmarking on ARM Edge Devices
Author:         Richard Wieser, BSc
Institution:    FH Technikum Wien, Master's Program in Embedded Systems
Year:           2026
Repository:     https://github.com/RichWiese/edge_offline_tts-benchmark_rpi5
License:        MIT License

Description:
    Automates speech naturalness evaluation (MOS) across generated WAV
    directories using UTMOSv2 on CPU. Masks CUDA devices for reproducible CPU
    execution, enriches predictions with sentence IDs and linguistic categories,
    and exports detailed and summary CSV files.
===============================================================================
"""

import os

# Mask CUDA devices before importing PyTorch to force deterministic CPU inference
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import re
import sys
from pathlib import Path
import pandas as pd
import torch
import utmosv2
from tqdm import tqdm

# =============================================================================
# CONFIGURATION
# =============================================================================
BASE_DIR = Path("../").resolve()
DEVICE = "cpu"

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

        # Skip the evaluation script directory and hidden folders (e.g., .git)
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


def evaluate_directory(model, audio_dir: Path):
    """Run UTMOSv2 CPU inference on all WAV files inside a directory."""
    wav_files = sorted(
        [p for p in audio_dir.iterdir() if p.suffix.lower() == ".wav"]
    )
    records = []

    for wav_path in tqdm(wav_files, desc="Inferenz (CPU)", leave=False):
        prompt_id = extract_prompt_id(wav_path.name)
        kategorie = CATEGORY_MAP.get(prompt_id, "Sonstige")

        try:
            # Explicitly run prediction on CPU
            score = model.predict(input_path=str(wav_path), device=DEVICE)

            if hasattr(score, "item"):
                mos_val = float(score.item())
            elif isinstance(score, (list, tuple)):
                mos_val = float(score[0])
            elif isinstance(score, dict):
                mos_val = float(
                    score.get("mos", score.get("score", list(score.values())[0]))
                )
            else:
                mos_val = float(score)
        except Exception as e:
            print(
                f"\n[Fehler bei Datei] {wav_path.name}: {e}", file=sys.stderr
            )
            mos_val = None

        records.append(
            {
                "Prompt_ID": prompt_id,
                "Kategorie": kategorie,
                "deg": wav_path.name,
                "mos_pred": round(mos_val, 4) if mos_val is not None else None,
                "model": "UTMOSv2",
            }
        )

    return pd.DataFrame(records)


def generate_summary(df_results: pd.DataFrame, summary_csv_path: Path):
    """Create an aggregated summary CSV per category (mean and standard deviation)."""
    summary_rows = []

    for cat, group in df_results.groupby("Kategorie"):
        valid_scores = group["mos_pred"].dropna()
        summary_rows.append(
            {
                "Kategorie": cat,
                "Anzahl": len(group),
                "MOS_Mean": round(valid_scores.mean(), 4)
                if not valid_scores.empty
                else None,
                "MOS_Std": round(valid_scores.std(ddof=0), 4)
                if len(valid_scores) > 1
                else 0.0,
            }
        )

    df_summary = pd.DataFrame(summary_rows)

    all_scores = df_results["mos_pred"].dropna()
    gesamt_row = {
        "Kategorie": "GESAMT",
        "Anzahl": len(df_results),
        "MOS_Mean": round(all_scores.mean(), 4)
        if not all_scores.empty
        else None,
        "MOS_Std": round(all_scores.std(ddof=0), 4)
        if len(all_scores) > 1
        else 0.0,
    }

    df_summary = pd.concat(
        [df_summary, pd.DataFrame([gesamt_row])], ignore_index=True
    )
    df_summary.to_csv(summary_csv_path, sep=";", index=False)
    print(f"  -> Zusammenfassung:   {summary_csv_path.name}")


def main():
    print(f"Lade UTMOSv2 Modell strikt auf: {DEVICE} ...")

    # Initialize pretrained UTMOSv2 model on CPU in evaluation mode
    try:
        model = utmosv2.create_model(pretrained=True, device=DEVICE)
    except TypeError:
        model = utmosv2.create_model(pretrained=True)

    if hasattr(model, "to"):
        model = model.to(DEVICE)
    if hasattr(model, "eval"):
        model.eval()

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

        dest_results_csv = parent_dir / f"{subfolder_name}_utmosv2_results.csv"
        dest_summary_csv = parent_dir / f"{subfolder_name}_utmosv2_summary.csv"

        print(f"\n{'='*70}")
        print(f"EVALUIERE: {os.path.relpath(audio_dir, BASE_DIR)}")
        print(f"Zielordner: {parent_dir}")
        print(f"{'='*70}")

        df_results = evaluate_directory(model, audio_dir)

        df_results.to_csv(dest_results_csv, sep=";", index=False)
        print(f"  -> Detail-Ergebnisse: {dest_results_csv.name}")

        generate_summary(df_results, dest_summary_csv)

    print(f"\n{'='*70}")
    print("UTMOSv2-Evaluation für alle Verzeichnisse erfolgreich beendet.")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
