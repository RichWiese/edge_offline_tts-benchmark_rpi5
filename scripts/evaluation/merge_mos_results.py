#!/usr/bin/env python3
"""
===============================================================================
Script:         merge_mos_results.py
Project:        Offline Text-to-Speech Benchmarking on ARM Edge Devices
Author:         Richard Wieser, BSc
Institution:    FH Technikum Wien, Master's Program in Embedded Systems
Year:           2026
Repository:     https://github.com/RichWiese/edge_offline_tts-benchmark_rpi5
License:        MIT License

Description:
    Merges individual MOS prediction CSV files (NISQA/UTMOSv2) into a single 
    validated Excel file. Normalizes prompt IDs and checks for missing data.
===============================================================================
"""

import os
import glob
import re
import pandas as pd

csv_files = glob.glob("*.csv")
all_data = []
total_csv_rows = 0

# Expected column structure for MOS prediction data
expected_columns = ["Prompt_ID", "Kategorie", "deg", "mos_pred", "model"]

def pad_single_digit(val):
    """Convert single-digit IDs like 'P1' to two-digit format 'P01' for sorting."""
    if isinstance(val, str):
        return re.sub(r"(?<=[a-zA-Z])(\d)\b", r"0\1", val)
    return val

for file in csv_files:
    for separator in [";", ",", "\t"]:
        try:
            df = pd.read_csv(file, sep=separator)
            if "model" in df.columns and "Prompt_ID" in df.columns:

                # Normalize prompt IDs
                df["Prompt_ID"] = df["Prompt_ID"].apply(pad_single_digit)

                # Check 1: Verify missing columns
                missing_cols = [c for c in expected_columns if c not in df.columns]
                if missing_cols:
                    print(f"⚠️ WARNUNG in {file}: Folgende Spalten fehlen: {missing_cols}")

                all_data.append(df)
                total_csv_rows += len(df)
                break
        except Exception:
            continue

if all_data:
    df_master = pd.concat(all_data, ignore_index=True)
    df_master = df_master.sort_values(by=["model", "Prompt_ID"])

    vorhandene_cols = [c for c in expected_columns if c in df_master.columns]
    andere_cols = [c for c in df_master.columns if c not in expected_columns]
    df_master = df_master[vorhandene_cols + andere_cols]

    # Check 2: Row count loss validation
    print("\n--- MOS VALIDIERUNGS-BERICHT ---")
    if len(df_master) == total_csv_rows:
        print(f"✅ Zeilen-Check erfolgreich: Alle {len(df_master)} Datensätze wurden verlustfrei übertragen.")
    else:
        print(f"❌ FEHLER: Zeilenanzahl weicht ab! CSVs gesamt: {total_csv_rows}, Excel: {len(df_master)}")

    # Check 3: Check critical MOS columns for missing values (NaN)
    critical_cols = ["deg", "mos_pred", "model"]
    for col in critical_cols:
        if col in df_master.columns:
            missing_count = df_master[col].isna().sum()
            if missing_count > 0:
                print(f"⚠️ WARNUNG: In Spalte '{col}' fehlen {missing_count} Messwerte (leere Zellen)!")
            else:
                print(f"✅ Spalte '{col}' ist vollständig.")

    # Export to distinct MOS Excel file
    ausgabe_name = "mos_nisqa_utmos_predictions.xlsx"
    df_master.to_excel(ausgabe_name, index=False)
    print(f"\nDatei '{ausgabe_name}' wurde erfolgreich erstellt.")
else:
    print("❌ Keine passenden MOS-CSV-Dateien gefunden.")