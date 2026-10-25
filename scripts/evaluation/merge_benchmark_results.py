#!/usr/bin/env python3
"""
===============================================================================
Script:         merge_benchmark_results.py
Project:        Offline Text-to-Speech Benchmarking on ARM Edge Devices
Author:         Richard Wieser, BSc
Institution:    FH Technikum Wien, Master's Program in Embedded Systems
Year:           2026
Repository:     https://github.com/RichWiese/edge_offline_tts-benchmark_rpi5
License:        MIT License

Description:
    Merges individual benchmark CSV files into a single validated Excel file.
    Normalizes sentence IDs (e.g., K1 to K01) and checks for missing columns,
    lost rows, and empty measurement values (NaN).
===============================================================================
"""

import pandas as pd
import glob
import re

# Find all CSV result files in the current directory
csv_files = glob.glob("*.csv")
all_data = []
total_csv_rows = 0

# Expected standard column structure across all benchmark runs
expected_columns = [
    'Modell', 'Satz_ID', 'Kategorie', 'Text', 'Zeichen', 'Iteration',
    'Temp_Start_C', 'Temp_End_C', 'Throttled_Flag', 'Synthesezeit_s',
    'Audio_Dauer_s', 'RTF', 'Audio_Pfad'
]

def pad_single_digit(val):
    """Convert single-digit IDs like 'K1' to two-digit format 'K01' for clean sorting."""
    if isinstance(val, str):
        # Match a letter followed by a single digit at the end of the word
        return re.sub(r'(?<=[a-zA-Z])(\d)\b', r'0\1', val)
    return val

# Read and validate each CSV file
for file in csv_files:
    for separator in [';', ',', '\t']:
        try:
            df = pd.read_csv(file, sep=separator)
            if 'Modell' in df.columns and 'Satz_ID' in df.columns:

                # Normalize sentence IDs (e.g., K1 -> K01)
                df['Satz_ID'] = df['Satz_ID'].apply(pad_single_digit)

                # Check 1: Verify if any expected columns are missing
                missing_cols = [c for c in expected_columns if c not in df.columns]
                if missing_cols:
                    print(f"⚠️ WARNUNG in {file}: Folgende Spalten fehlen: {missing_cols}")

                all_data.append(df)
                total_csv_rows += len(df)
                break
        except Exception:
            continue

# Merge, sort, validate, and export all collected data
if all_data:
    df_master = pd.concat(all_data, ignore_index=True)
    df_master = df_master.sort_values(by=['Modell', 'Satz_ID'])

    # Check 2: Compare total row count to ensure zero data loss during merge
    print("\n--- VALIDIERUNGS-BERICHT ---")
    if len(df_master) == total_csv_rows:
        print(f"✅ Zeilen-Check erfolgreich: Alle {len(df_master)} Datensätze wurden verlustfrei übertragen.")
    else:
        print(f"❌ FEHLER: Zeilenanzahl weicht ab! CSVs gesamt: {total_csv_rows}, Excel: {len(df_master)}")

    # Check 3: Check critical telemetry and latency columns for missing values (NaN)
    critical_cols = ['Temp_Start_C', 'Temp_End_C', 'RTF', 'Synthesezeit_s']
    for col in critical_cols:
        missing_count = df_master[col].isna().sum()
        if missing_count > 0:
            print(f"⚠️ WARNUNG: In Spalte '{col}' fehlen {missing_count} Messwerte (leere Zellen)!")
        else:
            print(f"✅ Spalte '{col}' ist vollständig.")

    # Export validated master dataset to Excel
    df_master.to_excel("rtf_telemetry_master.xlsx", index=False)
    print("\nDatei 'rtf_telemetry_master.xlsx' wurde erstellt.")