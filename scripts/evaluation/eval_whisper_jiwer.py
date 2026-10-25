#!/usr/bin/env python3
"""
===============================================================================
Script:         eval_whisper_wer-cer.py
Project:        Offline Text-to-Speech Benchmarking on ARM Edge Devices
Author:         Richard Wieser, BSc
Institution:    FH Technikum Wien, Master's Program in Embedded Systems
Year:           2026
Repository:     https://github.com/RichWiese/edge_offline_tts-benchmark_rpi5
License:        MIT License

Description:
    Evaluates phonetic intelligibility of generated TTS audio files by
    re-transcribing WAV samples with Faster-Whisper (medium, INT8, beam size 5).
    Normalizes digits and symbols into spoken German words via num2words and
    computes sentence-level and global Word Error Rate (WER) and Character
    Error Rate (CER) using JiWER.
===============================================================================
"""

import os
import glob
import re
import warnings
import pandas as pd
from faster_whisper import WhisperModel
from num2words import num2words
import jiwer

warnings.filterwarnings("ignore")

# =============================================================================
# CONFIGURATION
# =============================================================================
# Select target model directory containing benchmark CSVs and WAV subfolders:
# MODEL_DIR = "piper"
# MODEL_DIR = "coqui"
MODEL_DIR = "coqui_xtts"
# MODEL_DIR = "kikiri"
# MODEL_DIR = "kokoro_onnx"
# MODEL_DIR = "bark"
# MODEL_DIR = "parler"

LANGUAGE = "de"

# Prompt guiding Whisper to output numbers and times as full German words
INITIAL_PROMPT = (
    "Transkribiere alle Zahlen, Beträge, Uhrzeiten und Daten stets als voll "
    "ausgeschriebene deutsche Wörter. Zum Beispiel: zweitausendsechsundzwanzig, "
    "fünfzehn Uhr dreißig, dreiundzwanzig Komma fünf."
)
# =============================================================================


# =============================================================================
# NUMBER & TEXT NORMALIZATION
# =============================================================================
def verbalize_numbers(text):
    """Convert remaining digits, decimals, times, and symbols into German words."""
    text = text.replace("%", " Prozent")
    text = text.replace("€", " Euro")
    text = re.sub(r"(\d+)\.(\d+)\s*uhr", r"\1 uhr \2", text, flags=re.IGNORECASE)

    def repl_dec(m):
        g = num2words(int(m.group(1)), lang="de")
        n = num2words(int(m.group(2)), lang="de")
        return f"{g} Komma {n}"

    text = re.sub(r"\b(\d+)[,\.](\d+)\b", repl_dec, text)

    def repl_int(m):
        val = m.group(0).replace(".", "")
        try:
            return num2words(int(val), lang="de")
        except Exception:
            return m.group(0)

    text = re.sub(r"\b\d+\b", repl_int, text)
    return text


# Standard text normalization pipeline before Levenshtein comparison
clean_pipeline = jiwer.Compose([
    jiwer.ToLowerCase(),
    jiwer.RemovePunctuation(),
    jiwer.RemoveMultipleSpaces(),
    jiwer.Strip(),
])


# =============================================================================
# LOAD WHISPER ASR MODEL
# =============================================================================
print("Lade Whisper-Modell (medium / INT8)...")
model = WhisperModel("medium", device="cpu", compute_type="int8", cpu_threads=4)

# Collect benchmark CSV files while ignoring previously generated evaluation CSVs
all_csvs = glob.glob(os.path.join(MODEL_DIR, "*.csv"))
csv_candidates = [
    f for f in all_csvs if not os.path.basename(f).startswith("wer_eval_")
]

if not csv_candidates:
    raise FileNotFoundError(
        f"Keine passenden CSV-Dateien in '{MODEL_DIR}' gefunden!"
    )

print(f"Gefundene Datensätze: {len(csv_candidates)}\n")


# =============================================================================
# EVALUATE ALL BENCHMARK CSV FILES
# =============================================================================
for file_idx, csv_path in enumerate(csv_candidates, start=1):
    file_name = os.path.basename(csv_path)
    base_stem = os.path.splitext(file_name)[0]

    print("=" * 60)
    print(f"[{file_idx}/{len(csv_candidates)}] Verarbeite Datei: {file_name}")
    print("=" * 60)

    df = pd.read_csv(csv_path, header=0, sep=",", dtype=str)
    # Filter rows that reference an exported WAV file (column index 12: Audio_Pfad)
    df_valid = df[df.iloc[:, 12].str.strip() != "-"].copy()
    print(f"Gültige Audio-Dateien: {len(df_valid)}")

    references = []
    hypotheses = []
    log_results = []

    for idx, (_, row) in enumerate(df_valid.iterrows(), start=1):
        raw_text = str(row.iloc[3])
        audio_rel_path = str(row.iloc[12]).strip()
        audio_path = os.path.join(MODEL_DIR, audio_rel_path)

        # Fix potential mojibake encoding artifacts in ground-truth text
        try:
            clean_ref_raw = raw_text.encode("latin1").decode("utf-8")
        except (UnicodeEncodeError, UnicodeDecodeError):
            clean_ref_raw = raw_text

        if not os.path.exists(audio_path):
            print(f"WARNUNG: Datei fehlt: {audio_path}")
            continue

        # Transcribe audio using Faster-Whisper
        segments, _ = model.transcribe(
            audio_path,
            language=LANGUAGE,
            beam_size=5,
            initial_prompt=INITIAL_PROMPT,
        )
        hyp_raw = " ".join([seg.text for seg in segments]).strip()

        # Normalize numbers and text for both ground truth and hypothesis
        hyp_verbalized = verbalize_numbers(hyp_raw)
        ref_norm = clean_pipeline(clean_ref_raw)
        hyp_norm = clean_pipeline(hyp_verbalized)

        references.append(ref_norm)
        hypotheses.append(hyp_norm)

        # Compute word-level error metrics (WER)
        word_metrics = jiwer.process_words(ref_norm, hyp_norm)
        s = word_metrics.substitutions
        d = word_metrics.deletions
        i = word_metrics.insertions
        h = word_metrics.hits
        n = s + d + h

        single_wer = (s + d + i) / n if n > 0 else 0.0

        # Compute character-level error rate (CER)
        single_cer = jiwer.cer(ref_norm, hyp_norm)

        prompt_id = row.iloc[1]

        print(
            f"[{idx}/{len(df_valid)}] {prompt_id} | "
            f"WER: {single_wer:.1%} | CER: {single_cer:.1%} "
            f"(S:{s}, D:{d}, I:{i}, N:{n})"
        )

        log_results.append({
            "Prompt_ID": prompt_id,
            "Kategorie": row.iloc[2],
            "Original_Referenz": clean_ref_raw,
            "Whisper_Verbalisiert": hyp_verbalized,
            "S_Substitutions": s,
            "D_Deletions": d,
            "I_Insertions": i,
            "H_Hits": h,
            "N_Words": n,
            "Satz_WER": round(single_wer, 4),
            "Satz_CER": round(single_cer, 4),
        })

    # =========================================================================
    # AGGREGATE METRICS & EXPORT CSV
    # =========================================================================
    if references:
        global_words = jiwer.process_words(references, hypotheses)
        total_s = global_words.substitutions
        total_d = global_words.deletions
        total_i = global_words.insertions
        total_h = global_words.hits
        total_n = total_s + total_d + total_h
        total_wer = (total_s + total_d + total_i) / total_n if total_n > 0 else 0.0
        total_cer = jiwer.cer(references, hypotheses)

        print("\n" + "-" * 50)
        print(f"ERGEBNIS FÜR: {file_name}")
        print(f"Evaluierte Sätze        : {len(references)}")
        print(f"Word Error Rate (WER)   : {total_wer * 100:.2f} %")
        print(f"Char Error Rate (CER)   : {total_cer * 100:.2f} %")
        print("-" * 50)

        summary_row = {
            "Prompt_ID": "GESAMT",
            "Kategorie": "ALLE",
            "Original_Referenz": f"{len(references)} Sätze evaluiert",
            "Whisper_Verbalisiert": "-",
            "S_Substitutions": total_s,
            "D_Deletions": total_d,
            "I_Insertions": total_i,
            "H_Hits": total_h,
            "N_Words": total_n,
            "Satz_WER": round(total_wer, 4),
            "Satz_CER": round(total_cer, 4),
        }
        log_results.append(summary_row)

        out_path = os.path.join(
            MODEL_DIR, f"wer_eval_normalizedfix_detailed_{base_stem}_v2.csv"
        )
        pd.DataFrame(log_results).to_csv(out_path, index=False, sep=";")
        print(f"Gespeichert in: {out_path}\n")
