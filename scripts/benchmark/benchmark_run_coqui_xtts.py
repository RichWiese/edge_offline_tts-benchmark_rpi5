#!/usr/bin/env python3
"""
===============================================================================
Script:         benchmark_run_coqui_xtts.py
Project:        Offline Text-to-Speech Benchmarking on ARM Edge Devices
Author:         Richard Wieser, BSc
Institution:    FH Technikum Wien, Master's Program in Embedded Systems
Year:           2026
Repository:     https://github.com/RichWiese/edge_offline_tts-benchmark_rpi5
License:        MIT License

Description:
    Automated benchmark runner for Coqui XTTSv2 on Raspberry Pi 5.
    Evaluates autoregressive speech synthesis on a reduced test corpus using
    volatile RAM disk (/dev/shm) and records RTF, SoC temperature, and throttling.
===============================================================================
"""

import os
import time
import csv
import wave
import subprocess
import shutil
import torch
from TTS.api import TTS

# Patch torch.load for PyTorch 2.6+ compatibility (allow full checkpoint loading)
_orig_torch_load = torch.load
def _patched_torch_load(*args, **kwargs):
    kwargs["weights_only"] = False
    return _orig_torch_load(*args, **kwargs)
torch.load = _patched_torch_load

# --- STORAGE & DIRECTORY CONFIGURATION ---
# Use USB drive if mounted; otherwise fall back to local directory
BASE_DATA_DIR = "/media/pi/tts_data" if os.path.exists("/media/pi/tts_data") else "."
AUDIO_DIR = os.path.join(BASE_DATA_DIR, "coqui_xtts_outputs_normalized")
os.makedirs(AUDIO_DIR, exist_ok=True)
OUTPUT_CSV = os.path.join(BASE_DATA_DIR, "coqui_xtts_aaron_benchmark_results_normalized.csv")

# Volatile RAM disk (/dev/shm) for zero-overhead inference timing
TEMP_WAV = "/dev/shm/temp_coqui_xtts_eval.wav"
ITERATIONS = 2

# Predefined German speaker in the XTTSv2 checkpoint
SPEAKERS = {
    "coqui_male_aaron": "Aaron Dreschner"
}

# Reduced representative test corpus (K1-K4 and H6) due to heavyweight CPU load
TEST_CORPUS = [
    {"id": "K1", "cat": "Kurzbefehl", "text": "Licht aus."},
    {"id": "K2", "cat": "Kurzbefehl", "text": "Wie spät ist es?"},
    {"id": "K3", "cat": "Kurzbefehl", "text": "Alarm aktivieren."},
    {"id": "K4", "cat": "Kurzbefehl", "text": "Musik abspielen."},
    {"id": "H6", "cat": "Zahlen_Einheiten", "text": "Am zweiundzwanzigsten August zweitausendsechsundzwanzig um fünfzehn Uhr dreißig betrug die Temperatur exakt dreiundzwanzig Komma fünf Grad Celsius."}
]

# --- HARDWARE TELEMETRY FUNCTIONS ---
def get_cpu_temp():
    """Read SoC core temperature from Linux sysfs (in degrees Celsius)."""
    try:
        with open("/sys/class/thermal/thermal_zone0/temp", "r") as f:
            return float(f.read()) / 1000.0
    except Exception:
        return 0.0

def get_throttle_state():
    """Query Raspberry Pi firmware register to check for CPU/GPU throttling."""
    try:
        res = subprocess.check_output(["vcgencmd", "get_throttled"]).decode("utf-8")
        return res.strip().replace("throttled=", "")
    except Exception:
        return "N/A"

# Load official XTTSv2 model onto CPU (cached locally after initial download)
# Note: Extra warm-up synthesis is omitted for 1.8B heavyweight models to avoid pre-heating the passive heatsink
print("--- Lade XTTS-v2 Modell in den Arbeitsspeicher ---")
tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2", gpu=False)

# --- MAIN BENCHMARK EXECUTION ---
with open(OUTPUT_CSV, mode="w", newline="", encoding="utf-8") as csv_file:
    fieldnames = [
        "Modell", "Satz_ID", "Kategorie", "Text", "Zeichen",
        "Iteration", "Temp_Start_C", "Temp_End_C", "Throttled_Flag",
        "Synthesezeit_s", "Audio_Dauer_s", "RTF", "Audio_Pfad"
    ]
    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    writer.writeheader()

    for model_alias, speaker_name in SPEAKERS.items():
        print(f"\n--- Starte Benchmark für: {model_alias} ({speaker_name}) ---")

        for item in TEST_CORPUS:
            for it in range(1, ITERATIONS + 1):
                # Thermal safety pause: wait if SoC temperature exceeds 65.0 °C
                while get_cpu_temp() > 65.0:
                    print("CPU über 65°C. Warte 10s auf Abkühlung...")
                    time.sleep(10)

                temp_start = get_cpu_temp()

                # --- PURE INFERENCE TIMING (volatile synthesis to RAM disk /dev/shm) ---
                t0 = time.perf_counter()
                tts.tts_to_file(
                    text=item["text"],
                    speaker=speaker_name,
                    language="de",
                    file_path=TEMP_WAV
                )
                t1 = time.perf_counter()
                # --- END OF INFERENCE TIMING ---

                temp_end = get_cpu_temp()
                gen_time = t1 - t0
                throttled = get_throttle_state()

                dauerhafter_pfad = "-"
                try:
                    # Parse WAV header to read audio duration and calculate RTF
                    with wave.open(TEMP_WAV, "rb") as wav:
                        frames = wav.getnframes()
                        rate = wav.getframerate()
                        duration = frames / float(rate)
                    rtf = gen_time / duration if duration > 0 else 999.0

                    # Save representative sample (iteration 2) for offline ASR/MOS evaluation
                    if it == 2:
                        dauerhafter_pfad = os.path.join(AUDIO_DIR, f"{model_alias}_{item['id']}_normalized.wav")
                        shutil.copyfile(TEMP_WAV, dauerhafter_pfad)
                except Exception as e:
                    print(f"Fehler bei Audioanalyse: {e}")
                    duration = 0.0
                    rtf = 999.0
                    dauerhafter_pfad = "FEHLER"

                # Log all telemetry and latency metrics to CSV
                writer.writerow({
                    "Modell": model_alias,
                    "Satz_ID": item["id"],
                    "Kategorie": item["cat"],
                    "Text": item["text"],
                    "Zeichen": len(item["text"]),
                    "Iteration": it,
                    "Temp_Start_C": f"{temp_start:.1f}",
                    "Temp_End_C": f"{temp_end:.1f}",
                    "Throttled_Flag": throttled,
                    "Synthesezeit_s": f"{gen_time:.4f}",
                    "Audio_Dauer_s": f"{duration:.4f}",
                    "RTF": f"{rtf:.4f}",
                    "Audio_Pfad": dauerhafter_pfad
                })

                saved_info = f" -> Gespeichert: {dauerhafter_pfad}" if it == 2 else ""
                print(f"[{model_alias}] {item['id']} #{it} | RTF: {rtf:.3f} (Dauer: {duration:.2f}s, Synth: {gen_time:.2f}s){saved_info}")

                # Remove temporary WAV in RAM disk to keep memory free
                if os.path.exists(TEMP_WAV):
                    os.remove(TEMP_WAV)

print(f"\nMessung abgeschlossen. Gespeicherte Referenzdateien liegen in: {AUDIO_DIR}")
print(f"CSV-Ergebnisse liegen in: {OUTPUT_CSV}")