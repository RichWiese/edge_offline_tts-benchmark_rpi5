#!/usr/bin/env python3
"""
===============================================================================
Script:         benchmark_run_parler_lea.py
Project:        Offline Text-to-Speech Benchmarking on ARM Edge Devices
Author:         Richard Wieser, BSc
Institution:    FH Technikum Wien, Master's Program in Embedded Systems
Year:           2026
Repository:     https://github.com/RichWiese/edge_offline_tts-benchmark_rpi5
License:        MIT License

Description:
    Automated benchmark runner for Parler-TTS (Lea voice) on Raspberry Pi 5.
    Evaluates autoregressive speech synthesis on a reduced test corpus and
    records latency (RTF), SoC temperature, and throttling flags.
===============================================================================
"""

import os
import time
import csv
import subprocess
import torch
import numpy as np
import scipy.io.wavfile
from parler_tts import ParlerTTSForConditionalGeneration
from transformers import AutoTokenizer
from transformers import logging as hf_logging

# Hide cosmetic Hugging Face warnings in the terminal
hf_logging.set_verbosity_error()

# Limit CPU threads to 4 for ARM Cortex-A76 on Raspberry Pi 5
torch.set_num_threads(4)

# --- STORAGE & DIRECTORY CONFIGURATION ---
# Use USB drive if mounted; otherwise fall back to local directory
BASE_DATA_DIR = "/media/pi/tts_data" if os.path.exists("/media/pi/tts_data") else "."
AUDIO_DIR = os.path.join(BASE_DATA_DIR, "parler_outputs_normalized")
os.makedirs(AUDIO_DIR, exist_ok=True)
OUTPUT_CSV = os.path.join(BASE_DATA_DIR, "parler_lea_benchmark_results_normalized.csv")

ITERATIONS = 1

# Configuration for German female voice (Lea)
MODELS = {
    "parler_mini_multilingual_lea": {
        "repo_id": "parler-tts/parler-tts-mini-multilingual-v1.1",
        "description": "Lea's voice with a standard pace and clear enunciation."
    }
}

# Reduced representative test corpus (K1-K4 and H6) due to extreme CPU load
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

# --- MAIN BENCHMARK EXECUTION ---
with open(OUTPUT_CSV, mode="w", newline="", encoding="utf-8") as csv_file:
    fieldnames = [
        "Modell", "Satz_ID", "Kategorie", "Text", "Zeichen",
        "Iteration", "Temp_Start_C", "Temp_End_C", "Throttled_Flag",
        "Synthesezeit_s", "Audio_Dauer_s", "RTF", "Audio_Pfad"
    ]
    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    writer.writeheader()

    for model_name, cfg in MODELS.items():
        print(f"\n--- Lade Modell in den Arbeitsspeicher: {cfg['repo_id']} ({model_name}) ---")
        device = "cpu"

        # 1. Load prompt tokenizer
        prompt_tokenizer = AutoTokenizer.from_pretrained(cfg["repo_id"])

        # 2. Load model onto CPU
        model = ParlerTTSForConditionalGeneration.from_pretrained(cfg["repo_id"]).to(device)
        model.eval()

        # 3. Load description tokenizer
        try:
            description_tokenizer = AutoTokenizer.from_pretrained(model.config.text_encoder._name_or_path)
        except Exception:
            description_tokenizer = prompt_tokenizer

        sampling_rate = model.config.sampling_rate

        print(f"--- Starte Test für: {model_name} ---")
        print(f"--- Description: \"{cfg['description']}\" ---")

        # Pre-tokenize voice description once
        input_ids = description_tokenizer(cfg["description"], return_tensors="pt").input_ids.to(device)

        for item in TEST_CORPUS:
            for it in range(1, ITERATIONS + 1):
                # Thermal safety pause: wait if SoC temperature exceeds 65.0 °C
                while get_cpu_temp() > 65.0:
                    print("CPU über 65°C. Warte 10s auf Abkühlung...")
                    time.sleep(10)

                temp_start = get_cpu_temp()

                # --- PURE INFERENCE TIMING ---
                t0 = time.perf_counter()
                try:
                    prompt_input_ids = prompt_tokenizer(item["text"], return_tensors="pt").input_ids.to(device)

                    with torch.no_grad():
                        generation = model.generate(
                            input_ids=input_ids,
                            prompt_input_ids=prompt_input_ids,
                            do_sample=True,
                            temperature=1.0
                        )

                    audio_data = generation.squeeze().cpu().numpy()
                    t1 = time.perf_counter()
                    gen_time = t1 - t0
                    # --- END OF INFERENCE TIMING ---

                    duration = len(audio_data) / float(sampling_rate)
                    rtf = gen_time / duration if duration > 0 else 999.0

                    # Peak normalization and conversion to 16-bit PCM audio
                    max_val = np.max(np.abs(audio_data))
                    if max_val > 1.0:
                        audio_data = audio_data / max_val
                    elif max_val == 0.0:
                        print("Warnung: Generiertes Audio ist stumm.")

                    audio_data_int16 = (audio_data * 32767).astype(np.int16)

                    dauerhafter_pfad = os.path.join(AUDIO_DIR, f"{model_name}_{item['id']}_normalized.wav")
                    scipy.io.wavfile.write(dauerhafter_pfad, rate=sampling_rate, data=audio_data_int16)

                except Exception as e:
                    t1 = time.perf_counter()
                    gen_time = t1 - t0
                    print(f"Fehler bei {item['id']}: {e}")
                    duration = 0.0
                    rtf = 999.0
                    dauerhafter_pfad = "FEHLER"

                temp_end = get_cpu_temp()
                throttled = get_throttle_state()

                writer.writerow({
                    "Modell": model_name,
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

                print(f"[{model_name}] {item['id']} #{it} | RTF: {rtf:.3f} | Dauer: {duration:.2f}s | Synthese: {gen_time:.2f}s -> Gespeichert: {dauerhafter_pfad}")

print(f"\nMessung abgeschlossen. Gespeicherte Referenzdateien liegen in: {AUDIO_DIR}")
print(f"CSV-Ergebnisse liegen in: {OUTPUT_CSV}")