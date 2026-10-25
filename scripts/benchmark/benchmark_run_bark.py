#!/usr/bin/env python3
"""
===============================================================================
Script:         benchmark_run_bark.py
Project:        Offline Text-to-Speech Benchmarking on ARM Edge Devices
Author:         Richard Wieser, BSc
Institution:    FH Technikum Wien, Master's Program in Embedded Systems
Year:           2026
Repository:     https://github.com/RichWiese/edge_offline_tts-benchmark_rpi5
License:        MIT License

Description:
    Automated benchmark runner for Suno Bark (bark-small) on Raspberry Pi 5.
    Applies dynamic semantic token limits on a reduced corpus and records
    inference latency (RTF), SoC temperature, and CPU throttling flags.
===============================================================================
"""

import os
import time
import csv
import subprocess
import torch
import numpy as np
import scipy.io.wavfile
from transformers import AutoProcessor, BarkModel
from transformers import logging as hf_logging

# Hide cosmetic Hugging Face warnings in the terminal
hf_logging.set_verbosity_error()

# Limit CPU threads to 4 for ARM Cortex-A76 on Raspberry Pi 5
torch.set_num_threads(4)

# --- STORAGE & DIRECTORY CONFIGURATION ---
# Use USB drive if mounted; otherwise fall back to local directory
BASE_DATA_DIR = "/media/pi/tts_data" if os.path.exists("/media/pi/tts_data") else "."
AUDIO_DIR = os.path.join(BASE_DATA_DIR, "bark_outputs_normalized")
os.makedirs(AUDIO_DIR, exist_ok=True)
OUTPUT_CSV = os.path.join(BASE_DATA_DIR, "bark_benchmark_results_normalized.csv")

ITERATIONS = 1

# Bark model configuration: bark-small with German speaker preset
MODELS = {
    "bark_small_de0": {
        "repo_id": "suno/bark-small",
        "voice_preset": "v2/de_speaker_0"
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
        # Load Bark processor and model onto CPU in evaluation mode
        # Note: Extra warm-up synthesis is omitted for heavyweight models to avoid pre-heating the passive heatsink
        print(f"\n--- Lade Modell in den Arbeitsspeicher: {cfg['repo_id']} ---")
        processor = AutoProcessor.from_pretrained(cfg["repo_id"])
        model = BarkModel.from_pretrained(cfg["repo_id"]).to("cpu")
        model.eval()

        sampling_rate = model.generation_config.sample_rate

        print(f"--- Starte Test für: {model_name} (Voice: {cfg['voice_preset']}) ---")

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
                    inputs = processor(item["text"], voice_preset=cfg["voice_preset"])
                    inputs = {k: v.to("cpu") for k, v in inputs.items()}
                    if "attention_mask" not in inputs:
                        inputs["attention_mask"] = torch.ones_like(inputs["input_ids"])

                    # 1. Calculate dynamic token limit (~4 tokens per char + 40 buffer)
                    # Prevents endless silence/noise generation on short commands
                    dynamic_max_tokens = min(768, max(75, int(len(item["text"]) * 4.0) + 40))

                    # 2. Apply token limit and EOS parameters directly to the internal semantic model
                    model.semantic.generation_config.max_new_tokens = dynamic_max_tokens
                    model.semantic.generation_config.min_eos_p = 0.05
                    model.semantic.generation_config.pad_token_id = 10000
                    model.semantic.generation_config.eos_token_id = 10000

                    # Run deterministic multi-stage generation without gradient calculation
                    with torch.no_grad():
                        speech_output = model.generate(
                            **inputs,
                            do_sample=False,
                            pad_token_id=10000,
                            eos_token_id=10000
                        )

                    audio_data = speech_output.squeeze().cpu().numpy()
                    t1 = time.perf_counter()
                    gen_time = t1 - t0
                    # --- END OF INFERENCE TIMING ---

                    # Calculate audio duration and Real-Time Factor (RTF)
                    duration = len(audio_data) / float(sampling_rate)
                    rtf = gen_time / duration if duration > 0 else 999.0

                    # Peak normalization and conversion to 16-bit PCM audio
                    max_val = np.max(np.abs(audio_data))
                    if max_val > 1.0:
                        audio_data = audio_data / max_val
                    elif max_val == 0.0:
                        print("Warnung: Generiertes Audio ist stumm.")
                    audio_data_int16 = (audio_data * 32767).astype(np.int16)

                    # Save generated WAV file directly on iteration 1
                    dauerhafter_pfad = "-"
                    if it == 1:
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

                # Log all telemetry and latency metrics to CSV
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

                saved_info = f" -> Gespeichert: {dauerhafter_pfad}" if it == 1 else ""
                print(f"[{model_name}] {item['id']} #{it} | RTF: {rtf:.3f} (Dauer: {duration:.2f}s, Synth: {gen_time:.2f}s){saved_info}")

print(f"\nMessung abgeschlossen. Gespeicherte Referenzdateien liegen in: {AUDIO_DIR}")
print(f"CSV-Ergebnisse liegen in: {OUTPUT_CSV}")