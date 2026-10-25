# Benchmarking Offline German Text-to-Speech (TTS) Models on Raspberry Pi 5

This repository contains the benchmarking suite, evaluation pipelines, measurement datasets, and the standalone Terminal User Interface (TUI) prototype developed for the Master's thesis:

> **Benchmarking of Open-Source Text-to-Speech Models on Resource-Constrained ARM Edge Devices in Isolated Offline Operation**  
> *Master's Thesis in Embedded Systems, FH Technikum Wien (2026)*  
> **Author:** Richard Wieser, BSc  
> **Repository:** [https://github.com/RichWiese/edge_offline_tts-benchmark_rpi5](https://github.com/RichWiese/edge_offline_tts-benchmark_rpi5)

---

## Overview

Deploying neural Text-to-Speech (TTS) models directly on edge hardware eliminates cloud dependency, minimizes latency, and ensures user privacy in local smart-home and embedded assistant applications. This project systematically evaluates **seven open-source German TTS models** across **fourteen configurations** on a Raspberry Pi 5.

The evaluation framework analyzes the multi-dimensional trade-offs between:
* **Latency & Throughput:** Real-Time Factor (RTF) and synthesis throughput (samples/s)
* **Hardware Telemetry:** Core SoC temperature and firmware throttling registers (`vcgencmd get_throttled`)
* **Acoustic Quality & Naturalness:** Deep-learning MOS estimators (NISQA v2.0 and UTMOSv2)
* **Phonetic Intelligibility:** Word Error Rate (WER) and Character Error Rate (CER) via re-transcription with Faster-Whisper

### Evaluated Model Matrix (14 Configurations)
1. **Piper TTS** (ONNX Runtime): `thorsten_low`, `thorsten_medium`, `thorsten_emotional_medium`, `thorsten_high`
2. **Coqui-VITS** (Native PyTorch): `thorsten_vits`, `css10_vits_neon`
3. **Kokoro-82M** (ONNX Runtime): `kokoro_martin`
4. **Kikiri-TTS** (PyTorch + dynamic runtime weight injection): `kikiri_bernd`, `kikiri_martin`, `kikiri_victoria`
5. **Coqui XTTSv2** (Autoregressive Speech-LLM): `coqui_male_aaron`
6. **Parler-TTS** (Autoregressive Audio-LLM): `parler_mini_multilingual_christopher`, `parler_mini_multilingual_lea`
7. **Suno Bark** (Transformer cascade): `bark_small_de0`

---

## Hardware & Test Environment

* **Target Device:** Raspberry Pi 5 Model B (Broadcom BCM2712, quad-core ARM Cortex-A76 @ 2.4 GHz, 8 GB LPDDR4X)
* **Operating System:** Raspberry Pi OS Lite 64-bit (Debian 12 Bookworm, Linux Kernel 6.6+, Headless)
* **Cooling:** Passive aluminum heatsink (fanless, convection-cooled)
* **CPU Governor:** `performance` (pinned fixed clock frequency at 2.4 GHz across all 4 cores)
* **Memory Architecture:** In-memory audio synthesis using Linux RAM disk (`/dev/shm/`) to eliminate SD card/USB I/O latency bottlenecks
* **Evaluation Host:** Ubuntu 24.04 LTS (WSL2 / Linux PC) for post-processing with Faster-Whisper, JiWER, NISQA, and UTMOSv2

---

## Repository Structure

```text
├── textcorpus/
│   └── textcorpus_de.csv             # 43-phrase phonetically balanced & normalized German test corpus
├── docs/
│   ├── data_dictionary.md            # Data schema definition for telemetry, WER/CER, and MOS datasets
│   └── hardware_software_specs.md    # In-depth testbench hardware, SoC, and operating system specifications
├── environments/
│   ├── pi/                           # Frozen virtual environments for Raspberry Pi 5 benchmark runs
│   │   ├── pip_list_piper.txt
│   │   ├── pip_list_kokoro_onnx.txt
│   │   ├── pip_list_kikiri.txt
│   │   ├── pip_list_coqui.txt
│   │   ├── pip_list_parler.txt
│   │   └── pip_list_bark.txt
│   └── wsl2/                         # Frozen virtual environments for Ubuntu / WSL2 evaluation pipelines
│       ├── pip_list_whisper_jiwer.txt
│       ├── pip_list_nisqa.txt
│       └── pip_list_utmosv2.txt
├── results/                          # Validated master datasets (.xlsx and .csv)
│   ├── rtf_telemetry_master.xlsx
│   ├── rtf_telemetry_master.csv
│   ├── wer_cer_evaluation.xlsx
│   ├── wer_cer_evaluation.csv
│   ├── mos_nisqa_utmos_predictions.xlsx
│   └── mos_nisqa_utmos_predictions.csv
├── scripts/
│   ├── benchmark/                    # Benchmark runners executed directly on Raspberry Pi 5
│   │   ├── benchmark_run_piper.py
│   │   ├── benchmark_run_coqui_vits.py
│   │   ├── benchmark_run_kokoro_onnx.py
│   │   ├── benchmark_run_kikiri_bernd.py
│   │   ├── benchmark_run_kikiri_martin.py
│   │   ├── benchmark_run_kikiri_victoria.py
│   │   ├── benchmark_run_coqui_xtts.py
│   │   ├── benchmark_run_parler_christopher.py
│   │   ├── benchmark_run_parler_lea.py
│   │   └── benchmark_run_bark.py
│   └── evaluation/                   # Evaluation pipelines and dataset validation / aggregation scripts
│       ├── eval_whisper_jiwer.py
│       ├── eval_mos_nisqa.py
│       ├── eval_mos_utmosv2.py
│       ├── merge_benchmark_results.py
│       ├── merge_wer_results.py
│       └── merge_mos_results.py
├── tui_concept/                              # Standalone Go-based Terminal User Interface (Bubble Tea PoC)
│   ├── go.mod
│   ├── go.sum
│   └── main.go
├── LICENSE
└── README.md
```

---

## Benchmark Methodology & Execution Strategy

### 1. Zero-Overhead Timing & Dynamic Storage Fallback
All benchmark scripts write intermediate audio strictly to volatile memory (`/dev/shm/`) during latency measurement (`perf_counter()`), measuring pure algorithmic CPU inference time without disk I/O interference.

For data persistence, the scripts dynamically select the output directory:
* **USB Storage Mounted:** If an external drive is available under `/media/pi/tts_data/`, CSV files and audio samples are written directly to the external drive.
* **Local Fallback:** If no external drive is detected, the runner automatically saves files inside the local repository directory without throwing a `FileNotFoundError`.

### 2. Warm-up Strategy & Thermal Safeguards
* **Real-time Core Models (Piper, Kokoro-82M, Kikiri-TTS, Coqui-VITS):**  
  An initial in-memory warm-up phrase (`"Warmup."`) is synthesized before the test corpus starts. This compiles the computational graph and primes CPU caches to eliminate cold-start measurement bias.
* **Heavyweight Autoregressive Models (XTTSv2, Parler-TTS, Suno Bark):**  
  Warm-up inference is deliberately omitted. Due to multi-minute inference times per phrase on ARM cores, an extra unmeasured warm-up cycle would heat the passive heatsink past $70\,^\circ\text{C}$ before the benchmark begins, triggering premature thermal throttling.
* **Thermal Protection Loop:**  
  Before every sentence iteration, the SoC core temperature is queried via `/sys/class/thermal/thermal_zone0/temp`. If the SoC temperature exceeds $65.0\,^\circ\text{C}$, execution is suspended in 10-second sleep cycles until the processor cools down.

---

## Getting Started

### 1. Hardware Preparation (Raspberry Pi 5)
Lock the CPU scaling governor to `performance` to eliminate dynamic frequency scaling artifacts during timing measurements:

```bash
echo performance | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor
```

Install essential audio libraries and build dependencies on the host OS:

```bash
sudo apt update && sudo apt install -y python3-venv python3-pip libsndfile1 espeak-ng alsa-utils
```

### 2. Virtual Environment Setup
Due to conflicting dependency trees (e.g., ONNX Runtime vs. specialized PyTorch versions), each model family runs in its own isolated Python virtual environment. Exact package snapshots are provided in `environments/pi/` (target device) and `environments/wsl2/` (evaluation host).

Example for setting up the Piper environment:

```bash
python3 -m venv ~/piper_env
source ~/piper_env/bin/activate
pip install piper-tts onnxruntime soundfile numpy
```

Example for setting up the Kokoro ONNX environment:

```bash
python3 -m venv ~/kokoro_env
source ~/kokoro_env/bin/activate
pip install kokoro-onnx soundfile
```

### 3. Running Benchmark Scripts
Activate the corresponding virtual environment and run the benchmark script from the root of the repository.

Run Piper benchmark:
```bash
source ~/piper_env/bin/activate
python3 scripts/benchmark/benchmark_run_piper.py
```

Run Coqui-VITS benchmark:
```bash
source ~/coqui_env/bin/activate
python3 scripts/benchmark/benchmark_run_coqui_vits.py
```

Run Kokoro-82M ONNX benchmark:
```bash
source ~/kokoro_env/bin/activate
python3 scripts/benchmark/benchmark_run_kokoro_onnx.py
```

Run Parler-TTS benchmarks (executed individually to avoid memory fragmentation):
```bash
source ~/parler_env/bin/activate
python3 scripts/benchmark/benchmark_run_parler_christopher.py
python3 scripts/benchmark/benchmark_run_parler_lea.py
```

### 4. Output Data Format
The benchmark scripts continuously write results to CSV. Each row represents a single synthesis run:

```csv
Modell,Satz_ID,Kategorie,Text,Zeichen,Iteration,Temp_Start_C,Temp_End_C,Throttled_Flag,Synthesezeit_s,Audio_Dauer_s,RTF,Audio_Pfad
```

* **Modell:** Name of the model configuration (e.g., `thorsten_medium`)
* **Satz_ID:** Unique sentence identifier from `textcorpus/textcorpus_de.csv` (`K01` to `L05`)
* **Kategorie:** Linguistic category (e.g., `Kurzbefehl`, `Standard`, `Kompositum`, `Langtext`)
* **Text / Zeichen:** Ground-truth input string and character count
* **Iteration:** Measurement run index
* **Temp_Start_C / Temp_End_C:** SoC core temperature before and after synthesis ($^\circ\text{C}$)
* **Throttled_Flag:** Bitmask from `vcgencmd get_throttled` (`0x0` indicates no throttling occurred)
* **Synthesezeit_s:** Pure computation time in seconds (measured via `time.perf_counter()`)
* **Audio_Dauer_s:** Generated audio duration in seconds (parsed from WAV header/samples)
* **RTF:** Real-Time Factor ($\text{RTF} = \text{Synthesezeit} / \text{Audiodauer}$). Values $< 1.0$ indicate real-time capability
* **Audio_Pfad:** Relative file path to the persisted reference WAV file (`-` for RAM-only runs)

---

## Interactive TUI Prototype (Go)

A lightweight curses-style Terminal User Interface written in Go (using [Bubble Tea](https://github.com/charmbracelet/bubbletea)) is located in `tui/`. It serves as a Proof of Concept (PoC) for low-overhead local speech synthesis control over SSH.

### Features
* Minimal runtime footprint and zero interpreter overhead on ARM64
* Direct subprocess execution of local Piper TTS (`/home/pi/piper_env/bin/piper`)
* Dynamic destination selection: volatile RAM disk (`/dev/shm/`) or persistent storage
* Built-in binary WAV header parser extracting duration and calculating RTF directly on the edge device

### Running the TUI
```bash
cd tui
go run main.go
```

To compile a standalone binary:
```bash
cd tui
go build -o tui_synthesizer main.go
./tui_synthesizer
```

---

## Post-Processing & Evaluation Pipeline

To prevent thermal saturation and throttling on the Raspberry Pi 5, audio evaluations were executed on an external Ubuntu 24.04 LTS (WSL2) workstation using the scripts in `scripts/evaluation/`:

1. **Intelligibility (`eval_whisper_jiwer.py`):**  
   Exported WAV files were re-transcribed using `faster-whisper` (`medium`, `int8`, `beam_size=5`), normalized via German number verbalization (`num2words`), and evaluated against ground-truth text for Word Error Rate (WER) and Character Error Rate (CER) using Levenshtein distance (`jiwer`).
2. **Perceptual Speech Quality (`eval_mos_nisqa.py` & `eval_mos_utmosv2.py`):**  
   Mean Opinion Score (MOS) predictions were generated deterministically on CPU using NISQA (v2.0, `nisqa_tts.tar`) and UTMOSv2.
3. **Validation & Aggregation (`merge_*.py`):**  
   Individual CSV outputs were normalized, checked for row completeness and missing values (`NaN`), and consolidated into the master datasets in `results/`.

---

## Technical Documentation

For detailed schema definitions and hardware/software specifications, refer to the repository documentation:
* [Data Dictionary & Schema Definition](docs/data_dictionary.md)
* [Hardware & Software Specifications](docs/hardware_software_specs.md)

---

## License

This project is licensed under the MIT License.
