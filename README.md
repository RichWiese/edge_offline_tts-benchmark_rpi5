# Benchmarking Offline German Text-to-Speech (TTS) Models on Raspberry Pi 5

This repository contains the benchmarking framework, evaluation pipelines, measurement datasets, and prototype Terminal User Interface (TUI) developed for the Master's thesis:

> **Benchmarking of Open-Source Text-to-Speech Models on Resource-Constrained ARM Edge Devices in Isolated Offline Operation**

---

## Overview

Running state-of-the-art neural Text-to-Speech (TTS) models locally on edge hardware eliminates cloud dependency and ensures strict data privacy. This project systematically evaluates **seven open-source German TTS models** across **fourteen configurations** on a Raspberry Pi 5, analyzing the trade-offs between latency (Real-Time Factor, RTF), audio throughput, thermal stability, acoustic speech quality (NISQA), speech naturalness (UTMOSv2), and phonetic intelligibility (WER/CER).

### Evaluated TTS Candidates
1. **Piper TTS** (ONNX Runtime: `low`, `medium`, `emotional-medium`, `high`)
2. **Coqui-VITS** (Native PyTorch: `thorsten_vits`, `css10_vits_neon`)
3. **Kokoro-82M** (ONNX Runtime: `kokoro_martin`)
4. **Kikiri-TTS** (PyTorch + dynamic weight injection: `bernd`, `martin`, `victoria`)
5. **Coqui XTTSv2** (Autoregressive Speech-LLM: `coqui_male_aaron`)
6. **Parler-TTS** (Autoregressive Audio-LLM: `mini_christopher`, `mini_lea`)
7. **Suno Bark** (Transformer cascade: `bark_small_de0`)

---

## Hardware & Test Environment

* **Target Device:** Raspberry Pi 5 Model B (8 GB RAM)
* **Operating System:** Raspberry Pi OS Lite 64-bit (Debian 12 Bookworm, headless)
* **Cooling:** Passive aluminum heatsink (fanless, convection-cooled)
* **Clock Frequency:** CPU governor set to `performance` (fixed at 2.4 GHz)
* **Audio Processing:** Ephemeral in-memory generation via RAM disk (`/dev/shm/`)
* **Evaluation Host:** Ubuntu 24.04 LTS (WSL2) for automated Faster-Whisper, NISQA, and UTMOSv2 scoring

---

## Repository Structure

```text
├── corpus/               # 43-phrase phonetically balanced & normalized German test corpus
├── environments/         # Exact 'pip list' dependency snapshots for each isolated venv
├── results/              # Raw telemetry, latency, MOS predictions, and WER/CER CSV data
├── scripts/
│   ├── benchmark/        # Automated test runners with thermal and SoC throttling monitoring
│   └── evaluation/       # Whisper/JiWER ASR scoring, NISQA, UTMOSv2, and plotting scripts
├── tui/                  # Lightweight standalone Terminal User Interface for local TTS synthesis
└── README.md
