# System Architecture & Environment

This document outlines the precise hardware configuration, operating system layout, and software environment package versions utilized on the test bench.

---

## 1. Hardware Environment
* **Single Board Computer:** Raspberry Pi 5 Model B (Rev 1.0)
* **SoC:** Broadcom BCM2712 quad-core 64-bit ARM Cortex-A76 @ 2.4 GHz
* **Memory (RAM):** 8 GB LPDDR4X-4267 SDRAM
* **Storage:** SanDisk Extreme Pro 128 GB A2 MicroSDXC (OS) + External USB 3.0 SSD / Flash Drive (`/media/pi/tts_data/`) for telemetry logging and WAV persistence
* **Thermal Management:** Official Raspberry Pi 5 Passive Aluminum Heatsink (fanless convection cooling, no active fan)
* **Power Supply:** Official Raspberry Pi 27W USB-C Power Supply (delivering stable 5.1V / 5A)

---

## 2. Software & Operating System Stack
* **Operating System:** Raspberry Pi OS Lite (64-bit, Debian 12 Bookworm, Headless)
* **Linux Kernel:** Linux 6.6.20+rpt-rpi-2712 (ARM64)
* **Firmware Version:** Broadcom VideoCore VII firmware (supporting `vcgencmd get_throttled`)
* **Python Runtime:** Python 3.11.2 (compiled with GCC 12.2.0)
* **Core ML Frameworks:** 
  * PyTorch 2.x (CPU-optimized with OpenMP threading bound to 4 cores)
  * ONNX Runtime (`onnxruntime`) for accelerated inference on Piper and Kokoro-82M
  * Transformers (`transformers`) & Accelerate (`accelerate`) for Hugging Face model architectures (Parler-TTS, Suno Bark, Coqui XTTSv2)
  * Coqui TTS (`TTS`) for VITS and XTTS checkpoints