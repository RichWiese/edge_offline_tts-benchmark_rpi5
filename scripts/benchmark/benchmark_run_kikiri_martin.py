#!/usr/bin/env python3
"""
===============================================================================
Script:         benchmark_run_kikiri_martin.py
Project:        Offline Text-to-Speech Benchmarking on ARM Edge Devices
Author:         Richard Wieser, BSc
Institution:    FH Technikum Wien, Master's Program in Embedded Systems
Year:           2026
Repository:     https://github.com/RichWiese/edge_offline_tts-benchmark_rpi5
License:        MIT License

Description:
    Automated benchmark runner for Kikiri-TTS (Martin voice, PyTorch) on
    Raspberry Pi 5. Injects German model weights at runtime, executes a
    warm-up run, and records latency (RTF), SoC temperature, and throttling.
===============================================================================
"""

# Optimization: limit OpenMP threads to 4 for Raspberry Pi 5 CPU
import os
os.environ["OMP_NUM_THREADS"] = "4"
import time
import csv
import subprocess
import torch
# Flush very small floating-point numbers to zero and bind to 4 CPU threads
torch.set_flush_denormal(True)
torch.set_num_threads(4)
import numpy as np
import soundfile as sf
import kokoro.pipeline
from kokoro import KPipeline

# Patch German language code support into the Kokoro base pipeline
kokoro.pipeline.LANG_CODES["d"] = "de"
kokoro.pipeline.LANG_CODES["de"] = "de"
if hasattr(kokoro.pipeline.KPipeline, "LANG_CODES"):
    kokoro.pipeline.KPipeline.LANG_CODES["d"] = "de"
    kokoro.pipeline.KPipeline.LANG_CODES["de"] = "de"

# --- STORAGE & DIRECTORY CONFIGURATION ---
# Use USB drive if mounted; otherwise fall back to local directory
BASE_DATA_DIR = "/media/pi/tts_data" if os.path.exists("/media/pi/tts_data") else "."
AUDIO_DIR = os.path.join(BASE_DATA_DIR, "kikiri_martin_outputs_normalized")
os.makedirs(AUDIO_DIR, exist_ok=True)
OUTPUT_CSV = os.path.join(BASE_DATA_DIR, "kikiri_martin_pytorch_benchmark_results_normalized.csv")

ITERATIONS = 3
SAMPLE_RATE = 24000
DEVICE = "cpu"

MODELS = {
    "kikiri_martin_pytorch": {
        "model_path": "kikiri-german-martin/kikiri_german_martin_ep10.pth",
        "voice_path": "kikiri-german-martin/martin.pt",
        "lang": "d"
    }
}

# Standardized and graphemically normalized German test corpus (43 phrases)
TEST_CORPUS = [
    # --- KATEGORIE: KURZBEFEHLE (K1 - K10) ---
    {"id": "K1", "cat": "Kurzbefehl", "text": "Licht aus."},
    {"id": "K2", "cat": "Kurzbefehl", "text": "Wie spät ist es?"},
    {"id": "K3", "cat": "Kurzbefehl", "text": "Alarm aktivieren."},
    {"id": "K4", "cat": "Kurzbefehl", "text": "Musik abspielen."},
    {"id": "K5", "cat": "Kurzbefehl", "text": "Lautstärke höher."},
    {"id": "K6", "cat": "Kurzbefehl", "text": "Tür verriegeln."},
    {"id": "K7", "cat": "Kurzbefehl", "text": "Heizung an."},
    {"id": "K8", "cat": "Kurzbefehl", "text": "Bitte warten."},
    {"id": "K9", "cat": "Kurzbefehl", "text": "Fahre herunter."},
    {"id": "K10", "cat": "Kurzbefehl", "text": "Guten Morgen."},

    # --- KATEGORIE: STANDARD / ALLGEMEIN (M1 - M10) ---
    {"id": "M1", "cat": "Standard", "text": "Der Raspberry Pi führt die Sprachsynthese komplett lokal und ohne Internetverbindung aus."},
    {"id": "M2", "cat": "Standard", "text": "Künstliche Intelligenz verändert die Art und Weise, wie wir mit Computern interagieren."},
    {"id": "M3", "cat": "Standard", "text": "Das Betriebssystem basiert auf Debian und wurde für eingebettete Systeme optimiert."},
    {"id": "M4", "cat": "Standard", "text": "Ein lokal-betriebener Sprachassistent schützt die Privatsphäre der Benutzer im eigenen Smart Home."},
    {"id": "M5", "cat": "Standard", "text": "Die Evaluation von Algorithmen erfordert eine präzise Messung der Hardware-Ressourcen."},
    {"id": "M6", "cat": "Standard", "text": "Moderne Text-to-Speech-Systeme nutzen neuronale Netzwerke für eine natürliche Betonung."},
    {"id": "M7", "cat": "Standard", "text": "Die Datenverarbeitung auf der Edge-Ebene reduziert die Abhängigkeit von Cloud-Servern."},
    {"id": "M8", "cat": "Standard", "text": "Eine mathematische Analyse der Latenzzeiten ist für die Validierung unerlässlich."},
    {"id": "M9", "cat": "Standard", "text": "Die Ergebnisse dieses Benchmarks werden in einer tabellarischen Übersicht dargestellt."},
    {"id": "M10", "cat": "Standard", "text": "Dieses Python-Skript automatisiert den gesamten Prozess der Datenerhebung."},

    # --- KATEGORIE: SPRACHLICHE HÜRDEN & SONDERFÄLLE (H1 - H18) ---
    {"id": "H1", "cat": "Kompositum", "text": "Die Geschwindigkeitsüberschreitung in der Hauptbahnhofsunterführung verursachte Probleme."},
    {"id": "H2", "cat": "Kompositum", "text": "Das Bundesausbildungsförderungsgesetz regelt die finanzielle Unterstützung für Studierende."},
    {"id": "H3", "cat": "Kompositum", "text": "Ein Donaudampfschifffahrtsgesellschaftskapitän benötigt eine spezielle Ausbildung."},
    {"id": "H4", "cat": "Kompositum", "text": "Die Kraftfahrzeug-Haftpflichtversicherung ist für alle Fahrzeughalter gesetzlich vorgeschrieben."},
    {"id": "H5", "cat": "Kompositum", "text": "Arbeitsunfähigkeitsbescheinigungen müssen dem Arbeitgeber unverzüglich vorliegen."},
    {"id": "H6", "cat": "Zahlen_Einheiten", "text": "Am zweiundzwanzigsten August zweitausendsechsundzwanzig um fünfzehn Uhr dreißig betrug die Temperatur exakt dreiundzwanzig Komma fünf Grad Celsius."},
    {"id": "H7", "cat": "Zahlen_Einheiten", "text": "Der Preis stieg um fünfzehn Komma vier Prozent auf insgesamt eintausendzweihundertneunundvierzig Euro neunundneunzig pro Quadratmeter."},
    {"id": "H8", "cat": "Zahlen_Einheiten", "text": "Wir schreiben das Jahr zweitausendsechsundzwanzig und die Frequenz liegt bei zwei Komma vier Gigahertz."},
    {"id": "H9", "cat": "Zahlen_Einheiten", "text": "Die Auslastung des Speichers sank von fünfundachtzig Megabyte auf nur zwölf Komma fünf Kilobyte."},
    {"id": "H10", "cat": "Zahlen_Einheiten", "text": "Es wurden dreitausendvierhundertsechsundfünfzig Probanden in insgesamt zwölf Gruppen aufgeteilt."},
    {"id": "H11", "cat": "Homograph", "text": "Der August kam im Monat August an."},
    {"id": "H12", "cat": "Homograph", "text": "Die moderne Anlage erfordert eine präzise Anlage der finanziellen Mittel."},
    {"id": "H13", "cat": "Abkuerzung", "text": "Das betrifft unter anderem die Hardwarekomponenten beziehungsweise die Software-Architektur des Pi."},
    {"id": "H14", "cat": "Abkuerzung", "text": "Der Professor Doktor Müller erklärte die Details, das heißt die mathematischen Grundlagen."},
    {"id": "H15", "cat": "Abkuerzung", "text": "Messen Sie die Werte zum Beispiel alle zehn Sekunden für circa fünf Minuten kontinuierlich."},
    {"id": "H16", "cat": "Intonation", "text": "Achtung, das System reagiert nicht mehr! Ist die Stromversorgung unterbrochen?"},
    {"id": "H17", "cat": "Zahlen_Einheiten", "text": "Papst Johannes Paul der Zweite lebte im zwanzigsten Jahrhundert."},
    {"id": "H18", "cat": "Anglizismen", "text": "Das Deep-Learning-Framework nutzt den Cache des On-Device-Chips."},

    # --- KATEGORIE: LANGTEXTE / STRESSTEST (L1 - L5) ---
    {"id": "L1", "cat": "Langtext", "text": "Ein eingebettetes System ist ein Computer, der in einen technischen Kontext eingebunden ist. Dabei übernimmt der Rechner entweder Überwachungs-, Steuerungs- oder Regelungsfunktionen oder eine Form der Datenverarbeitung."},
    {"id": "L2", "cat": "Langtext", "text": "Edge Computing bezeichnet die dezentrale Datenverarbeitung an der Peripherie eines Netzwerks. Im Gegensatz zum Cloud Computing werden die Daten direkt dort verarbeitet, wo sie entstehen, was Latenzzeiten minimiert und die Bandbreite schont."},
    {"id": "L3", "cat": "Langtext", "text": "Die Evaluation von Text-to-Speech-Systemen auf Einplatinencomputern erfordert eine sorgfältige Trennung zwischen algorithmischer Effizienz und hardwarebedingten Restriktionen. Insbesondere thermische Effekte können die Reproduzierbarkeit der Messwerte stark beeinflussen."},
    {"id": "L4", "cat": "Langtext", "text": "Der Raspberry Pi fünf verfügt über eine deutlich gesteigerte Prozessorleistung im Vergleich zu seinen Vorgängern. Durch die modernisierte Architektur eignet er sich zunehmend für anspruchsvolle Aufgaben im Bereich des maschinellen Lernens direkt auf dem Endgerät."},
    {"id": "L5", "cat": "Langtext", "text": "Datenschutz und Informationssicherheit gewinnen in einer vernetzten Welt zunehmend an Bedeutung. Lokale Sprachmodelle bieten eine technologische Lösung, um hochgradig interaktive Systeme zu betreiben, ohne sensible Audiodaten an externe Serverstrukturen übertragen zu müssen."}
]

# --- HARDWARE TELEMETRY & WEIGHT INJECTION FUNCTIONS ---
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

def load_kikiri_weights(model, weights_path):
    """Map and inject fine-tuned German Kikiri weights into the Kokoro model."""
    checkpoint = torch.load(weights_path, map_location=DEVICE, weights_only=False)
    flat_weights = {}

    # Clean layer names to match internal Kokoro state_dict structure
    for sub_mod, sub_dict in checkpoint.items():
        if isinstance(sub_dict, dict):
            for param, tensor in sub_dict.items():
                clean_p = param.replace("module.", "")
                clean_p = clean_p.replace(".parametrizations.weight.original0", ".weight_g")
                clean_p = clean_p.replace(".parametrizations.weight.original1", ".weight_v")
                full_key = f"{sub_mod}.{clean_p}"
                flat_weights[full_key] = tensor
        elif isinstance(sub_dict, torch.Tensor):
            flat_weights[sub_mod] = sub_dict

    model_dict = model.state_dict()
    matched = {
        k: v for k, v in flat_weights.items()
        if k in model_dict and v.shape == model_dict[k].shape
    }

    print(f"-> Checkpoint enthält: {len(flat_weights)} Schichten.")
    print(f"-> Schichten-Abgleich: {len(matched)} von {len(model_dict)} Layern erfolgreich zugeordnet.")

    if len(matched) < 500:
        raise RuntimeError(f"FEHLER: Nur {len(matched)} Layer gemappt! Die Gewichte passen nicht.")

    model_dict.update(matched)
    model.load_state_dict(model_dict)
    print("Martin-Modellgewichte inklusive Decoder erfolgreich aktiv!\n")

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
        if not os.path.exists(cfg["model_path"]) or not os.path.exists(cfg["voice_path"]):
            print(f"Dateien für {model_name} nicht gefunden, überspringe.")
            continue

        print(f"--- Initialisiere Pipeline: {model_name} ---")
        pipeline = KPipeline(lang_code=cfg["lang"], repo_id="hexgrad/Kokoro-82M")

        # 1. Inject custom German Martin weights into the loaded base model
        load_kikiri_weights(pipeline.model, cfg["model_path"])

        # 2. Load Martin speaker style tensor
        voice_raw = torch.load(cfg["voice_path"], map_location=DEVICE, weights_only=False)
        if isinstance(voice_raw, dict):
            voice_tensor = next(iter(voice_raw.values()))
        else:
            voice_tensor = voice_raw

        if isinstance(voice_tensor, np.ndarray):
            voice_tensor = torch.from_numpy(voice_tensor)
        voice_tensor = voice_tensor.to(DEVICE)

        # 3. Warm-up run: initializes PyTorch pipeline and primes CPU caches
        print("Führe Warm-up aus...")
        for _, _, _ in pipeline("Warmup.", voice=voice_tensor, speed=1.0):
            pass

        print(f"--- Starte Benchmark für: {model_name} ---")

        for item in TEST_CORPUS:
            for it in range(1, ITERATIONS + 1):
                # Thermal safety pause: wait if SoC temperature exceeds 65.0 °C
                while get_cpu_temp() > 65.0:
                    print("CPU über 65°C. Warte 10s auf Abkühlung...")
                    time.sleep(10)

                temp_start = get_cpu_temp()

                # --- PURE INFERENCE TIMING (in-memory tensor generation) ---
                t0 = time.perf_counter()
                audio_segments = []
                for _, _, chunk in pipeline(item["text"], voice=voice_tensor, speed=1.0):
                    if chunk is not None:
                        audio_segments.append(chunk)
                t1 = time.perf_counter()
                # --- END OF INFERENCE TIMING ---

                temp_end = get_cpu_temp()
                gen_time = t1 - t0
                throttled = get_throttle_state()

                dauerhafter_pfad = "-"
                try:
                    # Concatenate generated audio chunks and calculate playback duration
                    if audio_segments:
                        if isinstance(audio_segments[0], torch.Tensor):
                            samples = torch.cat(audio_segments).cpu().numpy()
                        else:
                            samples = np.concatenate(audio_segments)
                        duration = len(samples) / float(SAMPLE_RATE)
                    else:
                        samples = np.array([], dtype=np.float32)
                        duration = 0.0

                    # Calculate Real-Time Factor (RTF)
                    rtf = gen_time / duration if duration > 0 else 999.0

                    # Save representative audio sample (iteration 2) for ASR/MOS evaluation
                    if it == 2 and len(samples) > 0:
                        dauerhafter_pfad = os.path.join(AUDIO_DIR, f"{model_name}_{item['id']}_normalized.wav")
                        sf.write(dauerhafter_pfad, samples, SAMPLE_RATE)
                except Exception as e:
                    print(f"Fehler bei Audioberechnung: {e}")
                    duration = 0.0
                    rtf = 999.0
                    dauerhafter_pfad = "FEHLER"

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

                saved_info = f" -> Gespeichert: {dauerhafter_pfad}" if it == 2 else ""
                print(f"[{model_name}] {item['id']} #{it} | RTF: {rtf:.3f} (Dauer: {duration:.2f}s, Synth: {gen_time:.2f}s){saved_info}")

print(f"\nMessung abgeschlossen. Gespeicherte Referenzdateien liegen in: {AUDIO_DIR}")
print(f"CSV-Ergebnisse liegen in: {OUTPUT_CSV}")