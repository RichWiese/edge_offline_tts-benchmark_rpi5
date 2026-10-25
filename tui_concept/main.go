/*
===============================================================================
Application:    main.go (Terminal User Interface)
Project:        Offline Text-to-Speech Benchmarking on ARM Edge Devices
Author:         Richard Wieser, BSc
Institution:    FH Technikum Wien, Master's Program in Embedded Systems
Year:           2026
Repository:     https://github.com/RichWiese/edge_offline_tts-benchmark_rpi5
License:        MIT License

Description:
    Interactive Terminal User Interface (TUI) prototype for remote headless
    TTS control over SSH on Raspberry Pi 5. Built with Bubble Tea.
    Automatically locks CPU governor to 'performance' during operation,
    executes Piper TTS inference, parses binary WAV headers, and reports
    real-time latency, throughput, and RTF metrics.
===============================================================================
*/

package main

import (
	"encoding/binary"
	"fmt"
	"io"
	"os"
	"os/exec"
	"os/signal"
	"path/filepath"
	"strings"
	"syscall"
	"time"

	"github.com/charmbracelet/bubbles/textinput"
	tea "github.com/charmbracelet/bubbletea"
)

// Default filesystem paths on Raspberry Pi 5
const (
	piperVenvPath = "/home/pi/piper_env"
	ramDiskPath   = "/dev/shm/piper_output.wav"
)

// -------------------------------------------------------------
// CPU Governor Management (Broadcom BCM2712 / RPi 5)
// -------------------------------------------------------------

// setCPUGovernor sets the CPU scaling governor across all 4 Cortex-A76 cores.
func setCPUGovernor(gov string) error {
	cmd := exec.Command("sudo", "sh", "-c", fmt.Sprintf("echo %s | tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor > /dev/null", gov))
	return cmd.Run()
}

// getCPUGovernor reads the current active governor from CPU core 0.
func getCPUGovernor() string {
	data, err := os.ReadFile("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor")
	if err != nil {
		return "ondemand"
	}
	return strings.TrimSpace(string(data))
}

// -------------------------------------------------------------
// Sequential Audio File Numbering (output_1.wav, output_2.wav)
// -------------------------------------------------------------

// getNextOutputFile finds the next unused filename to prevent overwriting past synthesis runs.
func getNextOutputFile() string {
	for i := 1; ; i++ {
		filename := fmt.Sprintf("output_%d.wav", i)
		if _, err := os.Stat(filename); os.IsNotExist(err) {
			return filename
		}
	}
}

// -------------------------------------------------------------
// Data Structures
// -------------------------------------------------------------

type ttsModel struct {
	Name          string
	Volume        int
	SaveFile      bool
	QualityLevels []string
	QualityIdx    int
}

type synthResult struct {
	ModelName     string
	Text          string
	OutputFile    string
	SynthDuration float64
	AudioDuration float64
	RTF           float64
	Throughput    float64
	SampleRate    uint32
	Err           error
}

type model struct {
	step         int
	projectName  textinput.Model
	models       []ttsModel
	modelIdx     int
	actionIdx    int
	isGenerating bool
	lastResult   *synthResult
}

// initialModel initializes the Bubble Tea application state and TTS candidate list.
func initialModel() model {
	ti := textinput.New()
	ti.Placeholder = "Text für die Sprachsynthese eingeben..."
	ti.Focus()

	models := []ttsModel{
		{
			Name:          "Piper TTS",
			Volume:        100,
			SaveFile:      false,
			QualityLevels: []string{"medium", "low", "high", "emotional-medium"},
			QualityIdx:    0,
		},
		{
			Name:          "Kokoro-82M (ONNX)",
			Volume:        100,
			SaveFile:      false,
			QualityLevels: []string{"martin"},
			QualityIdx:    0,
		},
		{
			Name:          "Coqui-VITS",
			Volume:        100,
			SaveFile:      false,
			QualityLevels: []string{"thorsten_vits", "css10_vits_neon"},
			QualityIdx:    0,
		},
		{
			Name:          "Kikiri-TTS",
			Volume:        100,
			SaveFile:      false,
			QualityLevels: []string{"bernd", "martin", "victoria"},
			QualityIdx:    0,
		},
	}

	return model{
		step:        1,
		projectName: ti,
		models:      models,
		modelIdx:    0,
		actionIdx:   0,
	}
}

// -------------------------------------------------------------
// Binary WAV Header Parser
// -------------------------------------------------------------

// readWavMetadata parses sample rate and calculates audio duration from the raw WAV header.
func readWavMetadata(filePath string) (float64, uint32, error) {
	file, err := os.Open(filePath)
	if err != nil {
		return 0, 0, err
	}
	defer file.Close()

	var header struct {
		RiffID   [4]byte
		FileSize uint32
		WaveID   [4]byte
	}
	if err := binary.Read(file, binary.LittleEndian, &header); err != nil {
		return 0, 0, err
	}

	if string(header.RiffID[:]) != "RIFF" || string(header.WaveID[:]) != "WAVE" {
		return 0, 0, fmt.Errorf("keine gültige WAV-Datei")
	}

	var sampleRate uint32
	var byteRate uint32
	var dataSize uint32

	for {
		var chunk struct {
			ID   [4]byte
			Size uint32
		}
		if err := binary.Read(file, binary.LittleEndian, &chunk); err != nil {
			break
		}

		chunkID := string(chunk.ID[:])
		if chunkID == "fmt " {
			var fmtChunk struct {
				AudioFormat   uint16
				NumChannels   uint16
				SampleRate    uint32
				ByteRate      uint32
				BlockAlign    uint16
				BitsPerSample uint16
			}
			if err := binary.Read(file, binary.LittleEndian, &fmtChunk); err != nil {
				return 0, 0, err
			}
			sampleRate = fmtChunk.SampleRate
			byteRate = fmtChunk.ByteRate

			if chunk.Size > 16 {
				file.Seek(int64(chunk.Size-16), io.SeekCurrent)
			}
		} else if chunkID == "data" {
			dataSize = chunk.Size
			break
		} else {
			file.Seek(int64(chunk.Size), io.SeekCurrent)
		}
	}

	if byteRate == 0 {
		return 0, 0, fmt.Errorf("ungültige Audiodaten im WAV-Header")
	}

	duration := float64(dataSize) / float64(byteRate)
	return duration, sampleRate, nil
}

// -------------------------------------------------------------
// Piper Model Locator & Inference Execution
// -------------------------------------------------------------

// locatePiperModel searches common local directories for the requested ONNX voice model.
func locatePiperModel(quality string) (string, error) {
	searchDirs := []string{
		"/home/pi/piper/models",
		filepath.Join(piperVenvPath, "models"),
		piperVenvPath,
		"./models",
		".",
	}

	pattern := fmt.Sprintf("*%s*.onnx", quality)

	for _, dir := range searchDirs {
		matches, err := filepath.Glob(filepath.Join(dir, pattern))
		if err == nil && len(matches) > 0 {
			return matches[0], nil
		}
	}

	// Fallback search: any .onnx model file
	for _, dir := range searchDirs {
		matches, err := filepath.Glob(filepath.Join(dir, "*.onnx"))
		if err == nil && len(matches) > 0 {
			return matches[0], nil
		}
	}

	return "", fmt.Errorf("kein Piper ONNX-Modell für '%s' gefunden", quality)
}

// executePiperSynthesis executes speech synthesis as an external CLI subprocess.
func executePiperSynthesis(m model) tea.Cmd {
	return func() tea.Msg {
		text := strings.TrimSpace(m.projectName.Value())
		if text == "" {
			return synthResult{Err: fmt.Errorf("bitte zuerst einen Text eingeben")}
		}

		selectedModel := m.models[m.modelIdx]
		if selectedModel.Name != "Piper TTS" {
			return synthResult{
				ModelName: selectedModel.Name,
				Err:       fmt.Errorf("das Modell '%s' ist im Prototyp noch nicht angebunden. Bitte 'Piper TTS' wählen", selectedModel.Name),
			}
		}

		piperBinary := filepath.Join(piperVenvPath, "bin", "piper")
		if _, err := os.Stat(piperBinary); os.IsNotExist(err) {
			if sysPath, err := exec.LookPath("piper"); err == nil {
				piperBinary = sysPath
			} else {
				return synthResult{Err: fmt.Errorf("piper Binary unter '%s' nicht gefunden", piperBinary)}
			}
		}

		quality := selectedModel.QualityLevels[selectedModel.QualityIdx]
		modelFile, err := locatePiperModel(quality)
		if err != nil {
			return synthResult{Err: err}
		}

		outputFile := ramDiskPath
		if selectedModel.SaveFile {
			outputFile = getNextOutputFile()
		}

		// Keep arguments minimal; Piper uses optimal model defaults from *.onnx.json
		args := []string{
			"--model", modelFile,
			"--output_file", outputFile,
		}

		cmd := exec.Command(piperBinary, args...)
		cmd.Stdin = strings.NewReader(text)

		startTime := time.Now()
		out, err := cmd.CombinedOutput()
		synthDuration := time.Since(startTime).Seconds()

		if err != nil {
			return synthResult{Err: fmt.Errorf("piper-Fehler: %v (%s)", err, string(out))}
		}

		audioDuration, sampleRate, err := readWavMetadata(outputFile)
		if err != nil {
			return synthResult{Err: fmt.Errorf("audio-Analysefehler: %v", err)}
		}

		rtf := 0.0
		throughput := 0.0
		if audioDuration > 0 {
			rtf = synthDuration / audioDuration
		}
		if synthDuration > 0 {
			throughput = (audioDuration * float64(sampleRate)) / synthDuration
		}

		// Asynchronously play synthesized audio via ALSA if available
		go func() {
			if aplayPath, err := exec.LookPath("aplay"); err == nil {
				exec.Command(aplayPath, "-q", outputFile).Run()
			}
		}()

		return synthResult{
			ModelName:     fmt.Sprintf("Piper (%s)", filepath.Base(modelFile)),
			Text:          text,
			OutputFile:    outputFile,
			SynthDuration: synthDuration,
			AudioDuration: audioDuration,
			RTF:           rtf,
			Throughput:    throughput,
			SampleRate:    sampleRate,
		}
	}
}

// -------------------------------------------------------------
// Bubble Tea Navigation & Event Loop
// -------------------------------------------------------------

func (m *model) setStep(step int) {
	m.step = step
	if m.step == 1 {
		m.projectName.Focus()
	} else {
		m.projectName.Blur()
	}
}

func (m *model) nextStep() {
	next := m.step + 1
	if next > 5 {
		next = 1
	}
	m.setStep(next)
}

func (m model) Init() tea.Cmd { return nil }

func (m model) Update(msg tea.Msg) (tea.Model, tea.Cmd) {
	var cmd tea.Cmd

	switch msg := msg.(type) {
	case synthResult:
		m.isGenerating = false
		m.lastResult = &msg
		return m, nil

	case tea.KeyMsg:
		if m.isGenerating {
			return m, nil
		}

		switch msg.String() {
		case "ctrl+c", "q":
			return m, tea.Quit
		case "tab":
			m.nextStep()
		case "enter":
			if m.step == 5 {
				if m.actionIdx == 0 {
					m.isGenerating = true
					m.lastResult = nil
					return m, executePiperSynthesis(m)
				}
				return m, tea.Quit
			}
			m.nextStep()
		case "up":
			if m.step == 2 && m.modelIdx > 0 {
				m.modelIdx--
			}
		case "down":
			if m.step == 2 && m.modelIdx < len(m.models)-1 {
				m.modelIdx++
			}
		case "left":
			if m.step == 3 {
				if m.models[m.modelIdx].QualityIdx > 0 {
					m.models[m.modelIdx].QualityIdx--
				}
			}
			if m.step == 5 {
				m.actionIdx = 0
			}
		case "right":
			if m.step == 3 {
				levels := m.models[m.modelIdx].QualityLevels
				if m.models[m.modelIdx].QualityIdx < len(levels)-1 {
					m.models[m.modelIdx].QualityIdx++
				}
			}
			if m.step == 5 {
				m.actionIdx = 1
			}
		case " ":
			if m.step == 4 {
				m.models[m.modelIdx].SaveFile = !m.models[m.modelIdx].SaveFile
			}
		}
	}

	if m.step == 1 {
		m.projectName, cmd = m.projectName.Update(msg)
	}

	return m, cmd
}

func (m model) View() string {
	s := "┌────────────────────────────────────────────────────────┐\n"
	s += "│   Offline Text-to-Speech Edge Synthesizer (RPi 5)      │\n"
	s += "└────────────────────────────────────────────────────────┘\n\n"

	getStepPointer := func(id int) string {
		if m.step == id {
			return "> "
		}
		return "  "
	}

	selectedModel := m.models[m.modelIdx]

	s += fmt.Sprintf("%sTEXT: %s\n\n", getStepPointer(1), m.projectName.View())

	s += fmt.Sprintf("%sMODELL AUSWAHL:\n", getStepPointer(2))
	for i, currentModel := range m.models {
		itemCursor := "    "
		if i == m.modelIdx {
			itemCursor = "  > "
		}
		s += fmt.Sprintf("%s%s\n", itemCursor, currentModel.Name)
	}
	s += "\n"

	currentQuality := selectedModel.QualityLevels[selectedModel.QualityIdx]
	s += fmt.Sprintf("%sMODELL-KONFIGURATION / STIMME: < %s >\n\n", getStepPointer(3), currentQuality)

	check := "[ ] (flüchtig in /dev/shm/piper_output.wav)"
	if selectedModel.SaveFile {
		check = "[x] (dauerhaft als ./output_<n>.wav)"
	}
	s += fmt.Sprintf("%sPERSISTIEREN: %s\n\n", getStepPointer(4), check)

	genBtn := "[ SYNTHESIZE ]"
	canBtn := "[ BEENDEN ]"
	if m.step == 5 {
		if m.actionIdx == 0 {
			genBtn = "> [ SYNTHESIZE ] <"
		} else {
			canBtn = "> [ BEENDEN ] <"
		}
	}
	if m.isGenerating {
		genBtn = "[ GENERIERT SPRACHE... ]"
	}
	s += fmt.Sprintf("%sAKTION: %s   %s\n\n", getStepPointer(5), genBtn, canBtn)

	// Result and telemetry display
	if m.lastResult != nil {
		res := m.lastResult
		s += "──────────────────────── TELEMETRIE & RESULTAT ────────────────────────\n"
		if res.Err != nil {
			s += fmt.Sprintf(" Status:   FEHLER\n Meldung:  %v\n", res.Err)
		} else {
			rtfStatus := "Echtzeit (E2E)"
			if res.RTF >= 1.0 {
				rtfStatus = "CLI-Kaltstart dominiert bei Kurzsätzen"
			}
			s += fmt.Sprintf(" Modell:        %s\n", res.ModelName)
			s += fmt.Sprintf(" Synthesezeit:  %.3f s (inkl. Prozessstart)\n", res.SynthDuration)
			s += fmt.Sprintf(" Audiodauer:    %.3f s (%.1f kHz)\n", res.AudioDuration, float64(res.SampleRate)/1000.0)
			s += fmt.Sprintf(" Real-Time F.:  %.4f [%s]\n", res.RTF, rtfStatus)
			s += fmt.Sprintf(" Durchsatz:     ~%.0f Samples/s\n", res.Throughput)
			s += fmt.Sprintf(" Ausgabepfad:   %s\n", res.OutputFile)
		}
		s += "───────────────────────────────────────────────────────────────────────\n"
	}

	s += "\n(Tab/Enter: Weiter | Pfeiltasten: Navigation / Stimme wählen | Leertaste: Umschalten | Q: Beenden)"
	return s
}

// -------------------------------------------------------------
// Main Entrypoint
// -------------------------------------------------------------

func main() {
	origGov := getCPUGovernor()

	// Switch CPU governor to performance for deterministic timing
	if err := setCPUGovernor("performance"); err != nil {
		fmt.Printf("Hinweis: Konnte Governor nicht auf 'performance' setzen: %v\n", err)
	} else {
		fmt.Println("⚡ CPU-Governor auf 'performance' gesetzt.")
	}

	// Capture interrupt signals (SIGINT, SIGTERM) to restore original governor on exit
	sigChan := make(chan os.Signal, 1)
	signal.Notify(sigChan, os.Interrupt, syscall.SIGTERM)
	go func() {
		<-sigChan
		_ = setCPUGovernor("ondemand")
		os.Exit(0)
	}()

	// Always restore governor when application terminates normally
	defer func() {
		targetGov := "ondemand"
		if origGov != "" && origGov != "performance" {
			targetGov = origGov
		}
		_ = setCPUGovernor(targetGov)
		fmt.Printf("🌿 CPU-Governor zurück auf '%s' gestellt.\n", targetGov)
	}()

	p := tea.NewProgram(initialModel())
	if _, err := p.Run(); err != nil {
		fmt.Printf("Fehler beim Starten des TUI: %v\n", err)
	}
}