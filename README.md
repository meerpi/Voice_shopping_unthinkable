---
title: Voice Shopping Assistant
emoji: 🛒
colorFrom: gray
colorTo: indigo
sdk: static
pinned: false
---

# Voice Shopping Assistant

A hands-free voice-driven shopping list manager with audio preprocessing, speech-to-intent grounding, neural speech feedback, and direct retailer product resolution.

---

## Overview

The application processes natural voice commands to manage a categorised grocery list, provide contextual substitutions and seasonal recommendations, and generate direct 1-click links to major e-commerce platforms. Built using FastAPI, PyQt6, Silero VAD, and Google Gemini Multimodal APIs.

---

## Engineering Approach (200-Word Summary)

Standard speech-to-text systems fail on commodity microphones due to high-frequency roll-off, ambient noise, and acoustic blending of brand names with phonetic priors. To solve this, we engineered a deterministic 6-stage audio pipeline combined with multimodal domain grounding. Raw audio is converted to 16kHz PCM, bandpass-filtered (300 to 7500 Hz), and cleaned via attenuation-limited spectral gating (`prop_decrease=0.75`) to suppress room noise without clipping weak consonants. Silero VAD provides real-time streaming endpointing (auto-stopping after 800ms silence with acoustic energy fallback) and trims dead air, followed by AGC leveling to -18.0 dBFS RMS. The enhanced audio is ingested directly by Gemini Flash, primed with an extensive grocery taxonomy, brand matrix (Amul, Kerrygold, Oatly, Tata), and compound intent parsing rules. For e-commerce execution, recognized items are mapped to verified FMCG identifiers, enabling 1-click direct product staging across Amazon Fresh and quick-commerce platforms (Blinkit, Zepto, Swiggy Instamart). Neural text-to-speech speaks back confirmations in real time. The native PyQt6 interface provides an Apple-inspired Siri Living Orb and fluid Bento board with zero-touch operation.

---

## Core System Architecture

### 1. Audio Processing Pipeline
* **Format Conversion:** Normalises arbitrary browser and desktop audio to 16kHz mono 32-bit float PCM via FFmpeg.
* **Formant Isolation:** 4th-order Butterworth bandpass filter spanning 300 Hz to 7500 Hz to isolate fundamental speech frequencies.
* **Spectral Gating:** Noise reduction with controlled attenuation (`prop_decrease=0.75`) to avoid artifact introduction in quiet phonemes.
* **Voice Activity Detection:** Deep learning VAD via Silero to strip leading/trailing silence and segment voiced speech.
* **Dynamic Normalisation:** Automatic Gain Control targeting -18 dBFS RMS with peak limiting at 0.95.

### 2. Hands-Free Conversational Endpointing
* Real-time 32ms audio buffer streaming into Silero VAD.
* Dual-condition trigger combining neural voice probability ($P > 0.25$) and RMS energy thresholds for sensitive microphone pickup.
* Automatic stream closure after 800ms of sustained silence following active speech.

### 3. Speech Synthesis Feedback
* Asynchronous neural text-to-speech synthesis using Edge-TTS with fallback to gTTS.
* Spoken read-back of cart modifications in user-selected locale (en-IN, en-US, en-GB, hi-IN, es-ES).

### 4. Retailer Resolution
* Direct catalog mapping for FMCG items to unique product identifiers.
* 1-click deep link resolution for Amazon Fresh, Blinkit, Zepto, Swiggy Instamart, and BigBasket.
* Structured clipboard export for offline review.

---

## Installation

### Prerequisites
* Python 3.10 or higher
* FFmpeg installed on system PATH
* Google AI Studio Gemini API Key

### Setup
```bash
git clone https://github.com/meerpi/Voice_shopping_unthinkable.git
cd Voice_shopping_unthinkable

python3 -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
```

### Environment Configuration
Create a `.env` file in the root directory:
```env
GEMINI_API_KEY=your_gemini_api_key_here
```

---

## Usage

### Desktop Application
Run the PyQt6 interface:
```bash
python desktop_app.py
```

### Web Application & REST API
Start the FastAPI server:
```bash
python -m uvicorn app:app --host 127.0.0.1 --port 8000
```
Open `http://localhost:8000` in a modern web browser.

---

## Verification Matrix

| Voice Input | Acoustic Challenge | Enhancement Applied | Resolved Entity | Store Action |
|---|---|---|---|---|
| *"Add 2 jackfruit"* | Fast connected speech | Bandpass filter + Silero VAD | Produce: Jackfruit (Qty: 2) | Direct product link |
| *"Add 2 packs of Kerrygold butter and oat milk"* | Multi-brand compound phrase | Spectral gating + AGC | Dairy: Kerrygold Butter (Qty: 2), Oat Milk (Qty: 1) | FMCG ASIN mapping |
| *"Add 1 kg potatoes and find Colgate toothpaste under $5"* | Mixed add and search intents | Gemini schema parsing | Produce: Potatoes (1 kg), Search: Colgate Toothpaste | Open Food Facts API lookup |

---

## Project Structure

```text
├── app.py                   # FastAPI backend, 6-stage audio DSP, Open Food Facts API & Gemini grounding
├── desktop_app.py           # Native PyQt6 desktop GUI (Hands-free VAD, Siri Orb, Bento shopping board)
├── retailer_cart_service.py # Direct retailer product resolution and deep-linking
├── tts_service.py           # Neural text-to-speech engine for spoken cart read-backs
├── static/                  # Responsive Web UI (index.html, manifest.json)
├── requirements.txt         # Pinned production dependencies
├── .gitignore               # Standard exclusions for environment, caches, and logs
└── README.md                # System documentation and engineering summary
```
