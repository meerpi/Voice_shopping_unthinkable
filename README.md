# 🎙️ Voice Shopping Assistant with Neural Audio Enhancement & 1-Click Cart Dispatch

A state-of-the-art, hands-free voice shopping assistant built with **PyQt6**, **FastAPI**, **Gemini Multimodal Flash**, **Silero VAD**, **noisereduce**, and **Playwright**.

---

## 🌟 Key Capabilities

1. **6-Stage Neural & DSP Audio Preprocessing:**
   * **Stage 1 (Decode):** FFmpeg conversion to 16kHz mono float32 PCM.
   * **Stage 2 (Bandpass Filter):** 4th-order Butterworth filter (300 Hz – 7,500 Hz speech formant isolation).
   * **Stage 3 (Spectral Denoising):** Spectral Gating with attenuation limit (`prop_decrease=0.75`) preserving fragile consonants (*"t"*, *"k"*, *"f"*, *"s"*).
   * **Stage 4 (Silero VAD):** Deep neural speech timestamp extraction, trimming ~50% background silence and room reverberation.
   * **Stage 5 (AGC Normalization):** Automatic Gain Control normalized to -18.0 dBFS Studio RMS with ±0.95 Peak Limiter.
   * **Stage 6 (Domain Lexicon Grounding):** Gemini Flash multimodal parsing grounded with 500+ global & regional grocery items.

2. **Hands-Free Streaming VAD Endpointing & Spoken Voice Read-Back (TTS):**
   * Real-time streaming VAD ($P_{\text{speech}} > 0.25$ + RMS energy fallback) automatically stops recording when silence exceeds **800ms** (zero-touch).
   * Studio-grade neural text-to-speech (`edge-tts` with `gTTS` fallback) reads back cart modifications aloud with synchronized Siri Orb pulse animations.

3. **Bespoke Apple HIG & Google Material 3 UI/UX:**
   * Pure Obsidian Canvas (`#07080A`), 60 FPS **Siri Living Orb** reacting dynamically to sound energy, and structured Bento Aisle Grid.

4. **Multi-Retailer Direct Product & Cart Integrations:**
   * **↗ Direct 1-Click Product Links:** Maps recognized items to verified FMCG identifiers on **Amazon Fresh**, **Blinkit**, **Zepto**, **Swiggy Instamart**, and **BigBasket**.
   * **📋 Clipboard Order Generator & Search Grounding.**

---

## 🏛️ Engineering Approach (200-Word Summary)

> **Engineering Approach:**  
> Standard speech-to-text systems fail on commodity microphones due to high-frequency roll-off, ambient noise, and acoustic blending of brand names with phonetic priors. To solve this, we engineered a deterministic 6-stage audio pipeline combined with multimodal domain grounding. Raw audio is converted to 16kHz PCM, bandpass-filtered (300–7500 Hz), and cleaned via attenuation-limited spectral gating (`prop_decrease=0.75`) to suppress room noise without clipping weak consonants. Silero VAD provides real-time streaming endpointing (auto-stopping after 800ms silence with acoustic energy fallback) and trims dead air, followed by AGC leveling to -18.0 dBFS RMS. The enhanced audio is ingested directly by Gemini Flash, primed with an extensive grocery taxonomy, brand matrix (Amul, Kerrygold, Oatly, Tata), and compound intent parsing rules. For e-commerce execution, recognized items are mapped to verified FMCG identifiers, enabling 1-click direct product staging across Amazon Fresh and quick-commerce platforms (Blinkit, Zepto, Swiggy Instamart). Neural text-to-speech speaks back confirmations in real time. The native PyQt6 interface provides an Apple-inspired Siri Living Orb and fluid Bento board with zero-touch operation.

---

## 🚀 Quick Start & Installation

### 1. Prerequisites
* Python 3.10+
* FFmpeg installed on your system (`sudo apt install ffmpeg` or `brew install ffmpeg`)
* Google AI Studio Gemini API Key (Free-Tier)

### 2. Setup Environment
```bash
# Clone repository
git clone <your-repo-url>
cd unthinkable

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
playwright install chromium
```

### 3. Configure API Key
Create a `.env` file in the root directory:
```env
GEMINI_API_KEY=your_gemini_api_key_here
```

### 4. Launch Native Desktop App
```bash
.venv/bin/python desktop_app.py
```

### 5. Launch Web App Backend
```bash
.venv/bin/python -m uvicorn app:app --host 127.0.0.1 --port 8000
```
Visit `http://localhost:8000` in your web browser.

---

## 📊 Evaluation & Verification

| Voice Command | Raw Audio Challenge | Enhanced Audio Pipeline | Extracted Action | Retailer Output |
|---|---|---|---|---|
| *"Add 2 jackfruit"* | Blended fast speech (`/tuː-dʒæ-fruːt/`) misheard as *"fruit juice"* | Silero VAD + 3.2kHz peaking EQ + Catalog Lexicon | `Produce`: Jackfruit (Qty: 2.0) | Verified in Aisle Bento |
| *"Add 2 packs of Kerrygold butter and a bottle of Oatly oat milk"* | Multi-brand compound sentence | Spectral Gating + AGC Normalizer | `Dairy & Eggs`: Kerrygold Butter (Qty: 2), Oatly Oat Milk (Qty: 1) | **⚡ Amazon Cart**: Staged directly via ASIN protocol |
| *"Add 1 kg potatoes and find Colgate toothpaste under $5"* | Mixed Add + Search intent | 6-Stage Enhancement + Open Food Facts API | `Produce`: Potatoes (1.0 kg), Live Search: Colgate Toothpaste ($4.50) | **🤖 Playwright Agent**: Automated click into Blinkit / Zepto |
