# Voice Shopping Assistant

A hands-free, voice-enabled grocery shopping assistant that converts spoken natural language into an organized, categorized shopping list with brand resolution, smart substitutions, and direct retailer checkout links.

🌐 **Live Hosted Application:** [https://voice-shopping-unthinkable.onrender.com/](https://voice-shopping-unthinkable.onrender.com/)

Available both as a **live web application (FastAPI)** and a **native desktop application (PyQt6)**.

---

## 200-Word Engineering Summary

Traditional speech-to-text pipelines struggle with grocery ordering because users speak quickly, list multiple items with mixed units in a single breath, use regional terms, and mention specific brand names that generic speech models mistranscribe. To address this, this project bypasses separate, fragile ASR steps and feeds audio directly to Google Gemini's multimodal audio API using strict structured JSON schemas and separated system instructions.

On the desktop client, audio is captured at 16 kHz mono via `sounddevice` and monitored in real time using Silero VAD to automatically detect speech start and stop boundaries (hands-free endpointing). The web interface uses browser-level audio constraints (`echoCancellation`, `noiseSuppression`, `autoGainControl`) and streams WebM Opus audio directly to the FastAPI backend. 

The backend validates inputs against an Indian FMCG brand taxonomy (Amul, Aashirvaad, Tata, Britannia, Fortune, MDH, Dettol, Colgate, etc.) and automatically categorizes items into Produce, Dairy & Eggs, Pantry, Bakery, and Household aisles. Validated items generate 1-click cart links to Amazon Fresh, Blinkit, Zepto, and Instamart, while returning spoken audio feedback and seasonal recommendations.

---

## Key Features

- **Hands-Free Voice Input:** Real-time Voice Activity Detection (Silero VAD) automatically detects when you finish speaking and processes the command without needing manual button presses.
- **Compound Intent Parsing:** Handles complex multi-item sentences like *"Add 4 apples, 5 oranges, and 1 packet of Amul butter"* in a single utterance.
- **Regional & FMCG Brand Disambiguation:** Recognizes popular Indian and international grocery brands (Amul, Mother Dairy, Tata, Aashirvaad, Parle-G, Britannia, Haldiram's, Surf Excel, etc.) and attaches verified brand badges.
- **Phonetic & Hindi/Hinglish Tolerance:** Handles common accent variations and bilingual phrases like *"2 kilo aalu"*, *"1 litre doodh"*, or *"packet cheeni"*.
- **Smart Suggestions & Substitutions:** Automatically suggests alternatives for out-of-stock or common items (e.g. Oat Milk for Whole Milk, Tofu for Paneer) and displays seasonal produce picks based on the current calendar month.
- **1-Click Retailer Deep Links:** Generates direct search and add-to-cart URLs for Amazon Fresh, Blinkit, Zepto, Swiggy Instamart, and BigBasket.
- **Spoken Audio Confirmations:** Reads back cart updates using text-to-speech with multi-language/accent support (`en-IN`, `en-US`, `en-GB`, `hi-IN`).
- **Product Nutrition & Price Lookup:** Queries the Open Food Facts API for product details, Nutri-Score grades, and price estimates.

---

## Architecture & How It Works

```
[ Microphones / Web Audio ]
           │
           ▼
[ Client-Side Audio Capture ]
   ├── Desktop: sounddevice 16kHz PCM stream + Silero VAD
   └── Web: MediaRecorder (Opus/WebM) + Web Audio API
           │
           ▼
[ FastAPI Backend / Python Controller ]
   ├── Audio Validation & Formatting
   └── Gemini Multimodal Pipeline (Structured Output Schema)
           │
           ├── Intent Extraction (ADD / REMOVE / SEARCH / CLEAR)
           ├── FMCG Brand & Aisle Categorization
           └── Quantity & Unit Normalization
           │
           ▼
[ Post-Processing & Integration ]
   ├── Open Food Facts API (Nutritional data & search)
   ├── Smart Substitution Engine (Heuristic + Seasonal graph)
   ├── Retailer Direct Linking (Amazon, Blinkit, Zepto, Instamart)
   └── Text-to-Speech Feedback (Edge-TTS / Web Speech API)
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| **Backend & API** | Python 3.10+, FastAPI, Uvicorn, Pydantic v2 |
| **Desktop Client** | PyQt6, QThread, sounddevice, NumPy |
| **Web Client** | HTML5, Vanilla JavaScript, CSS3, Web Audio API |
| **Speech & LLM** | Google GenAI SDK (`gemini-flash-lite-latest`), Silero VAD, PyTorch |
| **Audio Tools** | FFmpeg, SciPy, noisereduce |
| **External APIs** | Open Food Facts REST API |
| **Voice Synthesis** | Edge-TTS, gTTS, Web Speech Synthesis API |

---

## Getting Started

### Prerequisites
- Python 3.10 or higher
- `ffmpeg` installed and available on your system PATH
- A Google AI Studio Gemini API Key (free tier available at [aistudio.google.com](https://aistudio.google.com/))

### 1. Clone & Install Dependencies

```bash
git clone https://github.com/meerpi/Voice_shopping_unthinkable.git
cd Voice_shopping_unthinkable

# Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure Environment Variables

Create a `.env` file in the project root:

```env
GEMINI_API_KEY=your_gemini_api_key_here
```

---

## Running the Application

### Option A: Web Application
Start the FastAPI server:

```bash
python -m uvicorn app:app --host 127.0.0.1 --port 8000 --reload
```

Open `http://localhost:8000` in your web browser. Click the voice orb and speak your grocery items.

### Option B: Desktop GUI
Launch the PyQt6 native desktop app:

```bash
python desktop_app.py
```

The desktop app runs fully hands-free — click the mic orb once to start listening, and it will continuously process commands as you speak.

---

## Example Voice Commands

| Command | Action Taken |
|---|---|
| *"Add 4 apples and 5 oranges"* | Adds 4 Apples and 5 Oranges to **Produce** |
| *"Get 1 packet Amul butter and 2 packets bread"* | Adds Amul Butter (`[AMUL]`) to **Dairy & Eggs** and Bread to **Bakery** |
| *"Buy 2 kg Aashirvaad atta and 1 litre Fortune oil"* | Adds Atta and Oil with brand tags to **Pantry** |
| *"Remove apples"* | Removes Apples from the shopping list |
| *"Find Colgate toothpaste under 5 dollars"* | Searches Open Food Facts for Colgate items filtered by price |
| *"Clear my cart"* | Empties the shopping list |

---

## Project Structure

```
.
├── app.py                   # FastAPI backend, Gemini pipeline, brand & category logic
├── desktop_app.py           # PyQt6 desktop GUI with hands-free Silero VAD audio capture
├── retailer_cart_service.py # Retailer deep-link generator (Amazon, Blinkit, Zepto, etc.)
├── tts_service.py           # Text-to-speech audio feedback generator
├── static/                  # Web frontend assets
│   ├── index.html           # Single-page web application UI
│   └── manifest.json        # Web app manifest
├── .devcontainer/           # GitHub Codespaces container setup with auto-configured FFmpeg
├── requirements.txt         # Pinned Python package dependencies
├── .env.example             # Example environment variable file
└── README.md                # Project documentation
```

---

## License

This project is licensed under the MIT License.
