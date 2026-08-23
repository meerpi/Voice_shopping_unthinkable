import os
import json
import datetime
import io
import subprocess
from typing import List, Dict, Optional, Literal
from fastapi import FastAPI, UploadFile, File
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel, Field
import httpx
import numpy as np
from scipy.signal import butter, filtfilt
from dotenv import load_dotenv

load_dotenv(".env")

from google import genai
from google.genai import types

app = FastAPI(title="Voice Shopping Assistant")

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
gemini_client = genai.Client(api_key=GEMINI_API_KEY) if GEMINI_API_KEY else None

os.makedirs("debug_audio", exist_ok=True)

_vad_model = None
_nr_module = None

def _get_vad_model():
    global _vad_model
    if _vad_model is None:
        try:
            from silero_vad import load_silero_vad
            _vad_model = load_silero_vad()
        except Exception:
            pass
    return _vad_model

def _get_noisereduce():
    global _nr_module
    if _nr_module is None:
        try:
            import noisereduce as nr
            _nr_module = nr
        except ImportError:
            pass
    return _nr_module

def decode_webm_to_pcm(audio_bytes: bytes, target_sr: int = 16000) -> np.ndarray:
    cmd = [
        'ffmpeg', '-i', 'pipe:0',
        '-f', 'f32le',
        '-ac', '1',
        '-ar', str(target_sr),
        '-acodec', 'pcm_f32le',
        'pipe:1'
    ]
    proc = subprocess.run(cmd, input=audio_bytes, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=10)
    if proc.returncode != 0:
        raise RuntimeError("ffmpeg decode failed")
    return np.frombuffer(proc.stdout, dtype=np.float32)

def encode_pcm_to_webm(pcm: np.ndarray, sr: int = 16000) -> bytes:
    cmd = [
        'ffmpeg', '-y',
        '-f', 'f32le',
        '-ar', str(sr),
        '-ac', '1',
        '-i', 'pipe:0',
        '-c:a', 'libopus',
        '-b:a', '32k',
        '-f', 'webm',
        'pipe:1'
    ]
    proc = subprocess.run(cmd, input=pcm.astype(np.float32).tobytes(), stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=10)
    if proc.returncode != 0:
        raise RuntimeError("ffmpeg encode failed")
    return proc.stdout

def butterworth_bandpass(pcm: np.ndarray, sr: int = 16000, lowcut: float = 300.0, highcut: float = 7500.0, order: int = 4) -> np.ndarray:
    nyq = 0.5 * sr
    low = lowcut / nyq
    high = min(highcut / nyq, 0.99)
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, pcm).astype(np.float32)

def spectral_noise_reduce(pcm: np.ndarray, sr: int = 16000) -> np.ndarray:
    nr = _get_noisereduce()
    if nr is None:
        return pcm
    try:
        noise_clip = pcm[:int(sr * 0.3)] if len(pcm) > int(sr * 0.3) else None
        # prop_decrease=0.75 preserves low-SNR consonants from neural over-suppression
        return nr.reduce_noise(
            y=pcm, sr=sr, y_noise=noise_clip,
            prop_decrease=0.75, stationary=False,
            n_fft=512, freq_mask_smooth_hz=300
        ).astype(np.float32)
    except Exception:
        return pcm

def vad_trim(pcm: np.ndarray, sr: int = 16000) -> np.ndarray:
    import torch
    model = _get_vad_model()
    if model is None:
        return pcm
    try:
        from silero_vad import get_speech_timestamps
        speech_ts = get_speech_timestamps(
            torch.from_numpy(pcm).float(), model,
            sampling_rate=sr, threshold=0.3,
            min_speech_duration_ms=100, min_silence_duration_ms=100, speech_pad_ms=60
        )
        if not speech_ts:
            return pcm
        return np.concatenate([pcm[ts['start']:ts['end']] for ts in speech_ts]).astype(np.float32)
    except Exception:
        return pcm

def agc_normalize(pcm: np.ndarray, target_db: float = -18.0) -> np.ndarray:
    rms = np.sqrt(np.mean(pcm ** 2))
    if rms < 1e-8:
        return pcm
    current_db = 20 * np.log10(rms + 1e-9)
    gain_db = np.clip(target_db - current_db, -20.0, 30.0)
    return (pcm * (10 ** (gain_db / 20.0))).astype(np.float32)

def peak_limit(pcm: np.ndarray, ceiling: float = 0.95) -> np.ndarray:
    return np.clip(pcm, -ceiling, ceiling).astype(np.float32)

def enhance_audio(raw_webm_bytes: bytes) -> tuple:
    sr = 16000
    diag = {}
    pcm = decode_webm_to_pcm(raw_webm_bytes, target_sr=sr)
    diag['raw_duration_s'] = round(len(pcm) / sr, 2)
    diag['raw_peak_dbfs'] = round(float(20 * np.log10(np.max(np.abs(pcm)) + 1e-9)), 1)
    diag['raw_rms_dbfs'] = round(float(20 * np.log10(np.sqrt(np.mean(pcm**2)) + 1e-9)), 1)

    pcm = butterworth_bandpass(pcm, sr=sr)
    pcm = spectral_noise_reduce(pcm, sr=sr)
    pcm = vad_trim(pcm, sr=sr)
    diag['vad_duration_s'] = round(len(pcm) / sr, 2)

    pcm = agc_normalize(pcm, target_db=-18.0)
    pcm = peak_limit(pcm, ceiling=0.95)
    diag['enhanced_rms_dbfs'] = round(float(20 * np.log10(np.sqrt(np.mean(pcm**2)) + 1e-9)), 1)

    enhanced_webm = encode_pcm_to_webm(pcm, sr=sr)
    return enhanced_webm, diag

class ExtractedItem(BaseModel):
    product_name: str = Field(description="Normalized generic product name")
    brand_hint: Optional[str] = Field(default=None, description="Exact brand name if specified")
    variant: Optional[str] = Field(default=None, description="Product variant or specifier")
    quantity: float = Field(default=1.0, description="Numeric count")
    unit: str = Field(default="item", description="Unit of measurement")
    category: Literal["Produce", "Dairy & Eggs", "Meat & Seafood", "Pantry", "Bakery", "Frozen", "Beverages", "Snacks", "Household", "Personal Care"]
    max_price: Optional[float] = None

class VoiceCommandResult(BaseModel):
    intent: Literal["ADD", "REMOVE", "SEARCH", "COMPOUND", "GET_SUGGESTIONS", "CLEAR", "UNKNOWN"]
    detected_language: str = "en"
    transcript: str = Field(description="Exact verbatim transcription")
    items_to_add: List[ExtractedItem] = Field(default_factory=list)
    items_to_remove: List[str] = Field(default_factory=list)
    search_query: Optional[str] = None
    search_max_price: Optional[float] = None
    feedback_message: str

class ShoppingCart:
    def __init__(self):
        self.items: Dict[str, Dict] = {}
        self.history: List[str] = ["Milk", "Bread", "Eggs", "Bananas", "Butter", "Coffee", "Rice"]

    def add(self, item: ExtractedItem) -> str:
        display_name = f"{item.brand_hint} {item.product_name}".strip() if item.brand_hint else item.product_name
        if item.variant:
            display_name = f"{display_name} ({item.variant})"
        
        key = item.product_name.lower().strip()
        if key in self.items:
            self.items[key]["quantity"] += item.quantity
            return f"Updated {display_name} to {self.items[key]['quantity']:g} {item.unit}"
        else:
            self.items[key] = {
                "name": display_name,
                "base_name": item.product_name,
                "quantity": item.quantity,
                "unit": item.unit,
                "category": item.category,
                "brand_hint": item.brand_hint,
                "variant": item.variant,
                "completed": False
            }
            return f"Added {item.quantity:g} {item.unit} of {display_name}"

    def remove(self, product_name: str) -> str:
        key = product_name.lower().strip()
        matches = [k for k in self.items.keys() if key in k or k in key]
        if matches:
            removed = self.items.pop(matches[0])
            return f"Removed {removed['name']}"
        return f"'{product_name}' not found in cart"

    def get_categorized(self) -> Dict[str, List[Dict]]:
        cats: Dict[str, List[Dict]] = {}
        for itm in self.items.values():
            cats.setdefault(itm["category"], []).append(itm)
        return cats

    def clear(self):
        self.items.clear()

cart = ShoppingCart()

class SmartSuggestions:
    SUBS = {
        "milk": ["Oat Milk", "Almond Milk", "Soy Milk", "Fairlife Lactose-Free"],
        "butter": ["Kerrygold Butter", "Plant Butter", "Ghee"],
        "pasta": ["Chickpea Pasta", "Whole Wheat Pasta"],
        "sugar": ["Stevia", "Monk Fruit Sweetener", "Raw Honey"],
        "rice": ["Basmati Rice", "Quinoa", "Cauliflower Rice"],
        "jackfruit": ["Artichoke Hearts", "Organic Tofu", "Canned Jackfruit"],
        "bread": ["Organic Whole Grain", "Sourdough"],
        "coffee": ["Dark Roast", "Nespresso Pods", "Espresso Beans"]
    }
    SEASONS = {
        "Spring": ["Asparagus", "Strawberries", "Fresh Peas", "Jackfruit"],
        "Summer": ["Watermelon", "Mangoes", "Peaches", "Sweet Corn"],
        "Autumn": ["Honeycrisp Apples", "Pumpkin", "Squash", "Pomegranates"],
        "Winter": ["Citrus", "Fresh Guava", "Kale", "Brussels Sprouts"]
    }

    @classmethod
    def current_season(cls) -> str:
        m = datetime.datetime.now().month
        return "Spring" if m in [3,4,5] else "Summer" if m in [6,7,8] else "Autumn" if m in [9,10,11] else "Winter"

    @classmethod
    def get_substitutes(cls, name: str) -> List[str]:
        for k, v in cls.SUBS.items():
            if k in name.lower():
                return v
        return []

    @classmethod
    def get_history_recs(cls, current_keys: List[str]) -> List[str]:
        return [h for h in cart.history if h.lower() not in [k.lower() for k in current_keys]][:4]

async def search_open_food_facts(query: str, max_price: Optional[float] = None, limit: int = 4) -> List[Dict]:
    url = f"https://world.openfoodfacts.org/cgi/search.pl?search_terms={query}&search_simple=1&action=process&json=1&page_size=10"
    try:
        async with httpx.AsyncClient(timeout=4.0) as client:
            resp = await client.get(url, headers={"User-Agent": "VoiceShoppingAssistant/1.0"})
            data = resp.json()
            results = []
            for p in data.get("products", []):
                name = p.get("product_name", query.title())
                brand = p.get("brands", "Generic")
                price = round(2.50 + (abs(hash(name)) % 550) / 100.0, 2)
                if max_price is None or price <= max_price:
                    results.append({
                        "name": name,
                        "brand": brand,
                        "price": f"${price:.2f}",
                        "nutriscore": p.get("nutriscore_grade", "A").upper()
                    })
                if len(results) >= limit:
                    break
            return results or [{"name": query.title(), "brand": "Organic Choice", "price": "$4.50", "nutriscore": "A"}]
    except Exception:
        return [{"name": query.title(), "brand": "Generic Brand", "price": "$3.50", "nutriscore": "A"}]

FLASH_MODELS = [
    "gemini-3.5-flash-lite",
    "gemini-flash-latest",
    "gemini-3.5-flash",
    "gemini-3.7-flash"
]

EXPANDED_GROCERY_PROMPT = """
You are a voice shopping assistant intelligence system with acoustic tolerance and FMCG brand grounding.

Brand Disambiguation:
- Dairy/Milks: Amul, Kerrygold, Oatly, Silk, Horizon Organic, Fairlife, Chobani, Vital Farms, Mother Dairy.
- Pantry: Tata, Aashirvaad, Barilla, Heinz, Kraft, Kikkoman, San Marzano, King Arthur.
- Snacks/Breakfast: Kellogg's, Quaker, Lays, Doritos, Haldiram's, Oreo.
- Personal: Colgate, Sensodyne, Crest, Dettol, Dawn, Tide.

Produce/Staples Grounding:
- Tropical & Regional: Jackfruit, Dragonfruit, Mango, Papaya, Guava, Chikoo, Coconut, Pomegranate, Bananas, Bitter Gourd (Karela), Bottle Gourd (Lauki), Okra (Bhindi), Eggplant, Spinach (Palak), Ginger, Garlic, Potatoes, Tomatoes, Onions.
- Grains: Basmati Rice, Atta, Flour, Olive Oil, Mustard Oil, Ghee, Eggs, Milk.

Phonetic & Slur Handling:
- 'tu-ja-froot' / 'toojack' -> 'Two Jackfruit' (qty: 2, Produce)
- 'fedex' / 'five x' -> '5 Eggs' (qty: 5, Dairy & Eggs)
- 'little milk' / 'a litre milk' -> '1 Liter Milk' (qty: 1, Dairy & Eggs)
- 'ek kilo aalu' -> 'Potatoes' (qty: 1, unit: 'kg', Produce)

Parse into structured VoiceCommandResult with conversational feedback_message.
"""

@app.post("/api/voice-audio")
async def process_voice_audio(file: UploadFile = File(...)):
    audio_bytes = await file.read()
    mime_type = file.content_type or "audio/webm"

    try:
        send_bytes, diag = enhance_audio(audio_bytes)
        send_mime = "audio/webm"
    except Exception:
        send_bytes, send_mime, diag = audio_bytes, mime_type, {}

    cart_keys = list(cart.items.keys())
    cart_context = f"Current Cart: {', '.join(cart_keys)}" if cart_keys else "Cart is empty."
    full_prompt = f"{EXPANDED_GROCERY_PROMPT}\n\n{cart_context}"

    parsed_cmd = None
    if gemini_client:
        for m in FLASH_MODELS:
            try:
                res = gemini_client.models.generate_content(
                    model=m,
                    contents=[full_prompt, types.Part.from_bytes(data=send_bytes, mime_type=send_mime)],
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json",
                        response_schema=VoiceCommandResult,
                        temperature=0.1
                    )
                )
                parsed_cmd = VoiceCommandResult.model_validate_json(res.text)
                break
            except Exception:
                continue

    if not parsed_cmd:
        return JSONResponse({"error": "Audio parsing failed"}, status_code=500)

    messages, suggested_subs, search_results = [], [], []

    for itm in parsed_cmd.items_to_add:
        messages.append(cart.add(itm))
        subs = SmartSuggestions.get_substitutes(itm.product_name)
        if subs: suggested_subs.extend(subs)

    for rm_name in parsed_cmd.items_to_remove:
        messages.append(cart.remove(rm_name))

    if parsed_cmd.intent == "CLEAR":
        cart.clear()
        messages.append("Cart cleared.")

    if parsed_cmd.search_query:
        search_results = await search_open_food_facts(parsed_cmd.search_query, max_price=parsed_cmd.search_max_price)

    season = SmartSuggestions.current_season()
    return {
        "transcript": parsed_cmd.transcript,
        "intent": parsed_cmd.intent,
        "language": parsed_cmd.detected_language,
        "messages": messages,
        "tts_message": parsed_cmd.feedback_message,
        "enhancement": diag,
        "cart": cart.get_categorized(),
        "substitutes": list(set(suggested_subs)),
        "search_results": search_results,
        "seasonal_picks": SmartSuggestions.SEASONS[season][:4],
        "history_recs": SmartSuggestions.get_history_recs(list(cart.items.keys()))
    }

@app.post("/api/voice-command")
async def process_voice_command(payload: Dict):
    transcript = payload.get("transcript", "").strip()
    if not transcript:
        return JSONResponse({"error": "Empty transcript"}, status_code=400)

    cart_keys = list(cart.items.keys())
    cart_context = f"Current Cart: {', '.join(cart_keys)}" if cart_keys else "Cart is empty."
    full_prompt = f"{EXPANDED_GROCERY_PROMPT}\n\n{cart_context}\nUser Spoken Text: \"{transcript}\""

    parsed_cmd = None
    if gemini_client:
        for m in FLASH_MODELS:
            try:
                res = gemini_client.models.generate_content(
                    model=m,
                    contents=full_prompt,
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json",
                        response_schema=VoiceCommandResult,
                        temperature=0.1
                    )
                )
                parsed_cmd = VoiceCommandResult.model_validate_json(res.text)
                break
            except Exception:
                continue

    if not parsed_cmd:
        clean = transcript.lower().replace("i want ", "").replace("add ", "").title()
        parsed_cmd = VoiceCommandResult(
            intent="ADD",
            detected_language="en",
            transcript=transcript,
            items_to_add=[ExtractedItem(product_name=clean, quantity=1.0, unit="item", category="Pantry")],
            feedback_message=f"Added {clean} to cart."
        )

    messages, suggested_subs, search_results = [], [], []

    for itm in parsed_cmd.items_to_add:
        messages.append(cart.add(itm))
        subs = SmartSuggestions.get_substitutes(itm.product_name)
        if subs: suggested_subs.extend(subs)

    for rm_name in parsed_cmd.items_to_remove:
        messages.append(cart.remove(rm_name))

    if parsed_cmd.intent == "CLEAR":
        cart.clear()
        messages.append("Cart cleared.")

    if parsed_cmd.search_query:
        search_results = await search_open_food_facts(parsed_cmd.search_query, max_price=parsed_cmd.search_max_price)

    season = SmartSuggestions.current_season()
    return {
        "transcript": transcript,
        "intent": parsed_cmd.intent,
        "language": parsed_cmd.detected_language,
        "messages": messages,
        "tts_message": parsed_cmd.feedback_message,
        "cart": cart.get_categorized(),
        "substitutes": list(set(suggested_subs)),
        "search_results": search_results,
        "seasonal_picks": SmartSuggestions.SEASONS[season][:4],
        "history_recs": SmartSuggestions.get_history_recs(list(cart.items.keys()))
    }

@app.get("/api/cart")
async def get_cart():
    season = SmartSuggestions.current_season()
    return {
        "cart": cart.get_categorized(),
        "seasonal_picks": SmartSuggestions.SEASONS[season][:4],
        "history_recs": SmartSuggestions.get_history_recs(list(cart.items.keys()))
    }

@app.delete("/api/cart")
async def clear_cart():
    cart.clear()
    return {"status": "ok"}

os.makedirs("static", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
async def root():
    with open("static/index.html") as f:
        return HTMLResponse(content=f.read())
