import os
import json
import datetime
import io
import subprocess
import tempfile
from typing import List, Dict, Optional, Literal
from fastapi import FastAPI, Request, UploadFile, File
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
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

# ─── Audio Enhancement Pipeline ───────────────────────────────────────────────

_vad_model = None
_nr_imported = False
_nr_module = None

def _get_vad_model():
    """Load Silero VAD neural model."""
    global _vad_model
    if _vad_model is None:
        try:
            from silero_vad import load_silero_vad
            _vad_model = load_silero_vad()
            print("✅ Silero VAD model loaded")
        except Exception as e:
            print(f"⚠️ Silero VAD not available: {e}")
    return _vad_model

def _get_noisereduce():
    """Lazy-import noisereduce."""
    global _nr_imported, _nr_module
    if not _nr_imported:
        try:
            import noisereduce as nr
            _nr_module = nr
            _nr_imported = True
            print("✅ noisereduce loaded")
        except ImportError as e:
            print(f"⚠️ noisereduce not available: {e}")
            _nr_imported = True
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
    proc = subprocess.run(
        cmd, input=audio_bytes,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        timeout=10
    )
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg decode failed: {proc.stderr.decode()[:200]}")
    pcm = np.frombuffer(proc.stdout, dtype=np.float32)
    return pcm

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
    raw_bytes = pcm.astype(np.float32).tobytes()
    proc = subprocess.run(
        cmd, input=raw_bytes,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        timeout=10
    )
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg encode failed: {proc.stderr.decode()[:200]}")
    return proc.stdout

def butterworth_bandpass(pcm: np.ndarray, sr: int = 16000,
                         lowcut: float = 300.0, highcut: float = 7500.0,
                         order: int = 4) -> np.ndarray:
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
        reduced = nr.reduce_noise(
            y=pcm, sr=sr,
            y_noise=noise_clip,
            prop_decrease=0.75,
            stationary=False,
            n_fft=512,
            freq_mask_smooth_hz=300
        )
        return reduced.astype(np.float32)
    except Exception as e:
        print(f"⚠️ noisereduce error: {e}")
        return pcm

def vad_trim(pcm: np.ndarray, sr: int = 16000) -> np.ndarray:
    import torch
    model = _get_vad_model()
    if model is None:
        return pcm
    try:
        from silero_vad import get_speech_timestamps
        wav_tensor = torch.from_numpy(pcm).float()
        speech_ts = get_speech_timestamps(
            wav_tensor, model,
            sampling_rate=sr,
            threshold=0.3,
            min_speech_duration_ms=100,
            min_silence_duration_ms=100,
            speech_pad_ms=60
        )
        if not speech_ts:
            return pcm
        segments = [pcm[ts['start']:ts['end']] for ts in speech_ts]
        trimmed = np.concatenate(segments)
        trimmed_pct = (1.0 - len(trimmed) / len(pcm)) * 100
        print(f"✅ VAD: Trimmed {trimmed_pct:.0f}% silence ({len(speech_ts)} speech segments)")
        return trimmed.astype(np.float32)
    except Exception as e:
        print(f"⚠️ VAD error: {e}")
        return pcm

def agc_normalize(pcm: np.ndarray, target_db: float = -18.0) -> np.ndarray:
    rms = np.sqrt(np.mean(pcm ** 2))
    if rms < 1e-8:
        return pcm
    current_db = 20 * np.log10(rms + 1e-9)
    gain_db = np.clip(target_db - current_db, -20.0, 30.0)
    gain_linear = 10 ** (gain_db / 20.0)
    return (pcm * gain_linear).astype(np.float32)

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
    diag['bandpass'] = '300-7500Hz'

    pcm = spectral_noise_reduce(pcm, sr=sr)
    diag['noise_reduction'] = 'spectral_gating_0.75'

    pcm = vad_trim(pcm, sr=sr)
    diag['vad_duration_s'] = round(len(pcm) / sr, 2)

    pcm = agc_normalize(pcm, target_db=-18.0)
    diag['agc_target'] = '-18dBFS'

    pcm = peak_limit(pcm, ceiling=0.95)
    diag['enhanced_peak_dbfs'] = round(float(20 * np.log10(np.max(np.abs(pcm)) + 1e-9)), 1)
    diag['enhanced_rms_dbfs'] = round(float(20 * np.log10(np.sqrt(np.mean(pcm**2)) + 1e-9)), 1)

    enhanced_webm = encode_pcm_to_webm(pcm, sr=sr)
    diag['enhanced_size_kb'] = round(len(enhanced_webm) / 1024.0, 1)

    return enhanced_webm, diag


# ─── Pydantic Schemas (Expanded Brand & Compound Intent) ───────────────────────

class ExtractedItem(BaseModel):
    product_name: str = Field(description="Normalized general item name in English (e.g. 'Jackfruit', 'Milk', 'Butter', 'Eggs', 'Rice', 'Apples')")
    brand_hint: Optional[str] = Field(default=None, description="Exact brand name if mentioned by user (e.g. 'Amul', 'Kerrygold', 'Oatly', 'Tata', 'Kellogg\\'s', 'Colgate', 'Lays')")
    variant: Optional[str] = Field(default=None, description="Product variant/specifier (e.g. 'Whole', 'Low Fat', 'Brown', 'Gluten-Free', 'Organic', 'Basmati', 'Greek')")
    quantity: float = Field(default=1.0, description="Numeric quantity (e.g. 2.0 for two jackfruits, 0.5 for half a kilo, 12 for a dozen)")
    unit: str = Field(default="item", description="Standard unit (e.g. 'item', 'liter', 'kg', 'g', 'pack', 'bottle', 'carton', 'dozen', 'box', 'can', 'bunch')")
    category: Literal["Produce", "Dairy & Eggs", "Meat & Seafood", "Pantry", "Bakery", "Frozen", "Beverages", "Snacks", "Household", "Personal Care"]
    max_price: Optional[float] = None

class VoiceCommandResult(BaseModel):
    intent: Literal["ADD", "REMOVE", "SEARCH", "COMPOUND", "GET_SUGGESTIONS", "CLEAR", "UNKNOWN"]
    detected_language: str = "en"
    transcript: str = Field(description="Exact verbatim transcription of user speech")
    items_to_add: List[ExtractedItem] = Field(default_factory=list, description="Items to add to the shopping list")
    items_to_remove: List[str] = Field(default_factory=list, description="List of product names to remove from the cart")
    search_query: Optional[str] = Field(default=None, description="Search query string if user is looking for an item")
    search_max_price: Optional[float] = Field(default=None, description="Max price filter if specified")
    feedback_message: str = Field(description="Natural, polished spoken feedback message for TTS / display")


# ─── Shopping Cart & Suggestions ──────────────────────────────────────────────

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
            return f"Updated {display_name} quantity to {self.items[key]['quantity']:g} {item.unit}"
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
            return f"Added {item.quantity:g} {item.unit} of {display_name} to {item.category}"

    def remove(self, product_name: str) -> str:
        key = product_name.lower().strip()
        matches = [k for k in self.items.keys() if key in k or k in key]
        if matches:
            removed = self.items.pop(matches[0])
            return f"Removed {removed['name']} from cart"
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
        "milk": ["Oatly Oat Milk (Dairy-Free)", "Almond Milk", "Soy Milk", "Fairlife Lactose-Free"],
        "butter": ["Kerrygold Pure Irish Butter", "Country Crock Plant Butter", "Ghee"],
        "pasta": ["Banza Chickpea Pasta (Gluten-Free)", "Barilla Whole Wheat Pasta"],
        "sugar": ["Stevia Extract", "Monk Fruit Sweetener", "Organic Raw Honey"],
        "rice": ["Royal Basmati Rice", "Quinoa", "Cauliflower Rice"],
        "jackfruit": ["Artichoke Hearts", "Organic Tofu", "Canned Young Jackfruit"],
        "bread": ["Dave's Killer Organic Bread", "Ezekiel Sprouted Grain Bread"],
        "coffee": ["Starbucks Pike Place Roast", "Nespresso Pods", "Lavazza Super Crema"]
    }
    SEASONS = {
        "Spring": ["Asparagus", "Strawberries", "Fresh Peas", "Jackfruit", "Artichokes"],
        "Summer": ["Watermelon", "Alphonso Mangoes", "Peaches", "Sweet Corn", "Zucchini"],
        "Autumn": ["Honeycrisp Apples", "Pumpkin Puree", "Butternut Squash", "Pomegranates"],
        "Winter": ["Clementines / Citrus", "Fresh Guava", "Organic Kale", "Brussels Sprouts"]
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


# ─── Open Food Facts Search ──────────────────────────────────────────────────

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


# ─── Gemini Model Pool & Massive Grounding System ─────────────────────────────

FLASH_MODELS = [
    "gemini-3.5-flash-lite",
    "gemini-flash-latest",
    "gemini-3.5-flash",
    "gemini-3.7-flash",
    "gemini-3.6-flash"
]

EXPANDED_GROCERY_PROMPT = """
You are a state-of-the-art voice shopping assistant intelligence system designed with deep acoustic tolerance and a comprehensive global grocery and FMCG brand taxonomy.

### 1. BRAND INTELLIGENCE & DISAMBIGUATION:
Extract the exact brand if mentioned, and normalize the product name:
- Dairy & Plant Milks: 'Amul' -> brand: 'Amul', 'Oatly' -> brand: 'Oatly', 'Kerrygold' -> brand: 'Kerrygold', 'Fairlife' -> brand: 'Fairlife', 'Horizon' -> brand: 'Horizon Organic', 'Chobani' -> brand: 'Chobani', 'Vital Farms' -> brand: 'Vital Farms', 'Mother Dairy' -> brand: 'Mother Dairy', 'Silk' -> brand: 'Silk'.
- Pantry & Staples: 'Tata Salt' -> brand: 'Tata', 'Aashirvaad' -> brand: 'Aashirvaad', 'Barilla' -> brand: 'Barilla', 'Heinz' -> brand: 'Heinz', 'Kraft' -> brand: 'Kraft', 'Kikkoman' -> brand: 'Kikkoman', 'San Marzano' -> brand: 'San Marzano', 'King Arthur' -> brand: 'King Arthur Flour'.
- Breakfast & Snacks: 'Kellogg's' -> brand: 'Kellogg\\'s', 'Quaker Oats' -> brand: 'Quaker', 'Lays' -> brand: 'Lays', 'Doritos' -> brand: 'Doritos', 'Haldirams' -> brand: 'Haldiram\\'s', 'Nature Valley' -> brand: 'Nature Valley', 'Oreo' -> brand: 'Oreo'.
- Personal Care & Household: 'Colgate Total' -> brand: 'Colgate', product: 'Toothpaste', 'Sensodyne' -> brand: 'Sensodyne', 'Crest' -> brand: 'Crest', 'Dettol' -> brand: 'Dettol', 'Dawn' -> brand: 'Dawn Dish Soap', 'Tide' -> brand: 'Tide Laundry Detergent'.

### 2. GLOBAL & REGIONAL PRODUCE & PANTRY CATALOG:
- Tropical & Indian Produce: Jackfruit, Dragonfruit, Alphonso Mango, Papaya, Guava, Chikoo (Sapodilla), Custard Apple (Sitaphal), Coconut, Pomegranate, Bananas, Bitter Gourd (Karela), Bottle Gourd (Lauki), Okra (Bhindi), Eggplant (Brinjal), Spinach (Palak), Fenugreek (Methi), Coriander, Curry Leaves, Ginger, Garlic, Potatoes (Aalu), Tomatoes, Onions.
- Dairy & Proteins: Whole Milk, Oat Milk, Almond Milk, Greek Yogurt, Curd, Paneer, Tofu, Pure Butter, Ghee, Free-Range Eggs, Cheddar Cheese, Mozzarella, Chicken Breast, Salmon Fillet.
- Pantry & Grains: Royal Basmati Rice, Jasmine Rice, Atta (Whole Wheat Flour), Almond Flour, Extra Virgin Olive Oil, Avocado Oil, Mustard Oil, Saffron, Cardamom, Cumin, Turmeric, Black Pepper.

### 3. COMPOUND MULTI-INTENT PARSING:
Users frequently issue compound multi-part sentences in one breath:
Example: "Add two packs of Kerrygold butter and a litre of oat milk, remove eggs, and find me organic apples under $5"
Should parse as:
- items_to_add: [
    { brand_hint: 'Kerrygold', product_name: 'Butter', variant: 'Pure Butter', quantity: 2, unit: 'pack', category: 'Dairy & Eggs' },
    { brand_hint: 'Oatly', product_name: 'Oat Milk', quantity: 1, unit: 'liter', category: 'Dairy & Eggs' }
  ]
- items_to_remove: ['Eggs']
- search_query: 'Organic Apples'
- search_max_price: 5.0
- intent: 'COMPOUND'

### 4. ACOUSTIC & PHONETIC SLUR TOLERANCE:
Muffled mic consonants or slurred pronunciation:
- 'tu-ja-froot' / 'toojack' / 'foojoo' -> 'Two Jackfruit' (qty: 2, unit: 'item', Produce)
- 'fedex' / 'five x' -> '5 Eggs' (qty: 5, unit: 'item', Dairy & Eggs)
- 'little milk' / 'a litre milk' -> '1 Liter Milk' (qty: 1, unit: 'liter', Dairy & Eggs)
- 'ek kilo aalu' -> 'Potatoes' (qty: 1, unit: 'kg', Produce)
- 'half dozen eggs' -> quantity: 6
- 'couple of bananas' -> quantity: 2

Return a structured VoiceCommandResult with friendly spoken feedback message.
"""


# ─── API Endpoints ────────────────────────────────────────────────────────────

@app.post("/api/voice-audio")
async def process_voice_audio(file: UploadFile = File(...)):
    audio_bytes = await file.read()
    mime_type = file.content_type or "audio/webm"

    with open("debug_audio/last_recording.webm", "wb") as f:
        f.write(audio_bytes)

    try:
        enhanced_bytes, diag = enhance_audio(audio_bytes)
        send_mime = "audio/webm"
        send_bytes = enhanced_bytes
    except Exception as e:
        print(f"⚠️ Enhancement fallback: {e}")
        send_bytes = audio_bytes
        send_mime = mime_type
        diag = {"enhancement": "skipped", "reason": str(e)}

    cart_keys = list(cart.items.keys())
    cart_context = f"Current Cart: {', '.join(cart_keys)}" if cart_keys else "Cart is empty."

    full_prompt = f"{EXPANDED_GROCERY_PROMPT}\n\n{cart_context}"

    parsed_cmd = None
    if gemini_client:
        for m in FLASH_MODELS:
            try:
                res = gemini_client.models.generate_content(
                    model=m,
                    contents=[
                        full_prompt,
                        types.Part.from_bytes(data=send_bytes, mime_type=send_mime)
                    ],
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json",
                        response_schema=VoiceCommandResult,
                        temperature=0.1
                    )
                )
                parsed_cmd = VoiceCommandResult.model_validate_json(res.text)
                print(f"✅ Gemini [{m}] recognized: \"{parsed_cmd.transcript}\" -> Add: {len(parsed_cmd.items_to_add)}, Remove: {len(parsed_cmd.items_to_remove)}")
                break
            except Exception as err:
                print(f"⚠️ Model {m} error: {err}")
                continue

    if not parsed_cmd:
        return JSONResponse({"error": "Failed to parse audio"}, status_code=500)

    messages = []
    suggested_subs = []
    search_results = []

    # Execute Additions
    for itm in parsed_cmd.items_to_add:
        msg = cart.add(itm)
        messages.append(msg)
        subs = SmartSuggestions.get_substitutes(itm.product_name)
        if subs: suggested_subs.extend(subs)

    # Execute Removals
    for rm_name in parsed_cmd.items_to_remove:
        msg = cart.remove(rm_name)
        messages.append(msg)

    if parsed_cmd.intent == "CLEAR":
        cart.clear()
        messages.append("Shopping cart cleared.")

    # Execute Search if present
    if parsed_cmd.search_query:
        search_results = await search_open_food_facts(parsed_cmd.search_query, max_price=parsed_cmd.search_max_price)
        messages.append(f"Found {len(search_results)} products for '{parsed_cmd.search_query}'")

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

    messages = []
    suggested_subs = []
    search_results = []

    for itm in parsed_cmd.items_to_add:
        msg = cart.add(itm)
        messages.append(msg)
        subs = SmartSuggestions.get_substitutes(itm.product_name)
        if subs: suggested_subs.extend(subs)

    for rm_name in parsed_cmd.items_to_remove:
        msg = cart.remove(rm_name)
        messages.append(msg)

    if parsed_cmd.intent == "CLEAR":
        cart.clear()
        messages.append("Shopping cart cleared.")

    if parsed_cmd.search_query:
        search_results = await search_open_food_facts(parsed_cmd.search_query, max_price=parsed_cmd.search_max_price)
        messages.append(f"Found {len(search_results)} products for '{parsed_cmd.search_query}'")

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
