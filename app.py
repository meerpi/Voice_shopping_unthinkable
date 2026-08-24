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
            sampling_rate=sr, threshold=0.25,
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
    brand_hint: Optional[str] = Field(default=None, description="Exact brand name if specified or identified")
    variant: Optional[str] = Field(default=None, description="Product variant or specifier")
    quantity: float = Field(default=1.0, description="Numeric count")
    unit: str = Field(default="item", description="Unit of measurement (pieces, kg, litre, pack, item)")
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

# Category and Brand Lexicon Classifier
CATEGORY_LOOKUP = {
    "Produce": ["sweet corn", "corn", "apple", "apples", "banana", "bananas", "jackfruit", "mango", "mangoes", "orange", "oranges", "potato", "potatoes", "tomato", "tomatoes", "onion", "onions", "spinach", "garlic", "ginger", "watermelon", "peaches", "peas", "berries", "strawberry", "strawberries", "avocado", "grapes", "lemon", "lime", "carrot", "carrots", "cucumber", "lettuce", "broccoli"],
    "Dairy & Eggs": ["milk", "butter", "cheese", "eggs", "egg", "ghee", "paneer", "yogurt", "curd", "cream", "oat milk", "almond milk", "soy milk"],
    "Bakery": ["bread", "bagel", "bagels", "croissant", "croissants", "muffin", "muffins", "tortilla", "buns", "pita"],
    "Pantry": ["rice", "atta", "flour", "sugar", "salt", "oil", "olive oil", "sunflower oil", "mustard oil", "dal", "toor dal", "moong dal", "pasta", "noodles", "ketchup", "sauce", "cereal", "oats", "corn flakes", "honey", "spices"],
    "Beverages": ["tea", "coffee", "juice", "soda", "water", "sparkling water", "energy drink"],
    "Snacks": ["chips", "cookies", "biscuits", "popcorn", "nuts", "almonds", "cashews", "chocolate"],
    "Personal Care": ["toothpaste", "toothbrush", "soap", "shampoo", "handwash", "deodorant", "sanitizer"],
    "Household": ["detergent", "dish soap", "paper towels", "tissue", "cleaner", "trash bags"]
}

# Comprehensive Indian FMCG & Grocery Brand Database
BRAND_DEFAULTS = {
    # Dairy & Breakfast
    "butter": "Amul", "milk": "Amul", "ghee": "Amul", "paneer": "Amul", "cheese": "Amul",
    "curd": "Mother Dairy", "yogurt": "Epigamia", "dahi": "Mother Dairy",
    "eggs": "Eggoz", "bread": "Britannia", "corn flakes": "Kellogg's", "oats": "Quaker",
    "muesli": "Kellogg's", "poha": "Tata Sampann", "honey": "Dabur",
    # Staples, Grains & Pulses
    "rice": "Daawat", "basmati rice": "India Gate", "atta": "Aashirvaad", "flour": "Aashirvaad",
    "maida": "Aashirvaad", "besan": "Tata Sampann", "dal": "Tata Sampann",
    "toor dal": "Tata Sampann", "moong dal": "Tata Sampann", "chana dal": "Tata Sampann",
    "salt": "Tata", "sugar": "Madhur", "mustard oil": "Fortune", "sunflower oil": "Fortune",
    "oil": "Fortune", "refined oil": "Fortune", "olive oil": "Borges", "ghee": "Amul",
    # Spices & Condiments
    "turmeric": "MDH", "haldi": "MDH", "red chilli": "Everest", "garam masala": "Everest",
    "spices": "Catch", "ketchup": "Kissan", "sauce": "Maggi", "mayonnaise": "Veeba",
    # Tea, Coffee & Beverages
    "tea": "Tata Tea Gold", "chai": "Red Label", "green tea": "Tetley",
    "coffee": "Nescafe Classic", "instant coffee": "Bru", "juice": "Real",
    "coconut water": "Raw Pressery", "syrup": "Rooh Afza",
    # Snacks & Quick Cooking
    "noodles": "Maggi", "pasta": "Barilla", "biscuits": "Parle-G", "cookies": "Good Day",
    "rusk": "Britannia", "chips": "Lay's", "namkeen": "Haldiram's", "bhujia": "Bikaji",
    # Personal Care & Cleaning
    "toothpaste": "Colgate Total", "toothbrush": "Oral-B", "soap": "Dettol", "shampoo": "Dove",
    "detergent": "Surf Excel", "dish soap": "Vim", "handwash": "Dettol", "floor cleaner": "Lizol",
    # Produce & Speciality
    "sweet corn": "Del Monte", "jackfruit": "Nature's Charm", "mushrooms": "Urban Platter"
}

KNOWN_INDIAN_BRANDS = [
    "Amul", "Mother Dairy", "Nandini", "Gowardhan", "Epigamia", "Country Delight",
    "Aashirvaad", "Fortune", "Tata", "Tata Sampann", "Tata Tea", "Daawat", "India Gate",
    "Madhur", "Patanjali", "Dabur", "MDH", "Everest", "Catch", "Red Label", "Taj Mahal",
    "Wagh Bakri", "Nescafe", "Bru", "Parle-G", "Parle", "Britannia", "Sunfeast",
    "Haldiram's", "Haldiram", "Bikaji", "Balaji", "Lay's", "Lays", "Kurkure", "Bingo",
    "Maggi", "Top Ramen", "Yippee", "Kellogg's", "Quaker", "MTR", "Kissan", "Veeba",
    "Del Monte", "Colgate", "Sensodyne", "Dettol", "Lifebuoy", "Surf Excel", "Ariel",
    "Vim", "Lizol", "Harpic", "Godrej", "Dove", "Pears", "Oatly", "Kerrygold"
]

def infer_category_and_brand(name: str) -> tuple:
    n = name.lower().strip()
    category = "Pantry"
    for cat, keywords in CATEGORY_LOOKUP.items():
        if any(k in n or n in k for k in keywords):
            category = cat
            break

    # First check if user explicitly stated a known brand
    brand = None
    for b in KNOWN_INDIAN_BRANDS:
        if b.lower() in n:
            brand = b
            break

    # If no brand stated, look up curated default for the grocery staple
    if not brand:
        for item_key, default_brand in BRAND_DEFAULTS.items():
            if item_key in n:
                brand = default_brand
                break

    return category, brand

class ShoppingCart:
    def __init__(self):
        self.items: Dict[str, Dict] = {}
        self.history: List[str] = ["Milk", "Bread", "Eggs", "Bananas", "Butter", "Coffee", "Rice"]

    def add(self, item: ExtractedItem) -> str:
        # Determine category & brand if missing
        inferred_cat, inferred_brand = infer_category_and_brand(item.product_name)
        category = item.category if item.category != "Pantry" else inferred_cat
        brand = item.brand_hint or inferred_brand
        
        display_name = f"{brand} {item.product_name}".strip() if (brand and brand.lower() not in item.product_name.lower()) else item.product_name
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
                "category": category,
                "brand_hint": brand,
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
    "gemini-flash-lite-latest",
    "gemini-3.5-flash-lite",
    "gemini-3.1-flash-lite",
    "gemini-3.6-flash",
    "gemini-flash-latest"
]

# ─── PROMPT ENGINEERING (research-backed) ───
# Key insight: system_instruction is separated from content to prevent
# instruction contamination and transcription bias. No product-specific
# examples are included — they cause the model to hallucinate those products
# instead of transcribing what it actually hears.

SYSTEM_INSTRUCTION = """You are a grocery shopping voice assistant for Indian users.

<role>
Your job has two parts:
1. TRANSCRIBE: Write exactly what you hear in the audio. The audio is the ONLY source of truth. Never guess or substitute words. If you cannot hear clearly, write what you hear phonetically.
2. EXTRACT: From your transcription, extract each grocery item with its quantity, unit, category, and brand (if mentioned).
</role>

<rules>
- Extract EVERY item mentioned. Users often list multiple items in one utterance.
- If a number precedes a product name, that is the quantity.
- Default quantity is 1 if no number is spoken.
- Assign each item to exactly one category from: Produce, Dairy & Eggs, Meat & Seafood, Pantry, Bakery, Frozen, Beverages, Snacks, Household, Personal Care.
- If a known Indian brand is mentioned (e.g. Amul, Tata, Aashirvaad, Fortune, MDH, Haldiram's, Parle, Britannia, Colgate, Dettol, Surf Excel, Maggi, Nescafe), set brand_hint to that brand.
- Tolerate Indian English accents and Hindi words: "aalu"=Potatoes, "doodh"=Milk, "cheeni"=Sugar, "anda"=Eggs, "chawal"=Rice, "atta"=Flour, "sabzi"=Vegetables, "pyaaz"=Onions, "tamatar"=Tomatoes.
- intent should be "ADD" when adding items, "REMOVE" when removing, "CLEAR" when clearing cart, "SEARCH" when searching.
- feedback_message should naturally confirm what was added.
</rules>"""

AUDIO_CONTENT_PROMPT = "Listen to this audio carefully. Transcribe exactly what was said, then extract all grocery items mentioned."

TEXT_CONTENT_TEMPLATE = 'The user typed: "{transcript}"\n\nExtract all grocery items mentioned.'

@app.post("/api/voice-audio")
async def process_voice_audio(file: UploadFile = File(...)):
    import logging
    logger = logging.getLogger("voice_audio")

    audio_bytes = await file.read()
    mime_type = file.content_type or "audio/webm"
    logger.info(f"Received audio: {len(audio_bytes)} bytes, mime={mime_type}")

    # Save raw recording for debugging
    try:
        with open("debug_audio/last_recording.webm", "wb") as f:
            f.write(audio_bytes)
    except Exception:
        pass

    cart_keys = list(cart.items.keys())
    cart_context = f"Current Cart: {', '.join(cart_keys)}" if cart_keys else "Cart is empty."

    diag = {"raw_bytes": len(audio_bytes)}
    parsed_cmd = None
    last_error = ""

    if gemini_client:
        # STRATEGY 1: Send raw browser audio directly to Gemini.
        # Gemini natively understands WebM Opus — no DSP needed.
        # system_instruction is separated from content per Google best practices.
        for m in FLASH_MODELS:
            try:
                logger.info(f"Trying raw audio with model {m}")
                res = gemini_client.models.generate_content(
                    model=m,
                    contents=[
                        f"{AUDIO_CONTENT_PROMPT}\n{cart_context}",
                        types.Part.from_bytes(data=audio_bytes, mime_type=mime_type)
                    ],
                    config=types.GenerateContentConfig(
                        system_instruction=SYSTEM_INSTRUCTION,
                        response_mime_type="application/json",
                        response_schema=VoiceCommandResult,
                        temperature=0.1
                    )
                )
                candidate = VoiceCommandResult.model_validate_json(res.text)
                # Only accept if Gemini actually heard something
                if candidate.transcript and candidate.transcript.strip() and candidate.items_to_add:
                    parsed_cmd = candidate
                    diag["method"] = f"raw_audio_{m}"
                    logger.info(f"Raw audio success with {m}: '{candidate.transcript}'")
                    break
                else:
                    logger.info(f"Raw audio {m} returned empty transcript, trying next")
                    last_error = f"{m}: empty transcript"
            except Exception as e:
                last_error = f"{m}: {type(e).__name__}: {str(e)[:100]}"
                logger.warning(f"Raw audio {m} failed: {last_error}")
                continue

        # STRATEGY 2: If raw didn't work, try DSP-enhanced audio
        if not parsed_cmd:
            try:
                enhanced_bytes, enhance_diag = enhance_audio(audio_bytes)
                diag.update(enhance_diag)
                for m in FLASH_MODELS[:2]:
                    try:
                        logger.info(f"Trying enhanced audio with model {m}")
                        res = gemini_client.models.generate_content(
                            model=m,
                            contents=[
                                f"{AUDIO_CONTENT_PROMPT}\n{cart_context}",
                                types.Part.from_bytes(data=enhanced_bytes, mime_type="audio/webm")
                            ],
                            config=types.GenerateContentConfig(
                                system_instruction=SYSTEM_INSTRUCTION,
                                response_mime_type="application/json",
                                response_schema=VoiceCommandResult,
                                temperature=0.1
                            )
                        )
                        candidate = VoiceCommandResult.model_validate_json(res.text)
                        if candidate.transcript and candidate.transcript.strip() and candidate.items_to_add:
                            parsed_cmd = candidate
                            diag["method"] = f"enhanced_audio_{m}"
                            logger.info(f"Enhanced audio success with {m}: '{candidate.transcript}'")
                            break
                    except Exception as e:
                        logger.warning(f"Enhanced audio {m} failed: {e}")
                        continue
            except Exception as e:
                logger.warning(f"Audio enhancement itself failed: {e}")

    if not parsed_cmd:
        logger.error(f"All audio processing failed. Last error: {last_error}")
        parsed_cmd = VoiceCommandResult(
            intent="UNKNOWN",
            detected_language="en",
            transcript="",
            items_to_add=[],
            feedback_message=f"Could not understand speech. Please try again or type your items. ({last_error})"
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

    parsed_cmd = None
    if gemini_client:
        for m in FLASH_MODELS:
            try:
                res = gemini_client.models.generate_content(
                    model=m,
                    contents=f"{TEXT_CONTENT_TEMPLATE.format(transcript=transcript)}\n{cart_context}",
                    config=types.GenerateContentConfig(
                        system_instruction=SYSTEM_INSTRUCTION,
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
        clean = transcript.lower().replace("i want ", "").replace("add ", "").replace("buy ", "").strip().title()
        cat, brand = infer_category_and_brand(clean)
        parsed_cmd = VoiceCommandResult(
            intent="ADD",
            detected_language="en",
            transcript=transcript,
            items_to_add=[ExtractedItem(product_name=clean, brand_hint=brand, quantity=1.0, unit="item", category=cat)],
            feedback_message=f"Added {clean} to {cat}."
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
