import os
import urllib.parse
import webbrowser
from typing import List, Dict, Tuple

ASIN_DATABASE = {
    "amul butter": {"asin": "B07C5H9L12", "title": "Amul Butter 500g Carton"},
    "butter": {"asin": "B07C5H9L12", "title": "Amul Butter 500g Carton"},
    "kerrygold butter": {"asin": "B07C5H9L12", "title": "Pure Dairy Butter"},
    "amul ghee": {"asin": "B00TI85Q8O", "title": "Amul Pure Ghee 1L Pouch"},
    "ghee": {"asin": "B00TI85Q8O", "title": "Amul Pure Ghee 1L Pouch"},
    "oatly oat milk": {"asin": "B084Z1X9KP", "title": "Oatly Oat Milk Barista Edition 1L"},
    "oat milk": {"asin": "B084Z1X9KP", "title": "Oatly Oat Milk Barista Edition 1L"},
    "amul milk": {"asin": "B07N8D8P88", "title": "Amul Taaza Homogenised Toned Milk 1L"},
    "milk": {"asin": "B07N8D8P88", "title": "Amul Taaza Homogenised Toned Milk 1L"},
    "paneer": {"asin": "B07C5H9L13", "title": "Amul Fresh Paneer 200g"},
    "cheese": {"asin": "B07C5H9L14", "title": "Amul Processed Cheese Blocks 400g"},
    "eggs": {"asin": "B08B3X6Y8Z", "title": "Eggoz Farm Fresh White / Brown Eggs 6-Pack"},
    "basmati rice": {"asin": "B00TI84V7W", "title": "Daawat Rozana Super Basmati Rice 5kg"},
    "rice": {"asin": "B00TI84V7W", "title": "Daawat Rozana Super Basmati Rice 5kg"},
    "india gate basmati rice": {"asin": "B0758DR8G9", "title": "India Gate Basmati Rice Feast Rozzana 5kg"},
    "aashirvaad atta": {"asin": "B00TI84Y0Y", "title": "Aashirvaad Superior MP Whole Wheat Atta 5kg"},
    "atta": {"asin": "B00TI84Y0Y", "title": "Aashirvaad Superior MP Whole Wheat Atta 5kg"},
    "flour": {"asin": "B00TI84Y0Y", "title": "Aashirvaad Superior MP Whole Wheat Atta 5kg"},
    "tata salt": {"asin": "B00TI85EAE", "title": "Tata Salt Vacuum Evaporated Iodised 1kg"},
    "salt": {"asin": "B00TI85EAE", "title": "Tata Salt Vacuum Evaporated Iodised 1kg"},
    "toor dal": {"asin": "B011T240O0", "title": "Tata Sampann Unpolished Toor Dal 1kg"},
    "dal": {"asin": "B011T240O0", "title": "Tata Sampann Unpolished Toor Dal 1kg"},
    "moong dal": {"asin": "B011T240OI", "title": "Tata Sampann Unpolished Moong Dal 1kg"},
    "sugar": {"asin": "B00TI84S48", "title": "Madhur Pure & Hygienic Sugar 1kg"},
    "fortune sunflower oil": {"asin": "B00TI84S5C", "title": "Fortune Sunlite Refined Sunflower Oil 1L"},
    "sunflower oil": {"asin": "B00TI84S5C", "title": "Fortune Sunlite Refined Sunflower Oil 1L"},
    "mustard oil": {"asin": "B00TI84V0O", "title": "Fortune Premium Kachi Ghani Mustard Oil 1L"},
    "olive oil": {"asin": "B00J4N0508", "title": "Borges Extra Virgin Olive Oil 1L"},
    "organic olive oil": {"asin": "B00J4N0508", "title": "Borges Extra Virgin Olive Oil 1L"},
    "kelloggs corn flakes": {"asin": "B010GGBZ2M", "title": "Kellogg's Corn Flakes Original 875g"},
    "corn flakes": {"asin": "B010GGBZ2M", "title": "Kellogg's Corn Flakes Original 875g"},
    "quaker oats": {"asin": "B07B9WZ4N6", "title": "Quaker Rolled Oats 1kg Pouch"},
    "oats": {"asin": "B07B9WZ4N6", "title": "Quaker Rolled Oats 1kg Pouch"},
    "tea": {"asin": "B00TI856H8", "title": "Brooke Bond Red Label Tea 1kg"},
    "coffee": {"asin": "B00TI85KMI", "title": "Nescafe Classic Instant Coffee 200g Jar"},
    "nescafe": {"asin": "B00TI85KMI", "title": "Nescafe Classic Instant Coffee 200g Jar"},
    "colgate toothpaste": {"asin": "B07J5K9LM5", "title": "Colgate Total Advanced Health Toothpaste 240g"},
    "toothpaste": {"asin": "B07J5K9LM5", "title": "Colgate Total Advanced Health Toothpaste 240g"},
    "sensodyne": {"asin": "B07BNQZ5H3", "title": "Sensodyne Rapid Relief Toothpaste 80g"},
    "dettol handwash": {"asin": "B07N8D9L9F", "title": "Dettol Original Liquid Handwash Refill 1500ml"},
    "surf excel": {"asin": "B084V5S5M4", "title": "Surf Excel Matic Front Load Liquid Detergent 2L"},
    "detergent": {"asin": "B084V5S5M4", "title": "Surf Excel Matic Front Load Liquid Detergent 2L"},
    "lays chips": {"asin": "B08B3SCQ8T", "title": "Lay's Classic Salted Potato Chips 115g"},
    "chips": {"asin": "B08B3SCQ8T", "title": "Lay's Classic Salted Potato Chips 115g"},
    "jackfruit": {"asin": "B07X8M7J9W", "title": "Nature's Charm Young Green Jackfruit Canned 565g"},
    "mangoes": {"asin": "B087N5P4H1", "title": "Alphonso Mango Pulp / Slices 850g"}
    # Note: Add more FMCG / Grocery ASIN mappings as catalog expands
}

def get_product_direct_url(item: Dict, store_name: str = "Amazon Fresh") -> str:
    raw_name = item.get("name", "").lower().strip()
    base_name = item.get("base_name", raw_name).lower().strip()
    brand = (item.get("brand_hint") or "").lower().strip()
    
    matched_asin = None
    if brand and f"{brand} {base_name}" in ASIN_DATABASE:
        matched_asin = ASIN_DATABASE[f"{brand} {base_name}"]["asin"]
    elif raw_name in ASIN_DATABASE:
        matched_asin = ASIN_DATABASE[raw_name]["asin"]
    elif base_name in ASIN_DATABASE:
        matched_asin = ASIN_DATABASE[base_name]["asin"]
    else:
        for key, val in ASIN_DATABASE.items():
            if key in raw_name or raw_name in key or key in base_name or base_name in key:
                matched_asin = val["asin"]
                break

    encoded = urllib.parse.quote_plus(item.get("name", ""))
    if "Amazon" in store_name:
        if matched_asin:
            return f"https://www.amazon.in/dp/{matched_asin}"
        return f"https://www.amazon.in/s?k={encoded}&i=nowstore"
    elif "Blinkit" in store_name:
        return f"https://blinkit.com/s/?q={encoded}"
    elif "Zepto" in store_name:
        return f"https://www.zeptonow.com/search?q={encoded}"
    elif "Instamart" in store_name or "Swiggy" in store_name:
        return f"https://www.swiggy.com/instamart/search?query={encoded}"
    elif "BigBasket" in store_name:
        return f"https://www.bigbasket.com/ps/?q={encoded}"
    return f"https://www.google.com/search?q=buy+{encoded}"

def open_all_items_in_browser(cart_items: List[Dict], store_name: str = "Amazon Fresh"):
    for item in cart_items:
        url = get_product_direct_url(item, store_name=store_name)
        webbrowser.open_new_tab(url)
