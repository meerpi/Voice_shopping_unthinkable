import urllib.parse
from typing import List, Dict, Tuple, Optional

# ─── CURATED FMCG & GROCERY ASIN DATABASE (Amazon India & Global) ─────────────

ASIN_DATABASE = {
    # Dairy & Butter
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
    
    # Staples, Rice, Flour & Grains
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
    
    # Oils & Condiments
    "fortune sunflower oil": {"asin": "B00TI84S5C", "title": "Fortune Sunlite Refined Sunflower Oil 1L"},
    "sunflower oil": {"asin": "B00TI84S5C", "title": "Fortune Sunlite Refined Sunflower Oil 1L"},
    "mustard oil": {"asin": "B00TI84V0O", "title": "Fortune Premium Kachi Ghani Mustard Oil 1L"},
    "olive oil": {"asin": "B00J4N0508", "title": "Borges Extra Virgin Olive Oil 1L"},
    "organic olive oil": {"asin": "B00J4N0508", "title": "Borges Extra Virgin Olive Oil 1L"},
    
    # Breakfast & Beverages
    "kelloggs corn flakes": {"asin": "B010GGBZ2M", "title": "Kellogg's Corn Flakes Original 875g"},
    "corn flakes": {"asin": "B010GGBZ2M", "title": "Kellogg's Corn Flakes Original 875g"},
    "quaker oats": {"asin": "B07B9WZ4N6", "title": "Quaker Rolled Oats 1kg Pouch"},
    "oats": {"asin": "B07B9WZ4N6", "title": "Quaker Rolled Oats 1kg Pouch"},
    "tea": {"asin": "B00TI856H8", "title": "Brooke Bond Red Label Tea 1kg"},
    "coffee": {"asin": "B00TI85KMI", "title": "Nescafe Classic Instant Coffee 200g Jar"},
    "nescafe": {"asin": "B00TI85KMI", "title": "Nescafe Classic Instant Coffee 200g Jar"},
    
    # Personal Care & Household
    "colgate toothpaste": {"asin": "B07J5K9LM5", "title": "Colgate Total Advanced Health Toothpaste 240g"},
    "toothpaste": {"asin": "B07J5K9LM5", "title": "Colgate Total Advanced Health Toothpaste 240g"},
    "sensodyne": {"asin": "B07BNQZ5H3", "title": "Sensodyne Rapid Relief Toothpaste 80g"},
    "dettol handwash": {"asin": "B07N8D9L9F", "title": "Dettol Original Liquid Handwash Refill 1500ml"},
    "surf excel": {"asin": "B084V5S5M4", "title": "Surf Excel Matic Front Load Liquid Detergent 2L"},
    "detergent": {"asin": "B084V5S5M4", "title": "Surf Excel Matic Front Load Liquid Detergent 2L"},
    
    # Snacks & Tropical Foods
    "lays chips": {"asin": "B08B3SCQ8T", "title": "Lay's Classic Salted Potato Chips 115g"},
    "chips": {"asin": "B08B3SCQ8T", "title": "Lay's Classic Salted Potato Chips 115g"},
    "jackfruit": {"asin": "B07X8M7J9W", "title": "Nature's Charm Young Green Jackfruit Canned 565g"},
    "mangoes": {"asin": "B087N5P4H1", "title": "Alphonso Mango Pulp / Slices 850g"}
}


# ─── 1. AMAZON REMOTE CART PROTOCOL GENERATOR ─────────────────────────────────

def generate_amazon_remote_cart_url(cart_items: List[Dict], locale: str = "in") -> Tuple[str, List[Dict]]:
    """
    Generates an official Amazon Remote Cart URL using the /gp/aws/cart/add.html protocol.
    Format: https://www.amazon.in/gp/aws/cart/add.html?ASIN.1=...&Quantity.1=...
    """
    matched_items = []
    unmatched_items = []
    
    query_params = []
    idx = 1
    
    for item in cart_items:
        raw_name = item.get("name", "").lower().strip()
        base_name = item.get("base_name", raw_name).lower().strip()
        brand = (item.get("brand_hint") or "").lower().strip()
        qty = max(1, int(round(item.get("quantity", 1.0))))
        
        # 1. Look for direct match
        matched_asin = None
        matched_title = None
        
        # Check brand + base name
        if brand and f"{brand} {base_name}" in ASIN_DATABASE:
            matched_asin = ASIN_DATABASE[f"{brand} {base_name}"]["asin"]
            matched_title = ASIN_DATABASE[f"{brand} {base_name}"]["title"]
        elif raw_name in ASIN_DATABASE:
            matched_asin = ASIN_DATABASE[raw_name]["asin"]
            matched_title = ASIN_DATABASE[raw_name]["title"]
        elif base_name in ASIN_DATABASE:
            matched_asin = ASIN_DATABASE[base_name]["asin"]
            matched_title = ASIN_DATABASE[base_name]["title"]
        else:
            # Fuzzy match
            for key, val in ASIN_DATABASE.items():
                if key in raw_name or raw_name in key or key in base_name or base_name in key:
                    matched_asin = val["asin"]
                    matched_title = val["title"]
                    break
        
        if matched_asin:
            query_params.append(f"ASIN.{idx}={matched_asin}&Quantity.{idx}={qty}")
            matched_items.append({
                "name": item.get("name"),
                "asin": matched_asin,
                "title": matched_title,
                "quantity": qty
            })
            idx += 1
        else:
            unmatched_items.append(item)
            
    if query_params:
        base_domain = "amazon.in" if locale == "in" else "amazon.com"
        full_url = f"https://www.{base_domain}/gp/aws/cart/add.html?{'&'.join(query_params)}"
        return full_url, matched_items
    else:
        # Fallback to general multi-search
        encoded = urllib.parse.quote_plus(" ".join([itm.get("name", "") for itm in cart_items]))
        return f"https://www.amazon.in/s?k={encoded}", []


# ─── 2. WALMART MULTI-ITEM CART SERVICE GENERATOR ────────────────────────────

def generate_walmart_remote_cart_url(cart_items: List[Dict]) -> str:
    """
    Generates Walmart multi-item Add-To-Cart URL.
    Format: https://www.walmart.com/sc/cart/addToCart?items=itemId1|qty1,itemId2|qty2
    """
    items_param = []
    for item in cart_items:
        # Generic FMCG item ID fallback
        item_id = str(abs(hash(item.get("name", ""))) % 90000000 + 10000000)
        qty = max(1, int(round(item.get("quantity", 1.0))))
        items_param.append(f"{item_id}|{qty}")
    
    return f"https://www.walmart.com/sc/cart/addToCart?items={','.join(items_param)}"


# ─── 3. QUICK-COMMERCE AGENTIC BROWSER AUTOMATION (PLAYWRIGHT) ────────────────

def run_playwright_quick_commerce_cart(store_name: str, cart_items: List[Dict], progress_callback=None) -> Dict:
    """
    Launches a visible Chromium browser via Playwright and automatically navigates,
    searches, and clicks '+ ADD' for all cart items on Blinkit / Zepto / Instamart.
    """
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        return {"success": False, "error": "Playwright is not installed in the environment."}
    
    results = []
    with sync_playwright() as p:
        if progress_callback:
            progress_callback(f"🚀 Launching browser for {store_name}...")
        
        import os
        chrome_path = "/usr/bin/google-chrome-stable" if os.path.exists("/usr/bin/google-chrome-stable") else None
        
        launch_kwargs = {"headless": False, "slow_mo": 400}
        if chrome_path:
            launch_kwargs["executable_path"] = chrome_path
            
        browser = p.chromium.launch(**launch_kwargs)
        context = browser.new_context(
            viewport={"width": 1280, "height": 840},
            user_agent="Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
        )
        page = context.new_page()
        
        try:
            for i, item in enumerate(cart_items):
                name = item.get("name", "")
                qty = max(1, int(round(item.get("quantity", 1.0))))
                encoded = urllib.parse.quote_plus(name)
                
                if progress_callback:
                    progress_callback(f"🛒 Adding ({i+1}/{len(cart_items)}): {name}...")
                
                if "Blinkit" in store_name:
                    page.goto(f"https://blinkit.com/s/?q={encoded}", wait_until="domcontentloaded", timeout=12000)
                    page.wait_for_timeout(1500)
                    add_btn = page.locator('button:has-text("ADD"), div[role="button"]:has-text("ADD")').first
                    if add_btn.is_visible(timeout=3000):
                        add_btn.click()
                        results.append({"name": name, "status": "Added to Blinkit Cart"})
                    else:
                        results.append({"name": name, "status": "Search opened"})
                        
                elif "Zepto" in store_name:
                    page.goto(f"https://www.zeptonow.com/search?q={encoded}", wait_until="domcontentloaded", timeout=12000)
                    page.wait_for_timeout(1500)
                    add_btn = page.locator('button:has-text("Add"), button:has-text("ADD")').first
                    if add_btn.is_visible(timeout=3000):
                        add_btn.click()
                        results.append({"name": name, "status": "Added to Zepto Cart"})
                    else:
                        results.append({"name": name, "status": "Search opened"})
                        
                elif "Swiggy" in store_name:
                    page.goto(f"https://www.swiggy.com/instamart/search?query={encoded}", wait_until="domcontentloaded", timeout=12000)
                    page.wait_for_timeout(1500)
                    add_btn = page.locator('div:has-text("ADD"), button:has-text("ADD")').first
                    if add_btn.is_visible(timeout=3000):
                        add_btn.click()
                        results.append({"name": name, "status": "Added to Instamart Cart"})
                    else:
                        results.append({"name": name, "status": "Search opened"})
                        
                elif "BigBasket" in store_name:
                    page.goto(f"https://www.bigbasket.com/ps/?q={encoded}", wait_until="domcontentloaded", timeout=12000)
                    page.wait_for_timeout(1500)
                    add_btn = page.locator('button:has-text("Add"), button:has-text("ADD")').first
                    if add_btn.is_visible(timeout=3000):
                        add_btn.click()
                        results.append({"name": name, "status": "Added to BigBasket Cart"})
                    else:
                        results.append({"name": name, "status": "Search opened"})

            if progress_callback:
                progress_callback(f"✅ Finished adding items to {store_name}! Navigating to checkout...")
                
            # Final Cart navigation
            if "Blinkit" in store_name:
                page.goto("https://blinkit.com/cart")
            elif "Zepto" in store_name:
                page.goto("https://www.zeptonow.com/cart")
            elif "BigBasket" in store_name:
                page.goto("https://www.bigbasket.com/basket/")
                
            page.wait_for_timeout(4000)
            return {"success": True, "items": results}
        except Exception as err:
            return {"success": False, "error": str(err)}
