import sys
import os
import io
import time
import json
import math
import webbrowser
import threading
import numpy as np
import sounddevice as sd
import torch
from dotenv import load_dotenv

load_dotenv(".env")

from google import genai
from google.genai import types

# Import backend logic, model schema and enhancement pipeline
from app import (
    cart,
    SmartSuggestions,
    search_open_food_facts,
    enhance_audio,
    VoiceCommandResult,
    ExtractedItem,
    FLASH_MODELS,
    EXPANDED_GROCERY_PROMPT,
    gemini_client,
    _get_vad_model
)

from retailer_cart_service import (
    generate_amazon_remote_cart_url,
    generate_walmart_remote_cart_url,
    run_playwright_quick_commerce_cart
)

from tts_service import speak_text_sync

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QLineEdit, QComboBox, QScrollArea,
    QGridLayout, QFrame, QSizePolicy
)
from PyQt6.QtCore import Qt, QTimer, pyqtSignal, QObject, QThread, QPointF
from PyQt6.QtGui import (
    QColor, QFont, QPainter, QBrush, QLinearGradient, QRadialGradient,
    QPen, QPainterPath
)

# ─── BESPOKE APPLE / GOOGLE DESIGN SYSTEM ────────────────────────────────────

BESPOKE_STYLE = """
QMainWindow {
    background-color: #07080A;
}

QWidget {
    font-family: -apple-system, 'SF Pro Display', 'SF Pro Text', 'Segoe UI', 'Ubuntu', sans-serif;
    color: #F1F3F5;
}

/* Master Surfaces */
QWidget#masterContent {
    background-color: #07080A;
}

QScrollArea#masterScroll {
    background-color: #07080A;
    border: none;
}

QScrollArea#masterScroll > QWidget > QWidget {
    background-color: #07080A;
}

/* Left Hero Bento Card */
QFrame#heroPanel {
    background-color: #0E1015;
    border: 1px solid rgba(255, 255, 255, 0.07);
    border-radius: 28px;
}

/* Right Content Surface */
QFrame#bentoCard {
    background-color: #0E1015;
    border: 1px solid rgba(255, 255, 255, 0.07);
    border-radius: 22px;
}

QFrame#aisleCard {
    background-color: #13151C;
    border: 1px solid rgba(255, 255, 255, 0.05);
    border-radius: 18px;
}

QFrame#productCard {
    background-color: #13151C;
    border: 1px solid rgba(255, 255, 255, 0.06);
    border-radius: 16px;
}

/* Typography */
QLabel#brandTitle {
    font-size: 28px;
    font-weight: 800;
    color: #FFFFFF;
    letter-spacing: -0.8px;
}

QLabel#brandSubtitle {
    font-size: 13px;
    color: #717682;
    font-weight: 400;
}

QLabel#sectionTitle {
    font-size: 18px;
    font-weight: 700;
    color: #FFFFFF;
    letter-spacing: -0.3px;
}

QLabel#statusText {
    font-size: 13px;
    color: #9BA1B0;
    font-weight: 500;
}

QLabel#transcriptHero {
    font-size: 16px;
    font-weight: 600;
    color: #FFFFFF;
    line-height: 1.4;
    letter-spacing: -0.2px;
}

/* Minimalist Search & Input */
QLineEdit#commandInput {
    background-color: #141720;
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 16px;
    padding: 12px 18px;
    font-size: 13px;
    color: #FFFFFF;
}
QLineEdit#commandInput:focus {
    border: 1px solid #FFFFFF;
    background-color: #181C26;
}

QComboBox#langPill {
    background-color: #141720;
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 12px;
    padding: 5px 12px;
    font-size: 11px;
    color: #8E94A4;
    font-weight: 600;
}
QComboBox#langPill::drop-down {
    border: none;
}
QComboBox QAbstractItemView {
    background-color: #141720;
    color: white;
    selection-background-color: #222736;
    border: 1px solid rgba(255, 255, 255, 0.1);
    border-radius: 8px;
    padding: 4px;
}

/* Minimalist Buttons */
QPushButton#primarySendBtn {
    background-color: #FFFFFF;
    color: #07080A;
    border: none;
    border-radius: 16px;
    padding: 12px 20px;
    font-weight: 700;
    font-size: 13px;
}
QPushButton#primarySendBtn:hover {
    background-color: #E2E5EA;
}

QPushButton#clearCartBtn {
    background-color: transparent;
    color: #717682;
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 14px;
    padding: 8px 16px;
    font-weight: 600;
    font-size: 12px;
}
QPushButton#clearCartBtn:hover {
    color: #EF4444;
    border-color: rgba(239, 68, 68, 0.3);
    background-color: rgba(239, 68, 68, 0.08);
}

QPushButton#amazonExportBtn {
    background-color: #181B24;
    color: #F1F3F5;
    border: 1px solid rgba(255, 255, 255, 0.1);
    border-radius: 14px;
    padding: 8px 18px;
    font-weight: 600;
    font-size: 12px;
}
QPushButton#amazonExportBtn:hover {
    background-color: #222634;
    border-color: rgba(255, 255, 255, 0.2);
}

/* Subtle Apple Pill Tags */
QPushButton#suggestionPill {
    background-color: #13161F;
    border: 1px solid rgba(255, 255, 255, 0.07);
    color: #9BA1B0;
    border-radius: 14px;
    padding: 6px 14px;
    font-size: 12px;
    font-weight: 500;
}
QPushButton#suggestionPill:hover {
    background-color: #1A1E2B;
    color: #FFFFFF;
    border-color: rgba(255, 255, 255, 0.18);
}

QScrollBar:vertical {
    background: transparent;
    width: 6px;
    margin: 0;
}
QScrollBar::handle:vertical {
    background: #1E222D;
    min-height: 20px;
    border-radius: 3px;
}
QScrollBar::handle:vertical:hover {
    background: #2D3344;
}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
    height: 0px;
}
QScrollBar:horizontal {
    height: 0px;
}
"""

# ─── APPLE INTELLIGENCE / SIRI LIVING ORB ─────────────────────────────────────

class SiriLivingOrbWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(140, 140)
        self.is_recording = False
        self.is_speaking = False
        self.phase = 0.0
        self.energy = 0.05
        self.target_energy = 0.05

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.animate_frame)
        self.timer.start(16)  # 60 FPS

    def set_recording(self, state: bool):
        self.is_recording = state
        if not state:
            self.is_speaking = False
            self.target_energy = 0.05
        self.update()

    def set_speech_active(self, speaking: bool):
        self.is_speaking = speaking
        self.update()

    def update_energy(self, chunk):
        if len(chunk) > 0:
            rms = np.sqrt(np.mean(chunk**2))
            self.target_energy = np.clip(rms * 12.0, 0.08, 1.0)

    def animate_frame(self):
        speed = 0.06 if self.is_speaking else (0.025 if self.is_recording else 0.012)
        self.phase += speed
        self.energy += (self.target_energy - self.energy) * 0.18
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        center_x = self.width() / 2.0
        center_y = self.height() / 2.0
        base_r = 46.0 + (self.energy * 14.0)

        # 1. Outer Ambient Aura Glow
        if self.is_recording:
            aura = QRadialGradient(center_x, center_y, base_r * 1.6)
            if self.is_speaking:
                aura.setColorAt(0.0, QColor(255, 75, 75, 120))
                aura.setColorAt(0.5, QColor(155, 81, 224, 60))
            else:
                aura.setColorAt(0.0, QColor(56, 189, 248, 80))
                aura.setColorAt(0.5, QColor(99, 102, 241, 40))
            aura.setColorAt(1.0, QColor(7, 8, 10, 0))
            painter.setBrush(QBrush(aura))
            painter.setPen(Qt.PenStyle.NoPen)
            painter.drawEllipse(QPointF(center_x, center_y), base_r * 1.6, base_r * 1.6)

        # 2. Organic Sine Waveform Contour Path
        path = QPainterPath()
        points = 64
        for i in range(points):
            angle = (i / points) * 2 * math.pi
            wave = math.sin(angle * 3 + self.phase) * (self.energy * 9.0)
            wave2 = math.cos(angle * 5 - self.phase * 1.5) * (self.energy * 5.0)
            r = base_r + wave + wave2
            x = center_x + r * math.cos(angle)
            y = center_y + r * math.sin(angle)
            if i == 0:
                path.moveTo(x, y)
            else:
                path.lineTo(x, y)
        path.closeSubpath()

        # 3. Siri Iridescent Gradient Fill
        grad = QLinearGradient(0, 0, self.width(), self.height())
        if self.is_recording:
            if self.is_speaking:
                grad.setColorAt(0.0, QColor("#FF4B4B"))
                grad.setColorAt(0.35, QColor("#FF8A3D"))
                grad.setColorAt(0.7, QColor("#9B51E0"))
                grad.setColorAt(1.0, QColor("#3B82F6"))
            else:
                grad.setColorAt(0.0, QColor("#2563EB"))
                grad.setColorAt(0.5, QColor("#7C3AED"))
                grad.setColorAt(1.0, QColor("#38BDF8"))
        else:
            grad.setColorAt(0.0, QColor("#1A1D27"))
            grad.setColorAt(0.5, QColor("#222736"))
            grad.setColorAt(1.0, QColor("#141722"))

        painter.setBrush(QBrush(grad))
        painter.setPen(QPen(QColor(255, 255, 255, 50 if self.is_recording else 20), 1.5))
        painter.drawPath(path)

        # 4. Central Icon
        painter.setPen(QPen(QColor("#FFFFFF" if self.is_recording else "#717682")))
        font = QFont("-apple-system", 20, QFont.Weight.Bold)
        painter.setFont(font)
        icon = "●" if self.is_speaking else ("🎙" if not self.is_recording else "...")
        painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, icon)


# ─── REAL-TIME STREAMING SILERO VAD ENDPOINTING THREAD ────────────────────────

class HandsFreeStreamingAudioThread(QThread):
    energy_update = pyqtSignal(object)
    speech_state_changed = pyqtSignal(bool)  # True = speaking, False = silent
    auto_endpoint_triggered = pyqtSignal(bytes)
    error = pyqtSignal(str)

    def __init__(self, sample_rate=16000, silence_hangtime_ms=900):
        super().__init__()
        self.sample_rate = sample_rate
        self.chunk_size = 512  # 32ms at 16kHz (Standard Silero VAD window)
        self.silence_hangtime_ms = silence_hangtime_ms
        self.max_silence_frames = int(silence_hangtime_ms / 32)  # ~28 frames for 900ms
        self.is_recording = False
        self.audio_buffer = []
        self.vad_model = _get_vad_model()

    def run(self):
        self.is_recording = True
        self.audio_buffer = []
        speech_ever_started = False
        speech_currently_active = False
        consecutive_silence_frames = 0
        min_speech_frames = 6  # Require at least ~190ms of speech before endpointing

        try:
            with sd.InputStream(samplerate=self.sample_rate, channels=1, dtype='float32', blocksize=self.chunk_size) as stream:
                while self.is_recording:
                    chunk, overflowed = stream.read(self.chunk_size)
                    pcm_chunk = chunk[:, 0].copy()
                    self.audio_buffer.append(pcm_chunk)
                    self.energy_update.emit(pcm_chunk)

                    # Real-Time Silero VAD Inference on 32ms chunk
                    if self.vad_model is not None:
                        with torch.no_grad():
                            tensor_chunk = torch.from_numpy(pcm_chunk).float()
                            speech_prob = self.vad_model(tensor_chunk, self.sample_rate).item()

                        if speech_prob > 0.45:
                            if not speech_currently_active:
                                speech_currently_active = True
                                speech_ever_started = True
                                self.speech_state_changed.emit(True)
                            consecutive_silence_frames = 0
                        elif speech_prob < 0.25:
                            if speech_currently_active:
                                consecutive_silence_frames += 1
                                if consecutive_silence_frames >= 4:  # ~120ms of silence
                                    speech_currently_active = False
                                    self.speech_state_changed.emit(False)
                            elif speech_ever_started:
                                consecutive_silence_frames += 1

                        # AUTOMATIC HANDS-FREE DISPATCH:
                        # If user spoke and has now stopped for >= 900ms, AUTO DISPATCH!
                        if speech_ever_started and consecutive_silence_frames >= self.max_silence_frames and len(self.audio_buffer) >= min_speech_frames:
                            print(f"🎯 Auto-Endpoint Triggered! Silence frames: {consecutive_silence_frames} ({consecutive_silence_frames*32}ms)")
                            self.is_recording = False
                            break

            if self.audio_buffer:
                full_pcm = np.concatenate(self.audio_buffer, axis=0)
                from app import encode_pcm_to_webm
                webm_bytes = encode_pcm_to_webm(full_pcm, sr=self.sample_rate)
                self.auto_endpoint_triggered.emit(webm_bytes)
            else:
                self.error.emit("No audio recorded.")
        except Exception as e:
            self.error.emit(str(e))

    def stop(self):
        self.is_recording = False


# ─── THREAD-SAFE ASYNC GEMINI PARSER ──────────────────────────────────────────

class CommandResultWorker(QThread):
    command_ready = pyqtSignal(object, dict, list)
    error_occurred = pyqtSignal(str)

    def __init__(self, mode, payload):
        super().__init__()
        self.mode = mode
        self.payload = payload

    def run(self):
        try:
            diag = {}
            search_results = []

            if self.mode == 'AUDIO':
                enhanced_bytes, diag = enhance_audio(self.payload)
                send_bytes = enhanced_bytes
                send_mime = "audio/webm"
            else:
                send_bytes = None

            cart_keys = list(cart.items.keys())
            cart_context = f"Current Cart: {', '.join(cart_keys)}" if cart_keys else "Cart is empty."

            if self.mode == 'AUDIO':
                full_prompt = f"{EXPANDED_GROCERY_PROMPT}\n\n{cart_context}"
                contents = [full_prompt, types.Part.from_bytes(data=send_bytes, mime_type=send_mime)]
            else:
                full_prompt = f"{EXPANDED_GROCERY_PROMPT}\n\n{cart_context}\nUser Spoken Text: \"{self.payload}\""
                contents = full_prompt

            parsed_cmd = None
            for m in FLASH_MODELS:
                try:
                    res = gemini_client.models.generate_content(
                        model=m,
                        contents=contents,
                        config=types.GenerateContentConfig(
                            response_mime_type="application/json",
                            response_schema=VoiceCommandResult,
                            temperature=0.1
                        )
                    )
                    parsed_cmd = VoiceCommandResult.model_validate_json(res.text)
                    break
                except Exception as err:
                    print(f"⚠️ Model {m} note: {err}")
                    continue

            if not parsed_cmd:
                if self.mode == 'TEXT':
                    clean = self.payload.lower().replace("i want ", "").replace("add ", "").title()
                    parsed_cmd = VoiceCommandResult(
                        intent="ADD",
                        detected_language="en",
                        transcript=self.payload,
                        items_to_add=[ExtractedItem(product_name=clean, quantity=1.0, unit="item", category="Pantry")],
                        feedback_message=f"Added {clean} to cart."
                    )

            if parsed_cmd and parsed_cmd.search_query:
                import asyncio
                search_results = asyncio.run(search_open_food_facts(parsed_cmd.search_query, max_price=parsed_cmd.search_max_price))

            self.command_ready.emit(parsed_cmd, diag, search_results)
        except Exception as e:
            self.error_occurred.emit(str(e))


# ─── MAIN LUXURY DESKTOP WINDOW ───────────────────────────────────────────────

class LuxuryShoppingAssistantApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Voice Shopping Assistant")
        self.setMinimumSize(1160, 800)
        self.recorder_thread = None
        self.is_recording = False
        self.worker_thread = None

        self.setup_ui()
        self.setStyleSheet(BESPOKE_STYLE)
        self.refresh_cart_and_suggestions()

    def setup_ui(self):
        master_widget = QWidget(self)
        self.setCentralWidget(master_widget)
        master_layout = QHBoxLayout(master_widget)
        master_layout.setContentsMargins(28, 28, 28, 28)
        master_layout.setSpacing(24)

        # ═════════════════════════════════════════════════════════════════════
        # LEFT BENTO PANEL: SIRI LIVING VOICE ORB & CONVERSATION HUB (340px)
        # ═════════════════════════════════════════════════════════════════════
        left_panel = QFrame(self)
        left_panel.setObjectName("heroPanel")
        left_panel.setFixedWidth(340)
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(26, 26, 26, 26)
        left_layout.setSpacing(18)

        # Header Title & Language Switcher
        left_header = QHBoxLayout()
        title_box = QVBoxLayout()
        title_box.setSpacing(2)
        app_title = QLabel("Assistant", left_panel)
        app_title.setObjectName("brandTitle")
        app_sub = QLabel("Hands-Free VAD Endpointing", left_panel)
        app_sub.setObjectName("brandSubtitle")
        title_box.addWidget(app_title)
        title_box.addWidget(app_sub)
        left_header.addLayout(title_box)
        left_header.addStretch()

        self.lang_select = QComboBox(left_panel)
        self.lang_select.setObjectName("langPill")
        self.lang_select.addItems(["en-IN", "en-US", "en-GB", "hi-IN", "es-ES"])
        left_header.addWidget(self.lang_select)
        left_layout.addLayout(left_header)

        # Siri Living Orb Center
        orb_container = QVBoxLayout()
        orb_container.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.siri_orb = SiriLivingOrbWidget(left_panel)
        self.siri_orb.setCursor(Qt.CursorShape.PointingHandCursor)
        self.siri_orb.mousePressEvent = lambda e: self.toggle_recording()
        orb_container.addWidget(self.siri_orb)
        left_layout.addLayout(orb_container)

        # Status Line & Spoken Quote
        self.status_label = QLabel("Tap orb once & speak naturally", left_panel)
        self.status_label.setObjectName("statusText")
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        left_layout.addWidget(self.status_label)

        self.transcript_box = QLabel("“Auto-detects when you stop speaking”", left_panel)
        self.transcript_box.setObjectName("transcriptHero")
        self.transcript_box.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.transcript_box.setWordWrap(True)
        left_layout.addWidget(self.transcript_box)

        # Diagnostics line
        self.diag_label = QLabel("Real-Time Streaming VAD (900ms Hangtime)", left_panel)
        self.diag_label.setStyleSheet("color: #4A5060; font-size: 11px; font-weight: 500;")
        self.diag_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        left_layout.addWidget(self.diag_label)

        left_layout.addStretch()

        # Command Input Bar
        input_container = QVBoxLayout()
        input_container.setSpacing(10)
        self.text_input = QLineEdit(left_panel)
        self.text_input.setObjectName("commandInput")
        self.text_input.setPlaceholderText("Type a grocery request...")
        self.text_input.returnPressed.connect(self.handle_text_submit)
        input_container.addWidget(self.text_input)

        btn_row = QHBoxLayout()
        btn_row.setSpacing(10)
        send_btn = QPushButton("Send Command", left_panel)
        send_btn.setObjectName("primarySendBtn")
        send_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        send_btn.clicked.connect(self.handle_text_submit)
        btn_row.addWidget(send_btn)

        clear_btn = QPushButton("Clear", left_panel)
        clear_btn.setObjectName("clearCartBtn")
        clear_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        clear_btn.clicked.connect(self.handle_clear_cart)
        btn_row.addWidget(clear_btn)
        input_container.addLayout(btn_row)

        left_layout.addLayout(input_container)
        master_layout.addWidget(left_panel)

        # ═════════════════════════════════════════════════════════════════════
        # RIGHT PANEL: LIVING SHOPPING LIST & SMART RECOMMENDATIONS
        # ═════════════════════════════════════════════════════════════════════
        right_scroll = QScrollArea(self)
        right_scroll.setObjectName("masterScroll")
        right_scroll.setWidgetResizable(True)
        right_scroll.setMinimumWidth(640)
        right_scroll.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        master_layout.addWidget(right_scroll, 1)

        right_content = QWidget()
        right_content.setObjectName("masterContent")
        right_content.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        right_scroll.setWidget(right_content)
        right_layout = QVBoxLayout(right_content)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(18)

        # 1. Header Toolbar with Multi-Retailer Checkout
        right_toolbar = QHBoxLayout()
        list_header = QLabel("Shopping List", right_content)
        list_header.setObjectName("sectionTitle")
        right_toolbar.addWidget(list_header)
        right_toolbar.addStretch()

        self.store_select = QComboBox(right_content)
        self.store_select.setObjectName("langPill")
        self.store_select.addItems([
            "Amazon Fresh",
            "Blinkit (10-min)",
            "Zepto",
            "Swiggy Instamart",
            "BigBasket",
            "Instacart"
        ])
        right_toolbar.addWidget(self.store_select)

        amazon_remote_btn = QPushButton("⚡ 1-Click Amazon Cart", right_content)
        amazon_remote_btn.setObjectName("amazonExportBtn")
        amazon_remote_btn.setStyleSheet("""
            QPushButton#amazonExportBtn {
                background-color: #FF9900;
                color: #000000;
                font-weight: 700;
                border-radius: 12px;
                padding: 7px 14px;
            }
            QPushButton#amazonExportBtn:hover {
                background-color: #FFAC33;
            }
        """)
        amazon_remote_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        amazon_remote_btn.clicked.connect(self.export_amazon_remote_cart)
        right_toolbar.addWidget(amazon_remote_btn)

        auto_agent_btn = QPushButton("🤖 Auto-Cart Agent", right_content)
        auto_agent_btn.setObjectName("amazonExportBtn")
        auto_agent_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        auto_agent_btn.clicked.connect(self.launch_auto_cart_agent)
        right_toolbar.addWidget(auto_agent_btn)

        copy_btn = QPushButton("📋 Copy", right_content)
        copy_btn.setObjectName("amazonExportBtn")
        copy_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        copy_btn.clicked.connect(self.copy_list_to_clipboard)
        right_toolbar.addWidget(copy_btn)

        right_layout.addLayout(right_toolbar)

        # 2. Apple-Style Smart Suggestion Tags Carousel
        suggestions_box = QFrame(right_content)
        suggestions_box.setObjectName("bentoCard")
        suggestions_box.setFixedHeight(82)
        sug_inner = QVBoxLayout(suggestions_box)
        sug_inner.setContentsMargins(18, 10, 18, 10)
        sug_inner.setSpacing(6)

        sug_title = QLabel("SMART SUGGESTIONS", suggestions_box)
        sug_title.setStyleSheet("color: #717682; font-size: 11px; font-weight: 700; letter-spacing: 0.8px;")
        sug_inner.addWidget(sug_title)

        self.tag_scroll = QScrollArea(suggestions_box)
        self.tag_scroll.setFixedHeight(34)
        self.tag_scroll.setWidgetResizable(True)
        self.tag_scroll.setStyleSheet("background: transparent; border: none;")
        self.tag_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.tag_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        self.tag_widget = QWidget()
        self.tag_widget.setStyleSheet("background: transparent;")
        self.tag_container = QHBoxLayout(self.tag_widget)
        self.tag_container.setContentsMargins(0, 0, 0, 0)
        self.tag_container.setSpacing(8)
        self.tag_scroll.setWidget(self.tag_widget)
        sug_inner.addWidget(self.tag_scroll)
        right_layout.addWidget(suggestions_box)

        # 3. Live Product Search Results Section (Material 3 Cards)
        self.search_frame = QFrame(right_content)
        self.search_frame.setObjectName("bentoCard")
        self.search_layout = QVBoxLayout(self.search_frame)
        self.search_layout.setContentsMargins(18, 16, 18, 16)
        self.search_title = QLabel("SEARCH RESULTS", self.search_frame)
        self.search_title.setStyleSheet("color: #717682; font-size: 11px; font-weight: 700; letter-spacing: 0.8px;")
        self.search_layout.addWidget(self.search_title)
        self.product_grid_layout = QGridLayout()
        self.product_grid_layout.setSpacing(12)
        self.search_layout.addLayout(self.product_grid_layout)
        self.search_frame.hide()
        right_layout.addWidget(self.search_frame)

        # 4. Categorized Supermarket Aisles
        self.aisle_container = QWidget(right_content)
        self.aisle_container.setStyleSheet("background: transparent;")
        self.aisle_grid_layout = QGridLayout(self.aisle_container)
        self.aisle_grid_layout.setContentsMargins(0, 0, 0, 0)
        self.aisle_grid_layout.setSpacing(16)
        right_layout.addWidget(self.aisle_container)

        right_layout.addStretch()

    # ─── HANDS-FREE AUDIO STREAMING & RECOGNITION ─────────────────────────────

    def toggle_recording(self):
        if not self.is_recording:
            self.start_recording()
        else:
            self.stop_recording()

    def start_recording(self):
        self.is_recording = True
        self.siri_orb.set_recording(True)
        self.status_label.setText("Listening... (Speak now)")
        self.status_label.setStyleSheet("color: #38BDF8; font-weight: 700;")
        self.transcript_box.setText("Listening... Speak your grocery items")

        self.recorder_thread = HandsFreeStreamingAudioThread(sample_rate=16000, silence_hangtime_ms=900)
        self.recorder_thread.energy_update.connect(self.siri_orb.update_energy)
        self.recorder_thread.speech_state_changed.connect(self.on_speech_state_changed)
        self.recorder_thread.auto_endpoint_triggered.connect(self.handle_recording_finished)
        self.recorder_thread.error.connect(self.handle_recording_error)
        self.recorder_thread.start()

    def on_speech_state_changed(self, is_speaking: bool):
        self.siri_orb.set_speech_active(is_speaking)
        if is_speaking:
            self.status_label.setText("Hearing speech...")
            self.status_label.setStyleSheet("color: #FF5A3C; font-weight: 700;")
        else:
            self.status_label.setText("Detecting endpoint...")
            self.status_label.setStyleSheet("color: #9BA1B0; font-weight: 500;")

    def stop_recording(self):
        self.is_recording = False
        self.siri_orb.set_recording(False)
        self.status_label.setText("Enhancing & Parsing...")
        self.status_label.setStyleSheet("color: #9BA1B0; font-weight: 500;")
        if self.recorder_thread:
            self.recorder_thread.stop()

    def handle_recording_finished(self, raw_webm_bytes):
        self.is_recording = False
        self.siri_orb.set_recording(False)
        self.status_label.setText("Parsing with Gemini Flash...")
        self.status_label.setStyleSheet("color: #38BDF8; font-weight: 600;")

        self.worker_thread = CommandResultWorker(mode='AUDIO', payload=raw_webm_bytes)
        self.worker_thread.command_ready.connect(self.on_command_ready)
        self.worker_thread.error_occurred.connect(self.on_worker_error)
        self.worker_thread.start()

    def handle_text_submit(self):
        text = self.text_input.text().strip()
        if not text:
            return
        self.text_input.clear()
        self.status_label.setText("Processing...")
        self.transcript_box.setText(f"“{text}”")

        self.worker_thread = CommandResultWorker(mode='TEXT', payload=text)
        self.worker_thread.command_ready.connect(self.on_command_ready)
        self.worker_thread.error_occurred.connect(self.on_worker_error)
        self.worker_thread.start()

    def on_command_ready(self, parsed_cmd: VoiceCommandResult, diag: dict, search_results: list):
        if not parsed_cmd:
            self.status_label.setText("Speech not recognized")
            return

        self.transcript_box.setText(f"“{parsed_cmd.transcript}”")
        self.status_label.setText(parsed_cmd.feedback_message)

        if diag:
            raw_s = diag.get("raw_duration_s", 0)
            vad_s = diag.get("vad_duration_s", 0)
            rms = diag.get("enhanced_rms_dbfs", 0)
            self.diag_label.setText(f"VAD Trimmed {raw_s}s → {vad_s}s | Leveled: {rms} dBFS")

        # Execute Additions
        for itm in parsed_cmd.items_to_add:
            cart.add(itm)

        # Execute Removals
        for rm_name in parsed_cmd.items_to_remove:
            cart.remove(rm_name)

        if parsed_cmd.intent == "CLEAR":
            cart.clear()

        if search_results:
            self.render_search_results(search_results)
        else:
            self.search_frame.hide()

        self.refresh_cart_and_suggestions()

        # Speak back feedback aloud (TTS read-back)
        if parsed_cmd.feedback_message:
            self.play_tts_feedback(parsed_cmd.feedback_message)

    def play_tts_feedback(self, text: str):
        lang = self.lang_select.currentText().split()[0]
        
        class TTSThread(QThread):
            energy = pyqtSignal(object)
            finished_tts = pyqtSignal()
            
            def run(self):
                try:
                    speak_text_sync(text, lang_code=lang, visualizer_callback=self.energy.emit)
                except Exception as e:
                    print(f"TTS Thread error: {e}")
                finally:
                    self.finished_tts.emit()

        self.tts_thread = TTSThread()
        self.tts_thread.energy.connect(self.siri_orb.update_energy)
        self.tts_thread.finished_tts.connect(lambda: self.siri_orb.set_speech_active(False))
        self.siri_orb.set_speech_active(True)
        self.tts_thread.start()

    def on_worker_error(self, err_msg):
        self.status_label.setText(f"Error: {err_msg}")

    def handle_recording_error(self, err_msg):
        self.status_label.setText(f"Notice: {err_msg}")
        self.stop_recording()

    def handle_clear_cart(self):
        cart.clear()
        self.status_label.setText("Cart cleared")
        self.transcript_box.setText("“Shopping list is empty”")
        self.diag_label.setText("Real-Time Streaming VAD (900ms Hangtime)")
        self.search_frame.hide()
        self.refresh_cart_and_suggestions()

    # ─── REFRESH UI: AISLES & TAGS ────────────────────────────────────────────

    def refresh_cart_and_suggestions(self):
        while self.tag_container.count():
            item = self.tag_container.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        season = SmartSuggestions.current_season()
        seasonal = SmartSuggestions.SEASONS[season][:4]
        history = SmartSuggestions.get_history_recs(list(cart.items.keys()))[:3]

        for s in seasonal:
            btn = QPushButton(f"{s} (In-Season)", self.tag_widget)
            btn.setObjectName("suggestionPill")
            btn.setSizePolicy(QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Fixed)
            btn.setCursor(Qt.CursorShape.PointingHandCursor)
            btn.clicked.connect(lambda ch, item=s: self.add_quick_item(item))
            self.tag_container.addWidget(btn)

        for h in history:
            btn = QPushButton(f"Replenish {h}", self.tag_widget)
            btn.setObjectName("suggestionPill")
            btn.setSizePolicy(QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Fixed)
            btn.setCursor(Qt.CursorShape.PointingHandCursor)
            btn.clicked.connect(lambda ch, item=h: self.add_quick_item(item))
            self.tag_container.addWidget(btn)

        while self.aisle_grid_layout.count():
            item = self.aisle_grid_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        categorized = cart.get_categorized()
        if not categorized:
            empty_lbl = QLabel("Your shopping list is empty. Speak or type to add items.", self.aisle_container)
            empty_lbl.setStyleSheet("color: #4A5060; font-size: 13px; padding: 20px;")
            self.aisle_grid_layout.addWidget(empty_lbl, 0, 0)
            return

        row, col = 0, 0
        for category, items in categorized.items():
            card = QFrame(self.aisle_container)
            card.setObjectName("aisleCard")
            card_layout = QVBoxLayout(card)
            card_layout.setContentsMargins(18, 16, 18, 16)
            card_layout.setSpacing(12)

            cat_header = QLabel(category.upper(), card)
            cat_header.setStyleSheet("color: #717682; font-weight: 700; font-size: 11px; letter-spacing: 0.8px;")
            card_layout.addWidget(cat_header)

            for itm in items:
                itm_row = QHBoxLayout()
                itm_row.setSpacing(8)

                name_lbl = QLabel(itm['name'], card)
                name_lbl.setStyleSheet("color: #FFFFFF; font-size: 14px; font-weight: 600;")
                qty_lbl = QLabel(f"{itm['quantity']:g} {itm['unit']}", card)
                qty_lbl.setStyleSheet("color: #8E94A4; font-size: 12px; background: rgba(255,255,255,0.06); padding: 3px 8px; border-radius: 8px;")

                store_btn = QPushButton("↗", card)
                store_btn.setFixedSize(24, 24)
                store_btn.setStyleSheet("color: #38BDF8; background: rgba(56, 189, 248, 0.1); border-radius: 12px; font-size: 12px; font-weight: bold; border: none;")
                store_btn.setCursor(Qt.CursorShape.PointingHandCursor)
                store_btn.setToolTip("Find this item in store")
                store_btn.clicked.connect(lambda ch, n=itm['name']: self.open_single_item_in_store(n))

                del_btn = QPushButton("×", card)
                del_btn.setFixedSize(22, 22)
                del_btn.setStyleSheet("color: #717682; background: transparent; font-size: 16px; font-weight: bold; border: none;")
                del_btn.setCursor(Qt.CursorShape.PointingHandCursor)
                del_btn.clicked.connect(lambda ch, n=itm['base_name']: self.remove_item(n))

                itm_row.addWidget(name_lbl)
                itm_row.addWidget(qty_lbl)
                itm_row.addStretch()
                itm_row.addWidget(store_btn)
                itm_row.addWidget(del_btn)
                card_layout.addLayout(itm_row)

            self.aisle_grid_layout.addWidget(card, row, col)
            col += 1
            if col >= 2:
                col = 0
                row += 1

    def render_search_results(self, products):
        self.search_frame.show()
        while self.product_grid_layout.count():
            item = self.product_grid_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        for i, p in enumerate(products[:4]):
            p_card = QFrame(self.search_frame)
            p_card.setObjectName("productCard")
            p_layout = QVBoxLayout(p_card)
            p_layout.setContentsMargins(14, 14, 14, 14)
            p_layout.setSpacing(6)

            name_lbl = QLabel(p['name'], p_card)
            name_lbl.setStyleSheet("color: #FFFFFF; font-size: 13px; font-weight: 600;")
            brand_lbl = QLabel(f"{p['brand']} • Score {p['nutriscore']}", p_card)
            brand_lbl.setStyleSheet("color: #717682; font-size: 11px;")
            price_lbl = QLabel(p['price'], p_card)
            price_lbl.setStyleSheet("color: #FFFFFF; font-weight: 700; font-size: 14px;")

            add_btn = QPushButton("+ Add Item", p_card)
            add_btn.setStyleSheet("""
                QPushButton {
                    background: #1E2330;
                    color: #FFFFFF;
                    border: 1px solid rgba(255,255,255,0.08);
                    border-radius: 10px;
                    padding: 6px;
                    font-weight: 600;
                    font-size: 11px;
                }
                QPushButton:hover {
                    background: #2B3245;
                }
            """)
            add_btn.setCursor(Qt.CursorShape.PointingHandCursor)
            add_btn.clicked.connect(lambda ch, item=p['name']: self.add_quick_item(item))

            p_layout.addWidget(name_lbl)
            p_layout.addWidget(brand_lbl)
            p_layout.addWidget(price_lbl)
            p_layout.addWidget(add_btn)

            row = i // 2
            col = i % 2
            self.product_grid_layout.addWidget(p_card, row, col)

    def add_quick_item(self, name):
        self.text_input.setText(f"Add 1 {name}")
        self.handle_text_submit()

    def remove_item(self, name):
        cart.remove(name)
        self.status_label.setText(f"Removed {name}")
        self.refresh_cart_and_suggestions()

    def export_amazon_remote_cart(self):
        items = list(cart.items.values())
        if not items:
            self.status_label.setText("Cart is empty. Add items first!")
            return
        
        locale = "in" if "en-IN" in self.lang_select.currentText() or "hi-IN" in self.lang_select.currentText() else "com"
        remote_url, matched = generate_amazon_remote_cart_url(items, locale=locale)
        webbrowser.open(remote_url)
        if matched:
            self.status_label.setText(f"⚡ Staged {len(matched)} items directly into Amazon Cart! ({matched[0]['title']}...)")
        else:
            self.status_label.setText("Opened Amazon Grocery Search ↗")

    def launch_auto_cart_agent(self):
        items = list(cart.items.values())
        if not items:
            self.status_label.setText("Cart is empty. Add items first!")
            return
        
        store = self.store_select.currentText()
        self.status_label.setText(f"🤖 Starting Auto-Cart Agent for {store}...")
        
        class AutoCartWorker(QThread):
            progress = pyqtSignal(str)
            done = pyqtSignal(dict)
            
            def run(self):
                try:
                    res = run_playwright_quick_commerce_cart(store, items, progress_callback=self.progress.emit)
                    self.done.emit(res)
                except Exception as e:
                    self.done.emit({"success": False, "error": str(e)})
                
        self.auto_cart_worker = AutoCartWorker()
        self.auto_cart_worker.progress.connect(lambda msg: self.status_label.setText(msg))
        self.auto_cart_worker.done.connect(lambda res: self.status_label.setText(
            "✅ Auto-Cart Agent completed! Check browser window." if res.get("success") else f"Agent error: {res.get('error')}"
        ))
        self.auto_cart_worker.start()

    def get_store_search_url(self, item_name: str) -> str:
        store = self.store_select.currentText()
        import urllib.parse
        encoded = urllib.parse.quote_plus(item_name)
        if "Amazon" in store:
            return f"https://www.amazon.in/s?k={encoded}&i=nowstore"
        elif "Blinkit" in store:
            return f"https://blinkit.com/s/?q={encoded}"
        elif "Zepto" in store:
            return f"https://www.zeptonow.com/search?q={encoded}"
        elif "Swiggy" in store:
            return f"https://www.swiggy.com/instamart/search?query={encoded}"
        elif "BigBasket" in store:
            return f"https://www.bigbasket.com/ps/?q={encoded}"
        elif "Instacart" in store:
            return f"https://www.instacart.com/store/s?k={encoded}"
        return f"https://www.google.com/search?q=buy+{encoded}+grocery"

    def open_single_item_in_store(self, item_name: str):
        url = self.get_store_search_url(item_name)
        webbrowser.open(url)
        self.status_label.setText(f"Opened '{item_name}' in {self.store_select.currentText()}")

    def copy_list_to_clipboard(self):
        items = list(cart.items.values())
        if not items:
            self.status_label.setText("Cart is empty")
            return
        lines = ["🛒 Grocery Shopping List:"]
        for itm in items:
            lines.append(f"• {itm['quantity']:g} {itm['unit']} {itm['name']} ({itm['category']})")
        text = "\n".join(lines)
        clipboard = QApplication.clipboard()
        clipboard.setText(text)
        self.status_label.setText("📋 Copied formatted grocery list to clipboard!")


# ─── APPLICATION ENTRY POINT ──────────────────────────────────────────────────

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = LuxuryShoppingAssistantApp()
    window.show()
    sys.exit(app.exec())
