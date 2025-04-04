import ollama
import tkinter as tk
from tkinter import filedialog, scrolledtext, ttk
from PIL import Image, ImageTk
import pyttsx3
import threading
import queue
import re
import time
import whisper
import faster_whisper
import pyaudio
import wave
import numpy as np
import tempfile
import os
import torch
import torchaudio
from collections import deque
import traceback
import fitz
from tkinterdnd2 import TkinterDnD, DND_FILES

# --- Flask Imports (Phase 1) ---
from flask import Flask, request, jsonify, render_template, Response
import json
import sys


# ===================
# Constants
# ===================
# --- Audio ---
CHUNK = 1024            # Audio frames per buffer
VAD_CHUNK = 512         # Smaller chunk for VAD responsiveness
RATE = 16000            # Sampling rate (must be 16 kHz for Silero VAD & good for Whisper)
FORMAT = pyaudio.paInt16
CHANNELS = 1
SILENCE_THRESHOLD_SECONDS = 1.0 # How long silence triggers end of recording
MIN_SPEECH_DURATION_SECONDS = 0.2 # Ignore very short bursts
PRE_SPEECH_BUFFER_SECONDS = 0.3 # Keep audio before speech starts

# --- UI ---
APP_TITLE = "Ollama Multimodal Chat++ (Streaming, TTS, VAD) + Web UI"
WINDOW_GEOMETRY = "850x850"

# --- Models ---
DEFAULT_OLLAMA_MODEL = "gemma3:27b"
DEFAULT_WHISPER_MODEL_SIZE = "turbo-large" # Faster startup, change to 'medium' or 'large' for better accuracy

# --- Whisper ---
WHISPER_LANGUAGES = [
    ("Auto Detect", None), ("English", "en"), ("Finnish", "fi"), ("Swedish", "sv"),
    ("German", "de"), ("French", "fr"), ("Spanish", "es"), ("Italian", "it"),
    ("Russian", "ru"), ("Chinese", "zh"), ("Japanese", "ja")
]
WHISPER_MODEL_SIZES = ["tiny", "base", "small", "medium", "large", "turbo-tiny", "turbo-base", "turbo-small", "turbo-medium", "turbo-large"]

# --- Web Server ---
FLASK_PORT = 5000
FLASK_HOST = "0.0.0.0" # Accessible on local network

# ===================
# Globals
# ===================
# --- Ollama & Chat ---
messages = []
messages_lock = threading.Lock() # (Phase 2) Lock for shared access
selected_image_path = ""
image_sent_in_history = False
current_model = DEFAULT_OLLAMA_MODEL
stream_queue = queue.Queue()
stream_done_event = threading.Event()
stream_in_progress = False
chat_history_widget = None # Assign after Tkinter setup

# --- TTS ---
tts_engine = None
tts_queue = queue.Queue()
tts_thread = None
tts_sentence_buffer = ""
tts_enabled = None      # tk.BooleanVar
tts_rate = None         # tk.IntVar
tts_voice_id = None     # tk.StringVar
tts_busy = False
tts_busy_lock = threading.Lock()
tts_initialized_successfully = False

# --- Whisper/VAD ---
vad_model = None
vad_utils = None
vad_get_speech_ts = None
whisper_model = None
whisper_model_size = DEFAULT_WHISPER_MODEL_SIZE
whisper_initialized = False
vad_initialized = False
vad_thread = None
vad_stop_event = threading.Event()
whisper_queue = queue.Queue() # Queue for audio filenames to transcribe
whisper_processing_thread = None
whisper_language = None     # Set via UI, default None (auto)
voice_enabled = None        # tk.BooleanVar
recording_indicator_widget = None # Assign after Tkinter setup
auto_send_after_transcription = None
user_input_widget = None    # Assign after Tkinter setup

# --- Audio Handling ---
py_audio = None
audio_stream = None
is_recording_for_whisper = False # State flag for VAD-triggered recording
audio_frames_buffer = deque() # Holds raw audio data during VAD recording
vad_audio_buffer = deque(maxlen=int(RATE / VAD_CHUNK * 1.5)) # Rolling buffer for VAD analysis (~1.5 sec)
temp_audio_file_path = None # Path for the temporary WAV file

# --- Web Interface Communication (Phase 1) ---
web_input_queue = queue.Queue() # Messages from Web UI -> Main App Logic
web_output_queue = queue.Queue() # Stream chunks from Ollama -> Web UI (SSE)
flask_thread = None # Holder for the Flask thread


# ===================
# Flask Web Server (Phase 1 & 2)
# ===================
app = Flask(__name__)

# --- Routes ---
@app.route('/')
def index():
    """Serves the main HTML page."""
    # (Phase 3) Serve the HTML file
    return render_template('index.html')

@app.route('/send_message', methods=['POST'])
def handle_send_message():
    """Receives message from web UI and queues it."""
    # (Phase 2) Implementation
    data = request.get_json()
    user_text = data.get('message', '').strip()
    # Optional: Handle image data (more complex, start with text)
    if user_text:
        print(f"[Web] Received message via web: '{user_text}'")
        # Put the message onto the queue for the main thread to process
        web_input_queue.put({"type": "text_message", "content": user_text})
        return jsonify({"status": "success", "message": "Message queued"})
    else:
        return jsonify({"status": "error", "message": "Empty message received"}), 400

@app.route('/get_history')
def handle_get_history():
    """Returns the chat history (thread-safe)."""
    # (Phase 2) Implementation
    global messages, messages_lock
    history_copy = []
    try:
        with messages_lock: # Protect read access
            for msg in messages:
                msg_copy = msg.copy()
                # Ensure image data (bytes) isn't sent, replace or omit it
                if "images" in msg_copy:
                    # Represent image presence without sending bytes
                    if msg_copy.get("content"):
                         msg_copy["content"] += " [Image Attachment]"
                    else:
                         msg_copy["content"] = "[Image Attachment]"
                    del msg_copy["images"] # Don't send raw bytes over JSON
                history_copy.append(msg_copy)
    except Exception as e:
        print(f"[Web History] Error accessing messages: {e}")
        return jsonify({"error": "Failed to retrieve history"}), 500
    return jsonify({"history": history_copy})


@app.route('/stream')
def handle_stream():
    """Streams Ollama responses using Server-Sent Events (SSE)."""
    # (Phase 2) Implementation
    def event_stream():
        print("[Web SSE] Client connected to stream.")
        last_event_time = time.time()
        try:
            while True:
                # Wait for a new message from the Ollama worker via the queue
                try:
                    # Block for a short time, then send keepalive if nothing
                    data = web_output_queue.get(timeout=10) # Wait up to 10s

                    if data.get("type") == "chunk":
                        yield f"event: message\ndata: {json.dumps(data)}\n\n"
                    elif data.get("type") == "end":
                        yield f"event: end\ndata: {json.dumps(data)}\n\n"
                        # Keep listening for next response
                    elif data.get("type") == "error":
                        yield f"event: error\ndata: {json.dumps(data)}\n\n"

                    web_output_queue.task_done() # Mark item as processed
                    last_event_time = time.time()

                except queue.Empty:
                    # Send a keepalive comment if queue is empty after timeout
                    if time.time() - last_event_time > 15: # Send keepalive every ~15s of inactivity
                         # print("[Web SSE] Sending keepalive.")
                         yield ": keepalive\n\n"
                         last_event_time = time.time() # Reset timer after keepalive
                    else:
                         time.sleep(0.1) # Small sleep if queue was checked recently but empty

                except GeneratorExit:
                    print("[Web SSE] Client disconnected.")
                    break # Exit loop if client disconnects
                except Exception as e:
                    print(f"[Web SSE] Error in stream loop: {e}")
                    # Yield error to client if possible
                    try:
                         yield f"event: error\ndata: {json.dumps({'type': 'error', 'content': f'SSE internal error: {e}'})}\n\n"
                    except Exception as notify_e:
                         print(f"[Web SSE] Could not notify client of error: {notify_e}")
                    time.sleep(1) # Avoid rapid error loops

        finally:
            print("[Web SSE] Stopping event stream generator for a client.")
            # Clean up if necessary
    return Response(event_stream(), mimetype="text/event-stream")

# --- Flask Runner ---
def run_flask():
    """Function to run the Flask app in a thread."""
    print(f"[Web] Starting Flask server on http://{FLASK_HOST}:{FLASK_PORT}")
    print(f"[Web] Access the UI from another device on your network via: http://<YOUR_PC_IP>:{FLASK_PORT}")
    try:
        # Use 'allow_unsafe_werkzeug=True' only if developing and need reloader with threading issues
        # For production or safer development with threads, debug=False is recommended.
        app.run(host=FLASK_HOST, port=FLASK_PORT, threaded=True, debug=False)
    except Exception as e:
        print(f"[Web] Flask server failed to start: {e}")
        # Optionally, try to signal main thread about the failure if needed
        # web_output_queue.put({"type": "error", "content": "Flask server failed"})


# ===================
# TTS Setup & Control (No changes needed for basic web integration)
# ===================
def initialize_tts():
    """Initializes the TTS engine. Returns True on success."""
    global tts_engine, tts_initialized_successfully, tts_rate, tts_voice_id
    if tts_engine: return True # Already initialized

    try:
        print("[TTS] Initializing engine...")
        tts_engine = pyttsx3.init()
        if not tts_engine: raise Exception("pyttsx3.init() returned None")

        tts_engine.setProperty('rate', tts_rate.get())
        tts_engine.setProperty('volume', 0.9)
        if tts_voice_id.get():
            tts_engine.setProperty('voice', tts_voice_id.get())

        tts_initialized_successfully = True
        print("[TTS] Engine initialized successfully.")
        return True
    except Exception as e:
        print(f"[TTS] Error initializing engine: {e}")
        tts_engine = None
        tts_initialized_successfully = False
        return False

def get_available_voices():
    """Returns a list of available TTS voices (name, id)."""
    temp_engine = None
    try:
        temp_engine = pyttsx3.init()
        if not temp_engine: return []
        voices = temp_engine.getProperty('voices')
        voice_list = [(v.name[:30] + "..." if len(v.name) > 30 else v.name, v.id) for v in voices]
        temp_engine.stop()
        del temp_engine # Ensure release
        return voice_list
    except Exception as e:
        print(f"[TTS] Error getting voices: {e}")
        if temp_engine:
            try: temp_engine.stop()
            except: pass
            del temp_engine
        return []

def set_voice(event=None):
    """Sets the selected voice for the TTS engine."""
    global tts_engine, tts_voice_id, tts_initialized_successfully
    if not tts_initialized_successfully or not tts_engine: return
    try:
        # Ensure voice_selector exists and has a valid selection
        if 'voice_selector' in globals() and hasattr(voice_selector, 'current'):
            selected_idx = voice_selector.current()
            if selected_idx >= 0 and 'available_voices' in globals() and selected_idx < len(available_voices):
                voice_id = available_voices[selected_idx][1]
                tts_voice_id.set(voice_id)
                tts_engine.setProperty('voice', voice_id)
                print(f"[TTS] Voice set to: {available_voices[selected_idx][0]}")
            else:
                print(f"[TTS] Invalid voice selection index: {selected_idx}")
        else:
            print("[TTS] Voice selector not ready for setting voice.")
    except Exception as e:
        print(f"[TTS] Error setting voice: {e}")


def set_speech_rate(value=None): # Updated to accept value from scale command
    """Sets the speech rate for the TTS engine."""
    global tts_engine, tts_rate, tts_initialized_successfully
    if not tts_initialized_successfully or not tts_engine: return
    try:
        rate = int(float(value))
        tts_rate.set(rate)
        tts_engine.setProperty('rate', rate)
        # print(f"[TTS] Speech rate set to: {rate}") # Optional: Can be noisy
    except Exception as e:
        print(f"[TTS] Error setting speech rate: {e}")

def tts_worker():
    """Worker thread processing the TTS queue."""
    global tts_engine, tts_queue, tts_busy, tts_initialized_successfully, tts_rate, tts_voice_id
    print("[TTS Worker] Thread started.")
    while True:
        try:
            text_to_speak = tts_queue.get() # Blocks until item available
            if text_to_speak is None: # Sentinel for stopping
                print("[TTS Worker] Received stop signal.")
                break

            if tts_engine and tts_enabled.get() and tts_initialized_successfully:
                with tts_busy_lock:
                    tts_busy = True

                # Refresh voice and rate settings before each say() call
                try:
                    current_voice = tts_voice_id.get()
                    if current_voice:
                        tts_engine.setProperty('voice', current_voice)
                    tts_engine.setProperty('rate', tts_rate.get())
                except Exception as prop_err:
                    print(f"[TTS Worker] Error setting properties: {prop_err}")
                    # Decide if we should continue or skip this utterance
                    with tts_busy_lock: tts_busy = False
                    tts_queue.task_done()
                    continue # Skip this utterance

                # print(f"[TTS Worker] Speaking chunk ({len(text_to_speak)} chars)...")
                # start_time = time.time()
                tts_engine.say(text_to_speak)
                tts_engine.runAndWait() # This blocks the worker, not the main thread
                # print(f"[TTS Worker] Finished in {time.time() - start_time:.2f}s")

                with tts_busy_lock:
                    tts_busy = False
            else:
                # print("[TTS Worker] Engine disabled or uninitialized, discarding text.")
                pass # Just discard

            tts_queue.task_done() # Mark task as complete
        except RuntimeError as rt_err:
             if "run loop already started" in str(rt_err):
                  print("[TTS Worker] Warning: run loop already started, skipping runAndWait().")
                  # This might happen if runAndWait was interrupted abruptly before.
                  # Reset busy state carefully.
                  with tts_busy_lock: tts_busy = False
             else:
                  print(f"[TTS Worker] Runtime Error: {rt_err}")
                  traceback.print_exc()
                  with tts_busy_lock: tts_busy = False
        except Exception as e:
            print(f"[TTS Worker] Unexpected Error: {e}")
            traceback.print_exc()
            with tts_busy_lock: # Ensure busy flag is reset on error
                tts_busy = False
            # Consider adding a small sleep to prevent rapid error looping
            time.sleep(0.5)

    print("[TTS Worker] Thread finished.")

def start_tts_thread():
    """Starts the TTS worker thread if needed."""
    global tts_thread, tts_initialized_successfully
    if tts_thread is None or not tts_thread.is_alive():
        if not tts_initialized_successfully:
            tts_initialized_successfully = initialize_tts() # Attempt init again

        if tts_initialized_successfully:
            tts_thread = threading.Thread(target=tts_worker, daemon=True)
            tts_thread.start()
            print("[TTS] Worker thread started.")
        else:
            print("[TTS] Engine init failed. Cannot start TTS thread.")
            # Update UI to reflect failure (handled in toggle_tts)

def stop_tts_thread():
    """Signals the TTS worker thread to stop and cleans up."""
    global tts_thread, tts_engine, tts_queue, tts_busy, tts_busy_lock
    print("[TTS] Stopping worker thread...")
    if tts_engine:
        try:
            # Attempt to stop speaking immediately
            tts_engine.stop()
            print("[TTS] Engine stop requested.")
            # Try to cleanly interrupt runAndWait if it's blocking
            # Unfortunately, pyttsx3 doesn't offer a reliable non-blocking API or interruption mechanism easily.
        except Exception as e:
             print(f"[TTS] Error stopping engine: {e}")

    # Clear the queue
    while not tts_queue.empty():
        try: tts_queue.get_nowait()
        except queue.Empty: break
    print("[TTS] Queue cleared.")

    # Signal thread to stop and wait
    if tts_thread and tts_thread.is_alive():
        tts_queue.put(None) # Send sentinel
        print("[TTS] Waiting for worker thread to join...")
        tts_thread.join(timeout=3) # Increased timeout
        if tts_thread.is_alive():
            print("[TTS] Warning: Worker thread did not terminate gracefully after 3s.")
        else:
             print("[TTS] Worker thread joined.")
    tts_thread = None
    # Reset busy state just in case
    with tts_busy_lock: tts_busy = False


def toggle_tts():
    """Callback for the TTS enable/disable checkbox."""
    global tts_enabled, tts_sentence_buffer, tts_initialized_successfully, voice_selector, rate_scale
    if tts_enabled.get():
        if not tts_initialized_successfully:
            # Try initializing again when toggled on
            tts_initialized_successfully = initialize_tts()

        if tts_initialized_successfully:
            print("[TTS] Enabled by user.")
            start_tts_thread() # Ensure thread is running
            # Enable controls if they were disabled
            if 'voice_selector' in globals() and voice_selector: voice_selector.config(state="readonly")
            if 'rate_scale' in globals() and rate_scale: rate_scale.config(state="normal")
        else:
            print("[TTS] Enable failed - Engine initialization problem.")
            tts_enabled.set(False) # Uncheck the box
            add_message_to_ui("error", "TTS Engine failed to initialize. Cannot enable TTS.")
            # Ensure controls are disabled
            if 'voice_selector' in globals() and voice_selector: voice_selector.config(state="disabled")
            if 'rate_scale' in globals() and rate_scale: rate_scale.config(state="disabled")
    else:
        print("[TTS] Disabled by user.")
        # Stop speaking and clear buffer/queue immediately
        if tts_engine:
            try:
                 tts_engine.stop()
                 # Clear any pending commands in the engine's internal queue (if possible)
                 # This is often needed if runAndWait is blocking
            except Exception as e: print(f"[TTS] Error stopping on toggle off: {e}")
        tts_sentence_buffer = ""
        while not tts_queue.empty():
            try: tts_queue.get_nowait()
            except queue.Empty: break
        # Optional: Stop the worker thread if you want to save resources when disabled
        # stop_tts_thread() # Consider if needed, starting/stopping threads frequently can add overhead


def queue_tts_text(new_text):
    """Accumulates text for TTS, intended to be flushed later."""
    global tts_sentence_buffer
    if tts_enabled.get() and tts_initialized_successfully:
        tts_sentence_buffer += new_text

def try_flush_tts_buffer():
    """Sends complete sentences from the buffer to the TTS queue if TTS is idle."""
    global tts_sentence_buffer, tts_busy, tts_queue, tts_busy_lock
    if not tts_enabled.get() or not tts_initialized_successfully or not tts_engine:
        tts_sentence_buffer = "" # Clear buffer if TTS is off
        return

    # Use the lock to check if TTS is currently processing a previous chunk
    with tts_busy_lock:
        if tts_busy:
             # print("[TTS Flush] Engine busy, delaying flush.")
             return # Don't queue new text if already speaking

    # Check if there's anything substantial to process
    if not tts_sentence_buffer or tts_sentence_buffer.isspace():
         return

    # Split on sentence endings (. ! ? \n) followed by space or end of string
    # Keep delimiters attached to the preceding sentence part.
    # Handle potential multiple delimiters like "..." or "?!" correctly.
    sentences = re.split(r'([.!?\n]+(?:\s+|$))', tts_sentence_buffer)

    # Filter out empty strings that can result from splitting
    sentences = [s for s in sentences if s]

    chunk_to_speak = ""
    processed_len = 0
    temp_buffer = []

    # Process pairs of (sentence_part, delimiter_part) if available
    # Or just single parts if the split resulted in odd number of elements
    i = 0
    while i < len(sentences):
        sentence_part = sentences[i]
        delimiter_part = ""
        # Check if the next element looks like a delimiter part
        if (i + 1) < len(sentences) and re.match(r'^([.!?\n]+(?:\s+|$))', sentences[i+1]):
             delimiter_part = sentences[i+1]
             # Found a complete sentence segment
             temp_buffer.append(sentence_part + delimiter_part)
             processed_len += len(sentence_part + delimiter_part)
             i += 2 # Move past sentence and delimiter
        else:
             # This part doesn't have a delimiter following it *yet*.
             # It's the last part or the next part isn't a delimiter.
             break # Stop processing here, keep the rest in buffer

    # Determine what's left in the buffer
    if processed_len > 0:
         tts_sentence_buffer = tts_sentence_buffer[processed_len:]
         chunk_to_speak = "".join(temp_buffer).strip()
    else:
         # No complete sentences found, leave buffer as is
         chunk_to_speak = ""


    # Queue the chunk if we found complete sentences
    if chunk_to_speak:
        # print(f"[TTS Flush] Queuing chunk: '{chunk_to_speak[:50]}...' ({len(chunk_to_speak)} chars)")
        tts_queue.put(chunk_to_speak)
        # Worker thread will set tts_busy when it starts speaking

def periodic_tts_check():
    """Periodically checks if TTS buffer can be flushed."""
    try:
        try_flush_tts_buffer()
    except Exception as e:
        print(f"[TTS Check] Error during periodic flush: {e}")
    finally:
        # Reschedule even if flushing fails, to keep checking
        # Check if window still exists before scheduling
        if window and window.winfo_exists():
             window.after(200, periodic_tts_check) # Check every 200ms


# ========================
# Whisper & VAD Setup (No changes needed for basic web integration)
# ========================
def initialize_whisper():
    """Initializes the Whisper model. Returns True on success."""
    global whisper_model, whisper_initialized, whisper_model_size
    if whisper_initialized: return True

    update_vad_status(f"Loading Whisper ({whisper_model_size})...", "blue")
    try:
        print(f"[Whisper] Initializing model ({whisper_model_size})...")

        # Suppress symlink warnings if they cause issues (optional)
        os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
        # os.environ["HF_HUB_DISABLE_SYMLINKS"] = "1" # Use if needed

        # --- Handle potential FasterWhisper issues ---
        # Ensure NUMBA_CACHE_DIR is writable if using Numba (common with Whisper)
        try:
            numba_cache_dir = os.path.join(tempfile.gettempdir(), 'numba_cache')
            os.makedirs(numba_cache_dir, exist_ok=True)
            os.environ['NUMBA_CACHE_DIR'] = numba_cache_dir
            print(f"[Whisper] Set NUMBA_CACHE_DIR to: {numba_cache_dir}")
        except Exception as cache_err:
            print(f"[Whisper] Warning: Could not set NUMBA_CACHE_DIR: {cache_err}")
        # --- End FasterWhisper handling ---

        if whisper_model_size.startswith("turbo"):
            whisper_turbo_model_name = whisper_model_size.split("-", 1)[1] # e.g., "large-v3" or just "large"
            device = "cuda" if torch.cuda.is_available() else "cpu"
            # Recommended compute types based on device and capability
            if device == "cuda":
                # Check for float16 support (Ampere GPUs and later)
                if torch.cuda.get_device_capability(0)[0] >= 7:
                     compute_type = "float16"
                else:
                     compute_type = "float32" # Fallback for older GPUs
            else:
                 # Use int8 for CPU for better performance
                 compute_type = "int8"
            
            print(f"[Whisper] Using FasterWhisper: model={whisper_turbo_model_name}, device={device}, compute_type={compute_type}")
            
            # Add local_files_only=False initially, maybe set to True after first download
            whisper_model = faster_whisper.WhisperModel(
                 whisper_turbo_model_name, 
                 device=device, 
                 compute_type=compute_type,
                 # download_root=os.path.join(tempfile.gettempdir(), "faster_whisper_models") # Optional: Specify download location
                 # local_files_only=False, # Set to True after first successful download
            )
            print(f"[Whisper] FasterWhisper model loaded.")

        else: # Standard OpenAI Whisper
            print(f"[Whisper] Using OpenAI Whisper: model={whisper_model_size}")
            whisper_model = whisper.load_model(whisper_model_size)
            print("[Whisper] OpenAI Whisper model loaded.")

        whisper_initialized = True
        # Update status only if VAD is also ready or not needed
        if vad_initialized or not voice_enabled.get():
             update_vad_status(f"Whisper ({whisper_model_size}) ready.", "green")
        else:
             update_vad_status(f"Whisper OK, waiting VAD...", "blue")

        print("[Whisper] Model initialization successful.")
        return True

    except ImportError as ie:
         print(f"[Whisper] Import Error: {ie}. Is FasterWhisper installed (`pip install faster-whisper`)?")
         whisper_initialized = False
         whisper_model = None
         update_vad_status("Whisper Import Error!", "red")
         add_message_to_ui("error", f"Whisper Import Error: {ie}. Ensure necessary libraries are installed.")
         return False
    except Exception as e:
        print(f"[Whisper] Error initializing model: {e}")
        traceback.print_exc()
        whisper_initialized = False
        whisper_model = None
        update_vad_status("Whisper init failed!", "red")
        add_message_to_ui("error", f"Failed to load Whisper {whisper_model_size} model: {e}")
        return False

def initialize_vad():
    """Initializes the Silero VAD model. Returns True on success."""
    global vad_model, vad_utils, vad_get_speech_ts, vad_initialized
    if vad_initialized: return True

    update_vad_status("Loading VAD model...", "blue")
    try:
        print("[VAD] Initializing Silero VAD model...")
        # Workaround for potential torch hub path issues on some systems
        try:
             torch_hub_dir = os.path.join(tempfile.gettempdir(), "torch_hub_silero")
             os.makedirs(torch_hub_dir, exist_ok=True)
             torch.hub.set_dir(torch_hub_dir)
             print(f"[VAD] Set torch hub directory for Silero VAD: {torch_hub_dir}")
        except Exception as hub_dir_err:
             print(f"[VAD] Warning: Could not set custom torch hub directory: {hub_dir_err}")

        # Try loading the model
        vad_model, vad_utils = torch.hub.load(repo_or_dir='snakers4/silero-vad',
                                              model='silero_vad',
                                              force_reload=False, # Use cached if available
                                              onnx=False, # Set to True if ONNX version is preferred/installed
                                              trust_repo=True) # Required for custom models
        
        (vad_get_speech_ts, _, read_audio, VADIterator, _) = vad_utils # Get necessary functions
        
        vad_initialized = True
        print("[VAD] Model initialized successfully.")
        # Update status only if Whisper is also ready or not needed
        if whisper_initialized or not voice_enabled.get():
             update_vad_status("VAD Ready.", "green")
        else:
             update_vad_status("VAD OK, waiting Whisper...", "blue")
        return True
    except Exception as e:
        print(f"[VAD] Error initializing model: {e}")
        traceback.print_exc()
        vad_initialized = False
        vad_model = None
        update_vad_status("VAD init failed!", "red")
        add_message_to_ui("error", f"Failed to load Silero VAD model: {e}")
        return False

def initialize_audio_system():
    """Initializes PyAudio. Returns True on success."""
    global py_audio
    if py_audio: return True
    try:
        print("[Audio] Initializing PyAudio...")
        py_audio = pyaudio.PyAudio()
        print("[Audio] PyAudio initialized.")
        # Optional: Print available devices for debugging
        # print("[Audio] Available Input Devices:")
        # for i in range(py_audio.get_device_count()):
        #     dev_info = py_audio.get_device_info_by_index(i)
        #     if dev_info.get('maxInputChannels') > 0:
        #          print(f"  {i}: {dev_info.get('name')} (Channels: {dev_info.get('maxInputChannels')})")
        return True
    except Exception as e:
        print(f"[Audio] Error initializing PyAudio: {e}")
        traceback.print_exc()
        add_message_to_ui("error", f"Failed to initialize audio system: {e}. Check microphone permissions and drivers.")
        py_audio = None
        return False

def vad_worker():
    """Worker thread for continuous VAD and triggering recording."""
    global py_audio, audio_stream, vad_audio_buffer, audio_frames_buffer, is_recording_for_whisper
    global vad_model, vad_get_speech_ts, vad_stop_event, temp_audio_file_path, whisper_queue
    global tts_busy, tts_busy_lock, window # Access main window for scheduling UI updates

    print("[VAD Worker] Thread started.")

    stream = None
    try:
        # Check if audio system is initialized
        if not py_audio:
            print("[VAD Worker] PyAudio not initialized. Cannot start.")
            update_vad_status("Audio System Error!", "red")
            return

        # Open audio stream
        try:
             stream = py_audio.open(format=FORMAT,
                                    channels=CHANNELS,
                                    rate=RATE,
                                    input=True,
                                    frames_per_buffer=VAD_CHUNK) # Use smaller chunk for VAD
             print("[VAD Worker] Audio stream opened successfully.")
        except OSError as ose:
             print(f"[VAD Worker] OSError opening audio stream: {ose}")
             # Provide more specific feedback if possible (e.g., device unavailable)
             if "Invalid input device" in str(ose) or "No Default Input Device Available" in str(ose):
                  msg = "Audio Error: No valid input device found or device unavailable."
             else:
                  msg = f"Audio Error: {ose}"
             update_vad_status("Audio Input Error!", "red")
             add_message_to_ui("error", msg)
             return # Cannot continue without a stream
        except Exception as e:
             print(f"[VAD Worker] Unexpected error opening audio stream: {e}")
             update_vad_status("Audio Init Error!", "red")
             add_message_to_ui("error", f"Failed to open audio stream: {e}")
             return

        update_vad_status("Listening...", "gray")

        num_silence_frames = 0
        silence_frame_limit = int(SILENCE_THRESHOLD_SECONDS * RATE / VAD_CHUNK)
        # speech_detected_recently = False # Not strictly needed with current logic
        frames_since_last_speech = 0
        min_speech_frames = int(MIN_SPEECH_DURATION_SECONDS * RATE / VAD_CHUNK)
        pre_speech_buffer_frames = int(PRE_SPEECH_BUFFER_SECONDS * RATE / VAD_CHUNK)

        temp_pre_speech_buffer = deque(maxlen=pre_speech_buffer_frames)
        was_tts_busy = False  # Track previous TTS state

        while not vad_stop_event.is_set():
            try:
                # Read audio data
                data = stream.read(VAD_CHUNK, exception_on_overflow=False)

                # Check TTS state (thread-safe)
                with tts_busy_lock:
                    current_tts_busy = tts_busy

                # --- Handle TTS Interference ---
                if current_tts_busy:
                    if not was_tts_busy: # Only update status on transition
                        print("[VAD Worker] TTS active, pausing VAD.")
                        update_vad_status("VAD Paused (TTS Active)", "blue")
                        was_tts_busy = True
                    # If we were recording, discard it to avoid capturing TTS output
                    if is_recording_for_whisper:
                        print("[VAD Worker] Discarding active recording due to TTS.")
                        is_recording_for_whisper = False
                        audio_frames_buffer.clear()
                        temp_pre_speech_buffer.clear() # Also clear pre-buffer
                    # Skip VAD processing for this chunk
                    time.sleep(0.05) # Small sleep to yield CPU
                    continue
                elif was_tts_busy: # TTS just finished
                     print("[VAD Worker] TTS finished, resuming VAD.")
                     update_vad_status("Listening...", "gray")
                     was_tts_busy = False
                     # Reset silence counters after TTS finishes
                     frames_since_last_speech = 0
                     # Give it a very brief moment for audio pipe to clear
                     time.sleep(0.1)
                     continue # Process next chunk cleanly

                # --- Normal VAD Processing ---
                audio_chunk_np = np.frombuffer(data, dtype=np.int16)
                vad_audio_buffer.append(audio_chunk_np)
                temp_pre_speech_buffer.append(data) # Store raw bytes for pre-buffer

                # Only run VAD if buffer has enough data (e.g., >= 30ms chunk for Silero)
                # Silero VAD expects chunks of specific sizes (e.g., 256, 512, 768, 1024, 1536 samples @ 16kHz)
                # We use VAD_CHUNK=512, which is valid.
                audio_float32 = audio_chunk_np.astype(np.float32) / 32768.0 # Normalize
                audio_tensor = torch.from_numpy(audio_float32)

                # Get speech probability (more robust than just timestamps)
                speech_prob = vad_model(audio_tensor.unsqueeze(0), RATE).item()
                is_speech = speech_prob > 0.4 # Adjust threshold if needed

                if is_speech:
                    # print(f"Speech detected (Prob: {speech_prob:.2f})") # Debug
                    frames_since_last_speech = 0
                    if not is_recording_for_whisper:
                        print("[VAD Worker] Speech started, beginning recording.")
                        is_recording_for_whisper = True
                        audio_frames_buffer.clear()
                        # Add pre-speech buffer content
                        for frame_data in temp_pre_speech_buffer:
                            audio_frames_buffer.append(frame_data)
                        # Don't add current chunk here yet, add below
                        update_vad_status("Recording...", "red")

                    # Append current data *if* recording (or just started)
                    if is_recording_for_whisper:
                         audio_frames_buffer.append(data)

                else: # No speech detected in this chunk
                    frames_since_last_speech += 1
                    # If recording, still append data until silence threshold is met
                    if is_recording_for_whisper:
                         audio_frames_buffer.append(data)

                         # Check if silence duration exceeds threshold
                         if frames_since_last_speech > silence_frame_limit:
                            print(f"[VAD Worker] Silence detected ({SILENCE_THRESHOLD_SECONDS}s), stopping recording.")
                            is_recording_for_whisper = False # Stop recording *before* processing

                            # --- Process the recorded audio ---
                            total_frames = len(audio_frames_buffer)
                            recording_duration = total_frames * VAD_CHUNK / RATE
                            effective_speech_frames = total_frames - frames_since_last_speech # Approximate speech part

                            print(f"[VAD Worker] Total recording duration: {recording_duration:.2f}s")

                            # Check if meets minimum *speech* duration (ignoring pre/post buffer/silence)
                            if effective_speech_frames < min_speech_frames:
                                print(f"[VAD Worker] Effective speech too short ({effective_speech_frames * VAD_CHUNK / RATE:.2f}s < {MIN_SPEECH_DURATION_SECONDS:.2f}s), discarding.")
                                audio_frames_buffer.clear() # Discard data
                                update_vad_status("Too short, discarded", "orange")
                                # Schedule return to listening state after a brief pause
                                if window and window.winfo_exists():
                                     window.after(800, lambda: update_vad_status("Listening...", "gray"))

                            else: # Sufficient speech duration, save and queue
                                try:
                                     temp_audio_file = tempfile.NamedTemporaryFile(prefix="vad_rec_", suffix=".wav", delete=False)
                                     temp_audio_file_path = temp_audio_file.name
                                     temp_audio_file.close() # Close it so wave can open it

                                     wf = wave.open(temp_audio_file_path, 'wb')
                                     wf.setnchannels(CHANNELS)
                                     wf.setsampwidth(py_audio.get_sample_size(FORMAT))
                                     wf.setframerate(RATE)
                                     wf.writeframes(b''.join(audio_frames_buffer))
                                     wf.close()
                                     print(f"[VAD Worker] Audio saved to {temp_audio_file_path} ({recording_duration:.2f}s)")

                                     # Queue for transcription
                                     whisper_queue.put(temp_audio_file_path)
                                     update_vad_status("Processing...", "blue") # UI update continues in whisper worker

                                except Exception as save_err:
                                     print(f"[VAD Worker] Error saving audio file: {save_err}")
                                     update_vad_status("File Save Error", "red")
                                     # Schedule return to listening state
                                     if window and window.winfo_exists():
                                          window.after(1000, lambda: update_vad_status("Listening...", "gray"))

                            # Clear buffer for next recording regardless of outcome
                            audio_frames_buffer.clear()
                            # No need to clear pre-speech buffer here, it's rolling

                    # If not recording, update status to listening (unless transitioning from TTS)
                    if not is_recording_for_whisper and not was_tts_busy:
                         update_vad_status("Listening...", "gray")


            except IOError as e:
                if e.errno == pyaudio.paInputOverflowed:
                    print("[VAD Worker] Warning: Input overflowed. Skipping frame.")
                    # Consider adding a small delay or buffer clearing strategy if this happens frequently
                else:
                    print(f"[VAD Worker] Stream read error: {e}")
                    # Attempt to recover or signal critical error
                    update_vad_status("Audio Stream Error!", "red")
                    vad_stop_event.set() # Stop the worker on critical stream errors
                    break # Exit the loop
            except Exception as e:
                print(f"[VAD Worker] Unexpected error in loop: {e}")
                traceback.print_exc()
                time.sleep(0.1) # Avoid busy-looping on unexpected errors

    except Exception as e:
        # Errors during initialization or stream opening
        print(f"[VAD Worker] Initialization or stream opening error: {e}")
        # Status update likely already happened in the failing part
    finally:
        # Cleanup
        print("[VAD Worker] Cleaning up...")
        if stream:
            if stream.is_active():
                 try: stream.stop_stream()
                 except Exception as stop_e: print(f"[VAD Worker] Error stopping stream: {stop_e}")
            try: stream.close()
            except Exception as close_e: print(f"[VAD Worker] Error closing stream: {close_e}")
            print("[VAD Worker] Audio stream closed.")

        # Reset state variables
        is_recording_for_whisper = False
        audio_frames_buffer.clear()
        vad_audio_buffer.clear()
        temp_pre_speech_buffer.clear()

        # Update final status based on why we exited
        if vad_stop_event.is_set():
            # If stopped normally by user toggle or critical error handled above
            if "Audio Stream Error!" not in (recording_indicator_widget.cget("text") if recording_indicator_widget else ""):
                 # Avoid overwriting specific error messages
                 update_vad_status("Voice Disabled", "grey")
        else:
            # Exited due to an unhandled error within the loop
            update_vad_status("VAD Stopped (Error)", "red")

    print("[VAD Worker] Thread finished.")

def process_audio_worker():
    """Worker thread to transcribe audio files from the whisper_queue."""
    global whisper_model, whisper_initialized, whisper_queue, whisper_language, window
    global whisper_model_size # Access model size for logic
    print("[Whisper Worker] Thread started.")
    while True:
        try:
            audio_file_path = whisper_queue.get() # Blocks
            if audio_file_path is None: # Sentinel
                print("[Whisper Worker] Received stop signal.")
                break

            if not whisper_initialized or not voice_enabled.get():
                print("[Whisper Worker] Skipping transcription (Whisper/Voice disabled or not ready).")
                if audio_file_path and os.path.exists(audio_file_path):
                     try: os.unlink(audio_file_path); print(f"[Whisper Worker] Deleted unused audio file: {audio_file_path}")
                     except Exception as del_err: print(f"[Whisper Worker] Error deleting unused file: {del_err}")
                whisper_queue.task_done()
                # Ensure VAD status returns to listening if appropriate
                if voice_enabled.get() and vad_initialized and not is_recording_for_whisper:
                     update_vad_status("Listening...", "gray")
                continue

            print(f"[Whisper Worker] Processing audio file: {audio_file_path}")
            update_vad_status("Transcribing...", "orange")

            start_time = time.time()
            transcribed_text = ""
            detected_language = "N/A"
            try:
                 # Use selected language, None means auto-detect
                 lang_to_use = whisper_language if whisper_language else None
                 print(f"[Whisper Worker] Using language: {'Auto' if lang_to_use is None else lang_to_use}")

                 # --- Choose transcription method based on model type ---
                 if isinstance(whisper_model, faster_whisper.WhisperModel):
                     # FasterWhisper (turbo models)
                     # Consider adding beam_size, temperature etc. as parameters if needed
                     segments, info = whisper_model.transcribe(audio_file_path,
                                                               language=lang_to_use,
                                                               # beam_size=5, # Optional
                                                               # vad_filter=True, # Optional: Use FasterWhisper's VAD
                                                               # vad_parameters=dict(threshold=0.5), # Optional VAD params
                                                               )
                     transcribed_text = " ".join([segment.text for segment in segments]).strip()
                     detected_language = info.language if hasattr(info, 'language') else 'N/A'
                     print(f"[Whisper Worker] FasterWhisper detected language: {detected_language} (Prob: {info.language_probability if hasattr(info, 'language_probability') else 'N/A'})")
                 
                 elif isinstance(whisper_model, whisper.Whisper):
                    # OpenAI Whisper
                    result = whisper_model.transcribe(audio_file_path, language=lang_to_use)
                    transcribed_text = result["text"].strip()
                    detected_language = result.get("language", "N/A")
                    print(f"[Whisper Worker] OpenAI Whisper detected language: {detected_language}")
                    
                 else:
                     print("[Whisper Worker] Error: Unknown Whisper model type.")
                     raise TypeError("Unsupported Whisper model object")


                 duration = time.time() - start_time
                 print(f"[Whisper Worker] Transcription complete in {duration:.2f}s: '{transcribed_text}'")

                 # Schedule UI update on main thread
                 if window and window.winfo_exists():
                     if transcribed_text:
                         window.after(0, update_input_with_transcription, transcribed_text)
                         update_vad_status("Transcription Ready", "green")
                     else:
                         update_vad_status("No speech detected", "orange")
                         # Schedule return to listening state after a brief pause
                         window.after(800, lambda: update_vad_status("Listening...", "gray"))

            except Exception as e:
                print(f"[Whisper Worker] Error during transcription: {e}")
                traceback.print_exc()
                update_vad_status("Transcription Error", "red")
                 # Schedule return to listening state after error display
                if window and window.winfo_exists():
                     window.after(1500, lambda: update_vad_status("Listening...", "gray"))
            finally:
                # Clean up the temporary audio file
                if audio_file_path and os.path.exists(audio_file_path):
                    try: os.unlink(audio_file_path)
                    except Exception as e: print(f"[Whisper Worker] Error deleting temp file {audio_file_path}: {e}")

            whisper_queue.task_done()

        except Exception as e:
            # Catch errors in the loop itself (e.g., queue errors)
            print(f"[Whisper Worker] Error in main loop: {e}")
            traceback.print_exc()
            # Ensure task_done is called even if outer loop breaks somehow
            if 'audio_file_path' in locals() and audio_file_path is not None:
                 whisper_queue.task_done()


    print("[Whisper Worker] Thread finished.")

def update_input_with_transcription(text):
    """Updates the user input text box with the transcribed text (runs in main thread)."""
    global user_input_widget, auto_send_after_transcription, window
    if not user_input_widget: return

    try:
        current_text = user_input_widget.get("1.0", tk.END).strip()
        if current_text:
            user_input_widget.insert(tk.END, " " + text)
        else:
            user_input_widget.insert("1.0", text)

        # Scroll to the end of the input box
        user_input_widget.see(tk.END)
        user_input_widget.update_idletasks() # Ensure UI updates

        # Automatically send message if option is enabled
        if auto_send_after_transcription and auto_send_after_transcription.get():
            # Small delay to let UI update completely visually
            if window and window.winfo_exists():
                 window.after(150, send_message) # Increased delay slightly
        else:
             # If not auto-sending, return VAD status to Listening after showing result
             if window and window.winfo_exists():
                 window.after(800, lambda: update_vad_status("Listening...", "gray"))

    except tk.TclError as e:
         print(f"[UI Update] Error updating input widget (maybe closed?): {e}")
    except Exception as e:
         print(f"[UI Update] Unexpected error updating input: {e}")
         traceback.print_exc()


def toggle_voice_recognition():
    """Enables/disables VAD and Whisper (runs in main thread)."""
    global voice_enabled, whisper_initialized, vad_initialized, vad_thread, vad_stop_event
    global py_audio, whisper_processing_thread

    # Update language setting just in case it changed via UI but wasn't applied
    set_whisper_language()
    # Update model size setting similarly
    set_whisper_model_size()

    if voice_enabled.get():
        print("[Voice] Enabling voice recognition...")
        update_vad_status("Initializing...", "blue")
        all_initialized = True

        # Initialize components sequentially
        if not py_audio:
            print("[Voice] Initializing Audio System...")
            if not initialize_audio_system(): all_initialized = False
            else: print("[Voice] Audio System OK.")

        # Only proceed if audio system is OK
        if all_initialized and not whisper_initialized:
            print("[Voice] Initializing Whisper...")
            if not initialize_whisper(): all_initialized = False
            else: print("[Voice] Whisper OK.") # Status updated inside init

        # Only proceed if Whisper is OK (or wasn't needed)
        if all_initialized and not vad_initialized:
            print("[Voice] Initializing VAD...")
            if not initialize_vad(): all_initialized = False
            else: print("[Voice] VAD OK.") # Status updated inside init

        # --- Start threads if all components initialized ---
        if all_initialized:
            # Start Whisper processing thread if not running
            if whisper_processing_thread is None or not whisper_processing_thread.is_alive():
                print("[Voice] Starting Whisper processing thread...")
                whisper_processing_thread = threading.Thread(target=process_audio_worker, daemon=True)
                whisper_processing_thread.start()
            else:
                 print("[Voice] Whisper processing thread already running.")

            # Start VAD worker thread if not running
            if vad_thread is None or not vad_thread.is_alive():
                print("[Voice] Starting VAD worker thread...")
                vad_stop_event.clear() # Reset stop event
                vad_thread = threading.Thread(target=vad_worker, daemon=True)
                vad_thread.start()
            else:
                 print("[Voice] VAD worker thread already running.")

            # Final status update if VAD started successfully (VAD worker sets "Listening...")
            # We can set a temporary "Enabled" status here.
            if vad_thread and vad_thread.is_alive():
                 update_vad_status("Voice Enabled", "green")
                 print("[Voice] Voice recognition enabled successfully.")
            else:
                 # This case shouldn't happen if init was successful, but handle defensively
                 print("[Voice] Enabling failed: VAD thread did not start.")
                 update_vad_status("VAD Start Failed", "red")
                 voice_enabled.set(False) # Uncheck the box

        else: # Initialization failed somewhere
            print("[Voice] Enabling failed due to initialization errors.")
            voice_enabled.set(False) # Uncheck the box
            # Status should already be showing the specific error (Whisper/VAD/Audio)
            if not (whisper_initialized or vad_initialized or py_audio):
                 update_vad_status("Init Failed", "red") # Generic fallback

    else: # Disabling voice recognition
        print("[Voice] Disabling voice recognition...")
        update_vad_status("Disabling...", "grey")
        if vad_thread and vad_thread.is_alive():
            print("[Voice] Signaling VAD worker to stop...")
            vad_stop_event.set() # Signal VAD worker to stop
            # VAD worker handles stream closing and sets final status ("Voice Disabled")
            # We don't join here to keep UI responsive, worker cleans up in background
        else:
             print("[Voice] VAD worker already stopped.")
             # Manually set status if VAD wasn't running
             update_vad_status("Voice Disabled", "grey")

        # Note: We keep the whisper processing thread alive, it waits on the queue.
        # We also keep PyAudio initialized unless explicitly closed on app exit.
        print("[Voice] Voice recognition disabled.")

def extract_pdf_content(pdf_path):
    """Extracts text content from a PDF file."""
    try:
        doc = fitz.open(pdf_path)
        text_content = ""
        metadata = doc.metadata or {} # Ensure metadata is a dict

        # Add basic metadata if available
        title = metadata.get('title', 'N/A')
        author = metadata.get('author', 'N/A')
        if title != 'N/A' or author != 'N/A':
             text_content += f"--- PDF Info ---\nTitle: {title}\nAuthor: {author}\n----------------\n\n"

        # Extract text from each page
        num_pages = doc.page_count
        max_chars_per_pdf = 20000 # Limit total chars extracted
        current_chars = len(text_content)
        truncated = False

        for page_num in range(num_pages):
            page = doc.load_page(page_num)
            page_text = page.get_text("text", sort=True).strip() # Get text, sorted for reading order
            
            if not page_text: # Skip empty pages
                 continue

            page_header = f"--- Page {page_num+1} of {num_pages} ---\n"
            page_text_len = len(page_header) + len(page_text) + 2 # +2 for newlines

            if current_chars + page_text_len > max_chars_per_pdf:
                 # Calculate remaining chars and truncate page text
                 remaining_chars = max_chars_per_pdf - current_chars - len(page_header) - 20 # -20 for truncation msg
                 if remaining_chars > 0:
                     text_content += page_header + page_text[:remaining_chars] + "... [Page Truncated]\n\n"
                 truncated = True
                 break # Stop processing further pages
            else:
                 text_content += page_header + page_text + "\n\n"
                 current_chars += page_text_len

        doc.close()

        if truncated:
             text_content += "\n[PDF content truncated due to length limit. Only first ~20k characters extracted.]"

        return text_content.strip()
    except Exception as e:
        print(f"[PDF Extract] Error extracting PDF content from {pdf_path}: {e}")
        traceback.print_exc()
        return f"Error extracting content from PDF '{os.path.basename(pdf_path)}': {str(e)}"


def select_file():
    """Opens dialog to select any supported file type (runs in main thread)."""
    global selected_image_path, image_sent_in_history, window, user_input_widget
    try:
        file_path = filedialog.askopenfilename(
            title="Select File Attachment",
            filetypes=[
                ("Supported Files", "*.png;*.jpg;*.jpeg;*.gif;*.bmp;*.pdf;*.txt;*.md;*.py;*.js;*.html;*.css;*.json;*.log"),
                ("Image files", "*.png;*.jpg;*.jpeg;*.gif;*.bmp"),
                ("PDF files", "*.pdf"),
                ("Text files", "*.txt;*.md;*.py;*.js;*.html;*.css;*.json;*.log"),
                ("All files", "*.*")
            ]
        )
    except Exception as fd_err:
        print(f"[File Dialog] Error opening file dialog: {fd_err}")
        add_message_to_ui("error", "Could not open file selection dialog.")
        return

    if file_path:
        process_selected_file(file_path)

def process_selected_file(file_path):
    """Processes a file selected via dialog or dropped (runs in main thread)."""
    global selected_image_path, image_sent_in_history, window, user_input_widget
    if not file_path or not os.path.isfile(file_path):
        print(f"[File Process] Invalid file path provided: {file_path}")
        return

    file_name = os.path.basename(file_path)
    file_ext = os.path.splitext(file_path)[1].lower()
    print(f"[File Process] Processing file: {file_name} (ext: {file_ext})")

    # --- Handle Image Files ---
    if file_ext in ['.png', '.jpg', '.jpeg', '.gif', '.bmp']:
        selected_image_path = file_path
        image_sent_in_history = False # Reset flag for new image
        try:
            update_image_preview(file_path)
            image_indicator.config(text=f"✓ Img: {file_name[:20]}{'...' if len(file_name)>20 else ''}")
            add_message_to_ui("status", f"Image '{file_name}' attached.")
        except Exception as img_err:
            print(f"[File Process] Error processing image {file_name}: {img_err}")
            add_message_to_ui("error", f"Failed to load image: {file_name}")
            clear_image() # Clear preview on error

    # --- Handle PDF Files ---
    elif file_ext == '.pdf':
        add_message_to_ui("status", f"Loading PDF: {file_name}...")
        if 'window' in globals() and window: window.update_idletasks() # Update UI

        # Extract content in a separate thread to avoid freezing UI
        def extract_and_update():
            try:
                content = extract_pdf_content(file_path)
                # Update UI in main thread
                if window and window.winfo_exists():
                     window.after(0, update_input_with_text_content, content, file_name, "PDF")
            except Exception as pdf_thread_err:
                 print(f"[PDF Thread] Error in PDF extraction thread: {pdf_thread_err}")
                 if window and window.winfo_exists():
                      window.after(0, lambda: add_message_to_ui("error", f"Error processing PDF {file_name}"))

        thread = threading.Thread(target=extract_and_update, daemon=True)
        thread.start()

    # --- Handle Text Files ---
    elif file_ext in ['.txt', '.md', '.py', '.js', '.html', '.css', '.json', '.log', '.csv', '.xml', '.yaml', '.ini', '.sh', '.bat']:
         add_message_to_ui("status", f"Loading text file: {file_name}...")
         if 'window' in globals() and window: window.update_idletasks()

         def load_text_and_update():
             try:
                 with open(file_path, 'r', encoding='utf-8', errors='ignore') as f: # Use errors='ignore' for robustness
                     content = f.read()
                 # Truncate if very long
                 max_len = 20000
                 if len(content) > max_len:
                     content = content[:max_len] + f"\n\n[--- Content truncated at {max_len} characters ---]"

                 # Update UI in main thread
                 if window and window.winfo_exists():
                     window.after(0, update_input_with_text_content, content, file_name, "Text")
             except Exception as e:
                 print(f"[Text Load] Error loading text file {file_name}: {e}")
                 if window and window.winfo_exists():
                      window.after(0, lambda: add_message_to_ui("error", f"Error loading text file: {e}"))

         thread = threading.Thread(target=load_text_and_update, daemon=True)
         thread.start()
         
    # --- Handle other files as plain text (optional) ---
    else:
         print(f"[File Process] Unsupported file type '{file_ext}' for specific handling. Trying to load as text.")
         add_message_to_ui("status", f"Loading file as text: {file_name}...")
         if 'window' in globals() and window: window.update_idletasks()
         
         def load_generic_text_and_update():
             try:
                 with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                     content = f.read()
                 max_len = 10000 # Shorter limit for generic files
                 if len(content) > max_len:
                     content = content[:max_len] + f"\n\n[--- Content truncated at {max_len} characters ---]"
                 if window and window.winfo_exists():
                     window.after(0, update_input_with_text_content, content, file_name, "File")
             except Exception as e:
                 print(f"[Generic Load] Error loading file {file_name} as text: {e}")
                 if window and window.winfo_exists():
                     window.after(0, lambda: add_message_to_ui("error", f"Could not load file {file_name} as text."))

         thread = threading.Thread(target=load_generic_text_and_update, daemon=True)
         thread.start()


def update_input_with_text_content(content, file_name, file_type="File"):
    """Updates the user input with extracted text content (runs in main thread)."""
    global user_input_widget, window
    if not user_input_widget: return

    try:
        # Prepare header
        header = f"--- Content from {file_type}: {file_name} ---\n"
        full_content = header + content + f"\n--- End of {file_name} Content ---"

        # Clear current content and insert new
        user_input_widget.delete("1.0", tk.END)
        user_input_widget.insert("1.0", full_content)
        user_input_widget.see("1.0") # Scroll to top
        user_input_widget.update_idletasks()

        # Notify user
        add_message_to_ui("status", f"{file_type} content from '{file_name}' loaded into input area.")
        print(f"[UI Update] {file_type} content loaded into input: {file_name}")

    except tk.TclError as e:
         print(f"[UI Update] Error updating input widget (maybe closed?): {e}")
    except Exception as e:
         print(f"[UI Update] Unexpected error updating input with text content: {e}")
         traceback.print_exc()


def paste_image_from_clipboard(event=None):
    """Pastes an image from the clipboard (runs in main thread)."""
    global selected_image_path, image_sent_in_history, window
    try:
        # Use Pillow's ImageGrab (requires Pillow installation)
        from PIL import ImageGrab
        clipboard_content = ImageGrab.grabclipboard()

        if clipboard_content is None:
            # print("[Paste] No image found in clipboard (clipboard_content is None).")
            # add_message_to_ui("status", "Clipboard does not contain an image.")
            return # Don't interfere with text paste
        
        # Check if it's an image object (PIL/Pillow)
        if isinstance(clipboard_content, Image.Image):
             img = clipboard_content
             print("[Paste] Image found in clipboard.")
             
             # Save to temporary file
             temp_file = tempfile.NamedTemporaryFile(prefix="pasted_img_", suffix=".png", delete=False)
             temp_path = temp_file.name
             temp_file.close()

             try:
                 img.save(temp_path, "PNG")
             except Exception as save_err:
                 print(f"[Paste] Error saving clipboard image to {temp_path}: {save_err}")
                 add_message_to_ui("error", "Failed to save pasted image.")
                 if os.path.exists(temp_path): os.unlink(temp_path) # Clean up failed attempt
                 return

             # Update application state
             selected_image_path = temp_path
             image_sent_in_history = False

             # Update UI
             update_image_preview(temp_path)
             image_indicator.config(text=f"✓ Pasted image")
             add_message_to_ui("status", f"Image pasted from clipboard.")
             print(f"[Paste] Image saved to {temp_path}")
             return "break" # Indicate we handled the paste event (for image)

        # Handle case where clipboard contains file paths (e.g., copied file in Explorer)
        elif isinstance(clipboard_content, list) and len(clipboard_content) > 0:
             # Check if the first item is a valid image file path
             file_path = clipboard_content[0]
             if isinstance(file_path, str) and os.path.isfile(file_path):
                 file_ext = os.path.splitext(file_path)[1].lower()
                 if file_ext in ['.png', '.jpg', '.jpeg', '.gif', '.bmp']:
                     print(f"[Paste] Image file path found in clipboard: {file_path}")
                     process_selected_file(file_path) # Use the common file processing function
                     return "break" # Indicate handled

        # If it's not an image or recognized file path, let text paste proceed
        # print("[Paste] Clipboard content is not a direct image or recognized file path.")
        return None # Allow default paste behavior

    except ImportError:
         print("[Paste] Error: Pillow library not found. Cannot paste images. `pip install Pillow`")
         add_message_to_ui("error", "Pillow library needed for image pasting.")
         return None
    except Exception as e:
        # Catch tkdnd specific errors or others
        if "bad screen distance" in str(e): # Example of specific error handling
             print(f"[Paste] Tkinter error during paste (potential focus issue): {e}")
        else:
             print(f"[Paste] Error pasting image: {e}")
             traceback.print_exc()
        # add_message_to_ui("error", f"Failed to paste image: {e}")
        return None # Allow default behavior on error


def setup_paste_binding():
    """Sets up the keyboard binding for paste (runs in main thread)."""
    # Bind Ctrl+V / Cmd+V to the main window and input widget
    # The event handler will try to paste an image first.
    # If it returns None, Tkinter's default text paste proceeds.
    # If it returns "break", default paste is suppressed.
    if window and user_input_widget:
         # Use a lambda to pass the event object correctly
         paste_handler = lambda event: paste_image_from_clipboard(event)
         
         # Bind to user input widget (higher priority for focus)
         user_input_widget.bind("<Control-v>", paste_handler)
         user_input_widget.bind("<Command-v>", paste_handler) # Mac

         # Bind to main window as a fallback (might catch paste when input isn't focused)
         # window.bind("<Control-v>", paste_handler)
         # window.bind("<Command-v>", paste_handler) # Mac
         print("[UI] Paste bindings (Ctrl+V / Cmd+V) set up.")
    else:
         print("[UI] Error: Could not set up paste bindings (window or input widget not ready).")

# --- Drag and Drop Handlers (Main Thread) ---

def handle_file_drop(event):
    """Generic handler for files dropped onto registered widgets."""
    # Extract file path(s) from the event data
    # TkinterDnD usually provides paths enclosed in curly braces {} if they contain spaces
    raw_path_data = event.data.strip()
    # Simple parsing: remove braces and handle potential multiple files (take first)
    if raw_path_data.startswith('{') and raw_path_data.endswith('}'):
         file_path = raw_path_data[1:-1]
    else:
         file_path = raw_path_data
         
    # Rudimentary check for multiple files (split by space if not braced)
    # This is not foolproof for all filenames but covers common cases.
    potential_files = file_path.split()
    if len(potential_files) > 1 and not raw_path_data.startswith('{'):
        # Check if the first part looks like a valid file
        if os.path.isfile(potential_files[0]):
             file_path = potential_files[0]
             print(f"[Drop] Multiple files detected, processing first: {file_path}")
        else:
             # Fallback to the whole string if first part isn't a file (might be filename with spaces)
             print(f"[Drop] Ambiguous drop data, processing as single path: {file_path}")
    
    # Process the determined single file path
    if file_path and os.path.isfile(file_path):
         print(f"[Drop] File dropped: {file_path}")
         # Use the common processing function
         process_selected_file(file_path)

         # Provide visual feedback on the drop target widget
         widget = event.widget
         try:
             original_bg = widget.cget("background")
             widget.config(bg="#90EE90") # Light green feedback
             # Use schedule to revert color after a delay
             if window and window.winfo_exists():
                 window.after(500, lambda w=widget, bg=original_bg: w.config(bg=bg))
         except tk.TclError: # Handle cases where widget background cannot be configured (e.g., some ttk widgets)
             pass
         except Exception as e:
             print(f"[Drop] Error providing visual feedback: {e}")
             
    else:
         print(f"[Drop] Ignored drop event: Data does not seem to be a valid file path: '{event.data}'")

# --- Whisper Language/Model Setters (Main Thread) ---

def set_whisper_language(event=None):
    """Sets the language for Whisper transcription."""
    global whisper_language, whisper_language_selector
    # Check if selector exists and has a valid selection
    if 'whisper_language_selector' in globals() and hasattr(whisper_language_selector, 'current'):
        selected_idx = whisper_language_selector.current()
        if selected_idx >= 0 and selected_idx < len(WHISPER_LANGUAGES):
             lang_name, lang_code = WHISPER_LANGUAGES[selected_idx]
             if whisper_language != lang_code: # Only print if changed
                 whisper_language = lang_code
                 print(f"[Whisper] Language set to: {lang_name} ({whisper_language})")
        else:
            print(f"[Whisper] Invalid language selection index: {selected_idx}")
    else:
         print("[Whisper] Language selector not ready.")


def set_whisper_model_size(event=None):
    """Sets Whisper model size and triggers re-initialization if needed."""
    global whisper_model_size, whisper_model_size_selector, whisper_initialized, whisper_model
    if 'whisper_model_size_selector' not in globals():
         print("[Whisper] Model size selector not ready.")
         return
         
    new_size = whisper_model_size_selector.get()
    if not new_size: # Handle empty selection if possible
         print("[Whisper] No model size selected.")
         return

    if new_size == whisper_model_size and whisper_initialized:
        # print(f"[Whisper] Model size '{new_size}' already selected and initialized.")
        return # No change needed

    print(f"[Whisper] Model size changed to: {new_size}. Re-initialization required.")

    # Update global state
    whisper_model_size = new_size
    whisper_initialized = False # Force re-initialization
    
    # Release old model reference to allow garbage collection
    if whisper_model:
        print("[Whisper] Releasing old model object...")
        # Explicit deletion might help, especially with GPU memory in some frameworks
        try:
            del whisper_model
            if torch.cuda.is_available():
                torch.cuda.empty_cache() # Try to free CUDA memory if applicable
        except Exception as del_err:
            print(f"[Whisper] Error during old model cleanup: {del_err}")
        whisper_model = None

    # If voice recognition is currently enabled, trigger immediate re-initialization
    if voice_enabled and voice_enabled.get():
        print("[Whisper] Voice enabled, attempting immediate re-initialization...")
        initialize_whisper() # This will load the new model and update status label
    else:
        # If voice is disabled, just note that re-init will happen when enabled
        update_vad_status(f"Model set to {new_size}", "blue") # Show selection change
        print(f"[Whisper] Model set to {new_size}. Will initialize when voice is enabled.")


def update_vad_status(text, color):
    """Safely updates the VAD status label from any thread (schedules on main thread)."""
    global recording_indicator_widget, window
    if recording_indicator_widget and window and window.winfo_exists():
        try:
            # Use schedule to run on main thread
            window.after(0, lambda w=recording_indicator_widget, t=text, c=color: w.config(text=t, fg=c))
        except tk.TclError as e:
            # Handle case where window might be closing while after() is pending
            if "application has been destroyed" not in str(e):
                 print(f"[UI Status] TclError updating VAD status (widget likely destroyed): {e}")
        except Exception as e:
             print(f"[UI Status] Error updating VAD status: {e}")


# ===================
# Ollama / Chat Logic
# ===================
def fetch_available_models():
    """Fetches available Ollama models (robust parsing)."""
    try:
        print("[Ollama Fetch] Calling ollama.list()...")
        models_data = ollama.list()
        print(f"[Ollama Fetch] Received raw data type: {type(models_data)}")
        # print(f"[Ollama Fetch] Raw data: {models_data}") # Uncomment for deep debugging if needed
        
        valid_models = []

        # --- NEW: Check for object with 'models' attribute containing Model objects ---
        if hasattr(models_data, 'models') and isinstance(models_data.models, list):
            print("[Ollama Fetch] Parsing as object with 'models' list attribute...")
            for model_obj in models_data.models:
                if hasattr(model_obj, 'model') and isinstance(model_obj.model, str):
                    valid_models.append(model_obj.model)
                elif hasattr(model_obj, 'name') and isinstance(model_obj.name, str): # Backup check for 'name' attribute
                     valid_models.append(model_obj.name)
                else:
                    print(f"[Ollama Fetch] Warning: Model object missing 'model' or 'name' string attribute: {model_obj}")
            if valid_models:
                 print(f"[Ollama Fetch] Successfully parsed {len(valid_models)} models via object attribute.")
            else:
                 print("[Ollama Fetch] Parsed object attribute, but found no valid model names.")


        # --- Fallback: Check for dictionary structure (older versions?) ---
        elif isinstance(models_data, dict) and 'models' in models_data:
            print("[Ollama Fetch] Parsing as dictionary with 'models' key...")
            models_list = models_data.get('models', [])
            if isinstance(models_list, list):
                for model_info in models_list:
                    if isinstance(model_info, dict) and 'name' in model_info:
                        valid_models.append(model_info['name'])
                    else:
                        print(f"[Ollama Fetch] Warning: Skipping unexpected model entry format in dict: {model_info}")
                if valid_models:
                    print(f"[Ollama Fetch] Successfully parsed {len(valid_models)} models via dictionary key.")
                else:
                     print("[Ollama Fetch] Parsed dictionary key, but found no valid model names.")
            else:
                print("[Ollama Fetch] Warning: 'models' key does not contain a list in dictionary.")

        # --- Fallback: Check for direct list of strings/dicts ---
        elif isinstance(models_data, list):
             print("[Ollama Fetch] Parsing as direct list...")
             for item in models_data:
                  if isinstance(item, str):
                       valid_models.append(item)
                  elif isinstance(item, dict) and 'name' in item:
                       valid_models.append(item['name'])
             if valid_models:
                  print(f"[Ollama Fetch] Successfully parsed {len(valid_models)} models via direct list.")
             else:
                  print("[Ollama Fetch] Parsed direct list, but found no valid model names.")

        # --- Handle No Models Found ---
        if not valid_models:
             print("[Ollama Fetch] No valid models identified after parsing attempts.")
             print(f"[Ollama Fetch] Final raw data received was: {models_data}")
             # Provide common fallbacks
             valid_models = [DEFAULT_OLLAMA_MODEL, "llama3:8b", "phi3:mini", "gemma:7b"]
             print(f"[Ollama Fetch] Returning fallback models: {valid_models}")
             return valid_models # Return fallbacks

        # Sort models for better UI presentation
        valid_models.sort()
        print(f"[Ollama Fetch] Found and sorted models: {valid_models}")
        return valid_models

    except Exception as e:
        print(f"[Ollama Fetch] Error during ollama.list() or parsing: {e}")
        traceback.print_exc()
        # Use schedule to add error to UI safely if window exists
        if 'window' in globals() and window and window.winfo_exists():
             window.after(0, lambda: add_message_to_ui("error", f"Could not fetch Ollama models: {e}. Check Ollama status."))
        # Provide common fallbacks if API fails
        fallbacks = [DEFAULT_OLLAMA_MODEL, "llama3:8b", "phi3:mini", "gemma:7b"]
        print(f"[Ollama Fetch] Returning fallback models due to error: {fallbacks}")
        return fallbacks

def chat_worker(user_message_content, image_path=None):
    """Background worker for Ollama streaming chat (Thread-safe history)."""
    global messages, messages_lock, current_model, stream_queue, stream_done_event, stream_in_progress
    global web_output_queue # (Phase 2) Access web output queue

    # Construct the message with potential image
    current_message = {"role": "user", "content": user_message_content}
    image_bytes = None # Store bytes if image is processed successfully
    image_error = None # Store error if image processing fails

    if image_path:
        try:
            print(f"[Chat Worker] Reading image file: {image_path}")
            # Read the image as bytes - Ollama Python library expects bytes
            with open(image_path, 'rb') as f:
                image_bytes = f.read()
            # Only add 'images' key if bytes were read successfully
            current_message["images"] = [image_bytes] # Use the read bytes
            print(f"[Chat Worker] Image bytes loaded ({len(image_bytes)} bytes).")
        except FileNotFoundError:
             image_error = f"Error: Image file not found at '{image_path}'"
        except Exception as e:
            image_error = f"Error reading image file '{os.path.basename(image_path)}': {e}"
            print(f"[Chat Worker] {image_error}")

    # --- Handle Image Errors ---
    if image_error:
         # Put error message on both queues (UI and Web)
         stream_queue.put(("ERROR", image_error))
         web_output_queue.put({"type": "error", "content": image_error}) # (Phase 2)
         # Don't add user message to history if image failed critically before sending
         stream_in_progress = False
         stream_done_event.set()
         return # Stop processing

    # --- Update Shared History (Thread-Safe) ---
    try:
        with messages_lock:
            messages.append(current_message)
            # Make a copy for this request to avoid modification during iteration
            history_for_ollama = list(messages)
            # Important: Remove raw image bytes from history sent to Ollama if model doesn't support it or for efficiency
            # Ollama library *should* handle the 'images' key correctly, but let's keep the history lean.
            # The current message *sent* to ollama.chat() will contain the image.
            # The history list (`messages`) retains it for potential future context if needed.
            # Let's modify the copy sent, not the global `messages` here.
            history_copy_for_request = []
            for msg in history_for_ollama:
                msg_copy = msg.copy()
                if "images" in msg_copy:
                     # Keep 'images' key only for the *very last* message if sending image now
                     if msg is not current_message:
                          del msg_copy["images"] # Remove image bytes from past messages in request history
                     # Optionally, replace with placeholder for older messages:
                     # else: msg_copy['content'] += " [Image was attached]"
                history_copy_for_request.append(msg_copy)

    except Exception as lock_err:
         err_text = f"Error accessing chat history (lock): {lock_err}"
         print(f"[Chat Worker] {err_text}")
         stream_queue.put(("ERROR", err_text))
         web_output_queue.put({"type": "error", "content": err_text}) # (Phase 2)
         stream_in_progress = False
         stream_done_event.set()
         return

    # --- Call Ollama API ---
    assistant_response = "" # Accumulate full response for history
    try:
        print(f"[Chat Worker] Sending request to model {current_model}...")
        # Debug: Print messages being sent
        # print("[Chat Worker] Messages Sent:", history_copy_for_request)

        stream = ollama.chat(
            model=current_model,
            messages=history_copy_for_request, # Send the potentially modified history copy
            stream=True
        )

        first_chunk = True
        for chunk in stream:
            # Check for stop signal (e.g., from main thread if needed, less common here)
            # if stop_event.is_set(): break

            if 'message' in chunk and 'content' in chunk['message']:
                content_piece = chunk['message']['content']
                if first_chunk:
                     # Signal start only once
                     stream_queue.put(("START", None))
                     web_output_queue.put({"type": "start"}) # (Phase 2) Signal web UI too
                     first_chunk = False
                
                # Send chunk to both queues
                stream_queue.put(("CHUNK", content_piece))
                web_output_queue.put({"type": "chunk", "content": content_piece}) # (Phase 2)

                assistant_response += content_piece # Accumulate for history

            # Check for errors within the stream
            if 'error' in chunk:
                 err_text = f"Ollama stream error: {chunk['error']}"
                 print(f"[Chat Worker] {err_text}")
                 stream_queue.put(("ERROR", err_text))
                 web_output_queue.put({"type": "error", "content": err_text}) # (Phase 2)
                 assistant_response = None # Indicate error occurred, don't save partial response
                 break # Stop processing on stream error

            # Check for 'done' status (alternative way stream might end)
            if chunk.get('done', False):
                 print("[Chat Worker] Stream finished (received 'done' flag).")
                 break # Exit loop gracefully if Ollama signals done

        # --- After successful streaming ---
        if assistant_response is not None: # Check if error occurred mid-stream
             # Add the complete assistant response to shared history (thread-safe)
             try:
                 with messages_lock:
                     messages.append({"role": "assistant", "content": assistant_response})
                 print(f"[Chat Worker] Assistant response added to history ({len(assistant_response)} chars).")
             except Exception as lock_err:
                 # Log error, but don't send to UI again if stream ended ok
                 print(f"[Chat Worker] Error adding assistant response to history (lock): {lock_err}")

             # Signal end to both queues
             stream_queue.put(("END", None))
             web_output_queue.put({"type": "end"}) # (Phase 2)

    except ollama.ResponseError as ore:
         err_text = f"Ollama API Response Error: {ore.status_code} - {ore.error}"
         print(f"[Chat Worker] {err_text}")
         stream_queue.put(("ERROR", err_text))
         web_output_queue.put({"type": "error", "content": err_text}) # (Phase 2)
    except Exception as e:
        err_text = f"Ollama communication error: {e}"
        print(f"[Chat Worker] {err_text}")
        traceback.print_exc()
        stream_queue.put(("ERROR", err_text))
        web_output_queue.put({"type": "error", "content": err_text}) # (Phase 2)
    finally:
        # --- Finalization ---
        stream_in_progress = False
        stream_done_event.set() # Signal completion/error to main thread
        print("[Chat Worker] Worker finished.")


def process_stream_queue():
    """Processes items from Ollama stream queue for UI and TTS (runs in main thread)."""
    global stream_queue, chat_history_widget, tts_sentence_buffer, stream_in_progress, window

    if not stream_in_progress: # Optimization: Don't poll if nothing is expected
         return

    try:
        while True: # Process all available items without blocking
            item_type, item_data = stream_queue.get_nowait()

            if item_type == "START":
                 # Optional: Clear "Thinking..." message here if needed,
                 # but usually handled when first chunk arrives or in send_message().
                 print("[Stream Processor] Received START signal.")
                 pass # Placeholder
                 
            elif item_type == "CHUNK":
                # Append chunk to UI
                if chat_history_widget:
                    try:
                        chat_history_widget.config(state=tk.NORMAL)
                        chat_history_widget.insert(tk.END, item_data, "bot_message")
                        chat_history_widget.config(state=tk.DISABLED)
                        chat_history_widget.see(tk.END) # Auto-scroll
                    except tk.TclError as ui_err:
                        print(f"[Stream Processor] Error updating UI (widget destroyed?): {ui_err}")
                else:
                     print("[Stream Processor] Warning: chat_history_widget not available for chunk.")
                     
                # Queue text for TTS
                queue_tts_text(item_data)
                
            elif item_type == "END":
                print("[Stream Processor] Received END signal.")
                # Add final newline for spacing in chat history
                if chat_history_widget:
                     try:
                         chat_history_widget.config(state=tk.NORMAL)
                         chat_history_widget.insert(tk.END, "\n\n", "bot_message") # Ensure spacing
                         chat_history_widget.config(state=tk.DISABLED)
                         chat_history_widget.see(tk.END)
                     except tk.TclError as ui_err:
                         print(f"[Stream Processor] Error adding final newline (widget destroyed?): {ui_err}")
                         
                # Flush any remaining partial sentence in TTS buffer
                queue_tts_text("\n") # Add newline to help flush
                try_flush_tts_buffer()
                # Note: stream_in_progress is reset by chat_worker or check_done
                return # Stop polling loop for this response

            elif item_type == "ERROR":
                 print(f"[Stream Processor] Received ERROR signal: {item_data}")
                 add_message_to_ui("error", item_data)
                 # Ensure stream_in_progress is reset (should be done by worker, but defensive)
                 stream_in_progress = False
                 return # Stop polling loop

    except queue.Empty:
        # No more items in queue right now
        pass
    except Exception as e:
         print(f"[Stream Processor] Error processing stream queue: {e}")
         traceback.print_exc()
         stream_in_progress = False # Stop polling on unexpected error

    # Reschedule polling ONLY if stream is still marked as in progress
    # and the window still exists
    if stream_in_progress and window and window.winfo_exists():
        window.after(100, process_stream_queue) # Check again in 100ms


# =================================
# Web Input Queue Processing (Phase 2)
# =================================

def check_web_input_queue():
    """Periodically check the queue for messages from the Flask web UI."""
    global web_input_queue, window, user_input_widget

    try:
        # Process all messages currently in the queue
        while True:
            message_data = web_input_queue.get_nowait()
            print(f"[Main] Processing message from web queue: {message_data}")

            msg_type = message_data.get("type")

            if msg_type == "text_message":
                content = message_data.get("content", "").strip()
                if content:
                    # --- Refactored approach: Call core send logic directly ---
                    # process_user_message(content, source="web") # Needs refactoring send_message

                    # --- Alternative: Simulate GUI interaction (simpler initially) ---
                    if user_input_widget:
                        try:
                             print(f"[Main] Simulating GUI input for web message: '{content[:50]}...'")
                             user_input_widget.delete("1.0", tk.END) # Clear GUI input
                             user_input_widget.insert("1.0", content) # Put text in GUI input
                             send_message() # Trigger the existing send logic from GUI
                        except tk.TclError as e:
                             print(f"[Main] Error interacting with GUI for web message: {e}")
                             add_message_to_ui("error", f"Failed to process web message in UI: {e}")
                    else:
                         print("[Main] Error: User input widget not available for web message.")

            # Handle other message types from web if needed later
            # elif msg_type == "set_model":
            #     model_name = message_data.get("model")
            #     # ... logic to update model ...

            web_input_queue.task_done() # Mark task as complete

    except queue.Empty:
        pass # No message from web UI this time
    except Exception as e:
        print(f"[Main] Error processing web input queue: {e}")
        traceback.print_exc()
    finally:
        # Reschedule the check if the window still exists
        if window and window.winfo_exists():
            window.after(300, check_web_input_queue) # Check every 300ms


# ===================
# UI Helpers
# ===================

def clear_image():
    """Clears the selected image."""
    global selected_image_path, image_sent_in_history, image_preview, image_indicator
    selected_image_path = ""
    image_sent_in_history = False # Allow sending next time
    if image_preview:
        image_preview.configure(image="", text="No Image", width=20, height=10, bg="lightgrey")
        image_preview.image = None # Clear reference
    if image_indicator:
        image_indicator.config(text="No image attached")
    print("[UI] Image selection cleared.")


def update_image_preview(file_path):
    """Updates the image preview label (runs in main thread)."""
    global image_preview, window
    if not image_preview or not window or not window.winfo_exists():
        print("[UI] Image preview widget not ready.")
        return

    try:
        img = Image.open(file_path)
        # Resize for preview
        max_size = 140 # Max width/height for preview
        img.thumbnail((max_size, max_size), Image.Resampling.LANCZOS) # Use Resampling for newer Pillow
        
        photo = ImageTk.PhotoImage(img)
        image_preview.configure(image=photo, width=img.width, height=img.height, text="") # Clear text
        image_preview.image = photo # Keep reference to avoid garbage collection
        print(f"[UI] Image preview updated for: {os.path.basename(file_path)}")
    except FileNotFoundError:
        print(f"[UI] Error updating image preview: File not found - {file_path}")
        clear_image()
        image_indicator.config(text="File Not Found", fg="red")
    except Exception as e:
        print(f"[UI] Error updating image preview: {e}")
        clear_image() # Reset if preview fails
        if image_indicator: image_indicator.config(text="Preview Error", fg="red")


def add_message_to_ui(role, content, tag_suffix=""):
    """Adds a message to the chat history widget (thread-safe via `window.after`)."""
    global chat_history_widget, window

    if not content: # Don't add empty messages
         # print("[UI] Skipping empty message.")
         return
    
    # Ensure this runs on the main thread
    if threading.current_thread() is not threading.main_thread():
         if window and window.winfo_exists():
              # print(f"[UI] Scheduling message add from thread: {role} - {content[:30]}...")
              window.after(0, add_message_to_ui, role, content, tag_suffix)
         else:
              print("[UI] Warning: Window closed, cannot schedule message add.")
         return

    # --- Main thread execution ---
    if not chat_history_widget:
        print("[UI] Warning: chat_history_widget not ready.")
        return
        
    try:
        # Temporarily enable widget, add content, disable again
        chat_history_widget.config(state=tk.NORMAL)

        prefix = ""
        tag = ""
        content_tag = ""

        if role == "user":
            prefix = "You: "
            tag = "user_tag"
            content_tag = "user_message" + tag_suffix
        elif role == "assistant":
            # Note: Assistant messages are typically added chunk-by-chunk by process_stream_queue
            # This might be used for non-streamed messages or status updates styled as assistant.
            prefix = "Ollama: "
            tag = "bot_tag"
            content_tag = "bot_message" + tag_suffix
        elif role == "error":
            prefix = "Error: "
            tag = "error" + tag_suffix
            content_tag = tag # Use same tag for content
        elif role == "status":
            prefix = "" # Status messages might not need a prefix
            tag = "status" + tag_suffix
            content_tag = tag

        # Insert prefix with its tag
        if prefix:
             chat_history_widget.insert(tk.END, prefix, tag)
             
        # Insert content with its tag, add spacing
        chat_history_widget.insert(tk.END, content + "\n\n", content_tag)

        # Auto-scroll to the end
        chat_history_widget.see(tk.END)
        chat_history_widget.config(state=tk.DISABLED)
        
        # Force UI update (optional, can sometimes cause minor flicker)
        # window.update_idletasks()

    except tk.TclError as e:
        # Handle potential errors if the widget is destroyed mid-operation
        if "application has been destroyed" not in str(e):
             print(f"[UI] TclError adding message (widget likely destroyed): {e}")
    except Exception as e:
        print(f"[UI] Unexpected error adding message to UI: {e}")
        traceback.print_exc()


def select_model(event=None):
    """Updates the selected Ollama model (runs in main thread)."""
    global current_model, model_selector, model_status
    if not model_selector: return # Not ready

    selected = model_selector.get()
    if selected and selected != "No models found":
        if selected != current_model: # Only update if changed
             current_model = selected
             if model_status: model_status.config(text=f"Using: {current_model.split(':')[0]}") # Show base name
             print(f"[Ollama] Model selected: {current_model}")
             add_message_to_ui("status", f"Switched model to: {current_model}")
    elif not selected:
         print("[Ollama] No model selected in dropdown.")


def send_message(event=None):
    """Handles sending the user's message to Ollama (runs in main thread)."""
    global messages, selected_image_path, image_sent_in_history
    global stream_in_progress, stream_done_event, user_input_widget, chat_history_widget
    global tts_sentence_buffer, tts_engine, tts_enabled, tts_queue, tts_busy, tts_busy_lock, send_button

    # --- Input Validation ---
    if stream_in_progress:
        add_message_to_ui("error", "Please wait for the current response to complete.")
        return

    if not user_input_widget:
         add_message_to_ui("error", "Input widget not available.")
         return

    user_text = user_input_widget.get("1.0", tk.END).strip()
    # Determine image to send (only if it hasn't been sent *since it was last selected*)
    image_to_send = selected_image_path if selected_image_path and not image_sent_in_history else None

    if not user_text and not image_to_send:
        add_message_to_ui("error", "Please enter a message or attach a new image.")
        return

    # --- Prepare for Sending ---
    print("[UI] Preparing to send message...")
    if send_button: send_button.config(state=tk.DISABLED) # Disable send button
    stream_in_progress = True
    stream_done_event.clear()

    # --- Display User Message in UI ---
    display_text = user_text
    image_tag = ""
    if image_to_send:
        image_filename = os.path.basename(image_to_send)
        image_tag = f" [Image: {image_filename[:20]}{'...' if len(image_filename)>20 else ''}]"
        display_text += image_tag
    
    # Handle case where only image is sent
    if not user_text and image_to_send:
        display_text = image_tag.strip() # Use only the image tag as display text

    add_message_to_ui("user", display_text)

    # Clear input AFTER adding to UI
    user_input_widget.delete("1.0", tk.END)

    # --- Add "Thinking" Indicator ---
    thinking_start_index = None
    thinking_text = "Ollama: Thinking...\n"
    if chat_history_widget:
         try:
             chat_history_widget.config(state=tk.NORMAL)
             # Get index before inserting thinking text
             thinking_start_index = chat_history_widget.index(tk.END + "-1c") # Index before the last implicit newline
             chat_history_widget.insert(tk.END, thinking_text, ("bot_tag", "thinking"))
             chat_history_widget.see(tk.END)
             chat_history_widget.config(state=tk.DISABLED)
             if window: window.update_idletasks()
         except tk.TclError as e:
              print(f"[UI] Error adding 'Thinking...' indicator: {e}")
              thinking_start_index = None # Ensure it's None if failed
         except Exception as e:
              print(f"[UI] Unexpected error adding 'Thinking...' indicator: {e}")
              thinking_start_index = None


    # --- Stop any ongoing TTS and clear buffers ---
    if tts_engine and tts_enabled and tts_enabled.get() and tts_initialized_successfully:
        print("[UI] Stopping active TTS for new message...")
        try: tts_engine.stop()
        except Exception as tts_stop_err: print(f"[TTS] Minor error stopping engine: {tts_stop_err}")
        # Clear queue and buffer immediately
        tts_sentence_buffer = ""
        while not tts_queue.empty():
            try: tts_queue.get_nowait()
            except queue.Empty: break
        with tts_busy_lock: tts_busy = False # Reset busy state immediately
        print("[UI] TTS queue and buffer cleared.")

    # --- Start Ollama worker thread ---
    print(f"[UI] Starting chat worker thread (Image: {'Yes' if image_to_send else 'No'})...")
    thread = threading.Thread(target=chat_worker, args=(user_text, image_to_send), daemon=True)
    thread.start()

    # --- Update Image State ---
    if image_to_send:
        image_sent_in_history = True # Mark image as sent for this context
        # Optional: Clear image selection automatically after sending
        # clear_image() # Uncomment to automatically clear
        # Optional: Update indicator text to show it was sent
        if image_indicator: image_indicator.config(text=f"✓ Sent: {os.path.basename(image_to_send)[:20]}...")

    # --- Set up check for worker completion ---
    def check_done():
        global stream_in_progress # Allow modification
        if stream_done_event.is_set():
            print("[UI] Worker thread signaled completion.")
            stream_in_progress = False # Update state *before* UI changes

            # --- Remove "Thinking..." message (carefully) ---
            if thinking_start_index and chat_history_widget and window.winfo_exists():
                 try:
                     # Define the exact range of the thinking message
                     # Index calculation can be tricky with variable width fonts/newlines
                     # Let's try getting text from start index to current end and check
                     chat_history_widget.config(state=tk.NORMAL)
                     current_end_index = chat_history_widget.index(tk.END + "-1c")
                     
                     # Check if the "Thinking..." text is likely still at the end
                     # Fetch the last line or so to verify
                     check_range_start = f"{thinking_start_index} linestart"
                     check_text = chat_history_widget.get(check_range_start, tk.END).strip()

                     if thinking_text.strip() in check_text.splitlines()[-1]: # Check last line
                          # More precise end calculation needed if possible
                          end_thinking = f"{thinking_start_index}+{len(thinking_text)}c"
                          # Verify the content exactly before deleting
                          if chat_history_widget.get(thinking_start_index, end_thinking) == thinking_text:
                               print("[UI] Removing 'Thinking...' indicator.")
                               chat_history_widget.delete(thinking_start_index, end_thinking)
                          else:
                               print("[UI] 'Thinking...' text mismatch, not removing automatically.")
                     else:
                         # Response likely started, no need to delete "Thinking..."
                         print("[UI] Response likely started, skipping 'Thinking...' removal.")
                         pass
                     
                     chat_history_widget.config(state=tk.DISABLED)
                 except tk.TclError as e:
                     if "application has been destroyed" not in str(e):
                         print(f"[UI] TclError removing 'Thinking...' (widget destroyed?): {e}")
                 except Exception as e:
                      print(f"[UI] Error removing 'Thinking...' indicator: {e}")

            # --- Re-enable Send Button ---
            if send_button: send_button.config(state=tk.NORMAL)
            print("[UI] Send button re-enabled.")
            # --- Ensure stream processing stops polling ---
            # (process_stream_queue checks stream_in_progress, so setting it False here is enough)

        elif window and window.winfo_exists(): # Worker not done, reschedule check
            window.after(150, check_done) # Check again
        else: # Window closed, stop checking
             print("[UI] Window closed, stopping worker completion check.")


    # Start the first check for stream queue processing
    if window and window.winfo_exists():
         window.after(100, process_stream_queue)
         # Start the first check for worker completion
         window.after(150, check_done)
    else:
         print("[UI] Window closed, cannot start queue processing or completion checks.")


# ===================
# Main Application Setup & Loop
# ===================
def on_closing():
    """Handles application shutdown gracefully."""
    print("[Main] Closing application...")
    global window, flask_thread, vad_thread, whisper_processing_thread, tts_thread, py_audio

    # --- Signal background threads to stop ---

    # 1. Signal VAD thread
    if vad_thread and vad_thread.is_alive():
        print("[Main] Stopping VAD thread...")
        vad_stop_event.set()
        # Don't join here yet, allow audio cleanup first

    # 2. Stop Whisper processing thread
    if whisper_processing_thread and whisper_processing_thread.is_alive():
         print("[Main] Stopping Whisper processing thread...")
         whisper_queue.put(None) # Send sentinel
         # Don't join here yet

    # 3. Stop TTS thread
    stop_tts_thread() # Handles engine stop and thread join internally

    # --- Join worker threads (after signaling) ---
    # Give them some time to finish cleanly
    join_timeout = 2.0 # seconds

    if vad_thread and vad_thread.is_alive():
         print("[Main] Waiting for VAD thread to join...")
         vad_thread.join(timeout=join_timeout)
         if vad_thread.is_alive(): print("[Main] Warning: VAD thread did not exit cleanly.")
         else: print("[Main] VAD thread joined.")

    if whisper_processing_thread and whisper_processing_thread.is_alive():
         print("[Main] Waiting for Whisper thread to join...")
         whisper_processing_thread.join(timeout=join_timeout)
         if whisper_processing_thread.is_alive(): print("[Main] Warning: Whisper thread did not exit cleanly.")
         else: print("[Main] Whisper thread joined.")

    # --- Clean up resources ---

    # 4. Terminate PyAudio (after VAD stream is closed)
    if py_audio:
        print("[Main] Terminating PyAudio...")
        try:
             py_audio.terminate()
             print("[Main] PyAudio terminated.")
        except Exception as pa_term_err:
             print(f"[Main] Error terminating PyAudio: {pa_term_err}")

    # 5. Clean up temporary audio file if it exists
    global temp_audio_file_path
    if temp_audio_file_path and os.path.exists(temp_audio_file_path):
        try:
            print(f"[Main] Deleting temporary audio file: {temp_audio_file_path}")
            os.unlink(temp_audio_file_path)
        except Exception as e:
            print(f"[Main] Error deleting temp file: {e}")

    # 6. Shutdown Flask server (optional, daemon thread should exit anyway)
    # Forcing shutdown might require more complex signaling or accessing the Werkzeug server instance
    print("[Main] Flask thread is daemon, should exit automatically.")
    # If flask_thread needs explicit cleanup, it would be done here.

    # 7. Destroy the Tkinter window
    if window:
        print("[Main] Destroying Tkinter window...")
        try:
            window.destroy()
            print("[Main] Window destroyed.")
        except tk.TclError as e:
            print(f"[Main] Error destroying window (might already be closed): {e}")
        except Exception as e:
             print(f"[Main] Unexpected error destroying window: {e}")

    print("[Main] Application closed.")
    # Force exit if threads are stuck (use cautiously)
    # sys.exit(0)


# --- Build GUI ---
# Use TkinterDnD.Tk for the main window to enable drag & drop
try:
    window = TkinterDnD.Tk()
    print("[UI] TkinterDnD initialized.")
except Exception as dnd_err:
     print(f"[UI] Error initializing TkinterDnD: {dnd_err}")
     print("[UI] Falling back to standard Tkinter. Drag and Drop will be disabled.")
     window = tk.Tk() # Fallback

window.title(APP_TITLE)
window.geometry(WINDOW_GEOMETRY)

# --- Tkinter Variables ---
tts_enabled = tk.BooleanVar(value=True)
tts_rate = tk.IntVar(value=160) # Default TTS rate
tts_voice_id = tk.StringVar(value="") # Default voice (will be set later)
voice_enabled = tk.BooleanVar(value=True) # Default VAD/Whisper state
auto_send_after_transcription = tk.BooleanVar(value=True) # Default auto-send state
selected_whisper_language = tk.StringVar() # Bound to language dropdown
selected_whisper_model = tk.StringVar(value=whisper_model_size) # Bound to model size dropdown

# --- Main Frame ---
main_frame = tk.Frame(window, padx=10, pady=10)
main_frame.pack(fill=tk.BOTH, expand=True)

# --- Top Controls Frame ---
top_controls_frame = tk.Frame(main_frame)
top_controls_frame.pack(fill=tk.X, expand=False, pady=(0, 10))
top_controls_frame.columnconfigure(0, weight=3) # Model selection wider
top_controls_frame.columnconfigure(1, weight=2) # TTS controls
top_controls_frame.columnconfigure(2, weight=2) # Voice controls

# --- Model Selection ---
model_frame = tk.LabelFrame(top_controls_frame, text="Ollama Model", padx=5, pady=5)
model_frame.grid(row=0, column=0, sticky="ew", padx=(0, 5))

print("[UI] Fetching available Ollama models...")
available_models = fetch_available_models()
model_selector = ttk.Combobox(model_frame, values=available_models, state="readonly", width=25)

# Set initial model selection
current_model_set = False
if available_models:
    # Prioritize DEFAULT_OLLAMA_MODEL if available
    if DEFAULT_OLLAMA_MODEL in available_models:
        model_selector.set(DEFAULT_OLLAMA_MODEL)
        current_model = DEFAULT_OLLAMA_MODEL
        current_model_set = True
    # Otherwise, try the first model in the list
    elif available_models:
        model_selector.set(available_models[0])
        current_model = available_models[0]
        current_model_set = True

if not current_model_set:
    model_selector.set("No models found")
    model_selector.config(state=tk.DISABLED)
    current_model = None # Ensure current_model reflects lack of selection
    print("[UI] No Ollama models found or fetch failed.")

model_selector.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0,5))
model_selector.bind("<<ComboboxSelected>>", select_model)

model_status = tk.Label(model_frame, text=f"Using: {current_model.split(':')[0] if current_model else 'N/A'}", font=("Arial", 8), width=15, anchor="w")
model_status.pack(side=tk.LEFT)

# --- TTS Controls ---
tts_outer_frame = tk.LabelFrame(top_controls_frame, text="Text-to-Speech", padx=5, pady=5)
tts_outer_frame.grid(row=0, column=1, sticky="nsew", padx=5) # Use sticky nsew

tts_toggle_button = ttk.Checkbutton(tts_outer_frame, text="Enable TTS", variable=tts_enabled, command=toggle_tts)
tts_toggle_button.pack(anchor="w", pady=2)

# Voice Selector
voice_frame = tk.Frame(tts_outer_frame)
voice_frame.pack(fill=tk.X, pady=2)
tk.Label(voice_frame, text="Voice:", font=("Arial", 8)).pack(side=tk.LEFT)
print("[UI] Getting available TTS voices...")
available_voices = get_available_voices() # Populated by get_available_voices()
voice_names = [v[0] for v in available_voices] if available_voices else ["No voices found"]
voice_selector = ttk.Combobox(voice_frame, values=voice_names, state="disabled", width=18, font=("Arial", 8), textvariable=tts_voice_id) # Bind to variable

if available_voices:
    # Try to find a reasonable default voice
    default_voice_index = 0
    preferred_voices = ["Zira", "David", "Hazel", "Susan", "Microsoft ", "Google UK"] # Example preferences
    for i, (name, v_id) in enumerate(available_voices):
         if any(pref in name for pref in preferred_voices):
             default_voice_index = i
             break
    voice_selector.current(default_voice_index)
    tts_voice_id.set(available_voices[default_voice_index][1]) # Set initial voice ID variable
    voice_selector.bind("<<ComboboxSelected>>", set_voice)
    print(f"[UI] Default TTS voice set to: {available_voices[default_voice_index][0]}")
else:
    voice_selector.set("No voices found")
    print("[UI] No TTS voices found.")

voice_selector.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=2)


# Rate Control
rate_frame = tk.Frame(tts_outer_frame)
rate_frame.pack(fill=tk.X, pady=2)
tk.Label(rate_frame, text="Speed:", font=("Arial", 8)).pack(side=tk.LEFT)
rate_scale = ttk.Scale(rate_frame, from_=80, to=300, orient=tk.HORIZONTAL, variable=tts_rate, command=set_speech_rate, length=80, state="disabled") # Start disabled
rate_scale.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=2)
rate_value = tk.Label(rate_frame, textvariable=tts_rate, width=3, font=("Arial", 8))
rate_value.pack(side=tk.LEFT)


# --- Voice Recognition Controls ---
voice_outer_frame = tk.LabelFrame(top_controls_frame, text="Voice Input (VAD)", padx=5, pady=5)
voice_outer_frame.grid(row=0, column=2, sticky="nsew", padx=(5, 0)) # Use sticky nsew

voice_toggle_button = ttk.Checkbutton(voice_outer_frame, text="Enable Voice", variable=voice_enabled, command=toggle_voice_recognition)
voice_toggle_button.pack(anchor="w", pady=2)

# Whisper Settings Frame
whisper_settings_frame = tk.Frame(voice_outer_frame)
whisper_settings_frame.pack(fill=tk.X, pady=2)

# Language Selector
lang_frame = tk.Frame(whisper_settings_frame)
lang_frame.pack(fill=tk.X)
tk.Label(lang_frame, text="Lang:", font=("Arial", 8)).pack(side=tk.LEFT)
whisper_language_selector = ttk.Combobox(lang_frame, values=[lang[0] for lang in WHISPER_LANGUAGES],
                                         state="readonly", width=10, font=("Arial", 8),
                                         textvariable=selected_whisper_language)
whisper_language_selector.current(0) # Default to "Auto Detect" (index 0)
set_whisper_language() # Initialize whisper_language variable from initial selection
whisper_language_selector.pack(side=tk.LEFT, padx=2)
whisper_language_selector.bind("<<ComboboxSelected>>", set_whisper_language)


# Model Size Selector
size_frame = tk.Frame(whisper_settings_frame)
size_frame.pack(fill=tk.X, pady=(2,0))
tk.Label(size_frame, text="Model:", font=("Arial", 8)).pack(side=tk.LEFT)
whisper_model_size_selector = ttk.Combobox(size_frame, values=WHISPER_MODEL_SIZES,
                                           state="readonly", width=10, font=("Arial", 8),
                                           textvariable=selected_whisper_model)
# selected_whisper_model already set to default, selector will reflect it
whisper_model_size_selector.pack(side=tk.LEFT, padx=2)
whisper_model_size_selector.bind("<<ComboboxSelected>>", set_whisper_model_size)


# Auto-send checkbox (Corrected)
auto_send_checkbox = ttk.Checkbutton(voice_outer_frame, text="Auto-send",
                                     variable=auto_send_after_transcription)
                                     # tooltip="Automatically send message after voice transcription") # Removed this line
auto_send_checkbox.pack(anchor="w", pady=2)


# Recording Status Indicator
recording_indicator_widget = tk.Label(voice_outer_frame, text="Voice Disabled", font=("Arial", 9, "italic"), fg="grey", anchor="w")
recording_indicator_widget.pack(fill=tk.X, pady=(5,2), padx=2)


# --- Chat History ---
chat_frame = tk.LabelFrame(main_frame, text="Chat History", padx=5, pady=5)
chat_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))

chat_history_widget = scrolledtext.ScrolledText(chat_frame, wrap=tk.WORD, height=15, state=tk.DISABLED, font=("Arial", 10))
chat_history_widget.pack(fill=tk.BOTH, expand=True)

# Define text tags
chat_history_widget.tag_config("user_tag", foreground="#007bff", font=("Arial", 10, "bold"))
chat_history_widget.tag_config("user_message", foreground="black", font=("Arial", 10))
chat_history_widget.tag_config("bot_tag", foreground="#28a745", font=("Arial", 10, "bold"))
chat_history_widget.tag_config("bot_message", foreground="black", font=("Arial", 10))
chat_history_widget.tag_config("thinking", foreground="gray", font=("Arial", 10, "italic"))
chat_history_widget.tag_config("error", foreground="#dc3545", font=("Arial", 10, "bold")) # Red
chat_history_widget.tag_config("status", foreground="#6f42c1", font=("Arial", 9, "italic")) # Purple


# --- Bottom Frame (Image + Input) ---
bottom_frame = tk.Frame(main_frame)
bottom_frame.pack(fill=tk.X, expand=False)

# Image Frame (Left)
image_frame = tk.LabelFrame(bottom_frame, text="Attachments", padx=5, pady=5)
image_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))

image_preview = tk.Label(image_frame, text="No Image\nDrop Here", width=20, height=8,
                        bg="lightgrey", relief="sunken", anchor="center")
image_preview.pack(pady=5)

# Register image preview as a drop target if TkinterDnD is available
if isinstance(window, TkinterDnD.Tk):
     print("[UI] Registering image preview as drop target.")
     image_preview.drop_target_register(DND_FILES)
     image_preview.dnd_bind('<<Drop>>', handle_file_drop) # Use generic handler
else:
     print("[UI] TkinterDnD not available, image drop disabled.")


img_button_frame = tk.Frame(image_frame)
img_button_frame.pack(fill=tk.X, pady=2)
select_file_button = ttk.Button(img_button_frame, text="Open File", command=select_file, width=10)
select_file_button.pack(side=tk.LEFT, expand=True, padx=2)
clear_button = ttk.Button(img_button_frame, text="Clear Img", command=clear_image, width=10)
clear_button.pack(side=tk.LEFT, expand=True, padx=2)

image_indicator = tk.Label(image_frame, text="No image attached", font=("Arial", 8, "italic"), fg="grey")
image_indicator.pack(pady=(3,0))


# Input Frame (Right)
input_frame = tk.LabelFrame(bottom_frame, text="Your Message (Enter=Send, Shift+Enter=Newline, Drag Files)", padx=10, pady=5)
input_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

user_input_widget = scrolledtext.ScrolledText(input_frame, wrap=tk.WORD, height=4, font=("Arial", 10))
user_input_widget.pack(fill=tk.BOTH, expand=True, pady=(0, 5))
user_input_widget.focus_set()

# Register text input as a drop target if TkinterDnD is available
if isinstance(window, TkinterDnD.Tk):
     print("[UI] Registering text input as drop target.")
     user_input_widget.drop_target_register(DND_FILES)
     user_input_widget.dnd_bind('<<Drop>>', handle_file_drop) # Use generic handler
else:
     print("[UI] TkinterDnD not available, text input drop disabled.")


send_button = ttk.Button(input_frame, text="Send", command=send_message, width=10)
send_button.pack(pady=5, anchor='e') # Anchor to the right

# --- Enter Key Binding ---
def on_enter_press(event):
    """Handles Enter key press in the input widget."""
    # Check if Shift key is pressed (modifier state 1)
    if event.state & 0x0001:
        # Shift+Enter: Allow default behavior (insert newline)
        return
    else:
        # Enter only: Trigger send message
        send_message()
        # Prevent default behavior (which would insert newline)
        return "break"

user_input_widget.bind("<KeyPress-Return>", on_enter_press)


# --- Final Setup and Initialization ---
window.protocol("WM_DELETE_WINDOW", on_closing) # Set close handler

# Attempt to initialize TTS early but don't block startup
# Let toggle_tts handle enabling controls based on success
print("[Main] Pre-initializing TTS...")
try:
     if initialize_tts():
         # Enable controls immediately if successful
         if voice_selector: voice_selector.config(state="readonly")
         if rate_scale: rate_scale.config(state="normal")
         print("[Main] TTS pre-initialization successful.")
     else:
         # Add startup message about TTS failure
         def show_tts_init_error():
             add_message_to_ui("status", "Note: TTS engine failed to initialize. TTS controls disabled.")
         if window and window.winfo_exists(): window.after(500, show_tts_init_error) # Delay slightly
         print("[Main] TTS pre-initialization failed.")
except Exception as init_e:
     print(f"[Main] Error during TTS pre-initialization: {init_e}")


# --- Start Background Threads and Checks ---

# 1. Start the Flask server in a daemon thread (Phase 1)
flask_thread = threading.Thread(target=run_flask, daemon=True)
flask_thread.start()

# 2. Schedule initial state toggles after GUI is loaded
#    These will handle further initialization of TTS/VAD/Whisper if enabled by default.
if window and window.winfo_exists():
    window.after(1000, toggle_tts)  # Call after GUI is fully loaded
    window.after(1500, toggle_voice_recognition)  # Call with slight delay

# 3. Start periodic checks (TTS flush, Web Input)
if window and window.winfo_exists():
     window.after(200, periodic_tts_check)
     window.after(300, check_web_input_queue) # (Phase 2) Start checking web input

# 4. Set up paste binding (needs window and input widget to exist)
if window and window.winfo_exists():
    window.after(50, setup_paste_binding) # Short delay to ensure widgets are ready

# --- Add Welcome Message ---
def add_welcome():
     add_message_to_ui("status", f"Welcome! Model: {current_model or 'N/A'}. Web UI on http://<YOUR_IP>:{FLASK_PORT}")
if window and window.winfo_exists():
     window.after(100, add_welcome)

# --- Start Tkinter Main Loop ---
print("[Main] Starting Tkinter main loop...")
try:
    window.mainloop()
except KeyboardInterrupt:
     print("[Main] KeyboardInterrupt received. Closing...")
     on_closing()