# -*- coding: utf-8 -*-
import ollama
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
# import torchaudio # Not strictly needed for VAD/Whisper processing here
from collections import deque
import traceback
import fitz # For PDF processing
import json
import sys
import mimetypes # For checking file types
import atexit

# --- Flask Imports ---
from flask import Flask, request, jsonify, render_template, Response, send_from_directory
from werkzeug.utils import secure_filename

# ===================
# Constants
# ===================
# --- Audio ---
CHUNK = 1024
VAD_CHUNK = 512
RATE = 16000
FORMAT = pyaudio.paInt16
CHANNELS = 1
SILENCE_THRESHOLD_SECONDS = 1.0
MIN_SPEECH_DURATION_SECONDS = 0.3 # Slightly longer minimum to reduce noise triggers
PRE_SPEECH_BUFFER_SECONDS = 0.3

# --- Models ---
DEFAULT_OLLAMA_MODEL = "gemma3:27b"
DEFAULT_WHISPER_MODEL_SIZE = "turbo-large" # Defaulting to turbo large for speed/accuracy

# --- Whisper ---
WHISPER_LANGUAGES = {
    "Auto Detect": None, "English": "en", "Finnish": "fi", "Swedish": "sv",
    "German": "de", "French": "fr", "Spanish": "es", "Italian": "it",
    "Russian": "ru", "Chinese": "zh", "Japanese": "ja"
}
WHISPER_MODEL_SIZES = ["tiny", "base", "small", "medium", "large", "turbo-tiny", "turbo-base", "turbo-small", "turbo-medium", "turbo-large"]

# --- Web Server ---
FLASK_PORT = 5000
FLASK_HOST = "0.0.0.0" # Accessible on local network
UPLOAD_FOLDER = 'uploads' # Temporary folder for uploads
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'pdf', 'txt', 'md', 'py', 'js', 'html', 'css', 'json', 'log', 'csv', 'xml', 'yaml', 'ini', 'sh', 'bat'}

# ===================
# Globals / Backend State
# ===================
# --- Ollama & Chat ---
messages = []
messages_lock = threading.Lock()
selected_file_path = None # Store path of the currently attached file (cleared after send)
selected_file_type = None # Store type ('image', 'pdf', 'text')
ollama_model_list = [] # Cache for fetched models
current_model = DEFAULT_OLLAMA_MODEL
stream_in_progress = False
ollama_lock = threading.Lock() # Lock for sending messages to ollama

# --- TTS ---
tts_engine = None
tts_queue = queue.Queue()
tts_thread = None
tts_sentence_buffer = ""
tts_enabled_state = True # Default TTS state
tts_rate_state = 160 # Default rate
tts_voice_id_state = "" # Default voice ID
tts_available_voices = [] # Store available voices (id, name)
tts_busy = False
tts_busy_lock = threading.Lock()
tts_initialized_successfully = False
tts_init_lock = threading.Lock() # Lock for TTS initialization
periodic_tts_timer = None # Holder for the periodic timer thread

# --- Whisper/VAD ---
vad_model = None
vad_utils = None
vad_get_speech_ts = None # For older Silero versions maybe
whisper_model = None
whisper_model_size = DEFAULT_WHISPER_MODEL_SIZE
whisper_initialized = False
vad_initialized = False
vad_thread = None
vad_stop_event = threading.Event()
whisper_queue = queue.Queue() # Queue for audio filenames to transcribe
whisper_processing_thread = None
whisper_language_state = None # Default 'Auto Detect'
voice_enabled_state = True # Default VAD state
vad_status = {"text": "Voice Disabled", "color": "grey"} # Current VAD status for SSE
vad_status_lock = threading.Lock()

# --- Audio Handling ---
py_audio = None
is_recording_for_whisper = False
audio_frames_buffer = deque()
vad_audio_buffer = deque(maxlen=int(RATE / VAD_CHUNK * 1.5))
temp_audio_file_path = None # Path for the VAD temporary WAV file

# --- Web Communication ---
web_output_queue = queue.Queue() # SSE Queue for Ollama chunks, status updates, errors
flask_app = None # Holder for the Flask app instance
flask_server_running = True # Flag to control background loops

# ============================================================================
# Utility Functions
# ============================================================================

def get_vad_status():
    """Safely get the current VAD status."""
    with vad_status_lock:
        return vad_status.copy()

def update_vad_status(text, color):
    """Safely updates the VAD status and pushes update to SSE queue."""
    global vad_status
    with vad_status_lock:
        if vad_status.get("text") != text or vad_status.get("color") != color:
            vad_status = {"text": text, "color": color}
            web_output_queue.put({"type": "status_update", "source": "vad", "data": vad_status})

def send_error_to_web(error_message):
    """Sends an error message to the web UI via SSE."""
    print(f"[Web Error] {error_message}")
    web_output_queue.put({"type": "error", "data": str(error_message)})

def send_status_to_web(status_message):
    """Sends a general status message to the web UI via SSE."""
    print(f"[Web Status] {status_message}")
    web_output_queue.put({"type": "status", "data": str(status_message)})

def allowed_file(filename):
    """Checks if the uploaded file extension is allowed."""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# ============================================================================
# TTS Logic (Updates for robustness)
# ============================================================================

def initialize_tts():
    """Initializes the TTS engine. Returns True on success."""
    global tts_engine, tts_initialized_successfully, tts_rate_state, tts_voice_id_state, tts_available_voices

    with tts_init_lock:
        if tts_initialized_successfully: return True
        if tts_engine:
             try: tts_engine.stop(); del tts_engine; tts_engine = None
             except: pass

        try:
            print("[TTS] Initializing engine...")
            tts_engine = pyttsx3.init()
            if not tts_engine: raise Exception("pyttsx3.init() returned None")

            tts_engine.setProperty('rate', tts_rate_state)
            tts_engine.setProperty('volume', 0.9)

            voices = tts_engine.getProperty('voices')
            tts_available_voices = [{"id": v.id, "name": v.name} for v in voices]
            print(f"[TTS] Found {len(tts_available_voices)} voices.")

            current_voice_valid = any(v['id'] == tts_voice_id_state for v in tts_available_voices)
            if not tts_voice_id_state or not current_voice_valid:
                if tts_available_voices:
                    default_found = False
                    common_defaults = ["Zira", "David", "Hazel", "Susan", "Microsoft"]
                    for voice in tts_available_voices:
                        if any(common in voice['name'] for common in common_defaults):
                            tts_voice_id_state = voice['id']
                            default_found = True; break
                    if not default_found: tts_voice_id_state = tts_available_voices[0]['id']
                else: tts_voice_id_state = ""

            if tts_voice_id_state: tts_engine.setProperty('voice', tts_voice_id_state)
            else: print("[TTS] Warning: No voices found or could be set.")

            tts_initialized_successfully = True
            print(f"[TTS] Engine initialized successfully. Voice: {tts_voice_id_state}")
            send_status_to_web("TTS Engine Ready.")
            web_output_queue.put({"type": "voices_update", "data": tts_available_voices})
            return True
        except Exception as e:
            print(f"[TTS] Error initializing engine: {e}"); traceback.print_exc()
            tts_engine = None; tts_initialized_successfully = False; tts_available_voices = []
            send_error_to_web(f"TTS Initialization Failed: {e}")
            return False

def get_available_voices():
    """Returns the cached list of available TTS voices."""
    global tts_available_voices
    if not tts_available_voices and not tts_initialized_successfully:
        initialize_tts()
    return tts_available_voices

def set_tts_voice(voice_id):
    """Sets the selected voice ID for TTS."""
    global tts_voice_id_state, tts_engine, tts_initialized_successfully
    if not tts_initialized_successfully: send_error_to_web("TTS not initialized"); return False
    try:
        if any(v['id'] == voice_id for v in tts_available_voices):
            tts_voice_id_state = voice_id
            if tts_engine: tts_engine.setProperty('voice', voice_id)
            print(f"[TTS] Voice set to ID: {voice_id}")
            return True
        else: send_error_to_web(f"Invalid TTS Voice ID: {voice_id}"); return False
    except Exception as e: print(f"[TTS] Error setting voice: {e}"); send_error_to_web(f"Error setting TTS voice: {e}"); return False

def set_tts_rate(rate):
    """Sets the speech rate for TTS."""
    global tts_rate_state, tts_engine, tts_initialized_successfully
    if not tts_initialized_successfully: send_error_to_web("TTS not initialized"); return False
    try:
        rate = int(rate)
        if 80 <= rate <= 400:
            tts_rate_state = rate
            if tts_engine: tts_engine.setProperty('rate', rate)
            return True
        else: send_error_to_web(f"Invalid TTS Rate: {rate}. Must be 80-400."); return False
    except Exception as e: print(f"[TTS] Error setting rate: {e}"); send_error_to_web(f"Error setting TTS rate: {e}"); return False

def tts_worker():
    """Worker thread processing the TTS queue."""
    global tts_engine, tts_queue, tts_busy, tts_initialized_successfully, tts_enabled_state
    global tts_voice_id_state, tts_rate_state, flask_server_running
    print("[TTS Worker] Thread started.")
    while flask_server_running: # Check flag to allow clean exit
        text_to_speak = None
        try:
            # Use timeout to prevent blocking indefinitely if server stops
            text_to_speak = tts_queue.get(timeout=0.5)
            if text_to_speak is None:
                print("[TTS Worker] Received stop signal (None).")
                break # Exit loop on None sentinel

            if tts_engine and tts_enabled_state and tts_initialized_successfully:
                # Set busy flag *before* speaking
                with tts_busy_lock:
                    if tts_busy: # Double check busy state before proceeding
                         print("[TTS Worker] Warning: TTS busy flag already set? Re-queuing text.")
                         tts_queue.put(text_to_speak) # Put it back in queue
                         continue # Skip this iteration
                    tts_busy = True
                    print(f"[TTS Worker] Set Busy Flag: True") # DEBUG

                try:
                    # Ensure properties are set correctly before speaking
                    current_voice = tts_voice_id_state
                    current_rate = tts_rate_state
                    if current_voice: tts_engine.setProperty('voice', current_voice)
                    tts_engine.setProperty('rate', current_rate)

                    print(f"[TTS Worker] Speaking chunk ({len(text_to_speak)} chars): '{text_to_speak[:50]}...'") # Log speaking start
                    tts_engine.say(text_to_speak)
                    tts_engine.runAndWait() # Blocks this thread
                    print(f"[TTS Worker] Finished speaking chunk.") # Log speaking end

                except Exception as speak_err:
                    print(f"[TTS Worker] Error during say/runAndWait: {speak_err}")
                    traceback.print_exc()
                finally:
                    # Reset busy flag *only after* runAndWait completes or errors
                    with tts_busy_lock:
                         tts_busy = False
                         print(f"[TTS Worker] Set Busy Flag: False") # DEBUG

            else:
                # print("[TTS Worker] TTS disabled or uninitialized, discarding text.")
                pass

        except queue.Empty:
            # Timeout occurred, loop continues checking flask_server_running flag
            continue
        except RuntimeError as rt_err:
             print(f"[TTS Worker] Runtime Error: {rt_err}")
             traceback.print_exc()
             if "run loop already started" in str(rt_err):
                  try: tts_engine.stop()
                  except: pass
             # Ensure busy flag is cleared on runtime error
             with tts_busy_lock: tts_busy = False; print(f"[TTS Worker] Cleared Busy Flag (Runtime Error)") # DEBUG
        except Exception as e:
            print(f"[TTS Worker] Unexpected Error in worker loop: {e}")
            traceback.print_exc()
            # Ensure busy flag is cleared on unexpected error
            with tts_busy_lock: tts_busy = False; print(f"[TTS Worker] Cleared Busy Flag (Exception)") # DEBUG
            time.sleep(0.1)
        finally:
             if text_to_speak is not None: # Don't mark done for None or timeout
                try: tts_queue.task_done()
                except ValueError: pass # Ignore if already marked done

    print("[TTS Worker] Thread finished.")


def start_tts_thread():
    """Starts the TTS worker thread if needed."""
    global tts_thread, tts_initialized_successfully
    if tts_thread is not None and tts_thread.is_alive(): return
    if tts_thread is not None: tts_thread = None # Clear dead thread ref

    if not tts_initialized_successfully: initialize_tts()

    if tts_initialized_successfully:
        print("[TTS] Starting new worker thread...")
        tts_thread = threading.Thread(target=tts_worker, daemon=True)
        tts_thread.start()
        print("[TTS] Worker thread started.")
    else: print("[TTS] Engine init failed. Cannot start TTS thread.")


def stop_tts_thread():
    """Signals the TTS worker thread to stop and cleans up."""
    global tts_thread, tts_engine, tts_queue, tts_busy, tts_busy_lock
    print("[TTS] Stopping worker thread...")
    if tts_engine and tts_initialized_successfully:
        try: tts_engine.stop()
        except Exception as e: print(f"[TTS] Error stopping engine: {e}")

    while not tts_queue.empty(): # Clear queue first
        try: tts_queue.get_nowait()
        except queue.Empty: break

    if tts_thread and tts_thread.is_alive():
        tts_queue.put(None) # Send sentinel LAST
        print("[TTS] Waiting for worker thread to join...")
        tts_thread.join(timeout=2.0)
        if tts_thread.is_alive(): print("[TTS] Warning: Worker thread did not terminate gracefully.")
        else: print("[TTS] Worker thread joined.")
    tts_thread = None
    with tts_busy_lock: tts_busy = False
    print("[TTS] Worker thread stopped state set.")

def toggle_tts(enable):
    """Handles enabling/disabling TTS via API."""
    global tts_enabled_state, tts_sentence_buffer, tts_initialized_successfully
    if enable:
        if not tts_initialized_successfully: initialize_tts()
        if tts_initialized_successfully:
            tts_enabled_state = True; print("[TTS] Enabled by API request.")
            start_tts_thread(); send_status_to_web("TTS Enabled."); return True
        else:
            tts_enabled_state = False; print("[TTS] Enable failed - Init problem.")
            send_error_to_web("TTS Engine init failed. Cannot enable."); return False
    else:
        tts_enabled_state = False; print("[TTS] Disabled by API request.")
        if tts_engine and tts_initialized_successfully:
            try: tts_engine.stop()
            except Exception: pass
        tts_sentence_buffer = ""
        while not tts_queue.empty():
            try: tts_queue.get_nowait()
            except queue.Empty: break
        with tts_busy_lock: tts_busy = False # Clear busy flag
        send_status_to_web("TTS Disabled."); return True

def queue_tts_text(new_text):
    """Accumulates text for TTS."""
    global tts_sentence_buffer, tts_enabled_state, tts_initialized_successfully
    if tts_enabled_state and tts_initialized_successfully:
        tts_sentence_buffer += new_text
        # print(f"[TTS Queue Buffer] Added text. Buffer size: {len(tts_sentence_buffer)}") # DEBUG

# Define sentence-ending characters and potential following whitespace
sentence_enders = ".!?"
sentence_end_pattern = re.compile(r'([' + sentence_enders + r'])([\s\n]+|$)')

def try_flush_tts_buffer():
    """Sends complete sentences from the buffer to the TTS queue if TTS is idle."""
    global tts_sentence_buffer, tts_busy, tts_queue, tts_busy_lock
    global tts_enabled_state, tts_initialized_successfully

    if not tts_enabled_state or not tts_initialized_successfully:
        tts_sentence_buffer = ""
        return

    with tts_busy_lock:
        if tts_busy:
            # print("[TTS Flush] Skipping flush, TTS busy.") # DEBUG
            return

    if not tts_sentence_buffer or tts_sentence_buffer.isspace():
        return

    # print(f"[TTS Flush] Checking buffer: '{tts_sentence_buffer[:80]}...'") # DEBUG
    last_processed_pos = 0
    chunk_to_speak = ""

    # Iterate through matches of sentence endings
    for match in sentence_end_pattern.finditer(tts_sentence_buffer):
        end_pos = match.end()
        # Extract the sentence including the delimiter and trailing space/newline
        sentence = tts_sentence_buffer[last_processed_pos:end_pos].strip()
        if sentence:
            # print(f"[TTS Flush] Found sentence: '{sentence}'") # DEBUG
            chunk_to_speak += sentence + " " # Add space between sentences
            last_processed_pos = end_pos
        else: # Handle cases like multiple punctuation marks together
            last_processed_pos = end_pos


    # If we found sentences, queue the combined chunk
    if chunk_to_speak:
        chunk_to_speak = chunk_to_speak.strip() # Remove trailing space
        print(f"[TTS Flush] Queuing chunk: '{chunk_to_speak[:80]}...' ({len(chunk_to_speak)} chars)")
        tts_queue.put(chunk_to_speak)
        # Update the buffer with the remaining part
        tts_sentence_buffer = tts_sentence_buffer[last_processed_pos:]
        # print(f"[TTS Flush] Remaining buffer: '{tts_sentence_buffer[:80]}...'") # DEBUG
    # else: # No complete sentence found in the buffer yet
        # print("[TTS Flush] No complete sentence found.") # DEBUG

def periodic_tts_check():
    """Periodically checks if TTS buffer can be flushed."""
    global periodic_tts_timer, flask_server_running
    try:
        if flask_server_running: # Only run if server is active
             try_flush_tts_buffer()
    except Exception as e:
        print(f"[TTS Check] Error during periodic flush: {e}")
        traceback.print_exc()
    finally:
        # Reschedule the timer only if the server is still running
        if flask_server_running:
            periodic_tts_timer = threading.Timer(0.25, periodic_tts_check) # Increased interval slightly
            periodic_tts_timer.daemon = True
            periodic_tts_timer.start()

# ============================================================================
# Whisper & VAD Logic (Mostly unchanged)
# ============================================================================
# ... (Initialize, Setters, Workers for VAD/Whisper remain the same as previous good version) ...
# --- Make sure all global variable accesses within these functions are correct ---
# --- and that update_vad_status is called appropriately ---
def initialize_whisper():
    """Initializes the Whisper model."""
    global whisper_model, whisper_initialized, whisper_model_size
    if whisper_initialized: return True

    print(f"[Whisper] Attempting initialization (Model: {whisper_model_size})...")
    update_vad_status(f"Loading Whisper ({whisper_model_size})...", "blue")

    # Environment setup for Numba cache (helps Silero VAD sometimes)
    os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
    try:
        numba_cache_dir = os.path.join(tempfile.gettempdir(), 'numba_cache')
        os.makedirs(numba_cache_dir, exist_ok=True)
        os.environ['NUMBA_CACHE_DIR'] = numba_cache_dir
    except Exception: pass # Ignore if fails

    try:
        model_load_start = time.time()
        if whisper_model_size.startswith("turbo"):
            # FasterWhisper
            whisper_turbo_model_name = whisper_model_size.split("-", 1)[1]
            device = "cuda" if torch.cuda.is_available() else "cpu"
            # Determine best compute type based on GPU capability if using CUDA
            compute_type = "float16" if device == "cuda" and torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 7 else "int8"
            print(f"[Whisper] Using FasterWhisper: model={whisper_turbo_model_name}, device={device}, compute_type={compute_type}")
            whisper_model = faster_whisper.WhisperModel(whisper_turbo_model_name, device=device, compute_type=compute_type)
        else:
            # OpenAI Whisper
            print(f"[Whisper] Using OpenAI Whisper: model={whisper_model_size}")
            whisper_model = whisper.load_model(whisper_model_size)

        whisper_initialized = True
        print(f"[Whisper] Model initialization successful in {time.time() - model_load_start:.2f}s.")
        if vad_initialized or not voice_enabled_state:
             final_status_text = "Voice Ready" if vad_initialized else f"Whisper ({whisper_model_size}) OK"
             update_vad_status(final_status_text, "green")
        else:
             update_vad_status(f"Whisper OK, VAD...", "blue")
        return True

    except ImportError as ie:
         print(f"[Whisper] Import Error: {ie}. Is FasterWhisper installed (`pip install faster-whisper`)?")
         whisper_initialized = False; whisper_model = None
         update_vad_status("Whisper Import Error!", "red")
         send_error_to_web(f"Whisper Import Error: {ie}. Install required libraries.")
         return False
    except Exception as e:
        print(f"[Whisper] Error initializing model: {e}")
        traceback.print_exc()
        whisper_initialized = False; whisper_model = None
        update_vad_status("Whisper Init Failed!", "red")
        send_error_to_web(f"Failed to load Whisper {whisper_model_size} model: {e}")
        return False


def initialize_vad():
    """Initializes the Silero VAD model."""
    global vad_model, vad_utils, vad_get_speech_ts, vad_initialized
    if vad_initialized: return True

    print("[VAD] Attempting initialization...")
    update_vad_status("Loading VAD model...", "blue")
    try:
        # Set hub dir within try-except
        try:
             torch_hub_dir = os.path.join(tempfile.gettempdir(), "torch_hub_silero")
             os.makedirs(torch_hub_dir, exist_ok=True)
             torch.hub.set_dir(torch_hub_dir)
        except Exception: pass # Ignore failures

        # Load VAD model
        vad_model_obj, vad_utils_funcs = torch.hub.load(repo_or_dir='snakers4/silero-vad',
                                                  model='silero_vad',
                                                  force_reload=False,
                                                  onnx=False, # Set True if ONNX preferred and installed
                                                  trust_repo=True)
        vad_model = vad_model_obj # Assign to global
        vad_utils = vad_utils_funcs # Assign utils
        # Extract specific function needed (newer versions might return VADIterator)
        if hasattr(vad_utils, 'get_speech_ts'):
             vad_get_speech_ts = vad_utils.get_speech_ts # For older interface
             print("[VAD] Using 'get_speech_ts' method.")
        elif callable(vad_model):
             print("[VAD] Using model directly for probability.")
        else:
            print("[VAD] Warning: Unable to determine VAD processing method from loaded objects.")
            raise RuntimeError("Silero VAD loaded in an unexpected format.")


        vad_initialized = True
        print("[VAD] Model initialized successfully.")
        if whisper_initialized or not voice_enabled_state:
             final_status_text = "Voice Ready" if whisper_initialized else "VAD OK"
             update_vad_status(final_status_text, "green")
        else:
             update_vad_status("VAD OK, Whisper...", "blue")
        return True
    except Exception as e:
        print(f"[VAD] Error initializing model: {e}")
        traceback.print_exc()
        vad_initialized = False; vad_model = None
        update_vad_status("VAD Init Failed!", "red")
        send_error_to_web(f"Failed to load Silero VAD model: {e}")
        return False

def initialize_audio_system():
    """Initializes PyAudio."""
    global py_audio
    if py_audio: return True
    try:
        print("[Audio] Initializing PyAudio...")
        py_audio = pyaudio.PyAudio()
        print("[Audio] PyAudio initialized.")
        # Check for input devices
        found_input = False
        for i in range(py_audio.get_device_count()):
             dev_info = py_audio.get_device_info_by_index(i)
             if dev_info.get('maxInputChannels') > 0:
                  print(f"  - Found Input Device {i}: {dev_info.get('name')}")
                  found_input = True
        if not found_input:
            print("[Audio] Warning: No input devices found by PyAudio!")
            send_error_to_web("No audio input devices detected. Check microphone connection/permissions.")
            # Don't return False here, VAD worker will handle stream opening error
        return True
    except Exception as e:
        print(f"[Audio] Error initializing PyAudio: {e}")
        traceback.print_exc()
        send_error_to_web(f"Failed to initialize audio system: {e}. Check microphone permissions and drivers.")
        py_audio = None
        return False

def vad_worker():
    """Worker thread for continuous VAD and triggering recording."""
    global py_audio, vad_audio_buffer, audio_frames_buffer, is_recording_for_whisper
    global vad_model, vad_get_speech_ts, vad_stop_event, temp_audio_file_path, whisper_queue
    global tts_busy, tts_busy_lock, voice_enabled_state, vad_utils, flask_server_running

    print("[VAD Worker] Thread started.")
    stream = None
    try:
        if not py_audio:
            print("[VAD Worker] PyAudio not initialized. Exiting."); update_vad_status("Audio System Error!", "red"); return
        try:
            stream = py_audio.open(format=FORMAT, channels=CHANNELS, rate=RATE,
                                   input=True, frames_per_buffer=VAD_CHUNK)
            print("[VAD Worker] Audio stream opened.")
        except Exception as e:
            print(f"[VAD Worker] Failed to open audio stream: {e}"); update_vad_status("Audio Input Error!", "red")
            send_error_to_web(f"Failed to open audio stream ({e}). Check Mic access."); return

        update_vad_status("Listening...", "gray")

        frames_since_last_speech = 0
        silence_frame_limit = int(SILENCE_THRESHOLD_SECONDS * RATE / VAD_CHUNK)
        min_speech_frames = int(MIN_SPEECH_DURATION_SECONDS * RATE / VAD_CHUNK)
        pre_speech_buffer_frames = int(PRE_SPEECH_BUFFER_SECONDS * RATE / VAD_CHUNK)
        temp_pre_speech_buffer = deque(maxlen=pre_speech_buffer_frames)
        was_tts_busy = False

        while flask_server_running and not vad_stop_event.is_set():
            if not voice_enabled_state: print("[VAD Worker] Voice disabled state detected."); break
            try:
                data = stream.read(VAD_CHUNK, exception_on_overflow=False)
                with tts_busy_lock: current_tts_busy = tts_busy

                if current_tts_busy:
                    if not was_tts_busy: update_vad_status("VAD Paused (TTS)", "blue"); was_tts_busy = True
                    if is_recording_for_whisper:
                        is_recording_for_whisper = False; audio_frames_buffer.clear(); temp_pre_speech_buffer.clear()
                    time.sleep(0.05); continue
                elif was_tts_busy:
                     update_vad_status("Listening...", "gray"); was_tts_busy = False; frames_since_last_speech = 0; time.sleep(0.1); continue

                audio_chunk_np = np.frombuffer(data, dtype=np.int16)
                temp_pre_speech_buffer.append(data)
                audio_float32 = audio_chunk_np.astype(np.float32) / 32768.0
                audio_tensor = torch.from_numpy(audio_float32)
                is_speech = False
                try:
                    if callable(vad_model):
                        speech_prob = vad_model(audio_tensor, RATE).item(); is_speech = speech_prob > 0.45
                    else: print("[VAD Worker] Error: VAD model not callable.")
                except Exception as vad_err: print(f"[VAD Worker] Error during VAD inference: {vad_err}"); is_speech = False

                if is_speech:
                    frames_since_last_speech = 0
                    if not is_recording_for_whisper:
                        is_recording_for_whisper = True; audio_frames_buffer.clear(); audio_frames_buffer.extend(temp_pre_speech_buffer); update_vad_status("Recording...", "red")
                    audio_frames_buffer.append(data)
                else:
                    frames_since_last_speech += 1
                    if is_recording_for_whisper:
                        audio_frames_buffer.append(data)
                        if frames_since_last_speech > silence_frame_limit:
                            is_recording_for_whisper = False
                            total_frames = len(audio_frames_buffer); recording_duration = total_frames * VAD_CHUNK / RATE
                            effective_speech_frames = max(0, total_frames - frames_since_last_speech)
                            buffer_copy = list(audio_frames_buffer) # Copy buffer before clearing
                            audio_frames_buffer.clear() # Clear buffer immediately

                            if effective_speech_frames < min_speech_frames:
                                update_vad_status("Too short", "orange")
                            else:
                                try:
                                    temp_file = tempfile.NamedTemporaryFile(prefix="vad_rec_", suffix=".wav", delete=False)
                                    temp_audio_file_path = temp_file.name; temp_file.close()
                                    wf = wave.open(temp_audio_file_path, 'wb')
                                    wf.setnchannels(CHANNELS); wf.setsampwidth(py_audio.get_sample_size(FORMAT)); wf.setframerate(RATE)
                                    wf.writeframes(b''.join(buffer_copy)); wf.close() # Use the copied buffer
                                    print(f"[VAD Worker] Audio saved to {temp_audio_file_path} ({recording_duration:.2f}s)")
                                    whisper_queue.put(temp_audio_file_path); update_vad_status("Processing...", "blue")
                                except Exception as save_err: print(f"[VAD Worker] Error saving audio: {save_err}"); update_vad_status("File Save Error", "red")

                            # Reset to listening state after brief delay showing status
                            status_text = get_vad_status()["text"]
                            delay = 0.8 if status_text == "Too short" else (0.1 if status_text == "Processing..." else 0.5)
                            threading.Timer(delay, lambda: update_vad_status("Listening...", "gray") if voice_enabled_state and not is_recording_for_whisper else None).start()


                    elif not was_tts_busy and get_vad_status()["text"] not in ["Listening...", "Processing...", "Too short"]:
                          update_vad_status("Listening...", "gray")

            except IOError as e:
                if e.errno == pyaudio.paInputOverflowed: print("[VAD Worker] Warning: Input overflowed.")
                else: print(f"[VAD Worker] Stream Read/IO Error: {e}"); update_vad_status("Audio Stream Error!", "red"); send_error_to_web(f"Audio read error: {e}"); vad_stop_event.set(); break
            except Exception as e: print(f"[VAD Worker] Unexpected error in loop: {e}"); traceback.print_exc(); time.sleep(0.1)

    except Exception as e: print(f"[VAD Worker] Error during setup or loop: {e}"); traceback.print_exc(); update_vad_status("VAD Setup Error", "red")
    finally:
        print("[VAD Worker] Cleaning up...");
        if stream:
            try:
                if stream.is_active(): stream.stop_stream(); stream.close(); print("[VAD Worker] Audio stream closed.")
            except Exception as e: print(f"[VAD Worker] Error closing stream: {e}")
        is_recording_for_whisper = False; audio_frames_buffer.clear(); vad_audio_buffer.clear(); temp_pre_speech_buffer.clear()
        current_status = get_vad_status()["text"]
        if not flask_server_running: update_vad_status("Server Stopping", "grey") # Indicate shutdown
        elif vad_stop_event.is_set() and "Error" not in current_status and "Failed" not in current_status: update_vad_status("Voice Disabled", "grey")
        elif not voice_enabled_state and current_status != "Voice Disabled": update_vad_status("Voice Disabled", "grey")
        elif "Error" not in current_status and "Failed" not in current_status: update_vad_status("VAD Stopped (Error)", "red")
    print("[VAD Worker] Thread finished.")


def process_audio_worker():
    """Worker thread to transcribe audio files from the whisper_queue."""
    global whisper_model, whisper_initialized, whisper_queue, whisper_language_state
    global whisper_model_size, voice_enabled_state, flask_server_running
    print("[Whisper Worker] Thread started.")
    while flask_server_running:
        audio_file_path = None
        try:
            audio_file_path = whisper_queue.get(timeout=0.5) # Use timeout
            if audio_file_path is None: print("[Whisper Worker] Received stop signal (None)."); break

            if not whisper_initialized or not voice_enabled_state:
                whisper_queue.task_done()
                if audio_file_path and os.path.exists(audio_file_path):
                     try: os.unlink(audio_file_path)
                     except Exception: pass
                if voice_enabled_state and vad_initialized and not is_recording_for_whisper and get_vad_status()["text"] not in ["Listening...", "Recording..."]:
                     update_vad_status("Listening...", "gray")
                continue

            print(f"[Whisper Worker] Processing: {os.path.basename(audio_file_path)}")
            update_vad_status("Transcribing...", "orange")
            start_time = time.time()
            transcribed_text = ""; detected_language = "??"

            try:
                lang_to_use = whisper_language_state
                if isinstance(whisper_model, faster_whisper.WhisperModel):
                    segments, info = whisper_model.transcribe(audio_file_path, language=lang_to_use, beam_size=5, vad_filter=True, vad_parameters=dict(threshold=0.5))
                    transcribed_text = " ".join([segment.text for segment in segments]).strip()
                    detected_language = info.language if hasattr(info, 'language') else '??'
                elif isinstance(whisper_model, whisper.Whisper):
                    result = whisper_model.transcribe(audio_file_path, language=lang_to_use)
                    transcribed_text = result["text"].strip(); detected_language = result.get("language", "??")
                else: raise TypeError("Unsupported Whisper model object")

                duration = time.time() - start_time
                print(f"[Whisper Worker] Transcription ({detected_language}, {duration:.2f}s): '{transcribed_text}'")

                if transcribed_text:
                    web_output_queue.put({"type": "transcription", "data": {"text": transcribed_text, "language": detected_language}})
                    update_vad_status("Transcription Ready", "green")
                else: update_vad_status("No speech text", "orange")

                # Reset to Listening after a short delay
                delay = 1.2 if transcribed_text else 0.8
                threading.Timer(delay, lambda: update_vad_status("Listening...", "gray") if voice_enabled_state and not is_recording_for_whisper else None).start()

            except Exception as e:
                print(f"[Whisper Worker] Transcription Error: {e}"); traceback.print_exc()
                update_vad_status("Transcribe Error", "red"); send_error_to_web(f"Transcription failed: {e}")
                threading.Timer(1.5, lambda: update_vad_status("Listening...", "gray") if voice_enabled_state and not is_recording_for_whisper else None).start()

            finally:
                if audio_file_path and os.path.exists(audio_file_path):
                    try: os.unlink(audio_file_path)
                    except Exception as e: print(f"[Whisper Worker] Error deleting temp file {audio_file_path}: {e}")
                whisper_queue.task_done()

        except queue.Empty: continue # Timeout, check flask_server_running flag
        except Exception as e:
            print(f"[Whisper Worker] Error in main loop: {e}"); traceback.print_exc()
            if audio_file_path: # Ensure task done if error after get
                 try: whisper_queue.task_done()
                 except ValueError: pass
    print("[Whisper Worker] Thread finished.")

def set_whisper_language(lang_code):
    """Sets the language for Whisper transcription via API."""
    global whisper_language_state
    lang_name = "Unknown Code"; valid_code = False
    for name, code in WHISPER_LANGUAGES.items():
        if code == lang_code: lang_name = name; valid_code = True; break
    if valid_code:
        whisper_language_state = lang_code; print(f"[Whisper] Language set to: {lang_name} (Code: {whisper_language_state})")
        send_status_to_web(f"Whisper language: {lang_name}"); return True
    else: send_error_to_web(f"Invalid Whisper language code: {lang_code}"); return False

def set_whisper_model(size):
    """Sets Whisper model size and triggers re-initialization if needed via API."""
    global whisper_model_size, whisper_initialized, whisper_model
    if size not in WHISPER_MODEL_SIZES: send_error_to_web(f"Invalid Whisper model size: {size}"); return False
    if size == whisper_model_size and whisper_initialized: return True

    print(f"[Whisper] Model size change requested to: {size}. Re-initialization required."); send_status_to_web(f"Whisper model changing to {size}...")
    whisper_model_size = size; whisper_initialized = False
    if whisper_model:
        print("[Whisper] Releasing old model object...");
        try: del whisper_model; torch.cuda.empty_cache() if torch.cuda.is_available() else None
        except Exception as del_err: print(f"[Whisper] Error during old model cleanup: {del_err}")
        whisper_model = None
    if voice_enabled_state: threading.Thread(target=initialize_whisper, daemon=True).start()
    else: update_vad_status("Voice Disabled", "grey")
    return True

def toggle_voice_recognition(enable):
    """Enables/disables VAD and Whisper via API."""
    global voice_enabled_state, whisper_initialized, vad_initialized, vad_thread, vad_stop_event, py_audio, whisper_processing_thread

    if enable:
        if voice_enabled_state and vad_thread and vad_thread.is_alive(): return True
        print("[Voice] Enabling voice recognition by API request..."); voice_enabled_state = True; update_vad_status("Initializing...", "blue")
        def init_and_start_vad_sequence():
            global voice_enabled_state, py_audio, whisper_initialized, vad_initialized, whisper_processing_thread, vad_thread, vad_stop_event
            all_initialized = True
            if not py_audio: all_initialized = initialize_audio_system()
            if all_initialized and not whisper_initialized: all_initialized = initialize_whisper()
            if all_initialized and not vad_initialized: all_initialized = initialize_vad()
            if not voice_enabled_state: print("[Voice Init Thread] Voice disabled during init."); return
            if all_initialized:
                if whisper_processing_thread is None or not whisper_processing_thread.is_alive():
                    print("[Voice Init Thread] Starting Whisper thread..."); whisper_processing_thread = threading.Thread(target=process_audio_worker, daemon=True); whisper_processing_thread.start()
                if vad_thread is None or not vad_thread.is_alive():
                    print("[Voice Init Thread] Starting VAD thread..."); vad_stop_event.clear(); vad_thread = threading.Thread(target=vad_worker, daemon=True); vad_thread.start()
                else: # Ensure status is correct if already running
                     if get_vad_status()["text"] in ["Voice Disabled", "Initializing..."]: update_vad_status("Listening...", "gray")
                print("[Voice Init Thread] Voice enabled successfully."); send_status_to_web("Voice Input Enabled.")
            else:
                print("[Voice Init Thread] Enabling failed due to init errors."); voice_enabled_state = False; send_error_to_web("Voice Input enabling failed (init error).")
                if "Error" not in get_vad_status()["text"] and "Failed" not in get_vad_status()["text"]: update_vad_status("Init Failed", "red")
        threading.Thread(target=init_and_start_vad_sequence, daemon=True).start(); return True
    else:
        if not voice_enabled_state: return True
        print("[Voice] Disabling voice recognition by API request..."); voice_enabled_state = False; update_vad_status("Disabling...", "grey")
        if vad_thread and vad_thread.is_alive(): print("[Voice] Signaling VAD worker to stop."); vad_stop_event.set()
        else: update_vad_status("Voice Disabled", "grey")
        print("[Voice] Voice disabled command issued."); send_status_to_web("Voice Input Disabled."); return True
# ============================================================================
# File Processing Logic (Unchanged)
# ============================================================================
def extract_pdf_content(pdf_path):
    """Extracts text content from a PDF file, handles errors and truncation."""
    try:
        doc = fitz.open(pdf_path)
        text_content = ""
        metadata = doc.metadata or {}
        title = metadata.get('title', 'N/A')
        author = metadata.get('author', 'N/A')
        if title != 'N/A' or author != 'N/A':
             text_content += f"PDF Info: Title='{title}', Author='{author}'\n\n"

        num_pages = doc.page_count
        max_chars_per_pdf = 30000
        current_chars = len(text_content)
        truncated = False

        for page_num in range(num_pages):
            page_header = f"--- Page {page_num+1} ---\n"
            page_content = ""
            try:
                 page = doc.load_page(page_num)
                 page_content = page.get_text("text", sort=True).strip()
            except Exception as page_err:
                 print(f"[PDF Extract] Error processing page {page_num+1}: {page_err}")
                 page_content = "[Error extracting text from this page]"

            if not page_content: continue # Skip empty pages

            page_text_len = len(page_header) + len(page_content) + 2

            if current_chars + page_text_len > max_chars_per_pdf:
                 remaining_chars = max_chars_per_pdf - current_chars - len(page_header) - 20
                 if remaining_chars > 50:
                     text_content += page_header + page_content[:remaining_chars] + "...[Page Truncated]\n\n"
                 truncated = True
                 print(f"[PDF Extract] Truncating PDF content at page {page_num+1} due to size limit.")
                 break
            else:
                 text_content += page_header + page_content + "\n\n"
                 current_chars += page_text_len

        doc.close()
        if truncated:
             text_content += "\n[--- PDF content truncated due to length limit ---]"
        return text_content.strip()
    except Exception as e:
        print(f"[PDF Extract] Error opening or processing PDF '{os.path.basename(pdf_path)}': {e}")
        traceback.print_exc()
        return f"[Error extracting content from PDF: {str(e)}]"

def process_uploaded_file(file_path):
    """
    Processes the uploaded file saved at file_path.
    Updates global state (selected_file_path, selected_file_type).
    Returns extracted text content for PDF/Text, or None for images/errors.
    Deletes the file if unsupported.
    """
    global selected_file_path, selected_file_type
    file_name = os.path.basename(file_path)
    file_ext = os.path.splitext(file_path)[1].lower()
    content_for_input = None
    processed_ok = False

    print(f"[File Process] Processing uploaded file: {file_name}")
    mtype, _ = mimetypes.guess_type(file_path)
    guessed_type = mtype.split('/')[0] if mtype else 'unknown'

    if guessed_type == 'image' or file_ext in ['.png', '.jpg', '.jpeg', '.gif', '.bmp']:
        selected_file_path = file_path
        selected_file_type = 'image'
        content_for_input = None # No text content for images
        processed_ok = True
        print(f"[File Process] Type: Image")
        send_status_to_web(f"Image attached: {file_name}")

    elif (guessed_type == 'application' and file_ext == '.pdf'):
        selected_file_path = file_path
        selected_file_type = 'pdf'
        print(f"[File Process] Type: PDF - Extracting content...")
        send_status_to_web(f"Processing PDF: {file_name}...")
        content_for_input = extract_pdf_content(file_path)
        processed_ok = True # Even if extraction fails, we keep the PDF reference
        send_status_to_web(f"PDF processed: {file_name}")

    elif guessed_type == 'text' or file_ext in ['.txt', '.md', '.py', '.js', '.html', '.css', '.json', '.log', '.csv', '.xml', '.yaml', '.ini', '.sh', '.bat']:
        selected_file_path = file_path
        selected_file_type = 'text'
        print(f"[File Process] Type: Text - Reading content...")
        send_status_to_web(f"Reading text file: {file_name}...")
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            max_len = 30000
            if len(content) > max_len:
                content_for_input = content[:max_len] + f"\n\n[--- Text content truncated at {max_len} characters ---]"
                print(f"[File Process] Text file truncated: {file_name}")
            else:
                 content_for_input = content
            processed_ok = True
            send_status_to_web(f"Text file loaded: {file_name}")
        except Exception as e:
            print(f"[File Process] Error reading text file {file_name}: {e}")
            send_error_to_web(f"Error reading text file '{file_name}': {e}")
            content_for_input = f"[Error reading text file: {e}]"
            processed_ok = True # Keep reference even on read error

    else: # Unsupported type
        print(f"[File Process] Unsupported file type: {file_name} (Type: {mtype}, Ext: {file_ext})")
        send_error_to_web(f"Unsupported file type: '{file_name}'. Cannot attach.")
        selected_file_path = None # Clear selection
        selected_file_type = None
        if os.path.exists(file_path): # Clean up the unsupported upload
             try: os.unlink(file_path)
             except: pass
        return None # Indicate failure, return None

    return content_for_input

# ============================================================================
# Ollama Chat Logic (Unchanged)
# ============================================================================
def get_current_chat_history():
    """Safely gets a copy of the chat history, filtering image data for display."""
    with messages_lock:
        history_copy = []
        for msg in messages:
            msg_copy = msg.copy()
            if "images" in msg_copy:
                 img_count = len(msg_copy.get("images", []))
                 placeholder = f" [Image Data ({img_count})]"
                 if "content" in msg_copy and msg_copy["content"]:
                     if placeholder not in msg_copy["content"]: msg_copy["content"] += placeholder
                 else: msg_copy["content"] = placeholder.strip()
                 del msg_copy["images"] # Remove the actual bytes
            history_copy.append(msg_copy)
        return history_copy

def run_chat_interaction(user_message_content, attached_file_info=None):
    """
    Starts the Ollama chat interaction in a background thread.
    Handles message construction, history update, TTS stop, and file state reset.
    """
    global stream_in_progress, selected_file_path, selected_file_type
    global tts_sentence_buffer, tts_queue, tts_busy, tts_busy_lock, tts_engine

    if not ollama_lock.acquire(blocking=False):
        send_error_to_web("Ollama is processing. Please wait."); return
    stream_in_progress = True; print("[Chat Start] Acquired Ollama lock.")

    image_path_to_send = None; history_display_content = user_message_content
    if attached_file_info and attached_file_info.get('type') == 'image':
        image_path_to_send = attached_file_info['path']
        img_placeholder = f" [Image: {attached_file_info['filename']}]"
        if history_display_content: history_display_content += img_placeholder
        else: history_display_content = img_placeholder.strip()
        print(f"[Chat Start] Including image: {attached_file_info['filename']}")

    user_msg_obj = {"role": "user", "content": history_display_content}
    try:
        with messages_lock: messages.append(user_msg_obj)
        web_output_queue.put({"type": "history_update", "data": get_current_chat_history()})
    except Exception as lock_err:
         print(f"[Chat Start] Error adding user message to history: {lock_err}"); send_error_to_web("Failed to update history.")
         stream_in_progress = False; ollama_lock.release(); return

    if tts_engine and tts_enabled_state and tts_initialized_successfully:
        print("[Chat Start] Stopping active TTS...");
        try: tts_engine.stop()
        except: pass
        tts_sentence_buffer = "";
        while not tts_queue.empty():
            try: tts_queue.get_nowait()
            except queue.Empty: break
        with tts_busy_lock: tts_busy = False; print(f"[Chat Start] Cleared TTS Busy Flag") # DEBUG

    print("[Chat Start] Starting Ollama worker thread...")
    thread = threading.Thread(target=chat_worker, args=(user_message_content, image_path_to_send, attached_file_info), daemon=True)
    thread.start()

    selected_file_path = None; selected_file_type = None
    print("[Chat Start] File state cleared, worker thread launched.")


def chat_worker(user_message_content, image_path=None, attached_file_info=None):
    """
    Background worker for Ollama streaming chat. Handles API call and cleanup.
    Releases ollama_lock on completion or error.
    """
    global messages, messages_lock, current_model, web_output_queue, stream_in_progress, ollama_lock

    history_for_ollama = []; image_bytes_for_request = None
    try:
        if image_path:
            print(f"[Chat Worker] Reading image: {image_path}")
            try:
                with open(image_path, 'rb') as f: image_bytes_for_request = f.read()
                print(f"[Chat Worker] Image bytes loaded ({len(image_bytes_for_request)} bytes).")
            except Exception as e:
                 err_text = f"Error reading image file '{os.path.basename(image_path)}': {e}"
                 print(f"[Chat Worker] {err_text}"); send_error_to_web(err_text); stream_in_progress = False; ollama_lock.release(); return

        with messages_lock: history_copy = list(messages)

        if history_copy and history_copy[-1]["role"] == "user":
             last_msg_api = history_copy[-1].copy()
             if image_path and attached_file_info:
                  placeholder = f" [Image: {attached_file_info['filename']}]"
                  if last_msg_api["content"].endswith(placeholder): last_msg_api["content"] = last_msg_api["content"][:-len(placeholder)].rstrip()
             if image_bytes_for_request: last_msg_api["images"] = [image_bytes_for_request]
             history_for_ollama = history_copy[:-1] + [last_msg_api]
        else: print("[Chat Worker] Error: Could not find last user message."); history_for_ollama = history_copy

    except Exception as hist_err:
         err_text = f"Error preparing history for Ollama: {hist_err}"
         print(f"[Chat Worker] {err_text}"); send_error_to_web(err_text); stream_in_progress = False; ollama_lock.release(); return

    assistant_response = ""
    try:
        print(f"[Chat Worker] Sending request to model {current_model}...")
        stream = ollama.chat(model=current_model, messages=history_for_ollama, stream=True)
        first_chunk = True
        for chunk in stream:
            if 'message' in chunk and 'content' in chunk['message']:
                content_piece = chunk['message']['content']
                if first_chunk: web_output_queue.put({"type": "stream_start"}); first_chunk = False
                web_output_queue.put({"type": "stream_chunk", "data": content_piece})
                assistant_response += content_piece; queue_tts_text(content_piece)
            if 'error' in chunk:
                 err_text = f"Ollama stream error: {chunk['error']}"
                 print(f"[Chat Worker] {err_text}"); send_error_to_web(err_text); assistant_response = None; break
            if chunk.get('done'):
                if chunk.get('total_duration'): duration_ms = chunk.get('total_duration') / 1_000_000; print(f"[Chat Worker] Ollama processing complete. Duration: {duration_ms:.2f} ms")
                break

        if assistant_response is not None:
             try:
                 with messages_lock: messages.append({"role": "assistant", "content": assistant_response})
                 web_output_queue.put({"type": "history_update", "data": get_current_chat_history()})
             except Exception as lock_err: print(f"[Chat Worker] Error adding assistant response to history: {lock_err}")
             web_output_queue.put({"type": "stream_end"}); queue_tts_text("\n"); try_flush_tts_buffer()
             print(f"[Chat Worker] Assistant response completed ({len(assistant_response)} chars).")
        else: # Stream error occurred
             web_output_queue.put({"type": "stream_end"}) # Still signal end to unlock UI

    except ollama.ResponseError as ore:
         err_text = f"Ollama API Error: {ore.status_code} - {ore.error}"; print(f"[Chat Worker] {err_text}"); send_error_to_web(err_text)
         web_output_queue.put({"type": "stream_end"})
    except Exception as e:
        err_text = f"Ollama communication error: {e}"; print(f"[Chat Worker] {err_text}"); traceback.print_exc(); send_error_to_web(err_text)
        web_output_queue.put({"type": "stream_end"})
    finally:
        stream_in_progress = False; ollama_lock.release(); print("[Chat Worker] Released Ollama lock. Worker finished.")
        file_to_delete = image_path or (attached_file_info.get('path') if attached_file_info else None)
        if file_to_delete and os.path.exists(file_to_delete) and UPLOAD_FOLDER in os.path.abspath(file_to_delete):
             try: os.unlink(file_to_delete); print(f"[Chat Worker] Cleaned up processed file: {os.path.basename(file_to_delete)}")
             except Exception as del_err: print(f"[Chat Worker] Error deleting processed file {file_to_delete}: {del_err}")

# ============================================================================
# Ollama Model Fetching (More Robust)
# ============================================================================
def fetch_available_models():
    """Fetches available Ollama models and caches them."""
    global ollama_model_list
    try:
        print("[Ollama] Fetching available models...")
        models_data = ollama.list() # ollama-python >= 0.2.0 returns ModelResponse object
        valid_models = []

        # Check for newer ModelResponse format (ollama-python >= 0.2.0)
        if hasattr(models_data, 'models') and isinstance(models_data.models, list):
             for model_obj in models_data.models:
                  # The object within the list might have 'name' or 'model' attribute
                  if hasattr(model_obj, 'name'):
                       valid_models.append(model_obj.name)
                  elif hasattr(model_obj, 'model'): # Fallback for slightly different structure
                       valid_models.append(model_obj.model)
                  else:
                       print(f"[Ollama] Warning: Model object missing 'name'/'model': {model_obj}")

        # Check for older dict format (ollama-python < 0.2.0 ?)
        elif isinstance(models_data, dict) and 'models' in models_data and isinstance(models_data['models'], list):
             for model_dict in models_data['models']:
                  if isinstance(model_dict, dict) and 'name' in model_dict:
                       valid_models.append(model_dict['name'])
                  else:
                       print(f"[Ollama] Warning: Invalid model dict entry: {model_dict}")

        # Check for simple list of strings (unlikely but possible)
        elif isinstance(models_data, list) and all(isinstance(item, str) for item in models_data):
             valid_models = models_data

        else:
             print(f"[Ollama] Warning: Unexpected format from ollama.list(): {type(models_data)}")

        ollama_model_list = sorted(list(set(valid_models))) # Ensure unique and sorted
        print(f"[Ollama] Found models: {ollama_model_list}")

    except AttributeError as ae:
         # Specifically catch AttributeErrors that might occur with version mismatches
         print(f"[Ollama] Error accessing model attributes (likely ollama-python version issue): {ae}")
         send_error_to_web(f"Ollama library error fetching models: {ae}")
    except Exception as e:
        print(f"[Ollama] Error fetching models: {e}. Returning cached or default list.")
        send_error_to_web(f"Could not fetch Ollama models: {e}")

    # Return cached list or default if cache is empty or fetch failed
    if not ollama_model_list:
         print("[Ollama] No models found or fetch failed. Using default list.")
         ollama_model_list = [DEFAULT_OLLAMA_MODEL, "llama3:8b", "phi3:mini"]

    return ollama_model_list


# ============================================================================
# Flask Application Setup (Unchanged)
# ============================================================================
def create_flask_app():
    """Creates and configures the Flask application."""
    app = Flask(__name__, template_folder='templates')
    app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
    app.secret_key = os.urandom(24) # Good practice

    if not os.path.exists(UPLOAD_FOLDER):
        try: os.makedirs(UPLOAD_FOLDER); print(f"Created upload folder: {UPLOAD_FOLDER}")
        except OSError as e: print(f"Error creating upload folder {UPLOAD_FOLDER}: {e}", file=sys.stderr)

    @app.route('/')
    def index(): return render_template('index.html')

    @app.route('/api/send_message', methods=['POST'])
    def api_send_message():
        global selected_file_path, selected_file_type
        if stream_in_progress: return jsonify({"status": "error", "message": "Please wait."}), 429

        user_text = request.form.get('message', '').strip()
        file = request.files.get('file')
        processed_file_info = None

        if file and file.filename:
            if allowed_file(file.filename):
                filename = secure_filename(file.filename)
                temp_save_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{int(time.time())}_{filename}")
                try:
                    file.save(temp_save_path); print(f"[API Send] File '{filename}' uploaded to {temp_save_path}")
                    file_content_for_input = process_uploaded_file(temp_save_path)
                    if selected_file_path is None: return jsonify({"status": "error", "message": f"Unsupported type or processing error: {filename}"}), 400
                    else:
                        processed_file_info = {"path": selected_file_path, "type": selected_file_type, "filename": filename}
                        if file_content_for_input and selected_file_type in ['pdf', 'text']:
                             header = f"--- Content from {selected_file_type.upper()}: {filename} ---\n"
                             footer = f"\n--- End of {filename} Content ---"
                             if user_text: user_text = f"{header}{file_content_for_input}{footer}\n\n{user_text}"
                             else: user_text = f"{header}{file_content_for_input}{footer}"
                             print(f"[API Send] Prepended content from {filename}")
                except Exception as e: print(f"[API Send] Error handling upload '{filename}': {e}"); traceback.print_exc(); return jsonify({"status": "error", "message": f"Error handling file: {e}"}), 500
            else: return jsonify({"status": "error", "message": f"File type not allowed: {file.filename}"}), 400
        else: selected_file_path = None; selected_file_type = None

        if not user_text and (not processed_file_info or processed_file_info['type'] != 'image'):
             return jsonify({"status": "error", "message": "Please enter message or attach file."}), 400

        run_chat_interaction(user_text, processed_file_info)
        return jsonify({"status": "success", "message": "Processing..."})

    @app.route('/api/clear_attachment', methods=['POST'])
    def api_clear_attachment():
        global selected_file_path, selected_file_type
        cleared = False; path_to_delete = selected_file_path
        if path_to_delete:
            print(f"[API Clear] Clearing attachment state for: {path_to_delete}"); selected_file_path = None; selected_file_type = None; cleared = True
            if UPLOAD_FOLDER in os.path.abspath(path_to_delete) and os.path.exists(path_to_delete):
                try: os.unlink(path_to_delete); print(f"[API Clear] Deleted temp file: {os.path.basename(path_to_delete)}")
                except Exception as e: print(f"[API Clear] Error deleting file {path_to_delete}: {e}")
        if cleared: send_status_to_web("Attachment cleared."); return jsonify({"status": "success"})
        else: return jsonify({"status": "no_op"})

    @app.route('/api/history')
    def api_history(): return jsonify({"history": get_current_chat_history()})

    @app.route('/api/stream')
    def api_stream():
        def event_stream():
            try: yield f"event: full_status\ndata: {json.dumps(get_full_backend_status())}\n\n"
            except Exception as init_e: print(f"[SSE Stream] Error sending initial status: {init_e}")
            print("[SSE Stream] Client connected."); queue_timeout = 20
            while True:
                try:
                    data = web_output_queue.get(timeout=queue_timeout)
                    sse_event_name = data.get("type", "message")
                    event_data_payload = data.get("data", data)
                    try: sse_data_json = json.dumps(event_data_payload)
                    except TypeError as json_err: print(f"[SSE Stream] Error JSON encoding data for '{sse_event_name}': {json_err}"); sse_data_json = json.dumps({"error": "Data encoding failed"})
                    yield f"event: {sse_event_name}\ndata: {sse_data_json}\n\n"
                    web_output_queue.task_done()
                except queue.Empty: yield ": keepalive\n\n"
                except GeneratorExit: print("[SSE Stream] Client disconnected."); break
                except Exception as e: print(f"[SSE Stream] Error in loop: {e}"); traceback.print_exc(); time.sleep(1)
        headers = {'Content-Type': 'text/event-stream', 'Cache-Control': 'no-cache', 'Connection': 'keep-alive'}
        return Response(event_stream(), headers=headers)

    @app.route('/api/status')
    def api_status(): return jsonify(get_full_backend_status())

    @app.route('/api/models')
    def api_models(): models = ollama_model_list if ollama_model_list else fetch_available_models(); return jsonify({"models": models})

    @app.route('/api/voices')
    def api_voices(): voices = get_available_voices(); return jsonify({"voices": voices})

    @app.route('/api/control/model', methods=['POST'])
    def api_control_model():
        global current_model; data = request.get_json(); model_name = data.get('model')
        if not model_name: return jsonify({"status": "error", "message": "Missing 'model'."}), 400
        known_models = ollama_model_list if ollama_model_list else fetch_available_models()
        if model_name not in known_models: print(f"[API Control] Warning: Model '{model_name}' not in known list.")
        current_model = model_name; print(f"[API Control] Ollama model set to: {current_model}"); send_status_to_web(f"Ollama model: {current_model}"); return jsonify({"status": "success", "current_model": current_model})

    @app.route('/api/control/tts', methods=['POST'])
    def api_control_tts():
        data = request.get_json(); enable = data.get('enable')
        if enable is None or not isinstance(enable, bool): return jsonify({"status": "error", "message": "Missing/invalid 'enable'."}), 400
        toggle_tts(enable); return jsonify({"status": "success", "tts_enabled": tts_enabled_state})

    @app.route('/api/control/tts_settings', methods=['POST'])
    def api_control_tts_settings():
        data = request.get_json(); rate = data.get('rate'); voice_id = data.get('voice_id'); results = {}; change_made = False
        if rate is not None: results['rate_set'] = set_tts_rate(rate); change_made = True
        if voice_id is not None: results['voice_set'] = set_tts_voice(voice_id); change_made = True
        if not change_made: return jsonify({"status": "error", "message": "No settings provided."}), 400
        return jsonify({"status": "success", "results": results, "current_rate": tts_rate_state, "current_voice_id": tts_voice_id_state})

    @app.route('/api/control/vad', methods=['POST'])
    def api_control_vad():
        data = request.get_json(); enable = data.get('enable')
        if enable is None or not isinstance(enable, bool): return jsonify({"status": "error", "message": "Missing/invalid 'enable'."}), 400
        toggle_voice_recognition(enable); time.sleep(0.05); return jsonify({"status": "success", "vad_enabled": voice_enabled_state, "current_status": get_vad_status()})

    @app.route('/api/control/whisper_settings', methods=['POST'])
    def api_control_whisper_settings():
         data = request.get_json(); lang_code = data.get('language_code'); lang_change_requested = 'language_code' in data
         model_size = data.get('model_size'); results = {}; change_made = False
         if lang_change_requested: results['language_set'] = set_whisper_language(lang_code); change_made = True
         if model_size is not None: results['model_set'] = set_whisper_model(model_size); change_made = True
         if not change_made: return jsonify({"status": "error", "message": "No settings provided."}), 400
         return jsonify({"status": "success", "results": results, "current_language": whisper_language_state, "current_model": whisper_model_size})

    return app

def get_full_backend_status():
     """Helper to gather current state for the status API and SSE pushes."""
     voices = tts_available_voices if tts_initialized_successfully else []
     vad_s = get_vad_status()
     if voice_enabled_state and not (vad_initialized and whisper_initialized):
         if "Init" not in vad_s['text'] and "Error" not in vad_s['text']:
             vad_s = {"text": "Initializing...", "color": "blue"}
     return {
         "ollama_model": current_model, "ollama_models": ollama_model_list,
         "tts_enabled": tts_enabled_state, "tts_rate": tts_rate_state, "tts_voice_id": tts_voice_id_state,
         "tts_voices": voices, "tts_initialized": tts_initialized_successfully,
         "vad_enabled": voice_enabled_state, "vad_status": vad_s,
         "whisper_language": whisper_language_state, "whisper_languages": WHISPER_LANGUAGES,
         "whisper_model": whisper_model_size, "whisper_models": WHISPER_MODEL_SIZES,
         "whisper_initialized": whisper_initialized, "vad_initialized": vad_initialized,
         "attachment": {"filename": os.path.basename(selected_file_path) if selected_file_path else None, "type": selected_file_type}
     }

# ===================
# Main Execution & Lifecycle Management
# ===================
def initialize_backend():
    """Initializes components that need early setup. Runs sequentially."""
    global current_model, periodic_tts_timer, flask_server_running
    print("\n" + "="*30 + " Initializing Backend " + "="*30)
    flask_server_running = True # Set flag
    fetch_available_models()
    if DEFAULT_OLLAMA_MODEL in ollama_model_list: current_model = DEFAULT_OLLAMA_MODEL
    elif ollama_model_list: current_model = ollama_model_list[0]
    else: current_model = DEFAULT_OLLAMA_MODEL
    print(f"[Init] Default Ollama Model set to: {current_model}")

    initialize_audio_system()
    initialize_tts()

    print("[Init] Starting periodic TTS check timer.")
    periodic_tts_timer = threading.Timer(0.5, periodic_tts_check)
    periodic_tts_timer.daemon = True
    periodic_tts_timer.start()

    if tts_enabled_state: start_tts_thread()
    if voice_enabled_state: toggle_voice_recognition(True)
    print("="*30 + " Initialization Complete " + "="*30 + "\n")

def cleanup_on_exit():
    """Gracefully shutdown background threads and resources."""
    global flask_server_running, periodic_tts_timer
    print("\n" + "="*30 + " Starting Graceful Shutdown " + "="*30)
    flask_server_running = False # Signal background threads to stop

    # Cancel the periodic TTS timer if it's running
    if periodic_tts_timer and periodic_tts_timer.is_alive():
        print("[Cleanup] Cancelling periodic TTS timer...")
        periodic_tts_timer.cancel()

    # Stop threads (functions handle sentinels and joins)
    if vad_thread and vad_thread.is_alive(): print("[Cleanup] Stopping VAD thread..."); vad_stop_event.set()
    if whisper_processing_thread and whisper_processing_thread.is_alive(): print("[Cleanup] Stopping Whisper thread..."); whisper_queue.put(None)
    stop_tts_thread() # Handles TTS stop and join

    # --- Join remaining threads (VAD/Whisper) ---
    join_timeout = 2.0
    threads_to_join = [vad_thread, whisper_processing_thread]
    thread_names = ["VAD", "Whisper"]
    for i, t in enumerate(threads_to_join):
        if t and t.is_alive():
            print(f"[Cleanup] Waiting for {thread_names[i]} thread to join...")
            t.join(timeout=join_timeout)
            if t.is_alive(): print(f"[Cleanup] Warning: {thread_names[i]} thread did not exit cleanly.")
            else: print(f"[Cleanup] {thread_names[i]} thread joined.")

    # Terminate PyAudio
    if py_audio:
        print("[Cleanup] Terminating PyAudio...");
        try: py_audio.terminate()
        except Exception as pa_err: print(f"[Cleanup] PyAudio termination error: {pa_err}")

    # Clean up temporary files
    print("[Cleanup] Cleaning up temporary files..."); deleted_count = 0
    if os.path.exists(UPLOAD_FOLDER):
         for filename in os.listdir(UPLOAD_FOLDER):
             file_path = os.path.join(UPLOAD_FOLDER, filename)
             try: os.unlink(file_path); deleted_count += 1
             except Exception as e: print(f"[Cleanup] Error deleting upload {filename}: {e}")
    global temp_audio_file_path
    if temp_audio_file_path and os.path.exists(temp_audio_file_path):
        try: os.unlink(temp_audio_file_path); deleted_count += 1
        except Exception as e: print(f"[Cleanup] Error deleting temp VAD file: {e}")
    print(f"[Cleanup] Deleted {deleted_count} temporary files.")

    print("="*30 + " Shutdown Complete " + "="*30 + "\n")

if __name__ == '__main__':
    initialize_backend()
    flask_app = create_flask_app()
    atexit.register(cleanup_on_exit)

    print(f"\n[Flask] Starting server on http://{FLASK_HOST}:{FLASK_PORT}")
    print(f"[Flask] Access UI: http://127.0.0.1:{FLASK_PORT} or http://<YOUR_LOCAL_IP>:{FLASK_PORT}")
    print("[Flask] Press Ctrl+C to quit.")

    try:
        # Use waitress or another production WSGI server for better performance/stability
        # from waitress import serve
        # serve(flask_app, host=FLASK_HOST, port=FLASK_PORT)
        # Using Flask's built-in server for simplicity in this example:
        flask_app.run(host=FLASK_HOST, port=FLASK_PORT, threaded=True, debug=False, use_reloader=False)
    except KeyboardInterrupt:
        print("\n[Flask] Ctrl+C received. Initiating shutdown...")
        # Cleanup is handled by atexit
    except Exception as e:
        print(f"\n[Flask] Server exited with error: {e}")
        traceback.print_exc()
        # Cleanup should still be attempted by atexit