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
DEFAULT_WHISPER_MODEL_SIZE = "turbo-large" # Defaulting to medium for better balance

# --- Whisper ---
# Keep language codes, names are for UI display (handled in frontend if needed)
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
selected_file_path = None # Store path of the currently attached file
selected_file_type = None # Store type ('image', 'pdf', 'text')
file_processed_for_input = False # Flag if PDF/Text was loaded into input already
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
    # print(f"[Status Update] VAD: {text} ({color})") # Debug VAD status changes
    with vad_status_lock:
        if vad_status["text"] != text or vad_status["color"] != color:
            vad_status = {"text": text, "color": color}
            # Push update to web clients via SSE queue
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
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# ============================================================================
# TTS Logic (Mostly unchanged, interacts with state variables)
# ============================================================================

def initialize_tts():
    """Initializes the TTS engine. Returns True on success."""
    global tts_engine, tts_initialized_successfully, tts_rate_state, tts_voice_id_state, tts_available_voices

    with tts_init_lock:
        if tts_initialized_successfully:
            return True # Already initialized successfully

        if tts_engine: # Cleanup old engine if retry needed
             try:
                  del tts_engine
                  tts_engine = None
             except: pass

        try:
            print("[TTS] Initializing engine...")
            tts_engine = pyttsx3.init()
            if not tts_engine: raise Exception("pyttsx3.init() returned None")

            # Set initial properties from state variables
            tts_engine.setProperty('rate', tts_rate_state)
            tts_engine.setProperty('volume', 0.9)
            if tts_voice_id_state:
                tts_engine.setProperty('voice', tts_voice_id_state)

            # Get and store available voices
            voices = tts_engine.getProperty('voices')
            tts_available_voices = [{"id": v.id, "name": v.name} for v in voices]
            if not tts_voice_id_state and tts_available_voices: # Set a default voice if none was set
                tts_voice_id_state = tts_available_voices[0]['id']
                tts_engine.setProperty('voice', tts_voice_id_state)

            tts_initialized_successfully = True
            print(f"[TTS] Engine initialized successfully. Default Voice: {tts_voice_id_state}")
            send_status_to_web("TTS Engine Ready.")
            # Send updated voice list via SSE
            web_output_queue.put({"type": "voices_update", "data": tts_available_voices})
            return True
        except Exception as e:
            print(f"[TTS] Error initializing engine: {e}")
            traceback.print_exc()
            tts_engine = None
            tts_initialized_successfully = False
            tts_available_voices = []
            send_error_to_web(f"TTS Initialization Failed: {e}")
            return False

def get_available_voices():
    """Returns the cached list of available TTS voices."""
    global tts_available_voices
    if not tts_initialized_successfully:
        initialize_tts() # Attempt to initialize if not already done
    return tts_available_voices

def set_tts_voice(voice_id):
    """Sets the selected voice ID for TTS."""
    global tts_voice_id_state, tts_engine, tts_initialized_successfully
    if not tts_initialized_successfully:
        send_error_to_web("TTS not initialized, cannot set voice.")
        return False
    try:
        # Validate if voice_id exists (optional but good practice)
        if any(v['id'] == voice_id for v in tts_available_voices):
            tts_voice_id_state = voice_id
            # Set property on engine if it exists
            if tts_engine:
                tts_engine.setProperty('voice', voice_id)
            print(f"[TTS] Voice set to ID: {voice_id}")
            return True
        else:
            print(f"[TTS] Error: Voice ID '{voice_id}' not found.")
            send_error_to_web(f"Invalid TTS Voice ID: {voice_id}")
            return False
    except Exception as e:
        print(f"[TTS] Error setting voice: {e}")
        send_error_to_web(f"Error setting TTS voice: {e}")
        return False

def set_tts_rate(rate):
    """Sets the speech rate for TTS."""
    global tts_rate_state, tts_engine, tts_initialized_successfully
    if not tts_initialized_successfully:
        send_error_to_web("TTS not initialized, cannot set rate.")
        return False
    try:
        rate = int(rate)
        if 80 <= rate <= 400: # Basic range validation
            tts_rate_state = rate
            if tts_engine:
                tts_engine.setProperty('rate', rate)
            # print(f"[TTS] Rate set to: {rate}") # Can be noisy
            return True
        else:
            print(f"[TTS] Error: Rate {rate} out of range (80-400).")
            send_error_to_web(f"Invalid TTS Rate: {rate}. Must be between 80 and 400.")
            return False
    except Exception as e:
        print(f"[TTS] Error setting rate: {e}")
        send_error_to_web(f"Error setting TTS rate: {e}")
        return False

def tts_worker():
    """Worker thread processing the TTS queue."""
    global tts_engine, tts_queue, tts_busy, tts_initialized_successfully, tts_enabled_state
    global tts_voice_id_state, tts_rate_state
    print("[TTS Worker] Thread started.")
    while True:
        try:
            text_to_speak = tts_queue.get()
            if text_to_speak is None:
                print("[TTS Worker] Received stop signal.")
                break

            if tts_engine and tts_enabled_state and tts_initialized_successfully:
                with tts_busy_lock:
                    tts_busy = True

                try:
                    # Ensure properties are set correctly before speaking
                    tts_engine.setProperty('voice', tts_voice_id_state)
                    tts_engine.setProperty('rate', tts_rate_state)

                    # print(f"[TTS Worker] Speaking chunk ({len(text_to_speak)} chars)...")
                    tts_engine.say(text_to_speak)
                    tts_engine.runAndWait()
                    # print(f"[TTS Worker] Finished speaking chunk.")

                except Exception as speak_err:
                    print(f"[TTS Worker] Error during say/runAndWait: {speak_err}")
                    # Attempt to recover or just log
                finally:
                    with tts_busy_lock:
                        tts_busy = False
            else:
                # print("[TTS Worker] TTS disabled or uninitialized, discarding text.")
                pass

            tts_queue.task_done()
        except RuntimeError as rt_err:
            if "run loop already started" in str(rt_err):
                print("[TTS Worker] Warning: run loop already started error.")
                # This might require engine re-initialization in severe cases
                with tts_busy_lock: tts_busy = False
            else:
                print(f"[TTS Worker] Runtime Error: {rt_err}")
                traceback.print_exc()
                with tts_busy_lock: tts_busy = False
        except Exception as e:
            print(f"[TTS Worker] Unexpected Error: {e}")
            traceback.print_exc()
            with tts_busy_lock: tts_busy = False
            time.sleep(0.5) # Avoid rapid error loops

    print("[TTS Worker] Thread finished.")


def start_tts_thread():
    """Starts the TTS worker thread if needed."""
    global tts_thread, tts_initialized_successfully
    if tts_thread is None or not tts_thread.is_alive():
        # Try to initialize TTS if it hasn't succeeded yet
        if not tts_initialized_successfully:
            initialize_tts()

        # Start thread only if initialization was successful
        if tts_initialized_successfully:
            tts_thread = threading.Thread(target=tts_worker, daemon=True)
            tts_thread.start()
            print("[TTS] Worker thread started.")
        else:
            print("[TTS] Engine init failed previously. Cannot start TTS thread.")


def stop_tts_thread():
    """Signals the TTS worker thread to stop and cleans up."""
    global tts_thread, tts_engine, tts_queue, tts_busy, tts_busy_lock
    print("[TTS] Stopping worker thread...")
    if tts_engine and tts_initialized_successfully:
        try:
            tts_engine.stop()
            print("[TTS] Engine stop requested.")
        except Exception as e:
             print(f"[TTS] Error stopping engine: {e}")

    # Clear the queue
    while not tts_queue.empty():
        try: tts_queue.get_nowait()
        except queue.Empty: break
    print("[TTS] Queue cleared.")

    if tts_thread and tts_thread.is_alive():
        tts_queue.put(None) # Send sentinel
        print("[TTS] Waiting for worker thread to join...")
        tts_thread.join(timeout=2.5)
        if tts_thread.is_alive():
            print("[TTS] Warning: Worker thread did not terminate gracefully.")
        else:
             print("[TTS] Worker thread joined.")
    tts_thread = None
    with tts_busy_lock: tts_busy = False
    print("[TTS] Worker thread stopped.")

def toggle_tts(enable):
    """Handles enabling/disabling TTS via API."""
    global tts_enabled_state, tts_sentence_buffer
    global tts_initialized_successfully # Check initialization status

    if enable:
        if not tts_initialized_successfully:
             print("[TTS Toggle] Attempting to initialize TTS on enable...")
             initialize_tts() # Try again

        if tts_initialized_successfully:
            tts_enabled_state = True
            print("[TTS] Enabled by API request.")
            start_tts_thread() # Ensure worker is running
            send_status_to_web("TTS Enabled.")
            return True
        else:
            tts_enabled_state = False # Ensure state is false if init failed
            print("[TTS] Enable failed - Engine initialization problem.")
            send_error_to_web("TTS Engine failed to initialize. Cannot enable TTS.")
            return False
    else:
        tts_enabled_state = False
        print("[TTS] Disabled by API request.")
        # Stop speaking and clear buffer/queue immediately
        if tts_engine and tts_initialized_successfully:
            try: tts_engine.stop()
            except Exception as e: print(f"[TTS] Error stopping on toggle off: {e}")
        tts_sentence_buffer = ""
        while not tts_queue.empty():
            try: tts_queue.get_nowait()
            except queue.Empty: break
        # Optionally stop the thread to save resources: stop_tts_thread()
        send_status_to_web("TTS Disabled.")
        return True


def queue_tts_text(new_text):
    """Accumulates text for TTS, intended to be flushed later."""
    global tts_sentence_buffer, tts_enabled_state, tts_initialized_successfully
    if tts_enabled_state and tts_initialized_successfully:
        tts_sentence_buffer += new_text

def try_flush_tts_buffer():
    """Sends complete sentences from the buffer to the TTS queue if TTS is idle."""
    global tts_sentence_buffer, tts_busy, tts_queue, tts_busy_lock
    global tts_enabled_state, tts_initialized_successfully

    if not tts_enabled_state or not tts_initialized_successfully:
        tts_sentence_buffer = "" # Clear buffer if TTS is off
        return

    with tts_busy_lock:
        if tts_busy:
             return # Don't queue new text if already speaking

    if not tts_sentence_buffer or tts_sentence_buffer.isspace():
         return

    # Split logic remains the same
    sentences = re.split(r'([.!?\n]+(?:\s+|$))', tts_sentence_buffer)
    sentences = [s for s in sentences if s]

    chunk_to_speak = ""
    processed_len = 0
    temp_buffer = []
    i = 0
    while i < len(sentences):
        sentence_part = sentences[i]
        delimiter_part = ""
        if (i + 1) < len(sentences) and re.match(r'^([.!?\n]+(?:\s+|$))', sentences[i+1]):
             delimiter_part = sentences[i+1]
             temp_buffer.append(sentence_part + delimiter_part)
             processed_len += len(sentence_part + delimiter_part)
             i += 2
        else:
             break

    if processed_len > 0:
         tts_sentence_buffer = tts_sentence_buffer[processed_len:]
         chunk_to_speak = "".join(temp_buffer).strip()
    else:
         chunk_to_speak = ""

    if chunk_to_speak:
        # print(f"[TTS Flush] Queuing chunk: '{chunk_to_speak[:50]}...' ({len(chunk_to_speak)} chars)")
        tts_queue.put(chunk_to_speak)

def periodic_tts_check():
    """Periodically checks if TTS buffer can be flushed."""
    try:
        try_flush_tts_buffer()
    except Exception as e:
        print(f"[TTS Check] Error during periodic flush: {e}")
    finally:
        # Continue running as long as the application (Flask server) is running
        # Use a timer event that reschedules itself
        if flask_app: # Check if Flask app object exists
             timer = threading.Timer(0.2, periodic_tts_check) # Check every 200ms
             timer.daemon = True # Allow program to exit even if timer is pending
             timer.start()

# ============================================================================
# Whisper & VAD Logic (Adapted for backend state and SSE updates)
# ============================================================================

def initialize_whisper():
    """Initializes the Whisper model."""
    global whisper_model, whisper_initialized, whisper_model_size
    if whisper_initialized: return True

    print(f"[Whisper] Attempting initialization (Model: {whisper_model_size})...")
    update_vad_status(f"Loading Whisper ({whisper_model_size})...", "blue")

    # --- Environment Setup ---
    os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
    try:
        numba_cache_dir = os.path.join(tempfile.gettempdir(), 'numba_cache')
        os.makedirs(numba_cache_dir, exist_ok=True)
        os.environ['NUMBA_CACHE_DIR'] = numba_cache_dir
    except Exception as cache_err:
        print(f"[Whisper] Warning: Could not set NUMBA_CACHE_DIR: {cache_err}")

    try:
        if whisper_model_size.startswith("turbo"):
            # FasterWhisper
            whisper_turbo_model_name = whisper_model_size.split("-", 1)[1]
            device = "cuda" if torch.cuda.is_available() else "cpu"
            compute_type = "float16" if device == "cuda" and torch.cuda.get_device_capability(0)[0] >= 7 else ("float32" if device == "cuda" else "int8")
            print(f"[Whisper] Using FasterWhisper: model={whisper_turbo_model_name}, device={device}, compute_type={compute_type}")
            whisper_model = faster_whisper.WhisperModel(whisper_turbo_model_name, device=device, compute_type=compute_type)
        else:
            # OpenAI Whisper
            print(f"[Whisper] Using OpenAI Whisper: model={whisper_model_size}")
            whisper_model = whisper.load_model(whisper_model_size)

        whisper_initialized = True
        print("[Whisper] Model initialization successful.")
        if vad_initialized or not voice_enabled_state:
             update_vad_status(f"Whisper ({whisper_model_size}) ready.", "green")
        else:
             update_vad_status(f"Whisper OK, waiting VAD...", "blue")
        return True

    except ImportError as ie:
         print(f"[Whisper] Import Error: {ie}. Is FasterWhisper installed (`pip install faster-whisper`)?")
         whisper_initialized = False
         whisper_model = None
         update_vad_status("Whisper Import Error!", "red")
         send_error_to_web(f"Whisper Import Error: {ie}. Ensure libraries are installed.")
         return False
    except Exception as e:
        print(f"[Whisper] Error initializing model: {e}")
        traceback.print_exc()
        whisper_initialized = False
        whisper_model = None
        update_vad_status("Whisper init failed!", "red")
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
        except Exception as hub_dir_err:
             print(f"[VAD] Warning: Could not set custom torch hub directory: {hub_dir_err}")

        # Load VAD model
        vad_model_obj, vad_utils_funcs = torch.hub.load(repo_or_dir='snakers4/silero-vad',
                                                  model='silero_vad',
                                                  force_reload=False,
                                                  onnx=False, # Set True if ONNX preferred
                                                  trust_repo=True)
        vad_model = vad_model_obj # Assign to global
        vad_utils = vad_utils_funcs # Assign utils
        # Extract specific function needed if available, otherwise maybe use VADIterator
        if hasattr(vad_utils, 'get_speech_ts'):
             vad_get_speech_ts = vad_utils.get_speech_ts
        else:
            # Fallback or alternative approach might be needed if get_speech_ts isn't directly available
            print("[VAD] Warning: 'get_speech_ts' not found directly in vad_utils. VAD logic might need adjustment.")
            # For this implementation, we'll rely on the model's direct probability output instead.

        vad_initialized = True
        print("[VAD] Model initialized successfully.")
        if whisper_initialized or not voice_enabled_state:
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
        send_error_to_web(f"Failed to load Silero VAD model: {e}")
        return False

def initialize_audio_system():
    """Initializes PyAudio."""
    global py_audio
    if py_audio: return True
    try:
        print("[Audio] Initializing PyAudio...")
        py_audio = pyaudio.PyAudio()
        # Optional: Log available devices
        # num_devices = py_audio.get_device_count()
        # print(f"[Audio] Found {num_devices} devices.")
        # for i in range(num_devices):
        #     dev_info = py_audio.get_device_info_by_index(i)
        #     if dev_info.get('maxInputChannels') > 0:
        #         print(f"  Input Device {i}: {dev_info.get('name')}")
        print("[Audio] PyAudio initialized.")
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
    global vad_model, vad_stop_event, temp_audio_file_path, whisper_queue
    global tts_busy, tts_busy_lock, voice_enabled_state

    print("[VAD Worker] Thread started.")
    stream = None
    try:
        if not py_audio:
            print("[VAD Worker] PyAudio not initialized. Exiting.")
            update_vad_status("Audio System Error!", "red")
            return

        try:
            stream = py_audio.open(format=FORMAT, channels=CHANNELS, rate=RATE,
                                   input=True, frames_per_buffer=VAD_CHUNK)
            print("[VAD Worker] Audio stream opened.")
        except Exception as e:
            print(f"[VAD Worker] Failed to open audio stream: {e}")
            update_vad_status("Audio Input Error!", "red")
            send_error_to_web(f"Failed to open audio stream: {e}")
            return

        update_vad_status("Listening...", "gray")

        frames_since_last_speech = 0
        silence_frame_limit = int(SILENCE_THRESHOLD_SECONDS * RATE / VAD_CHUNK)
        min_speech_frames = int(MIN_SPEECH_DURATION_SECONDS * RATE / VAD_CHUNK)
        pre_speech_buffer_frames = int(PRE_SPEECH_BUFFER_SECONDS * RATE / VAD_CHUNK)

        temp_pre_speech_buffer = deque(maxlen=pre_speech_buffer_frames)
        was_tts_busy = False

        while not vad_stop_event.is_set():
            if not voice_enabled_state: # Check if VAD got disabled externally
                 print("[VAD Worker] Voice disabled, stopping listening loop.")
                 break
            try:
                data = stream.read(VAD_CHUNK, exception_on_overflow=False)

                with tts_busy_lock: current_tts_busy = tts_busy

                if current_tts_busy:
                    if not was_tts_busy:
                        # print("[VAD Worker] TTS active, pausing VAD.")
                        update_vad_status("VAD Paused (TTS Active)", "blue")
                        was_tts_busy = True
                    if is_recording_for_whisper:
                        # print("[VAD Worker] Discarding active recording due to TTS.")
                        is_recording_for_whisper = False
                        audio_frames_buffer.clear()
                        temp_pre_speech_buffer.clear()
                    time.sleep(0.05)
                    continue
                elif was_tts_busy:
                     # print("[VAD Worker] TTS finished, resuming VAD.")
                     update_vad_status("Listening...", "gray")
                     was_tts_busy = False
                     frames_since_last_speech = 0 # Reset silence count
                     time.sleep(0.1)
                     continue

                # --- Normal VAD ---
                audio_chunk_np = np.frombuffer(data, dtype=np.int16)
                temp_pre_speech_buffer.append(data)

                # Process chunk for VAD probability
                audio_float32 = audio_chunk_np.astype(np.float32) / 32768.0
                audio_tensor = torch.from_numpy(audio_float32)
                try:
                    # Ensure vad_model is callable (it should be the loaded model object)
                    if callable(vad_model):
                         speech_prob = vad_model(audio_tensor.unsqueeze(0), RATE).item()
                         is_speech = speech_prob > 0.45 # Slightly higher threshold maybe
                    else:
                         print("[VAD Worker] Error: vad_model is not callable.")
                         is_speech = False # Assume no speech if model isn't working
                except Exception as vad_err:
                     print(f"[VAD Worker] Error during VAD inference: {vad_err}")
                     is_speech = False # Assume no speech on error

                if is_speech:
                    # print(f"Speech detected (Prob: {speech_prob:.2f})") # Debug
                    frames_since_last_speech = 0
                    if not is_recording_for_whisper:
                        # print("[VAD Worker] Speech started, beginning recording.")
                        is_recording_for_whisper = True
                        audio_frames_buffer.clear()
                        audio_frames_buffer.extend(temp_pre_speech_buffer) # Add pre-buffer
                        update_vad_status("Recording...", "red")
                    if is_recording_for_whisper: # Should always be true here now
                         audio_frames_buffer.append(data)

                else: # No speech
                    frames_since_last_speech += 1
                    if is_recording_for_whisper:
                        audio_frames_buffer.append(data) # Keep recording until silence threshold

                        if frames_since_last_speech > silence_frame_limit:
                            # print(f"[VAD Worker] Silence detected ({SILENCE_THRESHOLD_SECONDS}s), stopping recording.")
                            is_recording_for_whisper = False
                            total_frames = len(audio_frames_buffer)
                            recording_duration = total_frames * VAD_CHUNK / RATE
                            effective_speech_frames = max(0, total_frames - frames_since_last_speech)

                            # print(f"[VAD Worker] Recording duration: {recording_duration:.2f}s")

                            if effective_speech_frames < min_speech_frames:
                                # print(f"[VAD Worker] Speech too short ({effective_speech_frames * VAD_CHUNK / RATE:.2f}s), discarding.")
                                audio_frames_buffer.clear()
                                update_vad_status("Too short, discarded", "orange")
                                time.sleep(0.8) # Pause before listening again
                                update_vad_status("Listening...", "gray")
                            else:
                                try:
                                    temp_audio_file = tempfile.NamedTemporaryFile(prefix="vad_rec_", suffix=".wav", delete=False)
                                    temp_audio_file_path = temp_audio_file.name
                                    temp_audio_file.close()

                                    wf = wave.open(temp_audio_file_path, 'wb')
                                    wf.setnchannels(CHANNELS)
                                    wf.setsampwidth(py_audio.get_sample_size(FORMAT))
                                    wf.setframerate(RATE)
                                    wf.writeframes(b''.join(audio_frames_buffer))
                                    wf.close()
                                    # print(f"[VAD Worker] Audio saved to {temp_audio_file_path} ({recording_duration:.2f}s)")
                                    whisper_queue.put(temp_audio_file_path)
                                    update_vad_status("Processing...", "blue")
                                except Exception as save_err:
                                     print(f"[VAD Worker] Error saving audio file: {save_err}")
                                     update_vad_status("File Save Error", "red")
                                     time.sleep(1)
                                     update_vad_status("Listening...", "gray")

                            audio_frames_buffer.clear() # Clear buffer after processing/discarding

                    # If not recording and not just finished TTS, ensure status is Listening
                    elif not was_tts_busy:
                          current_status = get_vad_status()
                          if current_status.get("text") not in ["Listening...", "Too short, discarded", "Processing..."]:
                              update_vad_status("Listening...", "gray")


            except IOError as e:
                if e.errno == pyaudio.paInputOverflowed:
                    print("[VAD Worker] Warning: Input overflowed.")
                else:
                    print(f"[VAD Worker] Stream read error: {e}")
                    update_vad_status("Audio Stream Error!", "red")
                    vad_stop_event.set() # Stop worker on critical error
                    break
            except Exception as e:
                print(f"[VAD Worker] Unexpected error in loop: {e}")
                traceback.print_exc()
                time.sleep(0.1)

    except Exception as e:
        print(f"[VAD Worker] Error during setup: {e}")
    finally:
        print("[VAD Worker] Cleaning up...")
        if stream:
            try:
                if stream.is_active(): stream.stop_stream()
                stream.close()
                print("[VAD Worker] Audio stream closed.")
            except Exception as e: print(f"[VAD Worker] Error closing stream: {e}")

        is_recording_for_whisper = False
        audio_frames_buffer.clear()
        vad_audio_buffer.clear()

        # Set final status based on why we exited
        if vad_stop_event.is_set():
             if get_vad_status()["text"] not in ["Audio Stream Error!", "Audio Input Error!"]:
                 update_vad_status("Voice Disabled", "grey")
        else: # Exited due to loop error or external state change
            if voice_enabled_state: # If voice still meant to be on, it's an error
                update_vad_status("VAD Stopped (Error)", "red")
            else: # If voice was disabled externally, that's expected
                 update_vad_status("Voice Disabled", "grey")


    print("[VAD Worker] Thread finished.")

def process_audio_worker():
    """Worker thread to transcribe audio files from the whisper_queue."""
    global whisper_model, whisper_initialized, whisper_queue, whisper_language_state
    global whisper_model_size, voice_enabled_state
    print("[Whisper Worker] Thread started.")
    while True:
        try:
            audio_file_path = whisper_queue.get()
            if audio_file_path is None:
                print("[Whisper Worker] Received stop signal.")
                break

            if not whisper_initialized or not voice_enabled_state:
                # print("[Whisper Worker] Skipping transcription (disabled or not ready).")
                if os.path.exists(audio_file_path):
                    try: os.unlink(audio_file_path)
                    except Exception as del_e: print(f"[Whisper Worker] Error deleting unused file: {del_e}")
                whisper_queue.task_done()
                # If VAD is still supposed to be running, reset its status
                if voice_enabled_state and vad_initialized and not is_recording_for_whisper:
                    update_vad_status("Listening...", "gray")
                continue

            print(f"[Whisper Worker] Processing: {os.path.basename(audio_file_path)}")
            update_vad_status("Transcribing...", "orange")
            start_time = time.time()
            transcribed_text = ""
            detected_language = "N/A"

            try:
                lang_to_use = whisper_language_state # Already set via API/default
                # print(f"[Whisper Worker] Using language: {'Auto' if lang_to_use is None else lang_to_use}")

                if isinstance(whisper_model, faster_whisper.WhisperModel):
                    segments, info = whisper_model.transcribe(audio_file_path, language=lang_to_use)
                    transcribed_text = " ".join([segment.text for segment in segments]).strip()
                    detected_language = info.language if hasattr(info, 'language') else 'N/A'
                elif isinstance(whisper_model, whisper.Whisper):
                    result = whisper_model.transcribe(audio_file_path, language=lang_to_use)
                    transcribed_text = result["text"].strip()
                    detected_language = result.get("language", "N/A")
                else:
                    raise TypeError("Unsupported Whisper model object")

                duration = time.time() - start_time
                print(f"[Whisper Worker] Transcription ({detected_language}, {duration:.2f}s): '{transcribed_text}'")

                if transcribed_text:
                    # Send transcription result via SSE
                    web_output_queue.put({
                        "type": "transcription",
                        "data": {
                            "text": transcribed_text,
                            "language": detected_language
                        }
                    })
                    update_vad_status("Transcription Ready", "green")
                    time.sleep(1.0) # Show "Ready" briefly
                    update_vad_status("Listening...", "gray")
                else:
                    update_vad_status("No speech detected", "orange")
                    time.sleep(0.8)
                    update_vad_status("Listening...", "gray")

            except Exception as e:
                print(f"[Whisper Worker] Transcription Error: {e}")
                traceback.print_exc()
                update_vad_status("Transcription Error", "red")
                send_error_to_web(f"Transcription failed: {e}")
                time.sleep(1.5)
                update_vad_status("Listening...", "gray")
            finally:
                if os.path.exists(audio_file_path):
                    try: os.unlink(audio_file_path)
                    except Exception as e: print(f"[Whisper Worker] Error deleting temp file {audio_file_path}: {e}")
            whisper_queue.task_done()

        except Exception as e:
            print(f"[Whisper Worker] Error in main loop: {e}")
            traceback.print_exc()
            # Ensure task_done if exception occurs after getting item
            if 'audio_file_path' in locals() and audio_file_path:
                whisper_queue.task_done()

    print("[Whisper Worker] Thread finished.")


def set_whisper_language(lang_code):
    """Sets the language for Whisper transcription via API."""
    global whisper_language_state
    # Find the language name for logging, but store the code
    lang_name = "Unknown"
    valid_code = False
    for name, code in WHISPER_LANGUAGES.items():
        if code == lang_code:
            lang_name = name
            valid_code = True
            break
    if lang_code is None: # Handle Auto-Detect explicitly
         lang_name = "Auto Detect"
         valid_code = True

    if valid_code:
        whisper_language_state = lang_code
        print(f"[Whisper] Language set to: {lang_name} ({whisper_language_state})")
        send_status_to_web(f"Whisper language set to: {lang_name}")
        return True
    else:
        print(f"[Whisper] Error: Invalid language code received: {lang_code}")
        send_error_to_web(f"Invalid Whisper language code: {lang_code}")
        return False

def set_whisper_model(size):
    """Sets Whisper model size and triggers re-initialization if needed via API."""
    global whisper_model_size, whisper_initialized, whisper_model
    if size not in WHISPER_MODEL_SIZES:
        print(f"[Whisper] Error: Invalid model size received: {size}")
        send_error_to_web(f"Invalid Whisper model size: {size}")
        return False

    if size == whisper_model_size and whisper_initialized:
        print(f"[Whisper] Model size '{size}' already selected and initialized.")
        return True # No change needed

    print(f"[Whisper] Model size change requested to: {size}. Re-initialization required.")
    whisper_model_size = size
    whisper_initialized = False
    if whisper_model:
        print("[Whisper] Releasing old model object...")
        try:
            del whisper_model
            if torch.cuda.is_available(): torch.cuda.empty_cache()
        except Exception as del_err: print(f"[Whisper] Error during old model cleanup: {del_err}")
        whisper_model = None

    send_status_to_web(f"Whisper model set to {size}. Will reload if/when voice input is active.")

    # If voice recognition is currently enabled, trigger immediate re-initialization
    if voice_enabled_state:
        print("[Whisper] Voice enabled, attempting immediate re-initialization...")
        threading.Thread(target=initialize_whisper, daemon=True).start() # Init in background
    return True

def toggle_voice_recognition(enable):
    """Enables/disables VAD and Whisper via API."""
    global voice_enabled_state, whisper_initialized, vad_initialized, vad_thread, vad_stop_event
    global py_audio, whisper_processing_thread # Make sure all globals accessed are listed

    if enable:
        if voice_enabled_state and vad_thread and vad_thread.is_alive():
             print("[Voice] Voice recognition already enabled.")
             return True # Already running

        print("[Voice] Enabling voice recognition by API request...")
        voice_enabled_state = True # Set desired state first
        update_vad_status("Initializing...", "blue")

        # --- Initialization Sequence (run in background to avoid blocking API response) ---
        def init_and_start_vad():
            # --- ADD THIS LINE ---
            global voice_enabled_state, py_audio, whisper_initialized, vad_initialized # Declare globals used within this nested function
            # --- END ADD ---

            all_initialized = True
            # Init components if needed
            if not py_audio:
                 if not initialize_audio_system(): all_initialized = False
            if all_initialized and not whisper_initialized:
                 if not initialize_whisper(): all_initialized = False
            if all_initialized and not vad_initialized:
                 if not initialize_vad(): all_initialized = False

            # Start threads if all OK and still enabled
            # Now this line will correctly read the global state
            if all_initialized and voice_enabled_state:
                # Access other globals needed for thread management
                global whisper_processing_thread, vad_thread, vad_stop_event

                if whisper_processing_thread is None or not whisper_processing_thread.is_alive():
                    print("[Voice] Starting Whisper processing thread...")
                    whisper_processing_thread = threading.Thread(target=process_audio_worker, daemon=True)
                    whisper_processing_thread.start()

                if vad_thread is None or not vad_thread.is_alive():
                    print("[Voice] Starting VAD worker thread...")
                    vad_stop_event.clear()
                    vad_thread = threading.Thread(target=vad_worker, daemon=True)
                    vad_thread.start()
                # VAD worker will set "Listening..." status
                print("[Voice] Voice recognition enabled successfully.")
                send_status_to_web("Voice Input Enabled.")
            elif voice_enabled_state: # Still enabled, but init failed
                print("[Voice] Enabling failed due to initialization errors.")
                # Status should reflect the specific error from init functions
                voice_enabled_state = False # This assignment makes Python assume it's local without 'global'
                send_error_to_web("Voice Input enabling failed (initialization error).")
            else: # Was disabled again before init finished
                 print("[Voice] Initialization aborted as voice was disabled.")

        threading.Thread(target=init_and_start_vad, daemon=True).start()
        return True # API call initiates the process

    else: # Disabling
        if not voice_enabled_state:
             print("[Voice] Voice recognition already disabled.")
             return True
        print("[Voice] Disabling voice recognition by API request...")
        voice_enabled_state = False # Set desired state
        update_vad_status("Disabling...", "grey")
        if vad_thread and vad_thread.is_alive():
            print("[Voice] Signaling VAD worker to stop...")
            vad_stop_event.set()
            # VAD worker will update status to Disabled when it exits
        else:
            update_vad_status("Voice Disabled", "grey") # Update status if thread wasn't running
        # Whisper thread keeps running, waits on queue.
        print("[Voice] Voice recognition disabled.")
        send_status_to_web("Voice Input Disabled.")
        return True

# ============================================================================
# File Processing Logic
# ============================================================================

def extract_pdf_content(pdf_path):
    """Extracts text content from a PDF file (same as before)."""
    try:
        doc = fitz.open(pdf_path)
        text_content = ""
        metadata = doc.metadata or {}
        title = metadata.get('title', 'N/A')
        author = metadata.get('author', 'N/A')
        if title != 'N/A' or author != 'N/A':
             text_content += f"--- PDF Info ---\nTitle: {title}\nAuthor: {author}\n----------------\n\n"

        num_pages = doc.page_count
        max_chars_per_pdf = 20000
        current_chars = len(text_content)
        truncated = False

        for page_num in range(num_pages):
            page = doc.load_page(page_num)
            page_text = page.get_text("text", sort=True).strip()
            if not page_text: continue

            page_header = f"--- Page {page_num+1} of {num_pages} ---\n"
            page_text_len = len(page_header) + len(page_text) + 2

            if current_chars + page_text_len > max_chars_per_pdf:
                 remaining_chars = max_chars_per_pdf - current_chars - len(page_header) - 20
                 if remaining_chars > 0:
                     text_content += page_header + page_text[:remaining_chars] + "... [Page Truncated]\n\n"
                 truncated = True
                 break
            else:
                 text_content += page_header + page_text + "\n\n"
                 current_chars += page_text_len
        doc.close()
        if truncated:
             text_content += "\n[PDF content truncated due to length limit.]"
        return text_content.strip()
    except Exception as e:
        print(f"[PDF Extract] Error: {e}")
        traceback.print_exc()
        return f"Error extracting content from PDF '{os.path.basename(pdf_path)}': {str(e)}"

def process_uploaded_file(file_path):
    """Processes the uploaded file, returns content or None for images."""
    global selected_file_path, selected_file_type, file_processed_for_input
    
    file_name = os.path.basename(file_path)
    file_ext = os.path.splitext(file_path)[1].lower()
    content_for_input = None # Content to potentially prepend to user message
    file_processed_for_input = False # Reset flag

    print(f"[File Process] Processing uploaded file: {file_name}")

    # Determine file type using mimetype guess first, fallback to extension
    mtype, _ = mimetypes.guess_type(file_path)
    guessed_type = mtype.split('/')[0] if mtype else None # 'image', 'text', 'application' etc.

    if guessed_type == 'image' or file_ext in ['.png', '.jpg', '.jpeg', '.gif', '.bmp']:
        selected_file_path = file_path
        selected_file_type = 'image'
        send_status_to_web(f"Image '{file_name}' attached.")
        print(f"[File Process] Type: Image")
        # Images are handled separately, no content returned for input box

    elif guessed_type == 'application' and file_ext == '.pdf':
        selected_file_path = file_path
        selected_file_type = 'pdf'
        send_status_to_web(f"Processing PDF '{file_name}'...")
        print(f"[File Process] Type: PDF")
        content_for_input = extract_pdf_content(file_path)
        file_processed_for_input = True
        send_status_to_web(f"PDF '{file_name}' content extracted.")

    elif guessed_type == 'text' or file_ext in ['.txt', '.md', '.py', '.js', '.html', '.css', '.json', '.log', '.csv', '.xml', '.yaml', '.ini', '.sh', '.bat']:
        selected_file_path = file_path
        selected_file_type = 'text'
        send_status_to_web(f"Loading text file '{file_name}'...")
        print(f"[File Process] Type: Text")
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            max_len = 20000
            if len(content) > max_len:
                content_for_input = content[:max_len] + f"\n\n[--- Content truncated at {max_len} characters ---]"
            else:
                 content_for_input = content
            file_processed_for_input = True
            send_status_to_web(f"Text file '{file_name}' content loaded.")
        except Exception as e:
            print(f"[File Process] Error reading text file {file_name}: {e}")
            send_error_to_web(f"Error reading text file '{file_name}': {e}")
            # Keep file selected but indicate error
            selected_file_path = file_path
            selected_file_type = 'text_error' # Special type?

    else: # Unsupported type
        print(f"[File Process] Unsupported file type: {file_name} (Type: {mtype}, Ext: {file_ext})")
        send_error_to_web(f"Unsupported file type: '{file_name}'. Cannot attach.")
        # Clear selection
        selected_file_path = None
        selected_file_type = None
        if os.path.exists(file_path): # Clean up the unsupported upload
             try: os.unlink(file_path)
             except: pass
        return None # Indicate failure

    # Return content only if it was extracted (PDF/Text)
    return content_for_input if file_processed_for_input else None

# ============================================================================
# Ollama Chat Logic (Adapted for web)
# ============================================================================

def get_current_chat_history():
    """Safely gets a copy of the chat history, filtering image data."""
    with messages_lock:
        history_copy = []
        for msg in messages:
            msg_copy = msg.copy()
            if "images" in msg_copy:
                 # Replace image bytes with a placeholder for history view
                 img_count = len(msg_copy.get("images", []))
                 placeholder = f" [Image Attachment ({img_count})]"
                 if msg_copy.get("content"):
                     msg_copy["content"] += placeholder
                 else:
                     msg_copy["content"] = placeholder
                 del msg_copy["images"]
            history_copy.append(msg_copy)
        return history_copy

def run_chat_interaction(user_message_content):
    """Starts the Ollama chat interaction in a background thread."""
    global stream_in_progress, selected_file_path, selected_file_type, file_processed_for_input
    global tts_sentence_buffer, tts_queue, tts_busy, tts_busy_lock, tts_engine

    # Prevent concurrent Ollama requests
    if not ollama_lock.acquire(blocking=False):
        send_error_to_web("Ollama is already processing a request. Please wait.")
        return

    stream_in_progress = True # Set flag immediately after acquiring lock

    # --- Prepare message and image ---
    image_path_to_send = None
    if selected_file_path and selected_file_type == 'image':
        image_path_to_send = selected_file_path
        print(f"[Chat Start] Including image: {os.path.basename(image_path_to_send)}")

    final_user_content = user_message_content
    
    # If a PDF/Text file was processed, prepend its content (if not already done implicitly)
    # This logic might be handled by frontend now, but keep for backend processing possibility
    # if file_processed_for_input and selected_file_type in ['pdf', 'text']:
    #    # Logic to potentially prepend file content header, or assume it's in user_message_content
    #    pass 


    # --- Add user message to history (thread-safe) ---
    # Construct the message object *before* locking
    user_msg_obj = {"role": "user", "content": final_user_content}
    # Add placeholder if image is being sent with this message
    if image_path_to_send:
         img_placeholder = f" [Image: {os.path.basename(image_path_to_send)}]"
         if user_msg_obj["content"]:
              user_msg_obj["content"] += img_placeholder
         else: # Handle image-only message
              user_msg_obj["content"] = img_placeholder.strip()


    try:
        with messages_lock:
            messages.append(user_msg_obj) # Add message without image bytes to history
        # Send history update via SSE AFTER releasing lock
        web_output_queue.put({"type": "history_update", "data": get_current_chat_history()})
    except Exception as lock_err:
         print(f"[Chat Start] Error adding user message to history: {lock_err}")
         send_error_to_web("Failed to update chat history.")
         stream_in_progress = False
         ollama_lock.release()
         return


    # --- Stop TTS ---
    if tts_engine and tts_enabled_state and tts_initialized_successfully:
        print("[Chat Start] Stopping active TTS...")
        try: tts_engine.stop()
        except: pass
        tts_sentence_buffer = ""
        while not tts_queue.empty():
            try: tts_queue.get_nowait()
            except queue.Empty: break
        with tts_busy_lock: tts_busy = False


    # --- Start Worker Thread ---
    print("[Chat Start] Starting Ollama worker thread...")
    thread = threading.Thread(target=chat_worker, args=(final_user_content, image_path_to_send), daemon=True)
    thread.start()

    # --- Reset file state after initiating the chat ---
    # We assume the file was 'consumed' by this message send
    selected_file_path = None
    selected_file_type = None
    file_processed_for_input = False
    # Send state clear confirmation? Maybe not necessary unless frontend needs it.

def chat_worker(user_message_content, image_path=None):
    """Background worker for Ollama streaming chat (releases lock on finish)."""
    global messages, messages_lock, current_model, web_output_queue, stream_in_progress, ollama_lock

    # Prepare message history for Ollama API call (needs image bytes for *this* message)
    history_for_ollama = []
    image_bytes_for_request = None
    try:
        # Read image bytes if provided for this request
        if image_path:
            print(f"[Chat Worker] Reading image for request: {image_path}")
            try:
                with open(image_path, 'rb') as f:
                    image_bytes_for_request = f.read()
                print(f"[Chat Worker] Image bytes loaded ({len(image_bytes_for_request)} bytes).")
            except Exception as e:
                 err_text = f"Error reading image file '{os.path.basename(image_path)}': {e}"
                 print(f"[Chat Worker] {err_text}")
                 send_error_to_web(err_text)
                 stream_in_progress = False # Reset flag
                 ollama_lock.release() # Release lock on error
                 return # Stop processing

        # Get history copy, prepare for Ollama format
        with messages_lock:
            history_copy = list(messages) # Get copy within lock

        # Construct messages list for API, potentially adding image bytes to last user message
        is_last_message = True
        for msg in reversed(history_copy): # Iterate backwards to easily find the last user message
            msg_for_api = msg.copy()
            # Remove internal placeholders or image data from history copy sent to API
            if "images" in msg_for_api: del msg_for_api["images"] # Remove byte arrays from history copy
            # If this is the last user message and we have image bytes, add them
            if msg_for_api["role"] == "user" and is_last_message and image_bytes_for_request:
                 msg_for_api["images"] = [image_bytes_for_request]
                 # Remove the placeholder text added earlier if present
                 # placeholder = f" [Image: {os.path.basename(image_path)}]"
                 # if msg_for_api["content"].endswith(placeholder):
                 #      msg_for_api["content"] = msg_for_api["content"][:-len(placeholder)].rstrip()

            history_for_ollama.insert(0, msg_for_api) # Insert at beginning to maintain order
            if msg_for_api["role"] == "user": is_last_message = False # Only add image to the very last one

    except Exception as hist_err:
         err_text = f"Error preparing chat history for Ollama: {hist_err}"
         print(f"[Chat Worker] {err_text}")
         send_error_to_web(err_text)
         stream_in_progress = False
         ollama_lock.release()
         return

    # --- Call Ollama API ---
    assistant_response = ""
    try:
        print(f"[Chat Worker] Sending request to model {current_model}...")
        # print(f"[Chat Worker] History for API: {history_for_ollama}") # Debug: Careful with image bytes in log

        stream = ollama.chat(model=current_model, messages=history_for_ollama, stream=True)

        first_chunk = True
        for chunk in stream:
            if 'message' in chunk and 'content' in chunk['message']:
                content_piece = chunk['message']['content']
                if first_chunk:
                     web_output_queue.put({"type": "stream_start"}) # Signal web UI
                     first_chunk = False
                web_output_queue.put({"type": "stream_chunk", "data": content_piece})
                assistant_response += content_piece
                queue_tts_text(content_piece) # Queue for TTS
            if 'error' in chunk:
                 err_text = f"Ollama stream error: {chunk['error']}"
                 print(f"[Chat Worker] {err_text}")
                 send_error_to_web(err_text)
                 assistant_response = None # Mark as error
                 break
            if chunk.get('done', False): break

        # --- After Streaming ---
        if assistant_response is not None:
             # Add complete assistant response to shared history
             try:
                 with messages_lock:
                     messages.append({"role": "assistant", "content": assistant_response})
                 # Send history update via SSE AFTER releasing lock
                 web_output_queue.put({"type": "history_update", "data": get_current_chat_history()})
                 print(f"[Chat Worker] Assistant response added to history.")
             except Exception as lock_err:
                 print(f"[Chat Worker] Error adding assistant response to history: {lock_err}")

             web_output_queue.put({"type": "stream_end"}) # Signal web UI
             queue_tts_text("\n") # Force TTS flush check
             try_flush_tts_buffer()

    except ollama.ResponseError as ore:
         err_text = f"Ollama API Error: {ore.status_code} - {ore.error}"
         print(f"[Chat Worker] {err_text}")
         send_error_to_web(err_text)
    except Exception as e:
        err_text = f"Ollama communication error: {e}"
        print(f"[Chat Worker] {err_text}")
        traceback.print_exc()
        send_error_to_web(err_text)
    finally:
        stream_in_progress = False
        ollama_lock.release() # IMPORTANT: Release the lock
        print("[Chat Worker] Worker finished.")
        # Clean up the sent image file if it exists
        if image_path and os.path.exists(image_path):
             try:
                  # Check if it's in the upload folder before deleting
                  if UPLOAD_FOLDER in os.path.abspath(image_path):
                       os.unlink(image_path)
                       print(f"[Chat Worker] Cleaned up sent image: {os.path.basename(image_path)}")
             except Exception as del_err:
                  print(f"[Chat Worker] Error deleting sent image {image_path}: {del_err}")

def fetch_available_models():
    """Fetches available Ollama models."""
    try:
        models_data = ollama.list()
        valid_models = []
        
        # Case 1: New format - models_data has a 'models' attribute that contains Model objects
        if hasattr(models_data, 'models') and isinstance(models_data.models, list):
            for model_obj in models_data.models:
                if hasattr(model_obj, 'model'):
                    valid_models.append(model_obj.model)
                else:
                    print(f"[Ollama] Warning: Model object missing 'model' attribute: {model_obj}")
        
        # Case 2: Old format - models_data is a dict with 'models' key containing dicts
        elif isinstance(models_data, dict) and 'models' in models_data:
            models_list = models_data.get('models', [])
            if isinstance(models_list, list):
                for model in models_list:
                    if isinstance(model, dict) and 'name' in model:
                        valid_models.append(model['name'])
                    else:
                        print(f"[Ollama] Warning: Skipping invalid model entry: {model}")
        
        # Log warning for unexpected format
        else:
            print(f"[Ollama] Warning: Unexpected format received from ollama.list(). Got: {models_data}")
        
        # Return valid models or fallback list if none found
        return valid_models if valid_models else [DEFAULT_OLLAMA_MODEL, "llama3:8b", "phi3:mini"]
    
    except Exception as e:
        print(f"[Ollama] Error fetching models: {e}")
        # Provide common fallbacks if API fails
        return [DEFAULT_OLLAMA_MODEL, "llama3:8b", "phi3:mini"]

# ============================================================================
# Flask Application Setup
# ============================================================================

def create_flask_app():
    """Creates and configures the Flask application."""
    app = Flask(__name__, template_folder='templates')
    app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
    app.secret_key = os.urandom(24) # Needed for session/flash messages if used

    # Create upload folder if it doesn't exist
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)

    # --- Basic Routes ---
    @app.route('/')
    def index():
        """Serves the main HTML page."""
        return render_template('index.html')

    @app.route('/styles.css')
    def styles():
         # Route for external CSS if you create one
         return send_from_directory('templates', 'styles.css')


    # --- API Routes ---
    @app.route('/api/send_message', methods=['POST'])
    def api_send_message():
        """Handles message submission (text + optional file)."""
        global selected_file_path, selected_file_type, file_processed_for_input

        if stream_in_progress:
            return jsonify({"status": "error", "message": "Please wait for the previous response."}), 429 # Too Many Requests

        user_text = request.form.get('message', '').strip()
        file = request.files.get('file')
        file_content_for_input = None

        # --- File Handling ---
        if file and file.filename:
            if allowed_file(file.filename):
                filename = secure_filename(file.filename)
                temp_save_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{int(time.time())}_{filename}")
                try:
                    file.save(temp_save_path)
                    print(f"[API Send] File '{filename}' uploaded and saved to {temp_save_path}")
                    # Process the saved file (determines type, extracts content if applicable)
                    file_content_for_input = process_uploaded_file(temp_save_path)
                    # If processing failed, selected_file_path/type will be None
                    if selected_file_path is None:
                        return jsonify({"status": "error", "message": f"Failed to process or unsupported file type: {filename}"}), 400

                except Exception as e:
                    print(f"[API Send] Error saving or processing upload '{filename}': {e}")
                    return jsonify({"status": "error", "message": f"Error handling uploaded file: {e}"}), 500
            else:
                return jsonify({"status": "error", "message": f"File type not allowed: {file.filename}"}), 400
        else:
             # Clear any previous file selection if no new file is uploaded
             # selected_file_path = None
             # selected_file_type = None
             # file_processed_for_input = False
             pass # Keep existing selection if no new file provided? Depends on desired UX. Let's clear it.
             if not request.form.get('keep_attachment'): # Add a flag from frontend if needed
                 selected_file_path = None
                 selected_file_type = None
                 file_processed_for_input = False


        # --- Message Content Preparation ---
        final_user_text = user_text
        # Option 1: Prepend extracted content automatically
        # if file_content_for_input:
        #     final_user_text = f"--- Content from {selected_file_type}: {os.path.basename(selected_file_path)} ---\n{file_content_for_input}\n\n---\n\n{user_text}"

        # Option 2: Assume frontend handles placing content in the message box if desired.
        # Backend just needs to know *if* a file (especially image) is attached.

        if not final_user_text and not selected_file_path:
             return jsonify({"status": "error", "message": "Please enter a message or attach a file."}), 400

        # --- Start Chat ---
        run_chat_interaction(final_user_text) # Handles Ollama call, history, TTS stop etc.

        return jsonify({"status": "success", "message": "Message received, processing..."})

    @app.route('/api/clear_attachment', methods=['POST'])
    def api_clear_attachment():
        global selected_file_path, selected_file_type, file_processed_for_input
        cleared = False
        if selected_file_path:
            print(f"[API Clear] Clearing attachment: {selected_file_path}")
            # Delete the temp file if it exists in uploads
            if UPLOAD_FOLDER in os.path.abspath(selected_file_path) and os.path.exists(selected_file_path):
                try:
                    os.unlink(selected_file_path)
                    print(f"[API Clear] Deleted file: {os.path.basename(selected_file_path)}")
                except Exception as e:
                    print(f"[API Clear] Error deleting file {selected_file_path}: {e}")
            selected_file_path = None
            selected_file_type = None
            file_processed_for_input = False
            cleared = True

        if cleared:
             send_status_to_web("Attachment cleared.")
             return jsonify({"status": "success", "message": "Attachment cleared."})
        else:
             return jsonify({"status": "no_op", "message": "No attachment to clear."})


    @app.route('/api/history')
    def api_history():
        """Returns current chat history."""
        return jsonify({"history": get_current_chat_history()})

    @app.route('/api/stream')
    def api_stream():
        """SSE endpoint for streaming updates."""
        def event_stream():
            keepalive_freq = 15 # seconds
            last_event_time = time.time()
            # Immediately send current status on connect
            try:
                yield f"event: full_status\ndata: {json.dumps(get_full_backend_status())}\n\n"
                # Send initial history too? Maybe better via separate /api/history call from client.
                # yield f"event: history_update\ndata: {json.dumps(get_current_chat_history())}\n\n"
                last_event_time = time.time()
            except Exception as init_e:
                 print(f"[SSE Stream] Error sending initial status: {init_e}")

            print("[SSE Stream] Client connected.")
            while True:
                try:
                    data = web_output_queue.get(timeout=keepalive_freq)
                    event_type = data.get("type", "message") # Default type if not specified
                    event_data = json.dumps(data.get("data", data)) # Send data part or whole object

                    # Map internal types to SSE event types
                    if event_type == "stream_start": sse_event = "stream_start"
                    elif event_type == "stream_chunk": sse_event = "stream_chunk"
                    elif event_type == "stream_end": sse_event = "stream_end"
                    elif event_type == "error": sse_event = "error"
                    elif event_type == "status": sse_event = "status" # General status text
                    elif event_type == "status_update": sse_event = "status_update" # Specific source status (like VAD)
                    elif event_type == "history_update": sse_event = "history_update"
                    elif event_type == "voices_update": sse_event = "voices_update"
                    elif event_type == "transcription": sse_event = "transcription"
                    else: sse_event = "message" # Fallback

                    yield f"event: {sse_event}\ndata: {event_data}\n\n"
                    web_output_queue.task_done()
                    last_event_time = time.time()

                except queue.Empty:
                    # Send keepalive comment
                    yield ": keepalive\n\n"
                    last_event_time = time.time() # Reset timer after keepalive
                except GeneratorExit:
                    print("[SSE Stream] Client disconnected.")
                    break
                except Exception as e:
                    print(f"[SSE Stream] Error in stream loop: {e}")
                    try: # Try to inform client
                         yield f"event: error\ndata: {json.dumps({'type': 'error', 'data': f'SSE internal error: {e}'})}\n\n"
                    except Exception: pass
                    time.sleep(1) # Avoid rapid error loops
        return Response(event_stream(), mimetype="text/event-stream")

    @app.route('/api/status')
    def api_status():
        """Returns the current full backend status."""
        return jsonify(get_full_backend_status())

    @app.route('/api/models')
    def api_models():
        """Returns available Ollama models."""
        # Consider caching this if fetching is slow/expensive
        models = fetch_available_models()
        return jsonify({"models": models})
    


    @app.route('/api/voices')
    def api_voices():
         """Returns available TTS voices."""
         voices = get_available_voices()
         return jsonify({"voices": voices})


    # --- Control Routes ---
    @app.route('/api/control/model', methods=['POST'])
    def api_control_model():
        data = request.get_json()
        model_name = data.get('model')
        if not model_name:
            return jsonify({"status": "error", "message": "Missing 'model' parameter."}), 400
        # Add validation if needed (check against fetch_available_models?)
        global current_model
        current_model = model_name
        print(f"[API Control] Ollama model set to: {current_model}")
        send_status_to_web(f"Ollama model changed to: {current_model}")
        return jsonify({"status": "success", "current_model": current_model})

    @app.route('/api/control/tts', methods=['POST'])
    def api_control_tts():
        data = request.get_json()
        enable = data.get('enable')
        if enable is None:
             return jsonify({"status": "error", "message": "Missing 'enable' parameter (true/false)."}), 400
        success = toggle_tts(bool(enable))
        return jsonify({"status": "success" if success else "error", "tts_enabled": tts_enabled_state})

    @app.route('/api/control/tts_settings', methods=['POST'])
    def api_control_tts_settings():
         data = request.get_json()
         rate = data.get('rate')
         voice_id = data.get('voice_id')
         results = {}
         if rate is not None:
             success_rate = set_tts_rate(rate)
             results['rate_set'] = success_rate
         if voice_id is not None:
             success_voice = set_tts_voice(voice_id)
             results['voice_set'] = success_voice

         return jsonify({
             "status": "success",
             "results": results,
             "current_rate": tts_rate_state,
             "current_voice_id": tts_voice_id_state
         })

    @app.route('/api/control/vad', methods=['POST'])
    def api_control_vad():
        data = request.get_json()
        enable = data.get('enable')
        if enable is None:
             return jsonify({"status": "error", "message": "Missing 'enable' parameter (true/false)."}), 400
        success = toggle_voice_recognition(bool(enable))
        # Give a slight delay for status update to potentially propagate via SSE
        time.sleep(0.1)
        return jsonify({"status": "success", "vad_enabled": voice_enabled_state, "current_status": get_vad_status()})

    @app.route('/api/control/whisper_settings', methods=['POST'])
    def api_control_whisper_settings():
         data = request.get_json()
         lang_code = data.get('language_code') # Expect code like 'en', 'fi', or null for auto
         model_size = data.get('model_size')
         results = {}
         if lang_code is not None or data.get('language_code') == None: # Allow null for auto
             success_lang = set_whisper_language(lang_code)
             results['language_set'] = success_lang
         if model_size is not None:
             success_model = set_whisper_model(model_size)
             results['model_set'] = success_model

         return jsonify({
             "status": "success",
             "results": results,
             "current_language": whisper_language_state,
             "current_model": whisper_model_size
         })

    return app

def get_full_backend_status():
     """Helper to gather current state for the status API."""
     return {
         "ollama_model": current_model,
         "tts_enabled": tts_enabled_state,
         "tts_rate": tts_rate_state,
         "tts_voice_id": tts_voice_id_state,
         "tts_initialized": tts_initialized_successfully,
         "vad_enabled": voice_enabled_state,
         "vad_status": get_vad_status(),
         "whisper_language": whisper_language_state,
         "whisper_model": whisper_model_size,
         "whisper_initialized": whisper_initialized,
         "vad_initialized": vad_initialized,
         "attachment": {
             "filename": os.path.basename(selected_file_path) if selected_file_path else None,
             "type": selected_file_type
         }
     }

# ===================
# Main Execution
# ===================
def initialize_backend():
    """Initializes components that need early setup."""
    print("[Backend Init] Initializing components...")
    initialize_audio_system()
    initialize_tts() # Try to init TTS early to get voices
    # VAD/Whisper init will be triggered by toggle_voice_recognition if enabled by default

    # Start periodic checks only once
    threading.Timer(0.2, periodic_tts_check).start() # Start TTS flush check timer

    # Start default services if enabled
    if tts_enabled_state:
        start_tts_thread()
    if voice_enabled_state:
        toggle_voice_recognition(True) # Start VAD/Whisper init

    print("[Backend Init] Initialization sequence complete.")


def cleanup_on_exit():
    """Gracefully shutdown background threads and resources."""
    print("[Cleanup] Starting shutdown process...")

    # 1. Signal VAD thread to stop
    if vad_thread and vad_thread.is_alive():
        print("[Cleanup] Stopping VAD thread...")
        vad_stop_event.set()

    # 2. Stop Whisper processing thread
    if whisper_processing_thread and whisper_processing_thread.is_alive():
         print("[Cleanup] Stopping Whisper processing thread...")
         whisper_queue.put(None)

    # 3. Stop TTS thread
    stop_tts_thread()

    # --- Join threads ---
    join_timeout = 2.0
    if vad_thread and vad_thread.is_alive():
         vad_thread.join(timeout=join_timeout)
         if vad_thread.is_alive(): print("[Cleanup] Warning: VAD thread did not exit.")
         else: print("[Cleanup] VAD thread joined.")
    if whisper_processing_thread and whisper_processing_thread.is_alive():
         whisper_processing_thread.join(timeout=join_timeout)
         if whisper_processing_thread.is_alive(): print("[Cleanup] Warning: Whisper thread did not exit.")
         else: print("[Cleanup] Whisper thread joined.")

    # 4. Terminate PyAudio
    if py_audio:
        print("[Cleanup] Terminating PyAudio...")
        try: py_audio.terminate()
        except Exception as pa_err: print(f"[Cleanup] PyAudio termination error: {pa_err}")

    # 5. Clean up temporary files (VAD recording, uploads)
    print("[Cleanup] Cleaning up temporary files...")
    if temp_audio_file_path and os.path.exists(temp_audio_file_path):
        try: os.unlink(temp_audio_file_path)
        except Exception as e: print(f"[Cleanup] Error deleting temp VAD file: {e}")

    if os.path.exists(UPLOAD_FOLDER):
         for filename in os.listdir(UPLOAD_FOLDER):
             file_path = os.path.join(UPLOAD_FOLDER, filename)
             try:
                 if os.path.isfile(file_path):
                     os.unlink(file_path)
             except Exception as e:
                 print(f"[Cleanup] Error deleting upload {filename}: {e}")
    # Optionally remove the upload folder itself
    # try: os.rmdir(UPLOAD_FOLDER)
    # except OSError as e: print(f"[Cleanup] Error removing upload folder: {e}")


    print("[Cleanup] Shutdown complete.")


if __name__ == '__main__':
    # Ensure upload folder exists
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)

    # Initialize backend components in a separate thread to allow Flask to start sooner?
    # Or do it sequentially before starting Flask. Sequential is simpler.
    initialize_backend()

    # Create and run Flask app
    flask_app = create_flask_app() # Assign to global variable

    print(f"\n[Flask] Starting server on http://{FLASK_HOST}:{FLASK_PORT}")
    print(f"[Flask] Access UI from this machine: http://127.0.0.1:{FLASK_PORT}")
    print(f"[Flask] Access UI from local network: http://<YOUR_LOCAL_IP>:{FLASK_PORT}")
    print("[Flask] Press Ctrl+C to quit.")

    # Register cleanup function
    import atexit
    atexit.register(cleanup_on_exit)

    # Run Flask server
    # debug=True causes issues with threading and restarts, keep False for this setup
    flask_app.run(host=FLASK_HOST, port=FLASK_PORT, threaded=True, debug=False)

    # Cleanup might not always run reliably on forced exit, atexit helps.