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
APP_TITLE = "Ollama Multimodal Chat++ (Streaming, TTS, VAD)"
WINDOW_GEOMETRY = "850x850"

# --- Models ---
DEFAULT_OLLAMA_MODEL = "gemma3:27b"
DEFAULT_WHISPER_MODEL_SIZE = "base" # Faster startup, change to 'medium' or 'large' for better accuracy

# --- Whisper ---
WHISPER_LANGUAGES = [
    ("Auto Detect", None), ("English", "en"), ("Finnish", "fi"), ("Swedish", "sv"),
    ("German", "de"), ("French", "fr"), ("Spanish", "es"), ("Italian", "it"),
    ("Russian", "ru"), ("Chinese", "zh"), ("Japanese", "ja")
]
WHISPER_MODEL_SIZES = ["tiny", "base", "small", "medium", "large", "turbo", "turbo-tiny", "turbo-base", "turbo-small", "turbo-medium", "turbo-large"]

# ===================
# Globals
# ===================
# --- Ollama & Chat ---
messages = []
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
user_input_widget = None    # Assign after Tkinter setup

# --- Audio Handling ---
py_audio = None
audio_stream = None
is_recording_for_whisper = False # State flag for VAD-triggered recording
audio_frames_buffer = deque() # Holds raw audio data during VAD recording
vad_audio_buffer = deque(maxlen=int(RATE / VAD_CHUNK * 1.5)) # Rolling buffer for VAD analysis (~1.5 sec)
temp_audio_file_path = None # Path for the temporary WAV file

# ===================
# TTS Setup & Control
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
        del temp_engine
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
        selected_idx = voice_selector.current()
        if selected_idx >= 0:
            voice_id = available_voices[selected_idx][1]
            tts_voice_id.set(voice_id)
            tts_engine.setProperty('voice', voice_id)
            print(f"[TTS] Voice set to: {available_voices[selected_idx][0]}")
    except Exception as e:
        print(f"[TTS] Error setting voice: {e}")

def set_speech_rate(value=None): # Updated to accept value from scale command
    """Sets the speech rate for the TTS engine."""
    global tts_engine, tts_rate, tts_initialized_successfully
    if not tts_initialized_successfully or not tts_engine: return
    try:
        rate = tts_rate.get()
        tts_engine.setProperty('rate', rate)
        # print(f"[TTS] Speech rate set to: {rate}") # Optional: Can be noisy
    except Exception as e:
        print(f"[TTS] Error setting speech rate: {e}")

def tts_worker():
    """Worker thread processing the TTS queue."""
    global tts_engine, tts_queue, tts_busy, tts_initialized_successfully
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
        except Exception as e:
            print(f"[TTS Worker] Error: {e}")
            with tts_busy_lock: # Ensure busy flag is reset on error
                tts_busy = False

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
    global tts_thread, tts_engine, tts_queue
    print("[TTS] Stopping worker thread...")
    if tts_engine:
        try: tts_engine.stop() # Stop any current speech
        except Exception as e: print(f"[TTS] Error stopping engine: {e}")

    # Clear the queue
    while not tts_queue.empty():
        try: tts_queue.get_nowait()
        except queue.Empty: break

    if tts_thread and tts_thread.is_alive():
        tts_queue.put(None) # Send sentinel
        tts_thread.join(timeout=2) # Wait for thread to finish
        if tts_thread.is_alive():
            print("[TTS] Warning: Worker thread did not terminate gracefully.")
    tts_thread = None
    print("[TTS] Worker thread stopped.")

def toggle_tts():
    """Callback for the TTS enable/disable checkbox."""
    global tts_enabled, tts_sentence_buffer, tts_initialized_successfully
    if tts_enabled.get():
        if not tts_initialized_successfully:
            # Try initializing again when toggled on
            tts_initialized_successfully = initialize_tts()

        if tts_initialized_successfully:
            print("[TTS] Enabled by user.")
            start_tts_thread() # Ensure thread is running
            # Enable controls if they were disabled
            if 'voice_selector' in globals(): voice_selector.config(state="readonly")
            if 'rate_scale' in globals(): rate_scale.config(state="normal")
        else:
            print("[TTS] Enable failed - Engine initialization problem.")
            tts_enabled.set(False) # Uncheck the box
            add_message_to_ui("error", "TTS Engine failed to initialize. Cannot enable TTS.")
            # Ensure controls are disabled
            if 'voice_selector' in globals(): voice_selector.config(state="disabled")
            if 'rate_scale' in globals(): rate_scale.config(state="disabled")
    else:
        print("[TTS] Disabled by user.")
        # Stop speaking and clear buffer/queue immediately
        if tts_engine:
            try: tts_engine.stop()
            except Exception as e: print(f"[TTS] Error stopping on toggle off: {e}")
        tts_sentence_buffer = ""
        while not tts_queue.empty():
            try: tts_queue.get_nowait()
            except queue.Empty: break
        # Optional: Stop the worker thread if you want to save resources when disabled
        # stop_tts_thread()

def queue_tts_text(new_text):
    """Accumulates text for TTS, intended to be flushed later."""
    global tts_sentence_buffer
    if tts_enabled.get() and tts_initialized_successfully:
        tts_sentence_buffer += new_text

def try_flush_tts_buffer():
    """Sends complete sentences from the buffer to the TTS queue if TTS is idle."""
    global tts_sentence_buffer, tts_busy, tts_queue
    if not tts_enabled.get() or not tts_initialized_successfully or not tts_engine:
        return

    with tts_busy_lock:
        if tts_busy: # Don't queue new text if already speaking
            return

    # Split on sentence endings (. ! ? \n) followed by space or end of string
    # Keep delimiters attached to the preceding sentence part.
    sentences = re.split(r'([.!?\n]+(?:\s+|$))', tts_sentence_buffer)

    # Process pairs of (sentence_part, delimiter_part)
    chunk_to_speak = ""
    processed_len = 0
    temp_buffer = []

    for i in range(0, len(sentences) - 1, 2):
        sentence_part = sentences[i]
        delimiter_part = sentences[i+1] if (i+1) < len(sentences) else ""
        
        # We consider a sentence complete if it has a delimiter
        if sentence_part: # Avoid empty parts
             temp_buffer.append(sentence_part + delimiter_part)

    # If the last part doesn't end with a delimiter, keep it in the buffer
    if len(sentences) % 2 == 1 and sentences[-1]:
        # Last part is not followed by delimiter, keep it
        leftover_start_index = len("".join(temp_buffer))
        tts_sentence_buffer = tts_sentence_buffer[leftover_start_index:]
    elif temp_buffer: # We processed everything into complete sentences
        tts_sentence_buffer = ""
    
    chunk_to_speak = "".join(temp_buffer).strip()

    if chunk_to_speak:
        # print(f"[TTS] Queuing chunk: '{chunk_to_speak[:50]}...' ({len(chunk_to_speak)} chars)")
        tts_queue.put(chunk_to_speak)
        # No need to set tts_busy here, worker thread handles it

def periodic_tts_check():
    """Periodically checks if TTS buffer can be flushed."""
    try_flush_tts_buffer()
    # Reschedule even if flushing fails, to keep checking
    window.after(200, periodic_tts_check) # Check every 200ms


# ========================
# Whisper & VAD Setup
# ========================
def initialize_whisper():
    """Initializes the Whisper model. Returns True on success."""
    global whisper_model, whisper_initialized, whisper_model_size
    if whisper_initialized: return True

    update_vad_status(f"Loading Whisper ({whisper_model_size})...", "blue")
    try:
        print(f"[Whisper] Initializing model ({whisper_model_size})...")

        os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
        os.environ["HF_HUB_DISABLE_SYMLINKS"] = "1"

        if whisper_model_size.startswith("turbo"):

            whisper_turbo_model = whisper_model_size.split("-")[1]

            device = "cuda" if torch.cuda.is_available() else "cpu"
            compute_type = "float16" if device == "cuda" else "int8"
            whisper_model = faster_whisper.WhisperModel(whisper_turbo_model, device=device, compute_type=compute_type)
            print(f"[Whisper] Turbo ({whisper_turbo_model}) model loaded on {device} using {compute_type}")
        else:
            whisper_model = whisper.load_model(whisper_model_size)

        whisper_initialized = True
        update_vad_status(f"Whisper ({whisper_model_size}) ready.", "green")
        print("[Whisper] Model initialized successfully.")
        return True
    
    except Exception as e:
        print(f"[Whisper] Error initializing model: {e}")
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
        torch.hub.set_dir(os.path.join(tempfile.gettempdir(), "torch_hub"))
        os.makedirs(torch.hub.get_dir(), exist_ok=True)
        
        vad_model, vad_utils = torch.hub.load(repo_or_dir='snakers4/silero-vad',
                                              model='silero_vad',
                                              force_reload=False, # Use cached if available
                                              trust_repo=True)
        (vad_get_speech_ts, _, _, _, _) = vad_utils
        vad_initialized = True
        print("[VAD] Model initialized successfully.")
        # Don't overwrite Whisper status here, wait for both
        return True
    except Exception as e:
        print(f"[VAD] Error initializing model: {e}")
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
        return True
    except Exception as e:
        print(f"[Audio] Error initializing PyAudio: {e}")
        add_message_to_ui("error", f"Failed to initialize audio system: {e}")
        py_audio = None
        return False

def vad_worker():
    """Worker thread for continuous VAD and triggering recording."""
    global py_audio, audio_stream, vad_audio_buffer, audio_frames_buffer, is_recording_for_whisper
    global vad_model, vad_get_speech_ts, vad_stop_event, temp_audio_file_path, whisper_queue

    print("[VAD Worker] Thread started.")
    
    stream = None
    try:
        stream = py_audio.open(format=FORMAT,
                               channels=CHANNELS,
                               rate=RATE,
                               input=True,
                               frames_per_buffer=VAD_CHUNK) # Use smaller chunk for VAD
        print("[VAD Worker] Audio stream opened.")
        update_vad_status("Listening...", "gray")

        num_silence_frames = 0
        silence_frame_limit = int(SILENCE_THRESHOLD_SECONDS * RATE / VAD_CHUNK)
        speech_detected_recently = False
        frames_since_last_speech = 0
        pre_speech_buffer_frames = int(PRE_SPEECH_BUFFER_SECONDS * RATE / VAD_CHUNK)

        temp_pre_speech_buffer = deque(maxlen=pre_speech_buffer_frames)

        while not vad_stop_event.is_set():
            try:
                data = stream.read(VAD_CHUNK, exception_on_overflow=False)
                audio_chunk_np = np.frombuffer(data, dtype=np.int16)
                vad_audio_buffer.append(audio_chunk_np)
                temp_pre_speech_buffer.append(data) # Store raw bytes for pre-buffer

                # Only run VAD if buffer has enough data (~0.5s is good)
                if len(vad_audio_buffer) >= int(RATE / VAD_CHUNK * 0.5):
                    audio_data_np = np.concatenate(list(vad_audio_buffer))
                    audio_float32 = audio_data_np.astype(np.float32) / 32768.0
                    audio_tensor = torch.from_numpy(audio_float32).unsqueeze(0)

                    # Lower threshold slightly for better sensitivity in quiet environments
                    speech_timestamps = vad_get_speech_ts(audio_tensor, vad_model, sampling_rate=RATE, threshold=0.4)

                    if speech_timestamps:
                        # print("Speech detected", speech_timestamps) # Debug
                        speech_detected_recently = True
                        frames_since_last_speech = 0
                        if not is_recording_for_whisper:
                            print("[VAD Worker] Speech started, beginning recording.")
                            is_recording_for_whisper = True
                            audio_frames_buffer.clear()
                            # Add pre-speech buffer content
                            for frame_data in temp_pre_speech_buffer:
                                audio_frames_buffer.append(frame_data)
                            update_vad_status("Recording...", "red")
                        # Append current data *after* checking state
                        audio_frames_buffer.append(data)

                    else: # No speech detected in this buffer window
                        frames_since_last_speech += 1
                        # Check if we *were* recording and have now hit silence threshold
                        if is_recording_for_whisper and frames_since_last_speech > silence_frame_limit:
                            print(f"[VAD Worker] Silence detected ({SILENCE_THRESHOLD_SECONDS}s), stopping recording.")
                            
                            # Save audio
                            temp_audio_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
                            temp_audio_file_path = temp_audio_file.name
                            temp_audio_file.close() # Close it so wave can open it

                            wf = wave.open(temp_audio_file_path, 'wb')
                            wf.setnchannels(CHANNELS)
                            wf.setsampwidth(py_audio.get_sample_size(FORMAT))
                            wf.setframerate(RATE)
                            wf.writeframes(b''.join(audio_frames_buffer))
                            wf.close()
                            print(f"[VAD Worker] Audio saved to {temp_audio_file_path} ({len(audio_frames_buffer) * VAD_CHUNK / RATE:.2f}s)")

                            # Queue for transcription
                            whisper_queue.put(temp_audio_file_path)
                            
                            # Reset state
                            is_recording_for_whisper = False
                            audio_frames_buffer.clear()
                            update_vad_status("Processing...", "blue") # UI update happens via whisper worker later
                        
                        # If not recording, just continue listening
                        if not is_recording_for_whisper:
                             update_vad_status("Listening...", "gray")


                # Prevent buffer from growing indefinitely if VAD somehow fails often
                # (deque maxlen should handle this, but as a safeguard)
                # if len(vad_audio_buffer) > vad_audio_buffer.maxlen * 2:
                #     print("[VAD Worker] Warning: Audio buffer seems overly large, clearing.")
                #     vad_audio_buffer.clear()


            except IOError as e:
                # Handle stream errors, e.g., input overflow
                if e.errno == pyaudio.paInputOverflowed:
                    print("[VAD Worker] Warning: Input overflowed. Skipping frame.")
                else:
                    print(f"[VAD Worker] Stream read error: {e}")
                    # Maybe try to reopen stream or signal error? For now, just continue.
                    time.sleep(0.1) # Avoid busy-looping on error
            except Exception as e:
                print(f"[VAD Worker] Unexpected error: {e}")
                time.sleep(0.1)

    except Exception as e:
        print(f"[VAD Worker] Failed to open audio stream: {e}")
        update_vad_status("Audio Error!", "red")
        # Signal failure? Maybe disable voice recognition?
    finally:
        if stream:
            try:
                stream.stop_stream()
                stream.close()
                print("[VAD Worker] Audio stream closed.")
            except Exception as e:
                print(f"[VAD Worker] Error closing stream: {e}")
        # Reset state on exit
        is_recording_for_whisper = False
        audio_frames_buffer.clear()
        vad_audio_buffer.clear()
        if not vad_stop_event.is_set(): # If exited due to error, not explicit stop
            update_vad_status("VAD Stopped (Error)", "red")
        else:
             update_vad_status("Voice Disabled", "grey") # Normal stop

    print("[VAD Worker] Thread finished.")


def process_audio_worker():
    """Worker thread to transcribe audio files from the whisper_queue."""
    global whisper_model, whisper_initialized, whisper_queue, whisper_language
    print("[Whisper Worker] Thread started.")
    while True:
        try:
            audio_file_path = whisper_queue.get() # Blocks
            if audio_file_path is None: # Sentinel
                print("[Whisper Worker] Received stop signal.")
                break

            if not whisper_initialized or not voice_enabled.get():
                print("[Whisper Worker] Skipping transcription (disabled or not initialized).")
                try: os.unlink(audio_file_path) # Clean up file
                except Exception: pass
                whisper_queue.task_done()
                continue

            print(f"[Whisper Worker] Processing audio file: {audio_file_path}")
            update_vad_status("Transcribing...", "orange")
            
            start_time = time.time()
            try:
                 # Use selected language, None means auto-detect
                lang_to_use = whisper_language if whisper_language else None

                if whisper_model_size.startswith("turbo"):
                    # faster-whisper (turbo) model
                    segments, info = whisper_model.transcribe(audio_file_path, language=lang_to_use)
                    transcribed_text = ""
                    for segment in segments:
                        transcribed_text += segment.text + " "
                    transcribed_text = transcribed_text.strip()
                else:
                    result = whisper_model.transcribe(audio_file_path, language=lang_to_use)
                    transcribed_text = result["text"].strip()

                print(f"[Whisper Worker] Transcription complete in {time.time() - start_time:.2f}s: '{transcribed_text}'")

                if transcribed_text:
                     # Schedule UI update on main thread
                    window.after(0, update_input_with_transcription, transcribed_text)
                    update_vad_status("Transcription Ready", "green")
                else:
                    update_vad_status("No speech detected", "orange")

            except Exception as e:
                print(f"[Whisper Worker] Error during transcription: {e}")

                traceback.print_exc() 
                update_vad_status("Transcription Error", "red")
            finally:
                # Clean up the temporary audio file
                try: os.unlink(audio_file_path)
                except Exception as e: print(f"[Whisper Worker] Error deleting temp file {audio_file_path}: {e}")

            whisper_queue.task_done()

        except Exception as e:
            # Catch errors in the loop itself
            print(f"[Whisper Worker] Error: {e}")

    print("[Whisper Worker] Thread finished.")


def update_input_with_transcription(text):
    """Updates the user input text box with the transcribed text."""
    global user_input_widget
    if not user_input_widget: return

    current_text = user_input_widget.get("1.0", tk.END).strip()
    if current_text:
        user_input_widget.insert(tk.END, " " + text)
    else:
        user_input_widget.insert("1.0", text)
    # Optionally: Automatically send message after transcription?
    # send_message()


def toggle_voice_recognition():
    """Enables/disables VAD and Whisper."""
    global voice_enabled, whisper_initialized, vad_initialized, vad_thread, vad_stop_event
    global py_audio, whisper_processing_thread

    if voice_enabled.get():
        print("[Voice] Enabling voice recognition...")
        all_initialized = True
        if not py_audio:
            if not initialize_audio_system(): all_initialized = False
        if not whisper_initialized:
            if not initialize_whisper(): all_initialized = False
        if not vad_initialized:
            if not initialize_vad(): all_initialized = False

        if all_initialized:
            # Start Whisper processing thread if not running
            if whisper_processing_thread is None or not whisper_processing_thread.is_alive():
                whisper_processing_thread = threading.Thread(target=process_audio_worker, daemon=True)
                whisper_processing_thread.start()

            # Start VAD worker thread if not running
            if vad_thread is None or not vad_thread.is_alive():
                vad_stop_event.clear() # Reset stop event
                vad_thread = threading.Thread(target=vad_worker, daemon=True)
                vad_thread.start()
            update_vad_status("Voice Enabled", "green") # Initial status before VAD starts listening
            print("[Voice] Voice recognition enabled.")
        else:
            print("[Voice] Enabling failed due to initialization errors.")
            voice_enabled.set(False) # Uncheck the box
            update_vad_status("Init Failed", "red")
            # UI cleanup/disable happens in the init functions or here if needed

    else: # Disabling voice recognition
        print("[Voice] Disabling voice recognition...")
        update_vad_status("Disabling...", "grey")
        if vad_thread and vad_thread.is_alive():
            vad_stop_event.set() # Signal VAD worker to stop
            vad_thread.join(timeout=2)
            if vad_thread.is_alive():
                print("[Voice] Warning: VAD thread did not stop gracefully.")
            vad_thread = None
        # VAD worker handles stream closing; status update happens in its finally block
        print("[Voice] Voice recognition disabled.")
        # Note: We keep the whisper processing thread alive, it waits on the queue.
        # We also keep PyAudio initialized unless explicitly closed on app exit.

def set_whisper_language(event=None):
    """Sets the language for Whisper transcription."""
    global whisper_language, whisper_language_selector
    selected_idx = whisper_language_selector.current()
    if selected_idx >= 0:
        lang_name, lang_code = WHISPER_LANGUAGES[selected_idx]
        whisper_language = lang_code
        print(f"[Whisper] Language set to: {lang_name} ({whisper_language})")

def set_whisper_model_size(event=None):
    """Sets Whisper model size and triggers re-initialization if needed."""
    global whisper_model_size, whisper_model_size_selector, whisper_initialized, whisper_model
    new_size = whisper_model_size_selector.get()
    if new_size == whisper_model_size and whisper_initialized:
        return # No change

    print(f"[Whisper] Model size changed to: {new_size}")

    whisper_model_size = new_size
    whisper_initialized = False # Force re-initialization
    whisper_model = None # Release old model (let GC handle it)

    # If voice is currently enabled, re-initialize immediately
    if voice_enabled.get():
        initialize_whisper() # This will update status label

def update_vad_status(text, color):
    """Safely updates the VAD status label from any thread."""
    if recording_indicator_widget and window:
        try:
            # Use schedule to run on main thread
            window.after(0, lambda: recording_indicator_widget.config(text=text, fg=color))
        except tk.TclError:
            # Handle case where window might be closing
            pass


# ===================
# Ollama / Chat Logic
# ===================
def fetch_available_models():
    """Fetches available Ollama models."""
    try:
        models_data = ollama.list()
        return [model['name'] for model in models_data.get('models', [])]
    except Exception as e:
        print(f"[Ollama] Error fetching models: {e}")
        # Provide common fallbacks if API fails
        return [DEFAULT_OLLAMA_MODEL, "llama3:8b", "phi3:mini"]

def chat_worker(user_message_content, image_path=None):
    """Background worker for Ollama streaming chat."""
    global messages, current_model, stream_queue, stream_done_event, stream_in_progress

    # Construct the message with potential image
    current_message = {"role": "user", "content": user_message_content}
    if image_path:
        try:
             # Read the image as bytes - Ollama Python library expects bytes
            with open(image_path, 'rb') as f:
                 image_bytes = f.read()
            current_message["images"] = [image_bytes] # Pass bytes directly
        except Exception as e:
             err_text = f"Error reading image file '{os.path.basename(image_path)}': {e}"
             stream_queue.put(("ERROR", err_text))
             stream_in_progress = False
             stream_done_event.set()
             return # Stop processing if image fails

    # Add user message to history *before* sending to Ollama
    messages.append(current_message)
    
    # Prepare message history for Ollama (copy to avoid modifying global during iteration)
    history_for_ollama = list(messages) 

    assistant_response = "" # Accumulate full response for history

    try:
        print(f"[Ollama] Sending request to model {current_model}...")
        stream = ollama.chat(model=current_model, messages=history_for_ollama, stream=True)
        
        first_chunk = True
        for chunk in stream:
            if 'message' in chunk and 'content' in chunk['message']:
                content_piece = chunk['message']['content']
                if first_chunk:
                     stream_queue.put(("START", None)) # Signal start of response
                     first_chunk = False
                stream_queue.put(("CHUNK", content_piece))
                assistant_response += content_piece # Accumulate for history

            # Check for potential errors within the stream (less common)
            if 'error' in chunk:
                 stream_queue.put(("ERROR", f"Ollama error: {chunk['error']}"))
                 break # Stop processing on stream error

        # After successful streaming, add the complete assistant response to history
        if assistant_response:
             messages.append({"role": "assistant", "content": assistant_response})
             
        stream_queue.put(("END", None)) # Signal end of response stream

    except Exception as e:
        err_text = f"Ollama communication error: {e}"
        stream_queue.put(("ERROR", err_text))
    finally:
        stream_in_progress = False
        stream_done_event.set() # Signal completion/error


def process_stream_queue():
    """Processes items from Ollama stream queue for UI and TTS."""
    global stream_queue, chat_history_widget, tts_sentence_buffer

    try:
        while True: # Process all available items
            item_type, item_data = stream_queue.get_nowait()

            if item_type == "START":
                 # Optional: Clear "Thinking..." message here if needed,
                 # but usually handled when first chunk arrives.
                 pass # Placeholder
            elif item_type == "CHUNK":
                 # Append chunk to UI
                if chat_history_widget:
                    chat_history_widget.config(state=tk.NORMAL)
                    chat_history_widget.insert(tk.END, item_data, "bot_message")
                    chat_history_widget.config(state=tk.DISABLED)
                    chat_history_widget.see(tk.END)
                # Queue text for TTS
                queue_tts_text(item_data)
            elif item_type == "END":
                 # Add final newline for spacing in chat history
                 if chat_history_widget:
                     chat_history_widget.config(state=tk.NORMAL)
                     chat_history_widget.insert(tk.END, "\n\n")
                     chat_history_widget.config(state=tk.DISABLED)
                     chat_history_widget.see(tk.END)
                 # Flush any remaining partial sentence in TTS buffer
                 queue_tts_text("\n") # Add newline to force potential flush
                 try_flush_tts_buffer()
                 return # Stop polling for this response
            elif item_type == "ERROR":
                 add_message_to_ui("error", item_data)
                 # Ensure Ollama state is reset
                 global stream_in_progress
                 stream_in_progress = False
                 return # Stop polling

    except queue.Empty:
        pass # No items currently in queue

    # Reschedule if Ollama is still processing
    if stream_in_progress:
        window.after(100, process_stream_queue) # Check again in 100ms


# ===================
# UI Helpers
# ===================
def select_image():
    """Opens dialog to select an image file."""
    global selected_image_path, image_sent_in_history
    file_path = filedialog.askopenfilename(
        title="Select Image File",
        filetypes=[("Image files", "*.png;*.jpg;*.jpeg;*.gif;*.bmp")]
    )
    if file_path:
        selected_image_path = file_path
        image_sent_in_history = False # Reset flag when new image selected
        update_image_preview(file_path)
        image_indicator.config(text=f"✓ {os.path.basename(file_path)}")

def clear_image():
    """Clears the selected image."""
    global selected_image_path, image_sent_in_history
    selected_image_path = ""
    image_sent_in_history = False
    image_preview.configure(image="", text="No Image", width=20, height=10, bg="lightgrey")
    image_preview.image = None # Clear reference
    image_indicator.config(text="No image attached")

def update_image_preview(file_path):
    """Updates the image preview label."""
    try:
        img = Image.open(file_path)
        max_size = 150 # Slightly smaller preview
        img.thumbnail((max_size, max_size), Image.LANCZOS)
        photo = ImageTk.PhotoImage(img)
        image_preview.configure(image=photo, width=img.width, height=img.height, text="")
        image_preview.image = photo # Keep reference
    except Exception as e:
        print(f"Error updating image preview: {e}")
        clear_image() # Reset if preview fails
        image_indicator.config(text="Preview Error", fg="red")

def add_message_to_ui(role, content, tag_suffix=""):
    """Adds a message to the chat history widget."""
    global chat_history_widget
    if not chat_history_widget:
        print("Warning: chat_history_widget not ready.")
        return

    if not content: # Don't add empty messages
         return

    chat_history_widget.config(state=tk.NORMAL)
    if role == "user":
        chat_history_widget.insert(tk.END, "You: ", "user_tag")
        chat_history_widget.insert(tk.END, content + "\n\n", "user_message"+tag_suffix)
    elif role == "assistant":
        chat_history_widget.insert(tk.END, "Ollama: ", "bot_tag")
        # Content appended piece by piece for streaming
    elif role == "error":
        chat_history_widget.insert(tk.END, f"Error: {content}\n\n", "error"+tag_suffix)
    elif role == "status":
         chat_history_widget.insert(tk.END, f"{content}\n\n", "status"+tag_suffix)

    chat_history_widget.see(tk.END)
    chat_history_widget.config(state=tk.DISABLED)
    if 'window' in globals() and window:
         try: window.update_idletasks() # Ensure UI updates promptly
         except tk.TclError: pass # Ignore errors if window is closing


def select_model(event=None):
    """Updates the selected Ollama model."""
    global current_model, model_selector, model_status
    selected = model_selector.get()
    if selected and selected != "No models found":
        current_model = selected
        model_status.config(text=f"Using: {current_model.split(':')[0]}") # Show base name
        print(f"[Ollama] Model selected: {current_model}")

def send_message(event=None):
    """Handles sending the user's message to Ollama."""
    global messages, selected_image_path, image_sent_in_history
    global stream_in_progress, stream_done_event, user_input_widget, tts_sentence_buffer

    if stream_in_progress:
        add_message_to_ui("error", "Please wait for the current response to complete.")
        return

    user_text = user_input_widget.get("1.0", tk.END).strip()
    image_to_send = selected_image_path if not image_sent_in_history else None

    if not user_text and not image_to_send:
        add_message_to_ui("error", "Please enter a message or attach a new image.")
        return

    # --- Prepare for sending ---
    send_button.config(state=tk.DISABLED) # Disable send button
    stream_in_progress = True
    stream_done_event.clear()

    # Add user message to UI immediately
    display_text = user_text
    if image_to_send:
        display_text += f" [Image: {os.path.basename(image_to_send)}]"
    add_message_to_ui("user", display_text if display_text else "[Image Attached]")

    # Clear input AFTER adding to UI
    user_input_widget.delete("1.0", tk.END)

    # Add "Thinking" indicator in the chat history
    chat_history_widget.config(state=tk.NORMAL)
    chat_history_widget.insert(tk.END, "Ollama: ", "bot_tag")
    thinking_text = "Thinking...\n"
    thinking_start_index = chat_history_widget.index(tk.INSERT)
    chat_history_widget.insert(tk.INSERT, thinking_text, "thinking")
    chat_history_widget.see(tk.END)
    chat_history_widget.config(state=tk.DISABLED)
    window.update_idletasks()

    # --- Stop any ongoing TTS and clear buffers ---
    if tts_engine and tts_enabled.get():
        try: tts_engine.stop()
        except: pass
        # Clear queue and buffer
        while not tts_queue.empty():
            try: tts_queue.get_nowait()
            except queue.Empty: break
        tts_sentence_buffer = ""
        with tts_busy_lock: tts_busy = False # Reset busy state

    # --- Start Ollama worker thread ---
    thread = threading.Thread(target=chat_worker, args=(user_text, image_to_send), daemon=True)
    thread.start()

    # --- Set flags and start polling ---
    if image_to_send:
        image_sent_in_history = True # Mark image as sent for this context
        # Optionally clear image selection after sending, or require manual clear
        # clear_image() # Uncomment to automatically clear after send

    # Function to run after worker finishes (success or error)
    def on_worker_done():
         # Remove "Thinking..." message
        try:
            chat_history_widget.config(state=tk.NORMAL)
            # Calculate end index precisely
            end_thinking = f"{thinking_start_index}+{len(thinking_text)}c"
            # Check if the "Thinking..." text is still where we expect it
            current_text = chat_history_widget.get(thinking_start_index, end_thinking)
            if current_text == thinking_text:
                 chat_history_widget.delete(thinking_start_index, end_thinking)
            else:
                 # Text has changed (response started), no need to delete
                 pass
            chat_history_widget.config(state=tk.DISABLED)
        except tk.TclError:
            pass # Ignore errors if widget is destroyed
        finally:
             send_button.config(state=tk.NORMAL) # Re-enable send button

    # Use stream_done_event to trigger UI update
    def check_done():
        if stream_done_event.is_set():
            window.after(0, on_worker_done)
        else:
            window.after(100, check_done) # Check again

    window.after(100, process_stream_queue) # Start processing queue immediately
    window.after(100, check_done) # Start checking if worker is done


# ===================
# Main Application Setup & Loop
# ===================
def on_closing():
    """Handles application shutdown gracefully."""
    print("[Main] Closing application...")

    # 1. Signal VAD thread to stop first (accesses audio resources)
    if vad_thread and vad_thread.is_alive():
        print("[Main] Stopping VAD thread...")
        vad_stop_event.set()
        vad_thread.join(timeout=2) # Wait for VAD to finish closing stream
        if vad_thread.is_alive(): print("[Main] Warning: VAD thread did not exit cleanly.")

    # 2. Stop Whisper processing thread
    if whisper_processing_thread and whisper_processing_thread.is_alive():
         print("[Main] Stopping Whisper processing thread...")
         whisper_queue.put(None) # Send sentinel
         whisper_processing_thread.join(timeout=2)
         if whisper_processing_thread.is_alive(): print("[Main] Warning: Whisper thread did not exit cleanly.")

    # 3. Stop TTS thread
    stop_tts_thread() # Handles engine stop and thread join

    # 4. Terminate PyAudio (after streams are closed by VAD worker)
    if py_audio:
        print("[Main] Terminating PyAudio...")
        py_audio.terminate()

    # 5. Clean up temporary file if it exists
    global temp_audio_file_path
    if temp_audio_file_path and os.path.exists(temp_audio_file_path):
        try:
            print(f"[Main] Deleting temporary audio file: {temp_audio_file_path}")
            os.unlink(temp_audio_file_path)
        except Exception as e:
            print(f"[Main] Error deleting temp file: {e}")

    # 6. Destroy the Tkinter window
    print("[Main] Destroying window...")
    window.destroy()
    print("[Main] Application closed.")


# --- Build GUI ---
window = tk.Tk()
window.title(APP_TITLE)
window.geometry(WINDOW_GEOMETRY)

# --- Tkinter Variables ---
tts_enabled = tk.BooleanVar(value=False)
tts_rate = tk.IntVar(value=160)
tts_voice_id = tk.StringVar(value="")
voice_enabled = tk.BooleanVar(value=False)
selected_whisper_language = tk.StringVar()
selected_whisper_model = tk.StringVar(value=whisper_model_size)

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

available_models = fetch_available_models()
model_selector = ttk.Combobox(model_frame, values=available_models, state="readonly", width=25)
if available_models:
    if current_model in available_models:
        model_selector.set(current_model)
    elif DEFAULT_OLLAMA_MODEL in available_models:
         model_selector.set(DEFAULT_OLLAMA_MODEL)
         current_model = DEFAULT_OLLAMA_MODEL
    else:
        model_selector.set(available_models[0])
        current_model = available_models[0]
else:
    model_selector.set("No models found")
    model_selector.config(state=tk.DISABLED)
    current_model = None

model_selector.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0,5))
model_selector.bind("<<ComboboxSelected>>", select_model)

model_status = tk.Label(model_frame, text=f"Using: {current_model.split(':')[0] if current_model else 'N/A'}", font=("Arial", 8), width=15, anchor="w")
model_status.pack(side=tk.LEFT)

# --- TTS Controls ---
tts_outer_frame = tk.LabelFrame(top_controls_frame, text="Text-to-Speech", padx=5, pady=5)
tts_outer_frame.grid(row=0, column=1, sticky="ns", padx=5)

tts_toggle_button = ttk.Checkbutton(tts_outer_frame, text="Enable TTS", variable=tts_enabled, command=toggle_tts)
tts_toggle_button.pack(anchor="w", pady=2)

# Voice Selector
voice_frame = tk.Frame(tts_outer_frame)
voice_frame.pack(fill=tk.X, pady=2)
tk.Label(voice_frame, text="Voice:", font=("Arial", 8)).pack(side=tk.LEFT)
available_voices = get_available_voices()
voice_names = [v[0] for v in available_voices]
voice_selector = ttk.Combobox(voice_frame, values=voice_names, state="disabled", width=18, font=("Arial", 8)) # Start disabled
if available_voices:
    # Try to find a default or common voice (e.g., Zira, David, Hazel for Windows/Mac)
    default_voice_index = 0
    for i, (name, v_id) in enumerate(available_voices):
        if any(common in name for common in ["Zira", "David", "Hazel", "Susan"]):
            default_voice_index = i
            break
    voice_selector.current(default_voice_index)
    tts_voice_id.set(available_voices[default_voice_index][1])
    voice_selector.bind("<<ComboboxSelected>>", set_voice)
voice_selector.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=2)

# Rate Control
rate_frame = tk.Frame(tts_outer_frame)
rate_frame.pack(fill=tk.X, pady=2)
tk.Label(rate_frame, text="Rate:", font=("Arial", 8)).pack(side=tk.LEFT)
rate_scale = ttk.Scale(rate_frame, from_=80, to=300, orient=tk.HORIZONTAL, variable=tts_rate, command=set_speech_rate, state="disabled") # Start disabled
rate_scale.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=2)
rate_value = tk.Label(rate_frame, textvariable=tts_rate, width=3, font=("Arial", 8))
rate_value.pack(side=tk.LEFT)

# --- Voice Recognition Controls ---
voice_outer_frame = tk.LabelFrame(top_controls_frame, text="Voice Input (VAD)", padx=5, pady=5)
voice_outer_frame.grid(row=0, column=2, sticky="ns", padx=(5, 0))

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
whisper_language_selector.current(0) # Default to Auto Detect
whisper_language_selector.pack(side=tk.LEFT, padx=2)
whisper_language_selector.bind("<<ComboboxSelected>>", set_whisper_language)

# Model Size Selector
size_frame = tk.Frame(whisper_settings_frame)
size_frame.pack(fill=tk.X, pady=(2,0))
tk.Label(size_frame, text="Model:", font=("Arial", 8)).pack(side=tk.LEFT)
whisper_model_size_selector = ttk.Combobox(size_frame, values=WHISPER_MODEL_SIZES,
                                           state="readonly", width=10, font=("Arial", 8),
                                           textvariable=selected_whisper_model)
whisper_model_size_selector.pack(side=tk.LEFT, padx=2)
whisper_model_size_selector.bind("<<ComboboxSelected>>", set_whisper_model_size)


# Recording Status Indicator (moved here for better placement)
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
chat_history_widget.tag_config("error", foreground="red", font=("Arial", 10, "bold"))
chat_history_widget.tag_config("status", foreground="purple", font=("Arial", 9, "italic"))


# --- Bottom Frame (Image + Input) ---
bottom_frame = tk.Frame(main_frame)
bottom_frame.pack(fill=tk.X, expand=False)

# Image Frame (Left)
image_frame = tk.LabelFrame(bottom_frame, text="Image Attachment", padx=5, pady=5)
image_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))

image_preview = tk.Label(image_frame, text="No Image", width=20, height=8, bg="lightgrey", relief="sunken") # Adjusted size
image_preview.pack(pady=5)

img_button_frame = tk.Frame(image_frame)
img_button_frame.pack(fill=tk.X, pady=2)
select_button = tk.Button(img_button_frame, text="Select", command=select_image, width=8)
select_button.pack(side=tk.LEFT, expand=True, padx=2)
clear_button = tk.Button(img_button_frame, text="Clear", command=clear_image, width=8)
clear_button.pack(side=tk.LEFT, expand=True, padx=2)

image_indicator = tk.Label(image_frame, text="No image attached", font=("Arial", 8, "italic"), fg="grey")
image_indicator.pack(pady=(3,0))


# Input Frame (Right - Takes remaining space)
input_frame = tk.LabelFrame(bottom_frame, text="Your Message (Press Enter to Send, Shift+Enter for Newline)", padx=10, pady=5)
input_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

user_input_widget = scrolledtext.ScrolledText(input_frame, wrap=tk.WORD, height=4, font=("Arial", 10)) # Adjusted height
user_input_widget.pack(fill=tk.BOTH, expand=True, pady=(0, 5))
user_input_widget.focus_set()

send_button = tk.Button(input_frame, text="Send", command=send_message, width=10)
send_button.pack(pady=5, anchor='e') # Anchor to the right

# --- Enter Key Binding ---
def on_enter_press(event):
    if event.state & 0x0001: # Check if Shift key is pressed
        # Allow default behavior (insert newline)
        return
    else:
        # Trigger send message
        send_message()
        # Prevent default behavior (which would insert newline)
        return "break"

user_input_widget.bind("<KeyPress-Return>", on_enter_press)


# --- Final Setup and Initialization ---
window.protocol("WM_DELETE_WINDOW", on_closing) # Set close handler

# Attempt to initialize TTS early but don't block startup
# Let toggle_tts handle enabling controls based on success
print("[Main] Pre-initializing TTS...")
initialize_tts()
if tts_initialized_successfully:
    voice_selector.config(state="readonly")
    rate_scale.config(state="normal")
else:
    # Add startup message about TTS failure
    def show_tts_init_error():
         add_message_to_ui("status", "Note: TTS engine failed to initialize. TTS controls disabled.")
    window.after(500, show_tts_init_error) # Delay slightly

# Start periodic checks (TTS flush)
window.after(200, periodic_tts_check)

# Add initial welcome message
add_message_to_ui("assistant", "Welcome! Select a model, type a message, attach an image, or enable voice input.")

# --- Start Main Loop ---
window.mainloop()