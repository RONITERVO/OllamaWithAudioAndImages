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
import pyaudio
import wave
import numpy as np
import tempfile
import os

# ===================
# Globals
# ===================
messages = []
selected_image_path = ""
image_sent_in_history = False
current_model = "gemma3:27b"  # Default model

# -- TTS Globals --
tts_engine = None
tts_queue = queue.Queue()
tts_thread = None
tts_sentence_buffer = ""
tts_enabled = None  # Will be a tk.BooleanVar, set after tk.Tk() is created

# We track if TTS is actively speaking. We'll poll in the main thread
# to decide when to send the next big chunk.
tts_busy = False
tts_busy_lock = threading.Lock()

# -- Streaming Globals --
stream_queue = queue.Queue()    # For collecting streaming chunks from Ollama
stream_done_event = threading.Event()  # Signals when worker has finished or errored
stream_in_progress = False      # Simple flag to prevent multiple concurrent sends

# -- Whisper/Voice Recognition Globals --
whisper_model = None
whisper_model_size = "large"  # Options: tiny, base, small, medium, large
whisper_initialized = False
whisper_thread = None
whisper_queue = queue.Queue()
recording = False
audio_stream = None
audio_frames = []
py_audio = None
record_button = None
recording_indicator = None
temp_audio_file = None
voice_enabled = None  # Will be a tk.BooleanVar
whisper_language = "en"  # Default language 
whisper_language_selector = None
whisper_model_size_selector = None

# Common languages - you can expand this list
WHISPER_LANGUAGES = [
    ("Auto Detect", None),
    ("English", "en"),
    ("Finnish", "fi"),
    ("Swedish", "sv"),
    ("German", "de"),
    ("French", "fr"),
    ("Spanish", "es"),
    ("Italian", "it"),
    ("Russian", "ru"),
    ("Chinese", "zh"),
    ("Japanese", "ja")
]

WHISPER_MODEL_SIZES = ["tiny", "base", "small", "medium", "large"]

# ===================
# TTS Setup
# ===================
def initialize_tts():
    """Initializes the TTS engine."""
    global tts_engine
    try:
        print("Initializing TTS engine...")
        tts_engine = pyttsx3.init()
        tts_engine.setProperty('rate', 160)   # Adjust speech rate
        tts_engine.setProperty('volume', 0.9) # Adjust volume
        print("TTS engine initialized successfully.")
        return True
    except Exception as e:
        print(f"Error initializing TTS engine: {e}")
        tts_engine = None
        return False

def tts_worker():
    """Worker function that processes text-to-speech in a loop."""
    global tts_engine, tts_queue, tts_busy
    print("TTS worker thread started.")
    while True:
        try:
            text_to_speak = tts_queue.get()  # blocks
            if text_to_speak is None:  # sentinel to stop
                print("TTS worker received stop signal.")
                break

            if tts_engine and tts_enabled.get():
                # Mark TTS busy
                with tts_busy_lock:
                    global tts_busy
                    tts_busy = True

                print(f"[TTS] Speaking chunk ({len(text_to_speak)} chars)...")
                start_time = time.time()
                tts_engine.say(text_to_speak)
                tts_engine.runAndWait()
                print(f"[TTS] Finished in {time.time() - start_time:.2f}s")

                # Mark TTS not busy
                with tts_busy_lock:
                    tts_busy = False
            else:
                print("[TTS] Disabled or uninitialized, discarding text.")

            tts_queue.task_done()
        except Exception as e:
            print(f"Error in TTS worker: {e}")

def start_tts_thread():
    """Starts the TTS worker thread if not already running."""
    global tts_thread, tts_engine_initialized_successfully
    if tts_thread is None or not tts_thread.is_alive():
        if tts_engine is None:
            tts_engine_initialized_successfully = initialize_tts()
        else:
            tts_engine_initialized_successfully = True

        if tts_engine_initialized_successfully:
            tts_thread = threading.Thread(target=tts_worker, daemon=True)
            tts_thread.start()
            print("[TTS] Thread started.")
        else:
            print("[TTS] Engine init failed. Disabling TTS.")
            if 'tts_toggle_button' in globals():
                tts_toggle_button.config(state=tk.DISABLED)
                tts_enabled.set(False)
                add_message_to_ui("error", "TTS Engine failed to initialize.")

def stop_tts_thread():
    """Signals the TTS worker thread to stop."""
    global tts_queue, tts_thread, tts_engine
    print("[TTS] Stopping TTS thread...")
    if tts_engine:
        try:
            tts_engine.stop()
        except Exception as e:
            print(f"Error stopping engine: {e}")

    if tts_thread and tts_thread.is_alive():
        tts_queue.put(None)  # sentinel
        tts_thread.join(timeout=2)
        if tts_thread.is_alive():
            print("[TTS] Warning: TTS thread did not terminate gracefully.")
    tts_thread = None
    print("[TTS] TTS thread stopped.")

def toggle_tts():
    """Callback for the TTS checkbox."""
    global tts_engine, tts_queue, tts_sentence_buffer
    if tts_enabled.get():
        print("[TTS] Enabled")
        start_tts_thread()
    else:
        print("[TTS] Disabled")
        # Clear queue
        while not tts_queue.empty():
            try:
                tts_queue.get_nowait()
            except queue.Empty:
                break
        if tts_engine:
            try:
                tts_engine.stop()
            except Exception as e:
                print(f"Error stopping TTS engine on toggle: {e}")
        tts_sentence_buffer = ""

def queue_tts_text(new_text):
    """
    Accumulate new streaming text into a global buffer.
    We'll flush to the TTS queue only after TTS finishes speaking
    the previous chunk.
    """
    global tts_sentence_buffer
    if not tts_enabled.get() or not tts_engine:
        return

    tts_sentence_buffer += new_text

def try_flush_tts_buffer():
    """
    If TTS is idle, extract all *complete sentences* from tts_sentence_buffer
    and send them as one large chunk to the TTS queue. 
    Leave any trailing partial sentence in the buffer.
    """
    global tts_sentence_buffer, tts_busy

    if not tts_enabled.get() or not tts_engine:
        return

    # Make sure TTS is not busy
    with tts_busy_lock:
        if tts_busy:
            return

    # Split on sentence endings (approx. . ! ? or newline)
    sentences = re.split(r'(?<=[.!?\n])(\s+)', tts_sentence_buffer)

    if len(sentences) < 2:
        return  # no complete sentence yet

    accumulated = []
    leftover = ""
    i = 0
    while i < len(sentences):
        sentence_part = sentences[i].rstrip("\n")
        whitespace = ""
        if i+1 < len(sentences):
            whitespace = sentences[i+1]
        i += 2

        if not sentence_part:
            continue

        if re.search(r'[.!?]$', sentence_part.strip()):
            accumulated.append(sentence_part + whitespace)
        else:
            leftover = sentence_part + whitespace
            break

    chunk_to_speak = "".join(accumulated).strip()
    leftover = leftover + "".join(sentences[i:])

    tts_sentence_buffer = leftover

    if chunk_to_speak:
        print(f"[TTS] Queuing chunk of length {len(chunk_to_speak)}")
        tts_queue.put(chunk_to_speak)
        with tts_busy_lock:
            tts_busy = True

def periodic_tts_check():
    """
    Periodically invoked in the main thread.
    If TTS is idle, attempt to flush TTS buffer with any complete sentences.
    Then re-schedule itself.
    """
    try_flush_tts_buffer()
    window.after(200, periodic_tts_check)

# ===================
# Whisper/Voice Setup
# ===================
def initialize_whisper():
    """Initialize the Whisper model."""
    global whisper_model, whisper_initialized
    try:
        print(f"Initializing Whisper model ({whisper_model_size})...")
        whisper_model = whisper.load_model(whisper_model_size)
        whisper_initialized = True
        print("Whisper model initialized successfully.")
        return True
    except Exception as e:
        print(f"Error initializing Whisper model: {e}")
        whisper_initialized = False
        return False

def start_recording():
    """Start recording audio from microphone."""
    global recording, audio_stream, audio_frames, py_audio, temp_audio_file
    
    if recording:
        return  # Already recording
    
    if not py_audio:
        try:
            py_audio = pyaudio.PyAudio()
        except Exception as e:
            add_message_to_ui("error", f"Failed to initialize audio: {e}")
            return
    
    temp_audio_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    temp_audio_file.close()
    
    audio_frames = []
    
    try:
        audio_stream = py_audio.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=16000,
            input=True,
            frames_per_buffer=1024,
            stream_callback=audio_callback
        )
        recording = True
        record_button.config(text="Stop Recording", bg="#ff6666")
        recording_indicator.config(text="Recording...", fg="red")
        print("Recording started")
    except Exception as e:
        add_message_to_ui("error", f"Failed to start recording: {e}")
        recording = False

def audio_callback(in_data, frame_count, time_info, status):
    """Callback for audio stream to collect frames."""
    global audio_frames
    audio_frames.append(in_data)
    return (in_data, pyaudio.paContinue)

def stop_recording():
    """Stop recording and process the audio."""
    global recording, audio_stream, audio_frames, py_audio, temp_audio_file
    
    if not recording:
        return
    
    try:
        audio_stream.stop_stream()
        audio_stream.close()
        audio_stream = None
        recording = False
        
        record_button.config(text="Record Voice", bg="#4CAF50")
        recording_indicator.config(text="Processing audio...", fg="blue")
        
        with wave.open(temp_audio_file.name, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(py_audio.get_sample_size(pyaudio.paInt16))
            wf.setframerate(16000)
            wf.writeframes(b''.join(audio_frames))
        
        whisper_queue.put(temp_audio_file.name)
        
        if not whisper_thread or not whisper_thread.is_alive():
            thread = threading.Thread(target=process_audio_worker, daemon=True)
            thread.start()
        
    except Exception as e:
        add_message_to_ui("error", f"Failed to stop recording: {e}")
        recording = False
        record_button.config(text="Record Voice", bg="#4CAF50")
        recording_indicator.config(text="Recording error", fg="red")

def toggle_recording():
    """Toggle between starting and stopping recording."""
    if recording:
        stop_recording()
    else:
        start_recording()

def process_audio_worker():
    """Worker thread to process audio with Whisper."""
    global whisper_model, whisper_initialized, voice_enabled, whisper_language
    
    while True:
        try:
            audio_file = whisper_queue.get()
            if audio_file is None:
                break
            
            if not whisper_initialized or not voice_enabled.get():
                continue
            
            window.after(0, lambda: recording_indicator.config(text="Transcribing...", fg="blue"))
            
            # Use the selected language for transcription (if None, Whisper will auto-detect)
            result = whisper_model.transcribe(
                audio_file,
                language=whisper_language
            )
            transcribed_text = result["text"].strip()
            
            window.after(0, lambda text=transcribed_text: update_with_transcription(text))
            
            try:
                os.unlink(audio_file)
            except Exception:
                pass
                
        except Exception as e:
            print(f"Error processing audio: {e}")
            window.after(0, lambda: recording_indicator.config(text="Transcription error", fg="red"))
        finally:
            whisper_queue.task_done()

def update_with_transcription(text):
    """Update the UI with transcribed text."""
    global recording_indicator
    
    if not text:
        recording_indicator.config(text="No speech detected", fg="orange")
        return
    
    current_text = user_input.get("1.0", tk.END).strip()
    if current_text:
        user_input.insert(tk.END, " " + text)
    else:
        user_input.insert("1.0", text)
    
    recording_indicator.config(text=f"Transcribed: {text[:20]}..." if len(text) > 20 else f"Transcribed: {text}", fg="green")

def toggle_voice_recognition():
    """Toggle voice recognition on/off."""
    global voice_enabled, whisper_initialized, recording
    
    if voice_enabled.get() and not whisper_initialized:
        success = initialize_whisper()
        if not success:
            voice_enabled.set(False)
            voice_toggle_button.config(state=tk.DISABLED)
            add_message_to_ui("error", "Failed to initialize Whisper model. Voice recognition disabled.")
            return
    
    if not voice_enabled.get() and recording:
        stop_recording()
    
    if voice_enabled.get():
        record_button.config(state=tk.NORMAL)
        print("Voice recognition enabled")
    else:
        record_button.config(state=tk.DISABLED)
        print("Voice recognition disabled")

def set_whisper_language(event=None):
    """Set the language for Whisper transcription."""
    global whisper_language, whisper_language_selector
    selected_idx = whisper_language_selector.current()
    if selected_idx >= 0:
        _, whisper_language = WHISPER_LANGUAGES[selected_idx]
        print(f"Whisper language set to: {whisper_language if whisper_language else 'Auto Detect'}")

def set_whisper_model_size(event=None):
    """Set the model size for Whisper and reinitialize if needed."""
    global whisper_model_size, whisper_model_size_selector, whisper_initialized, whisper_model
    
    new_size = whisper_model_size_selector.get()
    if new_size == whisper_model_size:
        return
    
    whisper_model_size = new_size
    print(f"Whisper model size changed to: {whisper_model_size}")
    
    # If whisper is already initialized, we need to reinitialize with the new model size
    if whisper_initialized:
        whisper_initialized = False
        whisper_model = None
        
        if voice_enabled.get():
            window.after(0, lambda: recording_indicator.config(text=f"Loading {whisper_model_size} model...", fg="blue"))
            success = initialize_whisper()
            if success:
                window.after(0, lambda: recording_indicator.config(text=f"{whisper_model_size.capitalize()} model loaded", fg="green"))
            else:
                window.after(0, lambda: recording_indicator.config(text="Model loading failed", fg="red"))
                voice_enabled.set(False)

# ===================
# Ollama / Chat Setup
# ===================
def fetch_available_models():
    try:
        models_data = ollama.list()
        return [model['name'] for model in models_data.get('models', [])]
    except Exception as e:
        print(f"Error fetching models: {e}")
        return ["gemma3:27b", "phi4:14b-fp16", "qwen2.5:32b"]

def chat_worker(user_message_data):
    """
    Background worker that calls Ollama in streaming mode,
    places chunks on stream_queue, and sets stream_done_event when finished.
    """
    global messages, current_model, stream_in_progress

    try:
        stream = ollama.chat(model=current_model, messages=messages, stream=True)
        for chunk in stream:
            if 'message' in chunk and 'content' in chunk['message']:
                content_piece = chunk['message']['content']
                stream_queue.put(content_piece)
        stream_queue.put(None)

    except Exception as e:
        err_text = f"Ollama communication error: {str(e)}"
        stream_queue.put(("ERROR", err_text))
    finally:
        stream_in_progress = False
        stream_done_event.set()

def process_stream_queue():
    """
    Periodically called in the main thread to retrieve chunks
    from the stream_queue and update the chat UI (and TTS).
    """
    global stream_queue, chat_history

    try:
        while True:
            item = stream_queue.get_nowait()
            if item is None:
                chat_history.config(state=tk.NORMAL)
                chat_history.insert(tk.END, "\n\n")
                chat_history.config(state=tk.DISABLED)
                chat_history.see(tk.END)
                return

            if isinstance(item, tuple) and item[0] == "ERROR":
                _, err_text = item
                add_message_to_ui("error", err_text)
                return

            chunk_text = item
            chat_history.config(state=tk.NORMAL)
            chat_history.insert(tk.END, chunk_text, "bot_message")
            chat_history.config(state=tk.DISABLED)
            chat_history.see(tk.END)

            queue_tts_text(chunk_text)

    except queue.Empty:
        pass

    if stream_in_progress:
        window.after(100, process_stream_queue)

# ===================
# Image / UI Helpers
# ===================
def select_image():
    global selected_image_path, image_sent_in_history
    file_path = filedialog.askopenfilename(
        title="Select Image File",
        filetypes=[("Image files", "*.png;*.jpg;*.jpeg;*.gif;*.bmp")]
    )
    if file_path:
        selected_image_path = file_path
        image_sent_in_history = False
        update_image_preview(file_path)
        image_indicator.config(text="✓ Image attached")

def clear_image():
    global selected_image_path, image_sent_in_history
    selected_image_path = ""
    image_sent_in_history = False
    
    # Clear the image reference
    image_preview.image = None
    
    # Reset the label with original size constraints preserved
    image_preview.configure(image="", text="No Image", width=20, height=10, bg="lightgrey")
    
    image_indicator.config(text="No image attached")

def update_image_preview(file_path):
    try:
        img = Image.open(file_path)
        width, height = img.size
        max_size = 180
        if width > height:
            new_width, new_height = max_size, int(height * (max_size / width))
        else:
            new_height, new_width = max_size, int(width * (max_size / height))

        img = img.resize((new_width, new_height), Image.LANCZOS)
        photo = ImageTk.PhotoImage(img)
        image_preview.configure(image=photo, width=max_size, height=max_size)
        image_preview.image = photo
    except Exception as e:
        image_preview.configure(image="", text="Preview Error")
        image_preview.image = None

def add_message_to_ui(role, content):
    """Helper to insert text into the chat history with proper tags."""
    if 'chat_history' not in globals():
        print("Warning: chat_history not ready.")
        return
    was_disabled = (chat_history.cget("state") == tk.DISABLED)
    if was_disabled:
        chat_history.config(state=tk.NORMAL)

    if role == "user":
        chat_history.insert(tk.END, "You: ", "user_tag")
        chat_history.insert(tk.END, content + "\n\n", "user_message")
    elif role == "assistant":
        chat_history.insert(tk.END, "Ollama: ", "bot_tag")
        chat_history.insert(tk.END, content, "bot_message")
    elif role == "error":
        chat_history.insert(tk.END, f"Error: {content}\n\n", "error")

    chat_history.see(tk.END)
    if was_disabled:
        chat_history.config(state=tk.DISABLED)
    if 'window' in globals():
        window.update_idletasks()

def select_model(event=None):
    global current_model
    current_model = model_selector.get()
    model_status.config(text=f"Using model: {current_model}")

def send_message(event=None):
    """Called when user presses Send or Enter."""
    global messages, image_sent_in_history, stream_done_event, stream_in_progress

    if stream_in_progress:
        return

    user_text = user_input.get("1.0", tk.END).strip()
    if not user_text and not selected_image_path:
        return

    user_message_data = {"role": "user", "content": user_text}
    display_text = user_text
    if selected_image_path and not image_sent_in_history:
        try:
            user_message_data["images"] = [selected_image_path]
            image_sent_in_history = True
            if not user_text:
                display_text = "[Image Attached]"
        except Exception as e:
            add_message_to_ui("error", f"Failed to prepare image: {e}")
            return

    add_message_to_ui("user", display_text)
    messages.append(user_message_data)
    user_input.delete("1.0", tk.END)

    chat_history.config(state=tk.NORMAL)
    chat_history.insert(tk.END, "Ollama: ", "bot_tag")
    thinking_text = "Thinking...\n"
    thinking_start_index = chat_history.index(tk.INSERT)
    chat_history.insert(tk.INSERT, thinking_text, "thinking")
    chat_history.see(tk.END)
    chat_history.config(state=tk.DISABLED)
    window.update_idletasks()

    if tts_engine and tts_enabled.get():
        try:
            tts_engine.stop()
        except:
            pass
        while not tts_queue.empty():
            try:
                tts_queue.get_nowait()
            except queue.Empty:
                pass

    global tts_sentence_buffer
    tts_sentence_buffer = ""

    stream_done_event.clear()
    stream_in_progress = True

    def worker_wrapper():
        chat_worker(user_message_data)

        def remove_thinking():
            try:
                chat_history.config(state=tk.NORMAL)
                end_thinking = f"{thinking_start_index}+{len(thinking_text)}c"
                chat_history.delete(thinking_start_index, end_thinking)
                chat_history.config(state=tk.DISABLED)
            except tk.TclError:
                pass

        window.after(0, remove_thinking)

    thread = threading.Thread(target=worker_wrapper, daemon=True)
    thread.start()

    window.after(100, process_stream_queue)

# ===================
# Tkinter GUI
# ===================
window = tk.Tk()
window.title("Ollama Multimodal Chat w/ Streaming + TTS + Voice")
window.geometry("800x800")

tts_enabled = tk.BooleanVar(value=False)
voice_enabled = tk.BooleanVar(value=False)

def on_closing():
    """Handles application shutdown."""
    print("[Main] Closing application...")
    stop_tts_thread()
    
    global recording, whisper_queue, temp_audio_file
    if recording:
        stop_recording()
    whisper_queue.put(None)
    
    if temp_audio_file and os.path.exists(temp_audio_file.name):
        try:
            os.unlink(temp_audio_file.name)
        except:
            pass
    
    if py_audio:
        py_audio.terminate()
    
    window.destroy()

window.protocol("WM_DELETE_WINDOW", on_closing)

main_frame = tk.Frame(window, padx=10, pady=10)
main_frame.pack(fill=tk.BOTH, expand=True)

top_controls_frame = tk.Frame(main_frame)
top_controls_frame.pack(fill=tk.X, expand=False, pady=(0, 10))

model_frame = tk.LabelFrame(top_controls_frame, text="Model Selection", padx=5, pady=5)
model_frame.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 10))

available_models = fetch_available_models()
model_selector = ttk.Combobox(model_frame, values=available_models, state="readonly")
if available_models:
    if current_model in available_models:
        model_selector.set(current_model)
    else:
        model_selector.set(available_models[0])
        current_model = model_selector.get()
else:
    model_selector.set("No models found")
    model_selector.config(state=tk.DISABLED)
    current_model = None

model_selector.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5, pady=5)
model_selector.bind("<<ComboboxSelected>>", select_model)

model_status = tk.Label(model_frame, text=f"Using model: {current_model if current_model else 'N/A'}", font=("Arial", 8))
model_status.pack(side=tk.LEFT, padx=10)

tts_frame = tk.LabelFrame(top_controls_frame, text="Speech", padx=5, pady=5)
tts_frame.pack(side=tk.LEFT, fill=tk.Y)

tts_toggle_button = ttk.Checkbutton(tts_frame, text="Enable TTS",
                                    variable=tts_enabled, command=toggle_tts)
tts_toggle_button.pack(padx=5, pady=5)

voice_frame = tk.LabelFrame(top_controls_frame, text="Voice Recognition", padx=5, pady=5)
voice_frame.pack(side=tk.LEFT, fill=tk.Y)

voice_toggle_button = ttk.Checkbutton(voice_frame, text="Enable Voice",
                                     variable=voice_enabled, command=toggle_voice_recognition)
voice_toggle_button.pack(padx=5, pady=2)

# Create a frame for language and model size selectors
whisper_settings_frame = tk.Frame(voice_frame)
whisper_settings_frame.pack(padx=5, pady=2, fill=tk.X)

# Add language selector 
lang_frame = tk.Frame(whisper_settings_frame)
lang_frame.pack(fill=tk.X, pady=2)
tk.Label(lang_frame, text="Language:", font=("Arial", 8)).pack(side=tk.LEFT)
whisper_language_selector = ttk.Combobox(lang_frame, 
                                         values=[lang[0] for lang in WHISPER_LANGUAGES],
                                         state="readonly", 
                                         width=10,
                                         font=("Arial", 8))
whisper_language_selector.current(0)  # Set to Auto Detect
whisper_language_selector.pack(side=tk.LEFT, padx=5)
whisper_language_selector.bind("<<ComboboxSelected>>", set_whisper_language)

# Add model size selector
size_frame = tk.Frame(whisper_settings_frame)
size_frame.pack(fill=tk.X, pady=2)
tk.Label(size_frame, text="Model:", font=("Arial", 8)).pack(side=tk.LEFT)
whisper_model_size_selector = ttk.Combobox(size_frame, 
                                           values=WHISPER_MODEL_SIZES,
                                           state="readonly", 
                                           width=10, 
                                           font=("Arial", 8))
whisper_model_size_selector.set(whisper_model_size)  # Set current model size
whisper_model_size_selector.pack(side=tk.LEFT, padx=5)
whisper_model_size_selector.bind("<<ComboboxSelected>>", set_whisper_model_size)

tts_engine_initialized_successfully = initialize_tts()
if not tts_engine_initialized_successfully:
    tts_toggle_button.config(state=tk.DISABLED)
    tts_enabled.set(False)
    def show_tts_error():
        add_message_to_ui("error", "TTS engine failed to initialize. Feature disabled.")
    window.after(0, show_tts_error)

start_tts_thread()

chat_frame = tk.LabelFrame(main_frame, text="Chat History", padx=5, pady=5)
chat_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))

chat_history = scrolledtext.ScrolledText(chat_frame, wrap=tk.WORD, height=15, state=tk.DISABLED, font=("Arial", 10))
chat_history.pack(fill=tk.BOTH, expand=True)

chat_history.tag_config("user_tag", foreground="#007bff", font=("Arial", 10, "bold"))
chat_history.tag_config("user_message", foreground="black", font=("Arial", 10))
chat_history.tag_config("bot_tag", foreground="#28a745", font=("Arial", 10, "bold"))
chat_history.tag_config("bot_message", foreground="black", font=("Arial", 10))
chat_history.tag_config("thinking", foreground="gray", font=("Arial", 10, "italic"))
chat_history.tag_config("error", foreground="red", font=("Arial", 10, "bold"))

bottom_frame = tk.Frame(main_frame)
bottom_frame.pack(fill=tk.X, expand=False)

image_frame = tk.LabelFrame(bottom_frame, text="Image Attachment", padx=10, pady=10)
image_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))

image_preview = tk.Label(image_frame, text="No Image", width=20, height=10, bg="lightgrey", relief="sunken")
image_preview.pack(pady=5)

img_button_frame = tk.Frame(image_frame)
img_button_frame.pack(fill=tk.X, pady=5)

select_button = tk.Button(img_button_frame, text="Select", command=select_image)
select_button.pack(side=tk.LEFT, expand=True, padx=2)

clear_button = tk.Button(img_button_frame, text="Clear", command=clear_image)
clear_button.pack(side=tk.LEFT, expand=True, padx=2)

image_indicator = tk.Label(image_frame, text="No image attached", font=("Arial", 8, "italic"), fg="grey")
image_indicator.pack(pady=(5,0))

voice_record_frame = tk.LabelFrame(bottom_frame, text="Voice Input", padx=10, pady=10)
voice_record_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))

record_button = tk.Button(voice_record_frame, text="Record Voice", 
                         command=toggle_recording, bg="#4CAF50", fg="white",
                         state=tk.DISABLED)
record_button.pack(pady=5, fill=tk.X)

recording_indicator = tk.Label(voice_record_frame, text="Voice recognition disabled", 
                              font=("Arial", 8, "italic"), fg="grey")
recording_indicator.pack(pady=5)

input_frame = tk.LabelFrame(bottom_frame, text="Your Message", padx=10, pady=10)
input_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

user_input = scrolledtext.ScrolledText(input_frame, wrap=tk.WORD, height=5, font=("Arial", 10))
user_input.pack(fill=tk.BOTH, expand=True, pady=(0, 5))
user_input.focus_set()

send_button = tk.Button(input_frame, text="Send (Enter)", command=send_message)
send_button.pack(pady=5, fill=tk.X)

def on_return_key_press(event):
    if not (event.state & 0x0001):
        return "break"

def on_return_key_release(event):
    if not (event.state & 0x0001):
        send_message()

user_input.bind("<KeyPress-Return>", on_return_key_press)
user_input.bind("<KeyRelease-Return>", on_return_key_release)

if tts_engine_initialized_successfully:
    add_message_to_ui("assistant",
                      "Hello! I'm ready to chat. Toggle TTS and Voice Recognition above. You can type, speak, or attach an image.")
else:
    add_message_to_ui("assistant",
                      "Hello! I'm ready to chat, but TTS is disabled due to an initialization error.")

window.after(200, periodic_tts_check)

window.mainloop()
