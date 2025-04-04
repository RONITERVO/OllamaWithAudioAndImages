# OllamaWithAudioAndImages
# Ollama Multimodal Chat (Tkinter GUI + Flask Web UI) 💬🖥️🌐

This project provides a Python-based chat application that interacts with a locally running Ollama instance. It features:

1.  A **Tkinter GUI** for direct interaction on the host machine, including:
    *   Chat history display.
    *   Text input and message sending.
    *   Ollama model selection.
    *   Image file attachment (for multimodal models).
    *   PDF/Text file loading into the input area.
    *   Drag & Drop support for files.
    *   Optional Text-to-Speech (TTS) output using `pyttsx3`.
    *   Optional Voice Activity Detection (VAD) and Speech-to-Text (STT) using Silero-VAD and Whisper/FasterWhisper via microphone input.
    *   Real-time streaming of Ollama responses.
2.  A **Flask Web Server** running concurrently, providing:
    *   A web interface accessible from other devices on your local network.
    *   Viewing chat history.
    *   Sending text messages to Ollama.
    *   Real-time streaming of Ollama responses to the web browser using Server-Sent Events (SSE).

The goal is to have a primary GUI application that can also be controlled or monitored remotely via a simple web interface.

![image](https://github.com/user-attachments/assets/1096393e-9697-46f7-80ce-698c56e7c3a8)


## ✨ Features

*   **Ollama Integration:** Connects to your local Ollama instance for chat completions.
*   **Streaming Responses:** Displays Ollama responses word-by-word in both GUIs.
*   **Multimodal Input (GUI):** Attach images via file dialog, drag & drop, or clipboard pasting (requires Pillow).
*   **Text/PDF Input (GUI):** Load content from text or PDF files directly into the input area via dialog or drag & drop.
*   **Tkinter GUI:** Native desktop interface with controls for model selection, TTS, and VAD.
*   **Flask Web UI:** Access core chat functionality (text send/receive, history) from a web browser on your LAN.
*   **Text-to-Speech (TTS):** Reads assistant messages aloud (toggleable, configurable voice/rate).
*   **Voice Input (VAD + Whisper):** Automatically detects speech using Silero-VAD, records, transcribes using Whisper/FasterWhisper, and (optionally) sends the message. (Toggleable, configurable language/model size).
*   **Concurrent Operation:** Runs the Tkinter GUI main loop and the Flask server simultaneously using threading.
*   **Thread-Safe Communication:** Uses queues and locks for safe data exchange between the Flask thread and the main Tkinter thread.

## 🛠️ Technology Stack

*   **Python 3.9+**
*   **Ollama:** Local LLM runner (needs to be installed and running separately).
*   **Tkinter:** Standard Python GUI library.
*   **TkinterDnD2:** For Drag & Drop support in Tkinter (`tkdnd2-lite`).
*   **Pillow:** Python Imaging Library (for image handling/preview/pasting).
*   **Flask:** Web microframework for the web UI.
*   **pyttsx3:** Text-to-Speech library.
*   **PyAudio:** For microphone audio input.
*   **numpy:** Numerical library (dependency for audio/ML).
*   **torch & torchaudio:** PyTorch library (for Silero-VAD and Whisper).
*   **whisper / faster-whisper:** OpenAI's Whisper model or the optimized FasterWhisper implementation for Speech-to-Text.
*   **python-fitz (PyMuPDF):** For extracting text from PDF files.
*   **requests (via ollama library):** For communicating with Ollama.

## ⚙️ Setup and Installation

**Prerequisites:**

1.  **Python:** Ensure you have Python 3.9 or newer installed.
2.  **Ollama:** Install and run Ollama locally. Make sure you have pulled the models you intend to use (e.g., `ollama pull gemma3:27b`). See [ollama.com](https://ollama.com/).
3.  **Microphone:** A working microphone is required for voice input features.
4.  **Speakers/Headphones:** Required for TTS output.
5.  **`ffmpeg`:** Whisper (especially the original implementation) often requires `ffmpeg` to be installed and available in your system's PATH for audio format conversion. Download from [ffmpeg.org](https://ffmpeg.org/download.html).
6.  **(Windows) Build Tools:** Installing `PyAudio` might require Microsoft Visual C++ Build Tools. You can download them from the [Visual Studio website](https://visualstudio.microsoft.com/visual-cpp-build-tools/).
7.  **(Linux) PortAudio:** Installing `PyAudio` usually requires `portaudio19-dev`: `sudo apt-get update && sudo apt-get install portaudio19-dev python3-pyaudio` (or similar for your distribution).

**Installation Steps:**

1.  **Clone the Repository:**
    ```bash
    git clone <your-repo-url>
    cd pythonOllama # Or your repository's root directory name
    ```

2.  **Create and Activate a Virtual Environment (Recommended):**
    ```bash
    # Windows
    python -m venv venv
    .\venv\Scripts\activate

    # macOS/Linux
    python3 -m venv venv
    source venv/bin/activate
    ```

3.  **Install Dependencies:**
    *   **PyTorch:** Install PyTorch first, following instructions from the [official website](https://pytorch.org/get-started/locally/) for your OS and CUDA version (if applicable). For CPU-only:
        ```bash
        # Example CPU-only install (check website for current command)
        pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu
        ```
    *   **Core Libraries:**
        ```bash
        pip install ollama Pillow pyttsx3 Flask numpy PyAudio python-fitz tkdnd2-lite
        ```
    *   **Whisper/VAD:** Choose *either* OpenAI Whisper *or* FasterWhisper:
        *   **FasterWhisper (Recommended for speed):**
            ```bash
            pip install faster-whisper
            ```
        *   **OpenAI Whisper:**
            ```bash
            pip install -U openai-whisper
            ```
            *(Note: OpenAI Whisper might pull in an older torch version; install PyTorch first as recommended above).*

    *   **(Alternative) Using `requirements.txt`:** If a `requirements.txt` file is provided in the repo:
        ```bash
        pip install -r requirements.txt
        ```
        *(Make sure PyTorch was installed correctly beforehand, as `requirements.txt` might not specify CPU/GPU versions correctly).*

## ▶️ Running the Application

1.  **Ensure Ollama is running** in the background.
2.  **Navigate to the root directory** of the project (the one containing `chat_app_web.py` and the `templates` folder) in your terminal.
3.  **Activate your virtual environment** (e.g., `.\venv\Scripts\activate` or `source venv/bin/activate`).
4.  **Run the Python script:**
    ```bash
    python chat_app_web.py
    ```

This will launch the Tkinter GUI *and* start the Flask web server. You will see output in the console indicating both are running.

## 🖱️ Usage

**1. Tkinter GUI:**

*   Interact directly with the GUI that appears on your screen.
*   **Model Selection:** Choose an installed Ollama model from the dropdown.
*   **TTS Toggle:** Enable/disable Text-to-Speech output. Configure voice and speed.
*   **Voice Input Toggle:** Enable/disable VAD/Whisper. Configure language and Whisper model size. The status indicator shows if it's listening, recording, or processing.
*   **File Attachment:**
    *   Click "Open File" to attach an image/PDF/text file.
    *   Drag & Drop image, PDF, or text files onto the "Attachments" area or the main text input area.
    *   Paste an image directly from the clipboard (Ctrl+V / Cmd+V).
    *   Click "Clear Img" to remove the attached image.
*   **Send Message:** Type your message (Shift+Enter for newline) and click "Send" or press Enter.

**2. Web Interface:**

*   **Access:** Open a web browser on your local network (same Wi-Fi/LAN).
    *   On the *same machine*, go to `http://localhost:5000` or `http://127.0.0.1:5000`.
    *   On *other devices*, find your host machine's local IP address (e.g., `192.168.1.10`) and go to `http://<YOUR_PC_IP>:5000`. (You might need to adjust your firewall settings to allow connections to port 5000).
*   **Chat History:** View the conversation history.
*   **Send Message:** Type a text message in the input box and click "Send".
*   **Streaming Response:** Ollama's response will be streamed into the chatbox.

## 🔧 Configuration

Key parameters can be adjusted directly in the "Constants" section near the top of the `chat_app_web.py` script:

*   `DEFAULT_OLLAMA_MODEL`: The Ollama model loaded by default.
*   `DEFAULT_WHISPER_MODEL_SIZE`: The Whisper/FasterWhisper model size used for transcription.
*   `FLASK_PORT`: The port number for the web server (default: 5000).
*   `FLASK_HOST`: The host address for the Flask server (default: "0.0.0.0" to be accessible on the network).
*   Audio parameters (`CHUNK`, `RATE`, `SILENCE_THRESHOLD_SECONDS`, etc.).

## ⚠️ Troubleshooting

*   **`jinja2.exceptions.TemplateNotFound: index.html`:** Make sure you are running the `chat_app_web.py` script from the project's root directory, and the `templates` folder containing `index.html` exists directly inside that root directory.
*   **Ollama Connection Errors:** Ensure the Ollama service is running locally. Check if the selected model is available in Ollama (`ollama list`).
*   **Web UI Not Accessible Remotely:** Check your host machine's firewall settings. Ensure incoming connections are allowed on the `FLASK_PORT` (e.g., 5000). Verify the IP address you are using is correct and both devices are on the same network.
*   **Audio Input/Output Issues:**
    *   Verify microphone/speaker selection in your OS.
    *   Check if `PyAudio` installed correctly (may require build tools or specific package versions). See Prerequisites.
    *   Ensure microphone permissions are granted to Python/Terminal.
*   **Whisper/VAD Initialization Errors:**
    *   Make sure `ffmpeg` is installed and in the PATH if using OpenAI Whisper.
    *   Ensure PyTorch was installed correctly (CPU or GPU matching your system).
    *   Model files might be downloading on first use; check console output.
*   **`TclError: unknown option "-tooltip"`:** Remove the `tooltip="..."` argument from the `ttk.Checkbutton` creation or install and use the `tktooltip` library properly.

## 🚀 Future Enhancements

*   **Web UI VAD/TTS:** Implementing reliable voice input/output directly via the web browser is significantly more complex (requires Web Audio API, potentially WebSockets/WebRTC).
*   **Web UI File Uploads:** Allow attaching images/files via the web interface.
*   **Bi-directional Sync:** Push messages sent via the Tkinter GUI to active web clients in real-time.
*   **Configuration UI:** Allow changing settings like models, ports, etc., via the GUI or web UI instead of code constants.
*   **Error Handling:** More robust error display and recovery across threads.
*   **Authentication:** Add basic authentication to the web interface.

## 📄 License

`LICENSE` file for more info
