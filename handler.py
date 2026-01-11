"""
RunPod Serverless Handler for Supertonic TTS
Accepts: { text: string, language: string, voice: string, speed: number }
Returns: { audio: base64, duration: number }
"""

import runpod
import base64
import io
import os
import numpy as np
import soundfile as sf
from helper import load_text_to_speech, load_voice_style

# Global model instance (loaded once on cold start)
tts_model = None
voice_styles = {}

# Paths configuration
ONNX_DIR = "assets/onnx"
VOICE_STYLES_DIR = "assets/voice_styles"

def load_model():
    """Load TTS model on cold start"""
    global tts_model
    if tts_model is None:
        print(f"Loading Supertonic TTS model from {ONNX_DIR}...")
        
        # Verify required files exist
        required_files = [
            "duration_predictor.onnx",
            "text_encoder.onnx", 
            "vector_estimator.onnx",
            "vocoder.onnx",
            "tts.json",
            "unicode_indexer.json"
        ]
        
        missing_files = []
        for f in required_files:
            path = os.path.join(ONNX_DIR, f)
            if not os.path.exists(path):
                missing_files.append(f)
        
        if missing_files:
            raise FileNotFoundError(f"Missing required model files: {missing_files}")
        
        tts_model = load_text_to_speech(ONNX_DIR, use_gpu=False)
        print("Model loaded successfully!")
    return tts_model

def get_voice_style(voice_name):
    """Load and cache voice style"""
    global voice_styles
    if voice_name not in voice_styles:
        voice_path = os.path.join(VOICE_STYLES_DIR, f"{voice_name}.json")
        
        if not os.path.exists(voice_path):
            available = [f.replace('.json', '') for f in os.listdir(VOICE_STYLES_DIR) if f.endswith('.json')]
            raise FileNotFoundError(f"Voice style '{voice_name}' not found. Available: {available}")
        
        voice_styles[voice_name] = load_voice_style([voice_path])
        print(f"Loaded voice style: {voice_name}")
    return voice_styles[voice_name]

def handler(job):
    """
    RunPod handler function
    Input: { text: string, language: string, voice: string, speed: number }
    Output: { audio: base64, duration: number }
    """
    job_input = job.get("input", {})

    # Extract parameters with defaults
    text = job_input.get("text", "Hello, this is a test.")
    language = job_input.get("language", "en")
    voice = job_input.get("voice", "M1")
    speed = job_input.get("speed", 1.05)
    total_step = job_input.get("total_step", 5)

    # Validate language
    available_langs = ["en", "ko", "es", "pt", "fr"]
    if language not in available_langs:
        return {"error": f"Invalid language '{language}'. Available: {available_langs}"}

    try:
        # Load model
        model = load_model()

        # Load voice style
        style = get_voice_style(voice)

        print(f"Generating speech: text='{text[:50]}...', lang={language}, voice={voice}, speed={speed}")

        # Generate speech
        wav, duration = model(text, language, style, total_step, speed)

        # Trim audio to actual duration
        wav_trimmed = wav[0, :int(model.sample_rate * duration[0].item())]

        # Convert to WAV bytes
        buffer = io.BytesIO()
        sf.write(buffer, wav_trimmed, model.sample_rate, format='WAV')
        buffer.seek(0)

        # Encode as base64
        audio_base64 = base64.b64encode(buffer.read()).decode('utf-8')

        print(f"Generated {duration[0].item():.2f}s of audio")

        return {
            "audio": audio_base64,
            "duration": float(duration[0].item()),
            "sample_rate": model.sample_rate
        }

    except FileNotFoundError as e:
        return {"error": str(e)}
    except Exception as e:
        print(f"Error during synthesis: {e}")
        return {"error": str(e)}

# Health check - verify model can load on startup
print("=== Supertonic TTS RunPod Handler ===")
print(f"ONNX directory: {ONNX_DIR}")
print(f"Voice styles directory: {VOICE_STYLES_DIR}")

# List available files for debugging
if os.path.exists(ONNX_DIR):
    print(f"ONNX files: {os.listdir(ONNX_DIR)}")
else:
    print(f"WARNING: ONNX directory not found at {ONNX_DIR}")

if os.path.exists(VOICE_STYLES_DIR):
    print(f"Voice styles: {[f.replace('.json', '') for f in os.listdir(VOICE_STYLES_DIR) if f.endswith('.json')]}")
else:
    print(f"WARNING: Voice styles directory not found at {VOICE_STYLES_DIR}")

# Start the serverless handler
runpod.serverless.start({"handler": handler})
