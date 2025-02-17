from record_audio import record_audio
import whisper
import logging
from logging_module import setup_logger

# Set up logger
logger = setup_logger()

# Load Whisper ASR model
logger.info("Loading Whisper model...")
try:
    model = whisper.load_model("tiny")
    logger.info("Whisper model loaded successfully.")
except Exception as e:
    logger.error(f"Error loading Whisper model: {e}")
    raise

def transcribe_audio(file_path):
    """Transcribes an audio file using OpenAI's Whisper model."""
    logger.info(f"Transcribing audio file: {file_path}")
    try:
        result = model.transcribe(file_path)
        logger.info(f"Transcription successful for file: {file_path}")
        return result["text"]
    except Exception as e:
        logger.error(f"Error transcribing file {file_path}: {e}")
        return None

def transcribe_live_audio(duration=5):
    """Records and transcribes live audio."""
    file_path = record_audio("live_audio.wav", duration)
    return transcribe_audio(file_path)

if __name__ == "__main__":
    logger.info("Real-time speech-to-text started. Speak now...")

    transcription = transcribe_live_audio(5)  # You can adjust the duration here
    if transcription:
        logger.info("Transcription result:")
        print(transcription)
    else:
        logger.error("Real-time transcription failed.")
