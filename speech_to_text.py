import whisper
import logging
from logging_module import setup_logger  # Import your logger setup function

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

import time

if __name__ == "__main__":
    file_path = "audio.mp3"

    # Start the timer
    start_time = time.time()

    transcription = transcribe_audio(file_path)

    # Calculate the time taken for transcription
    end_time = time.time()
    duration = end_time - start_time

    # Log the time taken
    logger.info(f"Time taken for transcription: {duration:.2f} seconds")

    if transcription:
        logger.info("Transcription result:")
        print(transcription)
        logging.info(transcription)
    else:
        logger.error(f"Transcription failed for file: {file_path}")
