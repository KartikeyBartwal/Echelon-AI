import pyaudio
import wave
from logging_module import setup_logger

# Set up logger
logger = setup_logger()

def record_audio(output_filename, duration=5):
    """Records audio from the microphone and saves it as a WAV file."""
    CHUNK = 1024
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 44100

    logger.info(f"Starting audio recording for {duration} seconds...")

    try:
        p = pyaudio.PyAudio()
        stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE,
                        input=True, frames_per_buffer=CHUNK)

        logger.info("Recording started.")
        frames = []
        for _ in range(int(RATE / CHUNK * duration)):
            frames.append(stream.read(CHUNK))

        logger.info("Recording complete.")
        stream.stop_stream()
        stream.close()
        p.terminate()

        # Saving the recorded audio
        with wave.open(output_filename, 'wb') as wf:
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(p.get_sample_size(FORMAT))
            wf.setframerate(RATE)
            wf.writeframes(b''.join(frames))
        
        logger.info(f"Audio successfully saved to {output_filename}")

    except Exception as e:
        logger.error(f"Error occurred during recording: {e}")

    return output_filename

if __name__ == "__main__":
    output_file = "output.wav"
    logger.info(f"Script started, recording audio to {output_file}...")
    record_audio(output_file, duration=5)
    logger.info("Script finished.")
