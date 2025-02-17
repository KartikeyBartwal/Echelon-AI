import torch
import torchaudio
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import time

# Load pre-trained Wav2Vec2 model and processor
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")

def transcribe_audio(file_path):
    # Load audio file
    waveform, sample_rate = torchaudio.load(file_path)
    
    # Resample if necessary to match the model's sample rate (16kHz for wav2vec2-base-960h)
    if sample_rate != 16000:
        resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
        waveform = resampler(waveform)
    
    # Process the audio file and get the input tensor
    input_values = processor(waveform, return_tensors="pt").input_values
    
    # Make sure the shape is correct (should be 2D or 3D tensor)
    if input_values.ndimension() == 3:
        input_values = input_values.squeeze(0)  # Remove the batch dimension if necessary

    # Run the model
    with torch.no_grad():
        logits = model(input_values).logits

    # Decode the logits to text
    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = processor.decode(predicted_ids[0])

    return transcription

# File path to the audio file
file_path = "Optimus vs Grindor.mp3"

# Start measuring time
start_time = time.time()

# Perform transcription
transcription = transcribe_audio(file_path)

# End measuring time
end_time = time.time()

# Calculate and print the transcription time
transcription_time = end_time - start_time
print("Transcription: ", transcription)
print(f"Time taken for transcription: {transcription_time:.2f} seconds")
