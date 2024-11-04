from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
from flask import Flask, request, jsonify
import torchaudio
import torch
from io import BytesIO

# Load wav2vec2-large-960h model
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-large-960h")
model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-large-960h")

# Create new Flask web application.Argument indicates where to look for 
# resources such as templates and static files.
app = Flask(__name__)
#app.config["DEBUG"] = True

# Route decorator indicates what url to trigger for application
# Use post method since data will be received and processed
@app.route("/asr", methods = ['POST'])

def asr():
    
    input = request.files['file']
    
    waveform, sample_rate = torchaudio.load(BytesIO(input.read()))
    duration = str(waveform.size(1) / sample_rate)
    required_rate = 16000

    if sample_rate != required_rate:
        resampler = torchaudio.transforms.Resample(sample_rate, required_rate)
        resampled_waveform = resampler(waveform)
    else:
        resampled_waveform = waveform
    input_values = processor(resampled_waveform[0], return_tensors="pt", padding="longest", sampling_rate = required_rate).input_values
    logits = model(input_values).logits
    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = processor.batch_decode(predicted_ids)

    return jsonify({"transcription": transcription, "duration": duration})

# Launch the Flask application. Run on port 8001 (default port 5000)
app.run(port = 8001)




