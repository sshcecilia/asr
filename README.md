# Audio Speech Recognition

## Overview

This project aims to fine-tune Facebook's wav2vec2-large-960h model using the Common Voice dataset.

## Installation

Step 1: Clone repository
```bash
git clone https://github.com/sshcecilia/asr.git
cd asr
```

Step 2: Install dependencies
```bash
pip install requirements.txt
```

Step 3: Download Model
1. Download file from: https://drive.google.com/file/d/1Uslo469jdxulRTTvMOpP4m9z9SAIuhf-/view?usp=drive_link
2. Save the file under the following directory: ./asr_train//wav2vec2-large-960h-cv

## Usage of Model

```python
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
import torchaudio
import torch

processor = Wav2Vec2Processor.from_pretrained('./asr_train/wav2vec2-large-960h-cv')
model = Wav2Vec2ForCTC.from_pretrained('./asr_train/wav2vec2-large-960h-cv')
filedir = ''

required_rate = 16000
waveform, sample_rate = torchaudio.load(filedir)
if sample_rate != required_rate:
    resampler = torchaudio.transforms.Resample(sample_rate, required_rate)
    resampled_waveform = resampler(waveform)
else:
    resampled_waveform = waveform

input_values = processor(resampled_waveform[0], return_tensors="pt", sampling_rate = required_rate).input_values

logits = model(input_values).logits
predicted_ids = torch.argmax(logits, dim=-1)
transcription = processor.batch_decode(predicted_ids)
```
