# Audio Forensics

Audio Forensics is a Streamlit screening tool for classifying an uploaded speech recording as **REAL** or **FAKE** using a Keras CNN and Mel spectrogram input.

## Features

- WAV, MP3, and OGG upload with in-app playback.
- Librosa preprocessing at 16 kHz, mono, and a fixed 3-second window.
- 128-band Mel spectrogram using `n_fft=2048` and `hop_length=512`.
- Magma visualization resized to a 128 × 128 RGB model input.
- Sigmoid prediction, raw fake probability, and confidence score.
- Downloadable PDF report with the input filename, timestamp, result, raw output, and spectrogram.
- Exact SHA-256 recognition for the 20 documented demo samples, with the dataset label
  shown separately from the CNN prediction.
- Responsive dark forensic UI with verified-demo and AI-model result treatments.
- Local, portable paths; no external service or network request is required by the app.

## Technology

Python, Streamlit, TensorFlow/Keras, Librosa, NumPy, Matplotlib, Pillow, FPDF2, and scikit-learn.

## Project structure

```text
audio_deepfake_ui.py       Streamlit audio analysis application
image_ui.py                Legacy spectrogram-image application
cnn_model_smaller.keras    Bundled trained model used by the app
preprocess_spectrograms.py Portable audio-to-image preprocessing script
train_cnn_spectrogram.py  Report-aligned CNN training script
style_all.css              Streamlit theme
requirements.txt           Python dependencies
IMPLEMENTATION_AUDIT.md    Requirement-by-requirement audit
test_audio/                10 labeled REAL and 10 labeled FAKE demo samples
test_audio_sources.txt     Source, license, labels, and validation record
```

The repository does not include the dataset or the original `archive (1)` / `lang` folders mentioned in older documentation. It also does not include a reproducible training split, so no accuracy claim is made here.

The `test_audio` folder contains a small, locally validated WaveFake demonstration
set: ten `R` reference recordings and ten `WF1` generated recordings. See
`test_audio_sources.txt` for provenance and license/attribution requirements.

## Installation on Windows

From PowerShell in this folder:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If PowerShell blocks activation, run the app with the environment's executable directly:

```powershell
.\.venv\Scripts\python.exe -m streamlit run .\audio_deepfake_ui.py
```

## Run the application

```powershell
python -m streamlit run .\audio_deepfake_ui.py
```

Then open `http://localhost:8501`. The bundled model is loaded once and cached for the Streamlit session.

## Prepare data and train a new report-aligned model

Arrange audio into labelled directories. Nested language folders are supported:

```text
dataset\
├── english\
│   ├── real\
│   └── fake\
└── lang\
    └── hindi\
        ├── real\
        └── fake\
```

Convert audio to spectrograms:

```powershell
python .\preprocess_spectrograms.py --input-dir .\dataset --output-dir .\processed_spectrograms
```

For training, point the script at one directory whose immediate children are `real` and `fake` (for example `processed_spectrograms\english`):

```powershell
python .\train_cnn_spectrogram.py --data-dir .\processed_spectrograms\english --output .\cnn_model_report_aligned.keras
```

The training script defines three convolutional blocks and a 128-unit dense layer, matching the academic specification. The bundled `cnn_model_smaller.keras` predates this script and intentionally remains unchanged.

## Limitations

- The bundled model is a two-convolution-block model with a 64-unit dense layer, not the report's three-block/128-unit architecture.
- The model's original training dataset and validation results are not present in this checkout.
- Predictions are screening results and should not be treated as conclusive forensic proof.
- For the bundled demo samples only, an exact SHA-256 match selects the verified
  WaveFake dataset label. Verified demo results do not display a fabricated
  confidence percentage; the CNN's independent prediction and confidence remain
  visible separately. This is sample recognition, not evidence that the CNN learned
  to classify the sample.
- SHAP and the legacy image UI remain available, but SHAP has not been made part of the audio-upload workflow.
