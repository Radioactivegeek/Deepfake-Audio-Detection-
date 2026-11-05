# Audio Forensics

Audio Forensics is a local Streamlit application for screening speech recordings
as **REAL** or **FAKE** using a bundled TensorFlow/Keras model and Mel
spectrograms.

> **Important:** This is a demonstration and screening tool, not definitive
> forensic proof. The bundled model's original training data and validation
> results are not included, so this project makes no accuracy claim.

## What the application does

- Accepts WAV, MP3, and OGG audio uploads.
- Plays the uploaded recording in the browser.
- Converts audio to mono at 16 kHz.
- Trims or zero-pads audio to a fixed 3-second window.
- Generates a 128-band Mel spectrogram using:
  - `n_fft=2048`
  - `hop_length=512`
  - Magma color mapping
- Resizes the spectrogram to the model's 128 × 128 RGB input.
- Displays the CNN's raw probability of FAKE and its confidence.
- Generates a downloadable PDF report containing the filename, timestamp,
  displayed result, CNN output, label source, and spectrogram.
- Runs locally; the application does not send uploaded audio to a cloud service.

## Verified demonstration samples

The repository includes 20 validated WaveFake samples:

```text
test_audio\
├── real\
│   ├── real_01.wav
│   └── ... real_10.wav
└── fake\
    ├── fake_01.wav
    └── ... fake_10.wav
```

The original dataset labels are documented in
[test_audio_sources.txt](test_audio_sources.txt):

- `R` recordings are treated as REAL reference samples.
- `WF1` recordings are treated as FAKE generated samples.
- `test_audio_download_metadata.json` stores the SHA-256 values used for exact
  sample recognition.

When an uploaded file exactly matches one of these unchanged demo files, the
application displays the verified dataset label and reports the CNN prediction
separately. This prevents a known demo label from being confused with CNN
performance. Unknown files use the CNN prediction normally.

Dataset references:

- WaveFake record: https://zenodo.org/records/5642694
- WaveFake research repository: https://github.com/RUB-SysSec/WaveFake
- CC BY-SA 4.0: https://creativecommons.org/licenses/by-sa/4.0/

Review the source license and retain attribution before redistributing the
sample files.

## Project structure

```text
audio_deepfake_ui.py       Active Streamlit audio application
cnn_model_smaller.keras    Bundled trained CNN model
style_all.css              Application styling
requirements.txt           Python dependencies
preprocess_spectrograms.py Portable audio-to-spectrogram converter
train_cnn_spectrogram.py   Optional report-aligned training script
test_audio/                10 REAL and 10 FAKE demonstration samples
test_audio_sources.txt     Sample provenance and validation record
test_audio_download_metadata.json
                           Sample labels, hashes, and validation metadata
```

## Installation on Windows

Open PowerShell in the project folder:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If PowerShell activation is restricted, use the environment directly:

```powershell
.\.venv\Scripts\python.exe -m streamlit run .\audio_deepfake_ui.py
```

## Run the application

```powershell
python -m streamlit run .\audio_deepfake_ui.py
```

Open [http://localhost:8501](http://localhost:8501) in a browser.

## Use the application

1. Select an audio file or one of the files in `test_audio`.
2. Press **RUN FORENSIC ANALYSIS**.
3. Review the generated spectrogram and result.
4. Check whether the result source is:
   - **SHA-256 verified dataset label** for an exact known demo sample, or
   - **CNN model prediction** for an unknown file.
5. Download the forensic PDF report if needed.

For `fake_05.wav`, the verified display label is **FAKE**, while the report
also shows the CNN's independent prediction. For `real_08.wav`, the verified
display label is **REAL**.

## Optional preprocessing and training

The repository does not include the original large training dataset. If you
have a compatible labelled dataset, arrange it like this:

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

Convert the audio to spectrogram images:

```powershell
python .\preprocess_spectrograms.py `
  --input-dir .\dataset `
  --output-dir .\processed_spectrograms
```

Train a new report-aligned model from a directory whose immediate children
are `real` and `fake`:

```powershell
python .\train_cnn_spectrogram.py `
  --data-dir .\processed_spectrograms\english `
  --output .\cnn_model_report_aligned.keras
```

The optional training script defines three convolutional blocks and a
128-unit dense layer. It does not replace the bundled model automatically.

## Model limitation

The bundled `cnn_model_smaller.keras` is an existing two-convolution-block
model with a 64-unit dense layer and one sigmoid output. It is intentionally
preserved and is not the same architecture as the report-aligned training
script. Do not interpret the verified demo-label feature as evidence that the
CNN itself correctly learned every sample.
