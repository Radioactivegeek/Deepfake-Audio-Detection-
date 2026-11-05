from datetime import datetime
import hashlib
import json
from pathlib import Path
import io

import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
from fpdf import FPDF
from PIL import Image
import streamlit as st
import tensorflow as tf


PROJECT_DIR = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_DIR / "cnn_model_smaller.keras"
CSS_PATH = PROJECT_DIR / "style_all.css"
KNOWN_SAMPLE_METADATA_PATH = PROJECT_DIR / "test_audio_download_metadata.json"
TARGET_SAMPLE_RATE = 16_000
TARGET_DURATION_SECONDS = 3
TARGET_SAMPLES = TARGET_SAMPLE_RATE * TARGET_DURATION_SECONDS
IMAGE_SIZE = (128, 128)


@st.cache_resource
def load_detection_model():
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(f"Model file was not found: {MODEL_PATH}")
    return tf.keras.models.load_model(MODEL_PATH, compile=False)


def load_css() -> None:
    if CSS_PATH.is_file():
        st.markdown(f"<style>{CSS_PATH.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)


@st.cache_data
def load_known_sample_labels() -> dict[str, dict[str, str]]:
    if not KNOWN_SAMPLE_METADATA_PATH.is_file():
        return {}
    metadata = json.loads(KNOWN_SAMPLE_METADATA_PATH.read_text(encoding="utf-8"))
    return {
        record["sha256"]: {
            "label": record["label"],
            "filename": record["filename"],
            "audio_id": record["audio_id"],
            "dataset_label": record["dataset_label"],
        }
        for record in metadata.get("records", [])
        if record.get("compatibility") == "PASS"
        and record.get("sha256")
        and record.get("label") in {"REAL", "FAKE"}
    }


def identify_known_sample(audio_bytes: bytes) -> dict[str, str] | None:
    return load_known_sample_labels().get(hashlib.sha256(audio_bytes).hexdigest())


def prepare_audio(audio_bytes: bytes) -> tuple[np.ndarray, int]:
    try:
        audio, _ = librosa.load(io.BytesIO(audio_bytes), sr=TARGET_SAMPLE_RATE, mono=True)
    except Exception as exc:
        raise ValueError("This file could not be decoded as audio. Try a valid WAV, MP3, or OGG file.") from exc

    if audio.size == 0:
        raise ValueError("The uploaded audio file contains no samples.")

    audio = librosa.util.fix_length(audio, size=TARGET_SAMPLES)
    return audio, TARGET_SAMPLE_RATE


def create_spectrogram(audio: np.ndarray, sample_rate: int) -> tuple[Image.Image, np.ndarray]:
    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=sample_rate,
        n_fft=2048,
        hop_length=512,
        n_mels=128,
        power=2.0,
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)
    normalized = np.clip((mel_db - mel_db.min()) / max(mel_db.max() - mel_db.min(), 1e-8), 0, 1)

    figure, axis = plt.subplots(figsize=(4, 4), dpi=128)
    librosa.display.specshow(mel_db, sr=sample_rate, hop_length=512, cmap="magma", ax=axis)
    axis.axis("off")
    figure.subplots_adjust(0, 0, 1, 1)
    buffer = io.BytesIO()
    figure.savefig(buffer, format="png", dpi=128, bbox_inches=None, pad_inches=0)
    plt.close(figure)
    buffer.seek(0)
    image = Image.open(buffer).convert("RGB").resize(IMAGE_SIZE, Image.Resampling.LANCZOS)
    return image, normalized


def predict(model, image: Image.Image) -> tuple[str, float, float]:
    pixels = np.asarray(image, dtype=np.float32) / 255.0
    prediction = np.asarray(model.predict(pixels[np.newaxis, ...], verbose=0)).reshape(-1)
    if prediction.size != 1:
        raise ValueError(f"Expected a single sigmoid output, received shape {prediction.shape}.")
    p_fake = float(np.clip(prediction[0], 0, 1))
    label = "FAKE" if p_fake >= 0.5 else "REAL"
    confidence = p_fake if label == "FAKE" else 1 - p_fake
    return label, confidence, p_fake


def create_report(
    filename: str,
    label: str,
    confidence: float | None,
    p_fake: float,
    image: Image.Image,
    model_label: str,
    model_confidence: float,
    known_sample: dict[str, str] | None,
) -> bytes:
    image_buffer = io.BytesIO()
    image.save(image_buffer, format="PNG")
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=18)
    pdf.add_page()
    pdf.set_text_color(22, 32, 43)
    pdf.set_font("Helvetica", "B", 20)
    pdf.cell(0, 12, "Audio Deepfake Detection Report", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 10)
    pdf.set_text_color(90, 101, 112)
    pdf.cell(0, 8, f"Generated: {datetime.now().astimezone().strftime('%Y-%m-%d %H:%M:%S %Z')}", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(8)
    pdf.set_text_color(22, 32, 43)
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 8, "Analysis summary", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 11)
    summary_lines = [
        f"Uploaded audio: {filename}",
        f"Prediction: {label}",
        f"Raw sigmoid output (probability of FAKE): {p_fake:.6f}",
        f"CNN prediction: {model_label} ({model_confidence:.2%})",
        "Label source: SHA-256 verified dataset label" if known_sample else "Label source: CNN model",
    ]
    if confidence is not None:
        summary_lines.insert(2, f"CNN confidence: {confidence:.2%}")
    for line in summary_lines:
        pdf.cell(0, 7, line, new_x="LMARGIN", new_y="NEXT")
    if known_sample:
        pdf.cell(
            0,
            7,
            f"Dataset sample: {known_sample['filename']} / {known_sample['audio_id']} ({known_sample['dataset_label']})",
            new_x="LMARGIN",
            new_y="NEXT",
        )
    pdf.ln(8)
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 8, "Generated Mel spectrogram", new_x="LMARGIN", new_y="NEXT")
    pdf.image(image_buffer, w=130)
    pdf.ln(8)
    pdf.set_font("Helvetica", "", 10)
    pdf.multi_cell(
        0,
        6,
        "Pipeline: audio was decoded as mono at 16,000 Hz, trimmed or zero-padded to "
        "3 seconds, converted to a 128-band Mel spectrogram, normalized to 128 x 128 "
        "RGB pixels, and evaluated by the bundled Keras classifier. This result is an "
        "automated screening aid, not a definitive authenticity determination. For a known "
        "demo sample, the displayed label is selected only after an exact SHA-256 match "
        "against the documented dataset metadata.",
    )
    return bytes(pdf.output())


def render_sidebar(model_status: str) -> None:
    with st.sidebar:
        st.markdown("## AUDIO FORENSICS")
        st.caption("AI-powered deepfake detection")
        st.markdown("---")
        st.markdown(f"**Model status**  \n`{model_status}`")
        st.markdown("**Processing**")
        st.markdown("- 16 kHz mono audio")
        st.markdown("- Fixed 3-second window")
        st.markdown("- 128 Mel bands")
        st.markdown("- 128 × 128 RGB input")
        st.markdown("**Accepted formats**  \nWAV · MP3 · OGG")
        st.markdown("---")
        st.caption("Use results as forensic screening evidence, not as sole proof.")


def main() -> None:
    st.set_page_config(page_title="Audio Forensics", page_icon="◈", layout="wide")
    load_css()
    try:
        model = load_detection_model()
        model_status = "READY"
    except Exception as exc:
        model = None
        model_status = "UNAVAILABLE"
        st.error(f"Model could not be loaded: {exc}")

    render_sidebar(model_status)
    st.markdown('<div class="app-kicker">AUDIO FORENSICS / SCREENING CONSOLE</div>', unsafe_allow_html=True)
    st.title("Detect synthetic speech before it becomes evidence.")
    st.markdown(
        '<p class="lead">Upload a short voice recording to generate a Mel spectrogram and '
        "run the bundled deepfake classifier.</p>",
        unsafe_allow_html=True,
    )

    uploader_col, info_col = st.columns([1.5, 1], gap="large")
    with uploader_col:
        st.markdown('<div class="panel-label">INPUT RECORDING</div>', unsafe_allow_html=True)
        audio_file = st.file_uploader("Drop an audio file here", type=["wav", "mp3", "ogg"], label_visibility="collapsed")
        if audio_file:
            st.audio(audio_file)
            st.caption(f"{audio_file.name} · {audio_file.size / 1024:.1f} KB")
    with info_col:
        st.markdown('<div class="panel-label">ANALYSIS WINDOW</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="info-panel"><strong>3 seconds</strong><br><span>Audio is resampled, '
            "converted to mono, then trimmed or padded for a consistent model input.</span></div>",
            unsafe_allow_html=True,
        )

    analyze = st.button("ANALYZE AUDIO", type="primary", use_container_width=True, disabled=not audio_file or model is None)
    if analyze and audio_file and model is not None:
        try:
            with st.spinner("Generating forensic analysis..."):
                audio, sample_rate = prepare_audio(audio_file.getvalue())
                spectrogram_image, _ = create_spectrogram(audio, sample_rate)
                model_label, model_confidence, p_fake = predict(model, spectrogram_image)
                known_sample = identify_known_sample(audio_file.getvalue())
                if known_sample:
                    label = known_sample["label"]
                    confidence = None
                    result_source = "SHA-256 verified dataset label"
                else:
                    label = model_label
                    confidence = model_confidence
                    result_source = "CNN model prediction"
            st.session_state["analysis"] = {
                "filename": audio_file.name,
                "image": spectrogram_image,
                "label": label,
                "confidence": confidence,
                "p_fake": p_fake,
                "model_label": model_label,
                "model_confidence": model_confidence,
                "known_sample": known_sample,
                "result_source": result_source,
            }
        except ValueError as exc:
            st.error(str(exc))
        except Exception as exc:
            st.error(f"Analysis failed: {exc}")

    analysis = st.session_state.get("analysis")
    if analysis:
        st.markdown("---")
        spectrogram_col, result_col = st.columns([1.15, 0.85], gap="large")
        with spectrogram_col:
            st.markdown('<div class="panel-label">GENERATED SPECTROGRAM</div>', unsafe_allow_html=True)
            st.image(analysis["image"], use_container_width=True)
            st.caption("Magma Mel spectrogram · 2048 FFT · 512 hop length · 128 Mel bands")
        with result_col:
            if analysis["known_sample"]:
                status_class = "result-fake" if analysis["label"] == "FAKE" else "result-real"
                st.markdown(
                    f'<div class="result-panel {status_class}"><span class="result-label">VERIFIED DEMO SAMPLE</span>'
                    f'<strong>{analysis["label"]}</strong><span>WaveFake dataset label · exact SHA-256 match</span></div>',
                    unsafe_allow_html=True,
                )
                st.success(
                    f"Verified source: {analysis['known_sample']['filename']} · "
                    f"dataset code {analysis['known_sample']['dataset_label']}"
                )
                st.caption(
                    f"CNN independently predicted {analysis['model_label']} "
                    f"with {analysis['model_confidence']:.2%} confidence."
                )
            else:
                status_class = "result-fake" if analysis["label"] == "FAKE" else "result-real"
                st.markdown(
                    f'<div class="result-panel {status_class}"><span class="result-label">AI MODEL PREDICTION</span>'
                    f'<strong>{analysis["label"]}</strong><span>CNN confidence {analysis["confidence"]:.2%}</span></div>',
                    unsafe_allow_html=True,
                )
                st.caption("Result source: bundled CNN model")
            st.metric("Raw probability of FAKE", f'{analysis["p_fake"]:.6f}')
            st.caption(f"Input: {analysis['filename']}")
            with st.expander("Technical analysis details"):
                st.markdown(
                    "- **Audio:** 16 kHz, mono, fixed 3-second window\n"
                    "- **Spectrogram:** 128 Mel bands · 2048 FFT · 512 hop length\n"
                    f"- **CNN output:** {analysis['model_label']} · {analysis['model_confidence']:.2%}\n"
                    f"- **Label source:** {analysis['result_source']}"
                )
            report = create_report(
                analysis["filename"],
                analysis["label"],
                analysis["confidence"],
                analysis["p_fake"],
                analysis["image"],
                analysis["model_label"],
                analysis["model_confidence"],
                analysis["known_sample"],
            )
            st.download_button(
                "DOWNLOAD FORENSIC REPORT",
                data=report,
                file_name="audio_deepfake_report.pdf",
                mime="application/pdf",
                use_container_width=True,
            )


if __name__ == "__main__":
    main()
