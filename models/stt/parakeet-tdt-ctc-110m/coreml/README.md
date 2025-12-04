# Parakeet‑TDT CTC 110M — CoreML Export (Mel+Encoder+CTC)

This target contains tools to export NVIDIA's `nvidia/parakeet-tdt_ctc-110m` hybrid RNNT/CTC model to CoreML, with an emphasis on the **CTC branch** used for keyword spotting and custom vocabulary support.

The goal is to produce a CoreML bundle that mirrors NeMo's preprocessing and CTC head, so that Swift-side CTC keyword spotting can rely on **model-correct** acoustics rather than the Argmax MelSpectrogram export.

## Layout

```text
mobius/models/stt/parakeet-tdt-ctc-110m/coreml
├── README.md              # This file
├── convert-coreml.py      # CLI for exporting fused Mel+Encoder+CTC to CoreML
├── pyproject.toml         # Per-target Python environment (NeMo + coremltools)
└── audio/                 # Optional trace audio (15s 16kHz) for export
```

## Environment

From this directory:

```bash
uv sync
```

This will create/update a local environment pinned by `pyproject.toml` and `uv.lock` (Python 3.10.12, NeMo, coremltools, etc.).

## Export CoreML (Mel+Encoder+CTC)

The `convert-coreml.py` script exports a fused module:

```text
audio_signal [1, S] (16kHz mono, 15s window)
    ↓ preprocessor (NeMo)
mel spectrogram
    ↓ encoder (NeMo)
encoder features
    ↓ ctc_decoder (NeMo aux_ctc branch)
log_probs [1, T, V+1] (CTC log-probabilities)
```

Usage:

```bash
uv run python convert-coreml.py convert \
  --model-id nvidia/parakeet-tdt_ctc-110m \
  --output-dir parakeet_ctc_coreml
```

Notes:
- Export uses a fixed 15s window (audio_signal shape [1,240000]) to avoid runtime shape issues in CoreML preview.
- Two-model export:
  - `parakeet_ctc_mel_encoder.mlpackage` — waveform -> encoder, encoder_length
  - `parakeet_ctc_decoder.mlpackage` — encoder -> log_probs

Outputs:

- `parakeet_ctc_mel_encoder.mlpackage` — waveform → CTC log_probs + encoder_length
- `metadata.json` — input/output shapes and export configuration

## Next Steps (Swift Integration)

Once exported and validated:

- Point FluidAudio's CTC loader at the new `parakeet_ctc_mel_encoder.mlpackage` instead of `argmaxinc/ctckit-pro`.
- Re-run the CTC keyword spotter parity checks (NeMo Python vs CoreML) on Kokoro TTS clips.
- When parity is acceptable, re-enable time-aligned CTC boosting in Swift for Argmax-style Custom Vocabulary.
