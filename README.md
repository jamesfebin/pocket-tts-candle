# Pocket TTS Rust

> **Archived**: This repository is archived and no longer maintained. Please use [babybirdprd/pocket-tts](https://github.com/babybirdprd/pocket-tts), which is a higher-quality and more performant implementation than this one.

A Rust port of [Kyutai's Pocket TTS](https://github.com/kyutai-labs/pocket-tts) using the [Candle](https://github.com/huggingface/candle) ML framework.

## About

This is an unofficial Rust implementation of Pocket TTS, a lightweight text-to-speech model designed to run efficiently on CPUs. The original model was developed by [Kyutai Labs](https://kyutai.org/) and is a 100M parameter transformer-based TTS system that delivers low latency (~200ms first chunk) and faster-than-realtime generation.

### Key Features

- **CPU-optimized**: Runs efficiently on consumer CPUs without requiring GPU acceleration
- **Low latency**: ~200ms to generate the first audio chunk
- **Streaming support**: Generate and stream audio in real-time
- **Voice cloning**: Clone voices from audio samples (requires `mimi` feature)
- **Metal support**: Hardware acceleration on macOS (requires `metal` feature)
- **Auto-download**: Automatically downloads model weights from Hugging Face

## Original Project

This port is based on the excellent work by the Kyutai team:

Paper: [Pocket TTS: A Small Model for High-Quality Speech Synthesis](https://arxiv.org/abs/2509.06926)

Original Authors: Manu Orsini*, Simon Rouard*, Gabriel De Marmiesse*, Václav Volhejn, Neil Zeghidour, Alexandre Défossez (*equal contribution)

Organization: [Kyutai Labs](https://kyutai.org/)

Original Repository: [kyutai-labs/pocket-tts](https://github.com/kyutai-labs/pocket-tts)

Model Card: [kyutai/pocket-tts on Hugging Face](https://huggingface.co/kyutai/pocket-tts)

Demo: [kyutai.org/tts](https://kyutai.org/tts)

## Installation

### Prerequisites

- Rust 2021 edition or later
- Optional: Hugging Face token (for voice cloning with gated models)

### Build from source

```bash
git clone <this-repository>
cd pocket-tts-rust
cargo build --release
```

## Usage

### Basic Text-to-Speech

Generate audio with a preset voice:

```bash
cargo run --release -- \
  --text "Hello from Pocket TTS." \
  --voice alba \
  --output out.wav
```

### Streaming Mode

Enable streaming for lower latency:

```bash
cargo run --release -- \
  --text "Hello from Pocket TTS." \
  --voice alba \
  --output out.wav \
  --streaming
```

### Metal Acceleration (macOS)

Use Metal for hardware acceleration on Apple Silicon:

```bash
cargo run --features metal --release -- \
  --text "Hello from Pocket TTS." \
  --voice alba \
  --output out.wav \
  --metal
```

### Voice Cloning

Clone a voice from an audio file (requires `mimi` feature and audio processing dependencies):

```bash
cargo run --features mimi --release -- \
  --text "Hello from Pocket TTS." \
  --voice /path/to/voice.wav \
  --output out.wav
```

**Note**: Voice cloning automatically downloads gated model weights. Set `HF_TOKEN` environment variable or run `huggingface-cli login` for authentication.

### Combined Features

Use Metal with voice cloning:

```bash
cargo run --features "mimi,metal" --release -- \
  --text "Hello from Pocket TTS." \
  --voice /path/to/voice.wav \
  --output out.wav \
  --metal
```

## Available Voices

The model includes several preset voice prompts. You can see the full catalog and licenses at the [tts-voices repository](https://huggingface.co/kyutai/tts-voices):

- **alba** - [Sample](https://huggingface.co/kyutai/tts-voices/blob/main/alba-mackenna/casual.wav)
- **marius** - [Sample](https://huggingface.co/kyutai/tts-voices/blob/main/voice-donations/Selfie.wav)
- **javert** - [Sample](https://huggingface.co/kyutai/tts-voices/blob/main/voice-donations/Butter.wav)
- **jean** - [Sample](https://huggingface.co/kyutai/tts-voices/blob/main/ears/p010/freeform_speech_01.wav)
- **fantine** - [Sample](https://huggingface.co/kyutai/tts-voices/blob/main/vctk/p244_023.wav)
- **cosette** - [Sample](https://huggingface.co/kyutai/tts-voices/blob/main/expresso/ex04-ex02_confused_001_channel1_499s.wav)
- **eponine** - [Sample](https://huggingface.co/kyutai/tts-voices/blob/main/vctk/p262_023.wav)
- **azelma** - [Sample](https://huggingface.co/kyutai/tts-voices/blob/main/vctk/p303_023.wav)

## Model Weights and Assets

The runner automatically downloads missing weights and voice embeddings from Hugging Face. By default, assets are stored in `./weights/` at the project root.

### Expected Files

```
./weights/
├── pocket-tts.safetensors          # Main model weights
├── tts_b6369a24.safetensors        # Voice cloning weights (downloaded when needed)
├── tokenizer.json                   # Tokenizer config
├── tokenizer.model                  # SentencePiece model
├── b6369a24.yaml                    # Model configuration
└── embeddings/                      # Voice prompt embeddings
```

### Manual Weight Management

You can specify custom paths for model assets:

```bash
cargo run --release -- \
  --text "Your text here" \
  --voice alba \
  --output out.wav \
  --weights /path/to/weights.safetensors \
  --tokenizer /path/to/tokenizer.json \
  --embeddings-dir /path/to/embeddings
```

### Configuration

Use a YAML config file for model variants:

```bash
cargo run --release -- \
  --text "Hello from Pocket TTS." \
  --voice alba \
  --output out.wav \
  --variant b6369a24
```

Or specify a custom config path:

```bash
cargo run --release -- \
  --text "Hello from Pocket TTS." \
  --voice alba \
  --output out.wav \
  --config ./weights/custom_config.yaml
```

## Cargo Features

- `metal` - Enable Metal acceleration for macOS
- `mimi` - Enable voice cloning from audio files (includes `symphonia` and `rubato` audio processing)
- `symphonia` - Audio decoding support
- `rubato` - Audio resampling support

## Implementation Notes

This port maintains compatibility with the original Candle example implementation while providing a complete CLI interface. The model architecture and inference logic remain faithful to the original Python implementation.

## Limitations

- English language only (same as original model)
- Requires CPU or Metal backend (GPU support not needed due to model efficiency)

## Prohibited Use

Use of this model must comply with all applicable laws and regulations and must not result in, involve, or facilitate any illegal, harmful, deceptive, fraudulent, or unauthorized activity. Prohibited uses include, without limitation:

- Voice impersonation or cloning without explicit and lawful consent
- Misinformation, disinformation, or deception (including fake news, fraudulent calls, or presenting generated content as genuine recordings)
- Generation of unlawful, harmful, libelous, abusive, harassing, discriminatory, hateful, or privacy-invasive content

We disclaim all liability for any non-compliant use.

## License

This port follows the same license as the original Pocket TTS project (MIT).

## Citation

If you use this model in your research or applications, please cite the original paper:

```bibtex
@article{orsini2025pockettts,
  title={Pocket TTS: A Small Model for High-Quality Speech Synthesis},
  author={Orsini, Manu and Rouard, Simon and De Marmiesse, Gabriel and Volhejn, Václav and Zeghidour, Neil and Défossez, Alexandre},
  journal={arXiv preprint arXiv:2509.06926},
  year={2025}
}
```

## Acknowledgments

This Rust port was made possible by:
- The original [Pocket TTS](https://github.com/kyutai-labs/pocket-tts) team at Kyutai Labs
- The [Candle](https://github.com/huggingface/candle) ML framework by Hugging Face
- The Rust ML community

All credit for the model architecture, training, and original implementation goes to the Kyutai team.
