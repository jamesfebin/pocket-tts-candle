# pocket-tts-rust

Rust port of Pocket TTS using Candle.

## Layout

- `src/main.rs` is copied from the Candle example (logic kept identical).
- The default weight paths expect assets under `weights/` at the project root.

By default the runner will auto‑download missing weights and voice embeddings from Hugging Face.
If you need gated weights (voice cloning), set `HF_TOKEN` (or run `huggingface-cli login`) before running.
When you pass `--voice /path/to/voice.wav`, the runner will attempt to download the gated weights.

You can also place the weights, tokenizer, and embeddings in:

```
./weights
```

Or pass explicit paths via `--weights`, `--tokenizer`, and `--embeddings-dir`.

Expected files (names match defaults in `main.rs`):
- `pocket-tts.safetensors`
- `tts_b6369a24.safetensors` (voice cloning weights when using `--voice /path/to.wav`)
- `tokenizer.json`
- `tokenizer.model`
- `embeddings/` (voice prompts)
- `b6369a24.yaml` (already included)

If `tokenizer.json` is missing, the runner will try to download it. If Hugging Face doesn’t
host it, the runner will generate `tokenizer.json` from `tokenizer.model` on first run
(uses `sp.proto` in this repo; no Python required).

## Build + Run

From this directory:

```bash
cargo run --release -- \
  --text "Hello from Pocket TTS." \
  --voice alba \
  --output out.wav
```

Streaming:

```bash
cargo run --release -- \
  --text "Hello from Pocket TTS." \
  --voice alba \
  --output out.wav \
  --streaming
```

Metal (macOS):

```bash
cargo run --features metal --release -- \
  --text "Hello from Pocket TTS." \
  --voice alba \
  --output out.wav \
  --metal
```

Voice cloning from audio (requires audio features):

```bash
cargo run --features mimi --release -- \
  --text "Hello from Pocket TTS." \
  --voice /path/to/voice.wav \
  --output out.wav
```

Note: when `--voice` points to an audio file, the runner will prefer the full
voice‑cloning weights and will download them to `weights/tts_b6369a24.safetensors`
by default. If you already have the correct weights elsewhere, pass `--weights`.

Metal + voice cloning:

```bash
cargo run --features "mimi,metal" --release -- \
  --text "Hello from Pocket TTS." \
  --voice /path/to/voice.wav \
  --output out.wav \
  --metal
```

## Config

Use a config YAML:

```bash
cargo run --release -- \
  --text "Hello from Pocket TTS." \
  --voice alba \
  --output out.wav \
  --variant b6369a24
```

Or pass an explicit config path:

```bash
cargo run --release -- \
  --text "Hello from Pocket TTS." \
  --voice alba \
  --output out.wav \
  --config ./weights/b6369a24.yaml
```



## License

This port follows the same license as the original Pocket TTS project (MIT).
