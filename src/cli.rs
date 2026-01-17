use std::path::PathBuf;

use clap::Parser;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
pub(crate) struct Args {
    /// Run on CPU.
    #[arg(long)]
    pub(crate) cpu: bool,

    /// Run on Metal (requires build with `--features metal`).
    #[arg(long, conflicts_with = "cpu")]
    pub(crate) metal: bool,

    /// Text to synthesize.
    #[arg(long)]
    pub(crate) text: Option<String>,

    /// Path to a text file to synthesize.
    #[arg(long)]
    pub(crate) text_file: Option<PathBuf>,

    /// Voice name from embeddings dir, or a .safetensors embedding path, or an audio file.
    #[arg(long, default_value = "alba")]
    pub(crate) voice: String,

    /// Output wav file path.
    #[arg(long, default_value = "out.wav")]
    pub(crate) output: PathBuf,

    /// Path to pocket-tts weights (safetensors).
    #[arg(long, default_value = "weights/pocket-tts.safetensors")]
    pub(crate) weights: PathBuf,

    /// Path to tokenizer.json.
    #[arg(long, default_value = "weights/tokenizer.json")]
    pub(crate) tokenizer: PathBuf,

    /// Path to model config YAML (overrides --variant lookup).
    #[arg(long)]
    pub(crate) config: Option<PathBuf>,

    /// Model variant name (used to locate a YAML in the weights dir).
    #[arg(long, default_value = "b6369a24")]
    pub(crate) variant: String,

    /// Directory containing voice embeddings (*.safetensors).
    #[arg(long, default_value = "weights/embeddings")]
    pub(crate) embeddings_dir: PathBuf,

    /// List available voice names and exit.
    #[arg(long)]
    pub(crate) list_voices: bool,

    /// Temperature for sampling.
    #[arg(long)]
    pub(crate) temperature: Option<f64>,

    /// Number of LSD decode steps.
    #[arg(long)]
    pub(crate) lsd_steps: Option<usize>,

    /// Clamp noise samples to [-noise_clamp, noise_clamp].
    #[arg(long)]
    pub(crate) noise_clamp: Option<f64>,

    /// EOS threshold.
    #[arg(long)]
    pub(crate) eos_threshold: Option<f64>,

    /// Extra frames to generate after EOS.
    #[arg(long)]
    pub(crate) frames_after_eos: Option<usize>,

    /// Use streaming/stateful inference for generation and decoding.
    #[arg(long)]
    pub(crate) streaming: bool,
}
