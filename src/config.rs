use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{bail, Context, Result};
use serde::Deserialize;

use crate::cli::Args;

#[derive(Debug, Clone, Deserialize)]
pub(crate) struct Config {
    pub(crate) flow_lm: FlowLmConfig,
    pub(crate) mimi: MimiConfig,
    pub(crate) weights_path: Option<String>,
    pub(crate) weights_path_without_voice_cloning: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
pub(crate) struct FlowLmConfig {
    pub(crate) dtype: String,
    pub(crate) flow: FlowConfig,
    pub(crate) transformer: FlowLmTransformerConfig,
    pub(crate) lookup_table: LookupTableConfig,
    pub(crate) weights_path: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
pub(crate) struct FlowConfig {
    pub(crate) dim: usize,
    pub(crate) depth: usize,
}

#[derive(Debug, Clone, Deserialize)]
pub(crate) struct FlowLmTransformerConfig {
    pub(crate) hidden_scale: usize,
    pub(crate) max_period: f64,
    pub(crate) d_model: usize,
    pub(crate) num_heads: usize,
    pub(crate) num_layers: usize,
}

#[derive(Debug, Clone, Deserialize)]
pub(crate) struct LookupTableConfig {
    pub(crate) dim: usize,
    pub(crate) n_bins: usize,
    pub(crate) tokenizer: String,
    pub(crate) tokenizer_path: String,
}

#[derive(Debug, Clone, Deserialize)]
pub(crate) struct MimiConfig {
    pub(crate) dtype: String,
    pub(crate) sample_rate: usize,
    pub(crate) channels: usize,
    pub(crate) frame_rate: f64,
    pub(crate) seanet: SeanetConfig,
    pub(crate) transformer: MimiTransformerConfig,
    pub(crate) quantizer: QuantizerConfig,
    pub(crate) weights_path: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
pub(crate) struct SeanetConfig {
    pub(crate) dimension: usize,
    pub(crate) channels: usize,
    pub(crate) n_filters: usize,
    pub(crate) n_residual_layers: usize,
    pub(crate) ratios: Vec<usize>,
    pub(crate) kernel_size: usize,
    pub(crate) residual_kernel_size: usize,
    pub(crate) last_kernel_size: usize,
    pub(crate) dilation_base: usize,
    pub(crate) pad_mode: String,
    pub(crate) compress: usize,
}

#[derive(Debug, Clone, Deserialize)]
pub(crate) struct MimiTransformerConfig {
    pub(crate) d_model: usize,
    pub(crate) input_dimension: usize,
    pub(crate) output_dimensions: Vec<usize>,
    pub(crate) num_heads: usize,
    pub(crate) num_layers: usize,
    pub(crate) layer_scale: f64,
    pub(crate) context: usize,
    #[serde(default = "default_mimi_max_period")]
    pub(crate) max_period: f64,
    pub(crate) dim_feedforward: usize,
}

#[derive(Debug, Clone, Deserialize)]
pub(crate) struct QuantizerConfig {
    pub(crate) dimension: usize,
    pub(crate) output_dimension: usize,
}

fn default_mimi_max_period() -> f64 {
    10_000.0
}

impl Config {
    pub(crate) fn default_b6369a24() -> Self {
        Self {
            weights_path: Some(
                "hf://kyutai/pocket-tts/tts_b6369a24.safetensors@427e3d61b276ed69fdd03de0d185fa8a8d97fc5b"
                    .to_string(),
            ),
            weights_path_without_voice_cloning: Some(
                "hf://kyutai/pocket-tts-without-voice-cloning/tts_b6369a24.safetensors@d4fdd22ae8c8e1cb3634e150ebeff1dab2d16df3"
                    .to_string(),
            ),
            flow_lm: FlowLmConfig {
                dtype: "float32".to_string(),
                flow: FlowConfig { depth: 6, dim: 512 },
                transformer: FlowLmTransformerConfig {
                    d_model: 1024,
                    hidden_scale: 4,
                    max_period: 10_000.0,
                    num_heads: 16,
                    num_layers: 6,
                },
                lookup_table: LookupTableConfig {
                    dim: 1024,
                    n_bins: 4000,
                    tokenizer: "sentencepiece".to_string(),
                    tokenizer_path: "hf://kyutai/pocket-tts-without-voice-cloning/tokenizer.model@d4fdd22ae8c8e1cb3634e150ebeff1dab2d16df3".to_string(),
                },
                weights_path: None,
            },
            mimi: MimiConfig {
                dtype: "float32".to_string(),
                sample_rate: 24_000,
                channels: 1,
                frame_rate: 12.5,
                seanet: SeanetConfig {
                    dimension: 512,
                    channels: 1,
                    n_filters: 64,
                    n_residual_layers: 1,
                    ratios: vec![6, 5, 4],
                    kernel_size: 7,
                    residual_kernel_size: 3,
                    last_kernel_size: 3,
                    dilation_base: 2,
                    pad_mode: "constant".to_string(),
                    compress: 2,
                },
                transformer: MimiTransformerConfig {
                    d_model: 512,
                    input_dimension: 512,
                    output_dimensions: vec![512],
                    num_heads: 8,
                    num_layers: 2,
                    layer_scale: 0.01,
                    context: 250,
                    max_period: 10_000.0,
                    dim_feedforward: 2048,
                },
                quantizer: QuantizerConfig {
                    dimension: 32,
                    output_dimension: 512,
                },
                weights_path: None,
            },
        }
    }
}

fn load_config(path: &Path) -> Result<Config> {
    let contents = fs::read_to_string(path)
        .with_context(|| format!("failed to read config {:?}", path))?;
    serde_yaml::from_str(&contents)
        .with_context(|| format!("failed to parse config {:?}", path))
}

fn resolve_config_path(args: &Args) -> Option<PathBuf> {
    if let Some(path) = &args.config {
        return Some(path.clone());
    }
    let weights_dir = args
        .weights
        .parent()
        .map(Path::to_path_buf)
        .unwrap_or_else(|| PathBuf::from("."));
    let by_variant = weights_dir.join(format!("{}.yaml", args.variant));
    if by_variant.exists() {
        return Some(by_variant);
    }
    let config_yaml = weights_dir.join("config.yaml");
    if config_yaml.exists() {
        return Some(config_yaml);
    }
    None
}

pub(crate) fn load_config_or_default(args: &Args) -> Result<Config> {
    if let Some(path) = &args.config {
        if !path.exists() {
            bail!("config not found: {:?}", path);
        }
        return load_config(path);
    }
    if let Some(path) = resolve_config_path(args) {
        return load_config(&path);
    }
    eprintln!("config not found; using built-in defaults");
    Ok(Config::default_b6369a24())
}
