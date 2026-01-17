use std::path::{Path, PathBuf};

use anyhow::{bail, Result};

use crate::cli::Args;
use crate::config::Config;
use crate::hf::{ensure_file_from_hf, parse_hf_uri};
use crate::tokenizer_utils::{
    tokenizer_is_model_path, tokenizer_model_local_path, write_tokenizer_json_from_spm,
};

const DEFAULT_WEIGHTS_PATH: &str = "weights/pocket-tts.safetensors";
const DEFAULT_VOICE_WEIGHTS_PATH: &str = "weights/tts_b6369a24.safetensors";

pub(crate) fn voice_is_audio_path(path: &Path) -> bool {
    path.exists()
        && !path
            .extension()
            .map(|ext| ext == "safetensors")
            .unwrap_or(false)
}

pub(crate) fn effective_weights_path(args: &Args, config: &Config, voice_is_audio: bool) -> PathBuf {
    if !voice_is_audio {
        return args.weights.clone();
    }
    let default_weights = PathBuf::from(DEFAULT_WEIGHTS_PATH);
    if args.weights != default_weights {
        return args.weights.clone();
    }
    if let Some(hf) = config.weights_path.as_deref().and_then(parse_hf_uri) {
        if let Some(filename) = Path::new(&hf.path).file_name().and_then(|f| f.to_str()) {
            return default_weights
                .parent()
                .unwrap_or_else(|| Path::new("."))
                .join(filename);
        }
    }
    PathBuf::from(DEFAULT_VOICE_WEIGHTS_PATH)
}

pub(crate) fn ensure_assets(
    args: &Args,
    config: &Config,
    weights_path: &Path,
    voice_is_audio: bool,
) -> Result<()> {
    let voice_path = PathBuf::from(&args.voice);

    if !weights_path.exists() {
        let weights_uri = if voice_is_audio {
            config.weights_path.as_deref()
        } else {
            config
                .weights_path_without_voice_cloning
                .as_deref()
                .or(config.weights_path.as_deref())
        };
        let weights_uri = weights_uri.ok_or_else(|| anyhow::anyhow!("no weights_path in config"))?;
        if weights_uri.starts_with("hf://") {
            println!("downloading weights from {weights_uri}");
            ensure_file_from_hf(weights_path, weights_uri)?;
        } else {
            bail!("weights not found: {:?} (uri {weights_uri})", weights_path);
        }
    }

    if tokenizer_is_model_path(&args.tokenizer) {
        if !args.tokenizer.exists() {
            let model_uri = tokenizer_model_uri(config)
                .ok_or_else(|| anyhow::anyhow!("tokenizer.model uri missing from config"))?;
            println!("downloading tokenizer.model from {model_uri}");
            ensure_file_from_hf(&args.tokenizer, &model_uri)?;
        }
    } else if !args.tokenizer.exists() {
        let mut fetched = false;
        if let Some(uri) = tokenizer_json_uri_from_config(config) {
            println!("downloading tokenizer.json from {uri}");
            if ensure_file_from_hf(&args.tokenizer, &uri).is_ok() {
                fetched = true;
            }
        }
        if !fetched {
            let model_uri = tokenizer_model_uri(config)
                .ok_or_else(|| anyhow::anyhow!("tokenizer.model uri missing from config"))?;
            let model_path = tokenizer_model_local_path(
                &args.tokenizer,
                &config.flow_lm.lookup_table.tokenizer_path,
            );
            if !model_path.exists() {
                println!("downloading tokenizer.model from {model_uri}");
                ensure_file_from_hf(&model_path, &model_uri)?;
            }
            if model_path.exists() {
                println!("generating tokenizer.json from {:?}", model_path);
                write_tokenizer_json_from_spm(&model_path, &args.tokenizer)?;
            } else {
                bail!("tokenizer.model not found: {:?}", model_path);
            }
        }
    }

    if !voice_path.exists() {
        if let Some(uri) = predefined_voice_uri(&args.voice, config) {
            let dest = args.embeddings_dir.join(format!("{}.safetensors", args.voice));
            if !dest.exists() {
                println!("downloading voice embedding from {uri}");
                ensure_file_from_hf(&dest, &uri)?;
            }
        }
    }

    if !weights_path.exists() {
        bail!("weights not found: {:?}", weights_path);
    }
    if !args.tokenizer.exists() {
        bail!("tokenizer not found: {:?}", args.tokenizer);
    }

    Ok(())
}

fn predefined_voice_uri(voice: &str, config: &Config) -> Option<String> {
    const VOICES: [&str; 8] = [
        "alba", "marius", "javert", "jean", "fantine", "cosette", "eponine", "azelma",
    ];
    if !VOICES.contains(&voice) {
        return None;
    }
    if let Some(base) = config.weights_path_without_voice_cloning.as_deref() {
        if let Some(hf) = parse_hf_uri(base) {
            let revision = hf.revision.unwrap_or_else(|| "main".to_string());
            return Some(format!(
                "hf://{}/embeddings/{voice}.safetensors@{revision}",
                hf.repo_id
            ));
        }
    }
    Some(format!(
        "hf://kyutai/pocket-tts-without-voice-cloning/embeddings/{voice}.safetensors@d4fdd22ae8c8e1cb3634e150ebeff1dab2d16df3"
    ))
}

fn tokenizer_json_uri_from_config(config: &Config) -> Option<String> {
    let hf = parse_hf_uri(&config.flow_lm.lookup_table.tokenizer_path)?;
    let dir = match hf.path.rsplit_once('/') {
        Some((dir, _)) => dir.to_string(),
        None => "".to_string(),
    };
    let path = if dir.is_empty() {
        "tokenizer.json".to_string()
    } else {
        format!("{dir}/tokenizer.json")
    };
    let revision = hf.revision.unwrap_or_else(|| "main".to_string());
    Some(format!(
        "hf://{}/{}@{}",
        hf.repo_id,
        path,
        revision
    ))
}

fn tokenizer_model_uri(config: &Config) -> Option<String> {
    let uri = config.flow_lm.lookup_table.tokenizer_path.clone();
    if uri.starts_with("hf://") {
        Some(uri)
    } else {
        None
    }
}
