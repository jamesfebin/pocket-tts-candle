use std::path::{Path, PathBuf};

use anyhow::{bail, Result};
use candle::{Device, Tensor};

use crate::model::PocketTts;

pub(crate) fn list_voices(embeddings_dir: &Path) -> Result<()> {
    if !embeddings_dir.exists() {
        println!("no embeddings dir: {:?}", embeddings_dir);
        return Ok(());
    }
    let mut names: Vec<_> = embeddings_dir
        .read_dir()?
        .filter_map(|entry| entry.ok())
        .filter(|entry| entry.path().extension().map(|e| e == "safetensors").unwrap_or(false))
        .filter_map(|entry| entry.path().file_stem().map(|s| s.to_string_lossy().to_string()))
        .collect();
    names.sort();
    if names.is_empty() {
        println!("no voices found in {:?}", embeddings_dir);
        return Ok(());
    }
    println!("available voices:");
    for name in names {
        println!("- {name}");
    }
    Ok(())
}

pub(crate) fn load_voice_prompt(
    voice: &str,
    embeddings_dir: &Path,
    device: &Device,
    model: &PocketTts,
) -> Result<Tensor> {
    let voice_path = PathBuf::from(voice);
    if voice_path.exists() {
        return load_voice_from_path(&voice_path, device, model);
    }
    let candidate = embeddings_dir.join(format!("{voice}.safetensors"));
    if candidate.exists() {
        return load_voice_embedding(&candidate, device);
    }
    bail!("unknown voice '{voice}', use --list-voices or pass a valid path")
}

fn load_voice_from_path(path: &Path, device: &Device, model: &PocketTts) -> Result<Tensor> {
    if path
        .extension()
        .map(|ext| ext == "safetensors")
        .unwrap_or(false)
    {
        return load_voice_embedding(path, device);
    }
    model.audio_prompt_from_path(path, device)
}

fn load_voice_embedding(path: &Path, device: &Device) -> Result<Tensor> {
    let tensors = candle::safetensors::load(path, device)?;
    let audio_prompt = tensors
        .get("audio_prompt")
        .ok_or_else(|| anyhow::anyhow!("missing audio_prompt in {:?}", path))?;
    Ok(audio_prompt.clone())
}
