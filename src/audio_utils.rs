use std::path::Path;

use anyhow::{bail, Result};

#[cfg(feature = "symphonia")]
pub(crate) fn load_audio_pcm(path: &Path) -> Result<(Vec<f32>, u32)> {
    crate::audio::pcm_decode(path)
        .map_err(|e| anyhow::anyhow!("failed to decode audio {:?}: {e}", path))
}

#[cfg(not(feature = "symphonia"))]
pub(crate) fn load_audio_pcm(_path: &Path) -> Result<(Vec<f32>, u32)> {
    bail!("audio input requires the symphonia feature, try --features mimi")
}

#[cfg(feature = "rubato")]
pub(crate) fn resample_pcm(pcm: &[f32], sr_in: u32, sr_out: u32) -> Result<Vec<f32>> {
    crate::audio::resample(pcm, sr_in, sr_out)
        .map_err(|e| anyhow::anyhow!("failed to resample audio {sr_in}->{sr_out}: {e}"))
}

#[cfg(not(feature = "rubato"))]
pub(crate) fn resample_pcm(_pcm: &[f32], _sr_in: u32, _sr_out: u32) -> Result<Vec<f32>> {
    bail!("audio resampling requires the rubato feature, try --features mimi")
}
