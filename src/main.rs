use std::fs;
use std::fs::File;
use std::io::BufWriter;
use std::path::PathBuf;

use anyhow::{bail, Result};
use candle::{DType, Device};
use candle_nn::VarBuilder;
use clap::Parser;

mod audio;
mod audio_utils;
mod assets;
mod cli;
mod config;
mod device;
mod hf;
mod model;
mod spm;
mod tokenizer_utils;
mod voice;
mod wav;

use assets::{effective_weights_path, ensure_assets, voice_is_audio_path};
use cli::Args;
use config::load_config_or_default;
use model::PocketTts;
use tokenizer_utils::{load_tokenizer, prepare_text_prompt, split_into_best_sentences, tokenize};
use voice::{list_voices, load_voice_prompt};

fn main() -> Result<()> {
    let args = Args::parse();
    let device = if args.metal {
        #[cfg(feature = "metal")]
        {
            Device::new_metal(0)?
        }
        #[cfg(not(feature = "metal"))]
        {
            bail!("--metal requested but binary built without metal support; re-run with --features metal");
        }
    } else {
        device::device(args.cpu)?
    };

    if args.list_voices {
        list_voices(&args.embeddings_dir)?;
        return Ok(());
    }

    let raw_text = match (args.text.as_deref(), args.text_file.as_ref()) {
        (Some(_), Some(_)) => {
            bail!("use either --text or --text-file, not both");
        }
        (Some(text), None) => text.to_string(),
        (None, Some(path)) => fs::read_to_string(path)
            .map_err(|e| anyhow::anyhow!("failed to read text file {:?}: {e}", path))?,
        (None, None) => bail!("missing input: provide --text or --text-file"),
    };

    let config = load_config_or_default(&args)?;
    let voice_path = PathBuf::from(&args.voice);
    let voice_is_audio = voice_is_audio_path(&voice_path);
    let weights_path = effective_weights_path(&args, &config, voice_is_audio);
    if voice_is_audio && weights_path != args.weights {
        println!("using voice-cloning weights at {:?}", weights_path);
    }
    ensure_assets(&args, &config, &weights_path, voice_is_audio)?;

    let tokenizer = load_tokenizer(&args.tokenizer, &config)?;

    let chunks = split_into_best_sentences(&tokenizer, &raw_text)?;

    let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[weights_path], DType::F32, &device)? };
    let model = PocketTts::new(vb, &config)?;
    let audio_prompt = load_voice_prompt(&args.voice, &args.embeddings_dir, &device, &model)?;

    let temperature = args.temperature.unwrap_or(0.7);
    let lsd_steps = args.lsd_steps.unwrap_or(1);
    let noise_clamp = args.noise_clamp;
    let eos_threshold = args.eos_threshold.unwrap_or(-4.0);

    let mut pcm = Vec::new();
    for chunk in chunks {
        let (text, frames_after_eos_guess) = prepare_text_prompt(&chunk);
        let frames_after_eos = args
            .frames_after_eos
            .unwrap_or(frames_after_eos_guess + 2);
        let tokens = tokenize(&tokenizer, &text)?;
        let mut chunk_pcm = if args.streaming {
            model.generate_audio_streaming(
                &tokens,
                &audio_prompt,
                temperature,
                lsd_steps,
                noise_clamp,
                eos_threshold,
                frames_after_eos,
            )?
        } else {
            model.generate_audio(
                &tokens,
                &audio_prompt,
                temperature,
                lsd_steps,
                noise_clamp,
                eos_threshold,
                frames_after_eos,
            )?
        };
        pcm.append(&mut chunk_pcm);
    }

    let mut output = BufWriter::new(File::create(&args.output)?);
    wav::write_pcm_as_wav(&mut output, &pcm, model.sample_rate as u32)?;
    println!("wrote {:?}", args.output);
    Ok(())
}
