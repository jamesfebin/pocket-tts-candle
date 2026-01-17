use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{bail, Context, Result};
use prost::Message;
use tokenizers::models::unigram::Unigram;
use tokenizers::normalizers::replace::ReplacePattern;
use tokenizers::normalizers::{NormalizerWrapper, Precompiled, Replace, Sequence};
use tokenizers::pre_tokenizers::metaspace::{Metaspace, PrependScheme};
use tokenizers::Tokenizer;

use crate::config::Config;
use crate::hf::parse_hf_uri;
use crate::spm;

const METASPACE_CHAR: char = '\u{2581}';

pub(crate) fn tokenizer_is_model_path(path: &Path) -> bool {
    path.extension()
        .and_then(|ext| ext.to_str())
        .map(|ext| ext.eq_ignore_ascii_case("model"))
        .unwrap_or(false)
}

pub(crate) fn tokenizer_model_local_path(
    tokenizer_path: &Path,
    tokenizer_path_config: &str,
) -> PathBuf {
    let filename = if let Some(hf) = parse_hf_uri(tokenizer_path_config) {
        Path::new(&hf.path)
            .file_name()
            .and_then(|f| f.to_str())
            .unwrap_or("tokenizer.model")
            .to_string()
    } else {
        Path::new(tokenizer_path_config)
            .file_name()
            .and_then(|f| f.to_str())
            .unwrap_or("tokenizer.model")
            .to_string()
    };
    tokenizer_path
        .parent()
        .unwrap_or_else(|| Path::new("."))
        .join(filename)
}

fn tokenizer_from_spm(model_path: &Path) -> Result<Tokenizer> {
    let data = fs::read(model_path)
        .with_context(|| format!("failed to read sentencepiece model {:?}", model_path))?;
    let model = spm::ModelProto::decode(data.as_slice())
        .map_err(|e| anyhow::anyhow!("failed to parse sentencepiece model: {e}"))?;
    let trainer = model
        .trainer_spec
        .as_ref()
        .ok_or_else(|| anyhow::anyhow!("sentencepiece model missing trainer_spec"))?;
    let model_type = trainer.model_type.unwrap_or(0);
    if model_type != 1 {
        bail!("unsupported sentencepiece model_type {model_type} (expected UNIGRAM=1)");
    }
    let unk_id = trainer.unk_id.unwrap_or(0) as usize;
    let byte_fallback = trainer.byte_fallback.unwrap_or(false);
    let vocab: Vec<(String, f64)> = model
        .pieces
        .iter()
        .map(|piece| {
            let token = piece.piece.clone().unwrap_or_default();
            let score = piece.score.unwrap_or(0.0) as f64;
            (token, score)
        })
        .collect();
    let unigram = Unigram::from(vocab, Some(unk_id), byte_fallback)
        .map_err(|e| anyhow::anyhow!("failed to build unigram tokenizer: {e}"))?;
    let mut tokenizer = Tokenizer::new(unigram);

    let mut normalizers = Vec::new();
    if let Some(spec) = model.normalizer_spec.as_ref() {
        if let Some(charsmap) = spec.precompiled_charsmap.as_ref() {
            if !charsmap.is_empty() {
                let precompiled = Precompiled::from(charsmap)
                    .map_err(|e| anyhow::anyhow!("failed to load precompiled chars map: {e}"))?;
                normalizers.push(NormalizerWrapper::Precompiled(precompiled));
            }
        }
    }
    let replace = Replace::new(ReplacePattern::Regex(" {2,}".to_string()), " ")
        .map_err(|e| anyhow::anyhow!("failed to build normalizer: {e}"))?;
    normalizers.push(NormalizerWrapper::Replace(replace));
    tokenizer.with_normalizer(Some(Sequence::new(normalizers)));

    let metaspace = Metaspace::new(METASPACE_CHAR, PrependScheme::Always, true);
    tokenizer.with_pre_tokenizer(Some(metaspace.clone()));
    tokenizer.with_decoder(Some(metaspace));

    Ok(tokenizer)
}

pub(crate) fn write_tokenizer_json_from_spm(model_path: &Path, json_path: &Path) -> Result<()> {
    let tokenizer = tokenizer_from_spm(model_path)?;
    if let Some(parent) = json_path.parent() {
        fs::create_dir_all(parent)?;
    }
    tokenizer
        .save(json_path, false)
        .map_err(|e| anyhow::anyhow!("failed to write tokenizer.json: {e}"))?;
    Ok(())
}

pub(crate) fn load_tokenizer(tokenizer_path: &Path, config: &Config) -> Result<Tokenizer> {
    if tokenizer_is_model_path(tokenizer_path) {
        return tokenizer_from_spm(tokenizer_path);
    }
    if tokenizer_path.exists() {
        return Tokenizer::from_file(tokenizer_path)
            .map_err(|e| anyhow::anyhow!("failed to load tokenizer: {e}"));
    }
    let model_path = tokenizer_model_local_path(tokenizer_path, &config.flow_lm.lookup_table.tokenizer_path);
    if model_path.exists() {
        return tokenizer_from_spm(&model_path);
    }
    bail!("tokenizer not found: {:?}", tokenizer_path);
}

pub(crate) fn tokenize(tokenizer: &Tokenizer, text: &str) -> Result<Vec<u32>> {
    let enc = tokenizer
        .encode(text, false)
        .map_err(|e| anyhow::anyhow!("tokenizer error: {e}"))?;
    Ok(enc.get_ids().iter().map(|&id| id as u32).collect())
}

pub(crate) fn prepare_text_prompt(text: &str) -> (String, usize) {
    let mut text = text.trim().replace('\n', " ").replace('\r', " ");
    while text.contains("  ") {
        text = text.replace("  ", " ");
    }
    let word_count = text.split_whitespace().count();
    let frames_after_eos_guess = if word_count <= 4 { 3 } else { 1 };

    if let Some(first) = text.chars().next() {
        if first.is_lowercase() {
            let mut chars = text.chars();
            let first = chars.next().unwrap();
            text = first.to_uppercase().collect::<String>() + chars.as_str();
        }
    }
    if text.chars().last().map(|c| c.is_alphanumeric()).unwrap_or(false) {
        text.push('.');
    }
    if word_count < 5 {
        text = format!("        {text}");
    }
    (text, frames_after_eos_guess)
}

pub(crate) fn split_into_best_sentences(tokenizer: &Tokenizer, text: &str) -> Result<Vec<String>> {
    let (text, _) = prepare_text_prompt(text);
    let text = text.trim();
    if text.is_empty() {
        bail!("text prompt cannot be empty");
    }
    let tokens = tokenizer
        .encode(text, false)
        .map_err(|e| anyhow::anyhow!("tokenizer error: {e}"))?;
    let token_ids = tokens.get_ids().to_vec();
    if token_ids.is_empty() {
        return Ok(vec![text.to_string()]);
    }

    let eos_tokens = tokenizer
        .encode(".!...?", false)
        .map_err(|e| anyhow::anyhow!("tokenizer error: {e}"))?;
    let eos_ids_all = eos_tokens.get_ids();
    let eos_ids: Vec<u32> = if eos_ids_all.len() > 1 {
        eos_ids_all[1..].to_vec()
    } else {
        eos_ids_all.to_vec()
    };

    let mut sentence_starts = vec![0usize];
    let mut prev_was_eos = false;
    for (idx, token) in token_ids.iter().enumerate() {
        let is_eos = eos_ids.contains(token);
        if is_eos {
            prev_was_eos = true;
        } else {
            if prev_was_eos {
                sentence_starts.push(idx);
            }
            prev_was_eos = false;
        }
    }
    sentence_starts.push(token_ids.len());

    let mut token_sentences = Vec::new();
    for win in sentence_starts.windows(2) {
        let start = win[0];
        let end = win[1];
        if start >= end {
            continue;
        }
        let piece = tokenizer
            .decode(&token_ids[start..end], false)
            .map_err(|e| anyhow::anyhow!("tokenizer error: {e}"))?;
        token_sentences.push((end - start, piece));
    }

    let max_tokens = 50usize;
    let mut chunks = Vec::new();
    let mut current = String::new();
    let mut current_tokens = 0usize;
    for (count, sentence) in token_sentences {
        if current.is_empty() {
            current = sentence;
            current_tokens = count;
            continue;
        }
        if current_tokens + count <= max_tokens {
            current.push_str(&sentence);
            current_tokens += count;
        } else {
            chunks.push(current);
            current = sentence;
            current_tokens = count;
        }
    }
    if !current.is_empty() {
        chunks.push(current);
    }
    Ok(chunks)
}
