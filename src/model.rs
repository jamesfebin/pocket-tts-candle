use std::path::Path;

use anyhow::{bail, Result};
use candle::{DType, Device, IndexOp, Module, Result as CandleResult, Tensor, D};
use candle_nn::{Conv1d, Conv1dConfig, ConvTranspose1d, ConvTranspose1dConfig, Embedding, Linear, VarBuilder};
use rand::thread_rng;
use rand_distr::{Distribution, Normal};

use crate::audio_utils::{load_audio_pcm, resample_pcm};
use crate::config::*;
pub(crate) struct PocketTts {
    flow_lm: FlowLmModel,
    mimi_encoder: MimiEncoder,
    mimi_decoder: MimiDecoder,
    pub(crate) sample_rate: usize,
}

impl PocketTts {
    pub(crate) fn new(vb: VarBuilder, config: &Config) -> CandleResult<Self> {
        let flow_lm = FlowLmModel::new(vb.pp("flow_lm"), &config.flow_lm, &config.mimi)?;
        let mimi_encoder = MimiEncoder::new(vb.pp("mimi"), &config.mimi)?;
        let mimi_decoder = MimiDecoder::new(vb.pp("mimi"), &config.mimi)?;
        Ok(Self {
            flow_lm,
            mimi_encoder,
            mimi_decoder,
            sample_rate: config.mimi.sample_rate,
        })
    }

    pub(crate) fn generate_audio(
        &self,
        tokens: &[u32],
        audio_prompt: &Tensor,
        temperature: f64,
        lsd_steps: usize,
        noise_clamp: Option<f64>,
        eos_threshold: f64,
        frames_after_eos: usize,
    ) -> CandleResult<Vec<f32>> {
        let device = self.flow_lm.device();
        let tokens = Tensor::from_vec(tokens.to_vec(), (1, tokens.len()), device)?;
        let text_embeddings = self.flow_lm.text_embeddings(&tokens)?;
        // Match Python prompt order: audio conditioning comes before text embeddings.
        let conditioning = Tensor::cat(&[audio_prompt.clone(), text_embeddings], 1)?;

        let mut sequence = self.flow_lm.bos_sequence()?;

        let max_gen_len = {
            let word_count = tokens.dim(1)? as f64;
            let gen_len_sec = word_count + 2.0;
            (gen_len_sec * self.mimi_encoder.frame_rate) as usize
        };

        let mut eos_step = None;
        for step in 0..max_gen_len {
            let (next_latent, eos_hit) = self.flow_lm.sample_next_latent(
                &sequence,
                &conditioning,
                temperature,
                lsd_steps,
                noise_clamp,
                eos_threshold,
            )?;
            sequence = Tensor::cat(&[sequence, next_latent.unsqueeze(1)?], 1)?;
            if eos_hit && eos_step.is_none() {
                eos_step = Some(step);
            }
            if let Some(eos_step) = eos_step {
                if step >= eos_step + frames_after_eos {
                    break;
                }
            }
        }

        let latents = sequence.narrow(1, 1, sequence.dim(1)? - 1)?;
        let audio = self
            .mimi_decoder
            .decode(&latents, &self.flow_lm.emb_mean, &self.flow_lm.emb_std)?;
        let audio = audio.i(0)?.i(0)?.to_dtype(DType::F32)?;
        Ok(audio.to_vec1::<f32>()?)
    }

    pub(crate) fn generate_audio_streaming(
        &self,
        tokens: &[u32],
        audio_prompt: &Tensor,
        temperature: f64,
        lsd_steps: usize,
        noise_clamp: Option<f64>,
        eos_threshold: f64,
        frames_after_eos: usize,
    ) -> CandleResult<Vec<f32>> {
        let device = self.flow_lm.device();
        let tokens = Tensor::from_vec(tokens.to_vec(), (1, tokens.len()), device)?;
        let text_embeddings = self.flow_lm.text_embeddings(&tokens)?;
        // Match Python prompt order: audio conditioning comes before text embeddings.
        let conditioning = Tensor::cat(&[audio_prompt.clone(), text_embeddings], 1)?;

        let mut flow_state = self.flow_lm.transformer.init_state();
        self.flow_lm.prefill_conditioning(&conditioning, &mut flow_state)?;

        let mut mimi_state = self.mimi_decoder.init_state(1)?;
        let mut current = self.flow_lm.bos_sequence()?;

        let max_gen_len = {
            let word_count = tokens.dim(1)? as f64;
            let gen_len_sec = word_count + 2.0;
            (gen_len_sec * self.mimi_encoder.frame_rate) as usize
        };

        let mut eos_step = None;
        let mut pcm = Vec::new();
        for step in 0..max_gen_len {
            let (next_latent, eos_hit) = self.flow_lm.sample_next_latent_streaming(
                &current,
                &mut flow_state,
                temperature,
                lsd_steps,
                noise_clamp,
                eos_threshold,
            )?;
            let next_latent = next_latent.unsqueeze(1)?;
            let audio_chunk = self.mimi_decoder.decode_step(
                &next_latent,
                &self.flow_lm.emb_mean,
                &self.flow_lm.emb_std,
                &mut mimi_state,
            )?;
            let audio_chunk = audio_chunk.i(0)?.i(0)?.to_dtype(DType::F32)?;
            let mut chunk = audio_chunk.to_vec1::<f32>()?;
            pcm.append(&mut chunk);

            current = next_latent;
            if eos_hit && eos_step.is_none() {
                eos_step = Some(step);
            }
            if let Some(eos_step) = eos_step {
                if step >= eos_step + frames_after_eos {
                    break;
                }
            }
        }

        Ok(pcm)
    }

    fn encode_audio_prompt(&self, audio: &Tensor) -> CandleResult<Tensor> {
        let encoded = self.mimi_encoder.encode(audio)?;
        let latents = encoded.transpose(1, 2)?;
        self.flow_lm.speaker_project(&latents)
    }

    pub(crate) fn audio_prompt_from_path(&self, path: &Path, device: &Device) -> Result<Tensor> {
        let (pcm, sample_rate) = load_audio_pcm(path)?;
        if sample_rate == 0 {
            bail!("missing sample rate in audio file {:?}", path);
        }
        let pcm = if sample_rate != self.sample_rate as u32 {
            resample_pcm(&pcm, sample_rate, self.sample_rate as u32)?
        } else {
            pcm
        };
        let pcm_len = pcm.len();
        let audio = Tensor::from_vec(pcm, (1, 1, pcm_len), device)?;
        let prompt = self.encode_audio_prompt(&audio)?;
        Ok(prompt)
    }
}

struct FlowLmModel {
    conditioner: Embedding,
    input_linear: Linear,
    transformer: Transformer,
    out_norm: LayerNorm,
    out_eos: Linear,
    flow_net: SimpleMlpAdaLn,
    speaker_proj_weight: Tensor,
    bos_emb: Tensor,
    emb_mean: Tensor,
    emb_std: Tensor,
}

impl FlowLmModel {
    fn new(vb: VarBuilder, cfg: &FlowLmConfig, mimi_cfg: &MimiConfig) -> CandleResult<Self> {
        let d_model = cfg.transformer.d_model;
        let ldim = mimi_cfg.quantizer.dimension;
        let n_bins = cfg.lookup_table.n_bins;
        if cfg.lookup_table.dim != d_model {
            return Err(candle::Error::Msg(format!(
                "lookup_table.dim {} does not match transformer.d_model {d_model}",
                cfg.lookup_table.dim
            )));
        }
        let conditioner = candle_nn::embedding(n_bins + 1, d_model, vb.pp("conditioner").pp("embed"))?;
        let input_linear = candle_nn::linear_no_bias(ldim, d_model, vb.pp("input_linear"))?;
        let transformer = Transformer::new(
            vb.pp("transformer"),
            d_model,
            cfg.transformer.num_heads,
            cfg.transformer.num_layers,
            cfg.transformer.hidden_scale,
            None,
            cfg.transformer.max_period,
        )?;
        let out_norm = LayerNorm::from_vb(vb.pp("out_norm"), d_model, 1e-5, true)?;
        let out_eos = candle_nn::linear(d_model, 1, vb.pp("out_eos"))?;
        let flow_net = SimpleMlpAdaLn::new(vb.pp("flow_net"), ldim, cfg.flow.dim, d_model, cfg.flow.depth)?;
        let speaker_proj_in = mimi_cfg.transformer.input_dimension;
        let speaker_proj_weight = vb.get((d_model, speaker_proj_in), "speaker_proj_weight")?;
        let bos_emb = vb.get((ldim,), "bos_emb")?;
        let emb_mean = vb.get((ldim,), "emb_mean")?;
        let emb_std = vb.get((ldim,), "emb_std")?;
        Ok(Self {
            conditioner,
            input_linear,
            transformer,
            out_norm,
            out_eos,
            flow_net,
            speaker_proj_weight,
            bos_emb,
            emb_mean,
            emb_std,
        })
    }

    fn device(&self) -> &Device {
        self.bos_emb.device()
    }

    fn bos_sequence(&self) -> CandleResult<Tensor> {
        let bos = self.bos_emb.reshape((1, 1, self.bos_emb.dims1()?))?;
        Ok(bos)
    }

    fn text_embeddings(&self, tokens: &Tensor) -> CandleResult<Tensor> {
        self.conditioner.forward(tokens)
    }

    fn speaker_project(&self, latents: &Tensor) -> CandleResult<Tensor> {
        let (b, t, c) = latents.dims3()?;
        let weight_t = self.speaker_proj_weight.transpose(0, 1)?;
        let (_, out_dim) = weight_t.dims2()?;
        let flat = latents.reshape((b * t, c))?;
        let proj = flat.matmul(&weight_t)?;
        proj.reshape((b, t, out_dim))
    }

    fn sample_next_latent(
        &self,
        sequence: &Tensor,
        conditioning: &Tensor,
        temperature: f64,
        lsd_steps: usize,
        noise_clamp: Option<f64>,
        eos_threshold: f64,
    ) -> CandleResult<(Tensor, bool)> {
        let input = self.input_linear.forward(sequence)?;
        let input = Tensor::cat(&[conditioning.clone(), input], 1)?;
        let mut transformer_out = self.transformer.forward(&input)?;
        transformer_out = self.out_norm.forward(&transformer_out)?;
        let seq_len = sequence.dim(1)?;
        transformer_out = transformer_out.narrow(1, transformer_out.dim(1)? - seq_len, seq_len)?;
        let last = transformer_out.i((.., seq_len - 1, ..))?;
        let eos_logit = self.out_eos.forward(&last)?;
        let eos_value = eos_logit.to_vec2::<f32>()?[0][0] as f64;
        let eos_hit = eos_value > eos_threshold;

        let std = temperature.sqrt();
        let noise = sample_noise(
            self.device(),
            (1, self.bos_emb.dims1()?),
            std,
            noise_clamp,
        )?;
        let next = lsd_decode(&self.flow_net, &last, &noise, lsd_steps)?;
        Ok((next, eos_hit))
    }

    fn prefill_conditioning(&self, conditioning: &Tensor, state: &mut TransformerState) -> CandleResult<()> {
        if conditioning.dim(1)? == 0 {
            return Ok(());
        }
        let _ = self.transformer.forward_streaming(conditioning, state)?;
        Ok(())
    }

    fn sample_next_latent_streaming(
        &self,
        current: &Tensor,
        state: &mut TransformerState,
        temperature: f64,
        lsd_steps: usize,
        noise_clamp: Option<f64>,
        eos_threshold: f64,
    ) -> CandleResult<(Tensor, bool)> {
        let input = self.input_linear.forward(current)?;
        let mut transformer_out = self.transformer.forward_streaming(&input, state)?;
        transformer_out = self.out_norm.forward(&transformer_out)?;
        let last = transformer_out.i((.., transformer_out.dim(1)? - 1, ..))?;
        let eos_logit = self.out_eos.forward(&last)?;
        let eos_value = eos_logit.to_vec2::<f32>()?[0][0] as f64;
        let eos_hit = eos_value > eos_threshold;

        let std = temperature.sqrt();
        let noise = sample_noise(
            self.device(),
            (1, self.bos_emb.dims1()?),
            std,
            noise_clamp,
        )?;
        let next = lsd_decode(&self.flow_net, &last, &noise, lsd_steps)?;
        Ok((next, eos_hit))
    }
}

struct SimpleMlpAdaLn {
    time_embed: Vec<TimestepEmbedder>,
    cond_embed: Linear,
    input_proj: Linear,
    res_blocks: Vec<ResBlock>,
    final_layer: FinalLayer,
}

impl SimpleMlpAdaLn {
    fn new(vb: VarBuilder, in_channels: usize, model_channels: usize, cond_channels: usize, depth: usize) -> CandleResult<Self> {
        let time_embed = vec![
            TimestepEmbedder::new(vb.pp("time_embed").pp("0"), model_channels, 256, 10_000.0)?,
            TimestepEmbedder::new(vb.pp("time_embed").pp("1"), model_channels, 256, 10_000.0)?,
        ];
        let cond_embed = candle_nn::linear(cond_channels, model_channels, vb.pp("cond_embed"))?;
        let input_proj = candle_nn::linear(in_channels, model_channels, vb.pp("input_proj"))?;
        let mut res_blocks = Vec::with_capacity(depth);
        for i in 0..depth {
            res_blocks.push(ResBlock::new(vb.pp("res_blocks").pp(&i.to_string()), model_channels)?);
        }
        let final_layer = FinalLayer::new(vb.pp("final_layer"), model_channels, in_channels)?;
        Ok(Self {
            time_embed,
            cond_embed,
            input_proj,
            res_blocks,
            final_layer,
        })
    }

    fn forward(&self, c: &Tensor, s: &Tensor, t: &Tensor, x: &Tensor) -> CandleResult<Tensor> {
        let mut x = self.input_proj.forward(x)?;
        let t0 = self.time_embed[0].forward(s)?;
        let t1 = self.time_embed[1].forward(t)?;
        let t_combined = t0.add(&t1)?.broadcast_div(&Tensor::from_vec(vec![2f32], (1,), t0.device())?)?;
        let c = self.cond_embed.forward(c)?;
        let y = t_combined.add(&c)?;
        for block in self.res_blocks.iter() {
            x = block.forward(&x, &y)?;
        }
        self.final_layer.forward(&x, &y)
    }
}

struct TimestepEmbedder {
    freqs: Tensor,
    linear1: Linear,
    linear2: Linear,
    rms_norm: RmsNorm,
}

impl TimestepEmbedder {
    fn new(vb: VarBuilder, hidden_size: usize, freq_size: usize, max_period: f64) -> CandleResult<Self> {
        let freqs = vb.get((freq_size / 2,), "freqs")?;
        let linear1 = candle_nn::linear(freq_size, hidden_size, vb.pp("mlp").pp("0"))?;
        let linear2 = candle_nn::linear(hidden_size, hidden_size, vb.pp("mlp").pp("2"))?;
        let rms_norm = RmsNorm::from_vb(vb.pp("mlp").pp("3"), hidden_size, 1e-5)?;
        Ok(Self {
            freqs,
            linear1,
            linear2,
            rms_norm,
        })
    }

    fn forward(&self, t: &Tensor) -> CandleResult<Tensor> {
        let t = t.broadcast_mul(&self.freqs)?;
        let cos = t.cos()?;
        let sin = t.sin()?;
        let emb = Tensor::cat(&[cos, sin], D::Minus1)?;
        let x = self.linear1.forward(&emb)?;
        let x = silu(&x)?;
        let x = self.linear2.forward(&x)?;
        self.rms_norm.forward(&x)
    }
}

struct ResBlock {
    in_ln: LayerNorm,
    linear1: Linear,
    linear2: Linear,
    ada_ln: Linear,
}

impl ResBlock {
    fn new(vb: VarBuilder, channels: usize) -> CandleResult<Self> {
        let in_ln = LayerNorm::from_vb(vb.pp("in_ln"), channels, 1e-6, true)?;
        let linear1 = candle_nn::linear(channels, channels, vb.pp("mlp").pp("0"))?;
        let linear2 = candle_nn::linear(channels, channels, vb.pp("mlp").pp("2"))?;
        let ada_ln = candle_nn::linear(channels, 3 * channels, vb.pp("adaLN_modulation").pp("1"))?;
        Ok(Self {
            in_ln,
            linear1,
            linear2,
            ada_ln,
        })
    }

    fn forward(&self, x: &Tensor, y: &Tensor) -> CandleResult<Tensor> {
        let ada = self.ada_ln.forward(&silu(y)?)?;
        let (shift, scale, gate) = split3(&ada)?;
        let h = self.in_ln.forward(x)?;
        let h = modulate(&h, &shift, &scale)?;
        let h = self.linear1.forward(&h)?;
        let h = silu(&h)?;
        let h = self.linear2.forward(&h)?;
        let gate = gate.broadcast_mul(&h)?;
        x.add(&gate)
    }
}

struct FinalLayer {
    norm: LayerNorm,
    linear: Linear,
    ada_ln: Linear,
}

impl FinalLayer {
    fn new(vb: VarBuilder, model_channels: usize, out_channels: usize) -> CandleResult<Self> {
        let norm = LayerNorm::from_vb_no_affine(model_channels, 1e-6, vb.pp("norm_final"))?;
        let linear = candle_nn::linear(model_channels, out_channels, vb.pp("linear"))?;
        let ada_ln = candle_nn::linear(model_channels, 2 * model_channels, vb.pp("adaLN_modulation").pp("1"))?;
        Ok(Self { norm, linear, ada_ln })
    }

    fn forward(&self, x: &Tensor, c: &Tensor) -> CandleResult<Tensor> {
        let ada = self.ada_ln.forward(&silu(c)?)?;
        let (shift, scale) = split2(&ada)?;
        let x = self.norm.forward(x)?;
        let x = modulate(&x, &shift, &scale)?;
        self.linear.forward(&x)
    }
}

struct ConvDownsample1d {
    conv: StreamingConv1d,
}

impl ConvDownsample1d {
    fn new(vb: VarBuilder, dimension: usize, stride: usize) -> CandleResult<Self> {
        let conv = StreamingConv1d::new(
            vb.pp("conv").pp("conv"),
            dimension,
            dimension,
            stride * 2,
            stride,
            1,
            PadMode::Replicate,
            false,
        )?;
        Ok(Self { conv })
    }

    fn forward(&self, x: &Tensor) -> CandleResult<Tensor> {
        self.conv.forward(x)
    }
}

struct MimiEncoder {
    encoder: SeanetEncoder,
    transformer: Transformer,
    downsample: Option<ConvDownsample1d>,
    sample_rate: usize,
    frame_rate: f64,
}

impl MimiEncoder {
    fn new(vb: VarBuilder, cfg: &MimiConfig) -> CandleResult<Self> {
        if cfg.transformer.d_model != cfg.seanet.dimension {
            return Err(candle::Error::Msg(format!(
                "mimi transformer d_model {} does not match seanet.dimension {}",
                cfg.transformer.d_model, cfg.seanet.dimension
            )));
        }
        if cfg.transformer.input_dimension != cfg.seanet.dimension {
            return Err(candle::Error::Msg(format!(
                "mimi transformer input_dimension {} does not match seanet.dimension {}",
                cfg.transformer.input_dimension, cfg.seanet.dimension
            )));
        }
        if cfg.transformer.output_dimensions.len() != 1
            || cfg.transformer.output_dimensions[0] != cfg.transformer.d_model
        {
            return Err(candle::Error::Msg(
                "mimi transformer output_dimensions must contain only d_model".to_string(),
            ));
        }
        let encoder = SeanetEncoder::new(vb.pp("encoder"), &cfg.seanet)?;
        let sample_rate = cfg.sample_rate;
        let frame_rate = cfg.frame_rate;
        let encoder_frame_rate = sample_rate as f64 / encoder.hop_length as f64;
        let stride_f = encoder_frame_rate / frame_rate;
        let downsample_stride = stride_f.round() as usize;
        if (stride_f - downsample_stride as f64).abs() > 1e-6 {
            return Err(candle::Error::Msg(
                "mimi downsample stride must be an integer".to_string(),
            ));
        }
        let downsample = if (encoder_frame_rate - frame_rate).abs() > f64::EPSILON {
            Some(ConvDownsample1d::new(
                vb.pp("downsample"),
                cfg.seanet.dimension,
                downsample_stride,
            )?)
        } else {
            None
        };
        let hidden_scale = match cfg.transformer.dim_feedforward.checked_div(cfg.transformer.d_model) {
            Some(scale) if scale > 0 && scale * cfg.transformer.d_model == cfg.transformer.dim_feedforward => scale,
            _ => {
                return Err(candle::Error::Msg(
                    "mimi dim_feedforward must be a multiple of d_model".to_string(),
                ))
            }
        };
        let transformer = Transformer::new(
            vb.pp("encoder_transformer").pp("transformer"),
            cfg.transformer.d_model,
            cfg.transformer.num_heads,
            cfg.transformer.num_layers,
            hidden_scale,
            Some(cfg.transformer.context),
            cfg.transformer.max_period,
        )?;
        Ok(Self {
            encoder,
            transformer,
            downsample,
            sample_rate,
            frame_rate,
        })
    }

    fn encode(&self, audio: &Tensor) -> CandleResult<Tensor> {
        let frame_size = (self.sample_rate as f64 / self.frame_rate).round() as usize;
        let audio = pad_for_conv1d(audio, frame_size, frame_size)?;
        let emb = self.encoder.forward(&audio)?;
        let emb = self.transformer.forward(&emb.transpose(1, 2)?)?;
        let mut emb = emb.transpose(1, 2)?;
        if let Some(downsample) = &self.downsample {
            emb = downsample.forward(&emb)?;
        }
        Ok(emb)
    }
}

struct MimiDecoder {
    quantizer: Conv1d,
    upsample: Option<StreamingConvTranspose1d>,
    transformer: Transformer,
    decoder: SeanetDecoder,
}

struct MimiDecoderState {
    upsample: Option<ConvTranspose1dState>,
    transformer: TransformerState,
    decoder: SeanetDecoderState,
}

impl MimiDecoder {
    fn new(vb: VarBuilder, cfg: &MimiConfig) -> CandleResult<Self> {
        if cfg.quantizer.output_dimension != cfg.transformer.d_model {
            return Err(candle::Error::Msg(format!(
                "mimi quantizer output_dimension {} does not match transformer.d_model {}",
                cfg.quantizer.output_dimension, cfg.transformer.d_model
            )));
        }
        if cfg.transformer.input_dimension != cfg.quantizer.output_dimension {
            return Err(candle::Error::Msg(format!(
                "mimi transformer input_dimension {} does not match quantizer output_dimension {}",
                cfg.transformer.input_dimension, cfg.quantizer.output_dimension
            )));
        }
        let quantizer = {
            let conv_cfg = Conv1dConfig {
                padding: 0,
                stride: 1,
                dilation: 1,
                groups: 1,
                cudnn_fwd_algo: None,
            };
            candle_nn::conv1d_no_bias(
                cfg.quantizer.dimension,
                cfg.quantizer.output_dimension,
                1,
                conv_cfg,
                vb.pp("quantizer").pp("output_proj"),
            )?
        };

        let encoder_hop_length: usize = cfg.seanet.ratios.iter().product();
        let encoder_frame_rate = cfg.sample_rate as f64 / encoder_hop_length as f64;
        let stride_f = encoder_frame_rate / cfg.frame_rate;
        let upsample_stride = stride_f.round() as usize;
        if (stride_f - upsample_stride as f64).abs() > 1e-6 {
            return Err(candle::Error::Msg(
                "mimi upsample stride must be an integer".to_string(),
            ));
        }
        let upsample = if (encoder_frame_rate - cfg.frame_rate).abs() > f64::EPSILON {
            Some(StreamingConvTranspose1d::new(
                vb.pp("upsample").pp("convtr").pp("convtr"),
                cfg.quantizer.output_dimension,
                cfg.quantizer.output_dimension,
                upsample_stride * 2,
                upsample_stride,
                cfg.quantizer.output_dimension,
                false,
            )?)
        } else {
            None
        };

        let hidden_scale = match cfg.transformer.dim_feedforward.checked_div(cfg.transformer.d_model) {
            Some(scale) if scale > 0 && scale * cfg.transformer.d_model == cfg.transformer.dim_feedforward => scale,
            _ => {
                return Err(candle::Error::Msg(
                    "mimi dim_feedforward must be a multiple of d_model".to_string(),
                ))
            }
        };
        let transformer = Transformer::new(
            vb.pp("decoder_transformer").pp("transformer"),
            cfg.transformer.d_model,
            cfg.transformer.num_heads,
            cfg.transformer.num_layers,
            hidden_scale,
            Some(cfg.transformer.context),
            cfg.transformer.max_period,
        )?;

        let decoder = SeanetDecoder::new(vb.pp("decoder"), &cfg.seanet)?;

        Ok(Self {
            quantizer,
            upsample,
            transformer,
            decoder,
        })
    }

    fn init_state(&self, batch_size: usize) -> CandleResult<MimiDecoderState> {
        Ok(MimiDecoderState {
            upsample: match &self.upsample {
                Some(upsample) => Some(upsample.init_state(batch_size)?),
                None => None,
            },
            transformer: self.transformer.init_state(),
            decoder: self.decoder.init_state(batch_size)?,
        })
    }

    fn decode(&self, latents: &Tensor, emb_mean: &Tensor, emb_std: &Tensor) -> CandleResult<Tensor> {
        let mut latents = latents.broadcast_mul(emb_std)?.broadcast_add(emb_mean)?;
        latents = latents.transpose(1, 2)?; // (B, C, T)
        let quantized = latents.apply(&self.quantizer)?;
        let upsampled = match &self.upsample {
            Some(upsample) => upsample.forward(&quantized)?,
            None => quantized,
        };
        let transformed = self.transformer.forward(&upsampled.transpose(1, 2)?)?;
        let transformed = transformed.transpose(1, 2)?;
        self.decoder.forward(&transformed)
    }

    fn decode_step(
        &self,
        latents: &Tensor,
        emb_mean: &Tensor,
        emb_std: &Tensor,
        state: &mut MimiDecoderState,
    ) -> CandleResult<Tensor> {
        let mut latents = latents.broadcast_mul(emb_std)?.broadcast_add(emb_mean)?;
        latents = latents.transpose(1, 2)?; // (B, C, T)
        let quantized = latents.apply(&self.quantizer)?;
        let upsampled = match (&self.upsample, &mut state.upsample) {
            (Some(upsample), Some(state)) => upsample.forward_streaming(&quantized, state)?,
            (None, None) => quantized,
            _ => {
                return Err(candle::Error::Msg(
                    "upsample state mismatch".to_string(),
                ))
            }
        };
        let transformed = self
            .transformer
            .forward_streaming(&upsampled.transpose(1, 2)?, &mut state.transformer)?;
        let transformed = transformed.transpose(1, 2)?;
        self.decoder.forward_streaming(&transformed, &mut state.decoder)
    }
}

struct Transformer {
    layers: Vec<TransformerLayer>,
}

struct AttentionState {
    k: Option<Tensor>,
    v: Option<Tensor>,
}

impl AttentionState {
    fn new() -> Self {
        Self { k: None, v: None }
    }

    fn past_len(&self) -> CandleResult<usize> {
        match &self.k {
            Some(k) => Ok(k.dim(2)?),
            None => Ok(0),
        }
    }

    fn append(&mut self, k: &Tensor, v: &Tensor) -> CandleResult<(Tensor, Tensor)> {
        let (k_all, v_all) = match (&self.k, &self.v) {
            (Some(k_cache), Some(v_cache)) => {
                let k_all = Tensor::cat(&[k_cache.clone(), k.clone()], 2)?;
                let v_all = Tensor::cat(&[v_cache.clone(), v.clone()], 2)?;
                (k_all, v_all)
            }
            _ => (k.clone(), v.clone()),
        };
        self.k = Some(k_all.clone());
        self.v = Some(v_all.clone());
        Ok((k_all, v_all))
    }
}

struct TransformerState {
    layers: Vec<AttentionState>,
}

impl TransformerState {
    fn new(num_layers: usize) -> Self {
        let mut layers = Vec::with_capacity(num_layers);
        for _ in 0..num_layers {
            layers.push(AttentionState::new());
        }
        Self { layers }
    }
}

impl Transformer {
    fn new(
        vb: VarBuilder,
        d_model: usize,
        num_heads: usize,
        num_layers: usize,
        hidden_scale: usize,
        context: Option<usize>,
        max_period: f64,
    ) -> CandleResult<Self> {
        let mut layers = Vec::with_capacity(num_layers);
        for i in 0..num_layers {
            layers.push(TransformerLayer::new(
                vb.pp("layers").pp(&i.to_string()),
                d_model,
                num_heads,
                hidden_scale,
                context,
                max_period,
            )?);
        }
        Ok(Self { layers })
    }

    fn forward(&self, x: &Tensor) -> CandleResult<Tensor> {
        let mut x = x.clone();
        for layer in self.layers.iter() {
            x = layer.forward(&x)?;
        }
        Ok(x)
    }

    fn init_state(&self) -> TransformerState {
        TransformerState::new(self.layers.len())
    }

    fn forward_streaming(&self, x: &Tensor, state: &mut TransformerState) -> CandleResult<Tensor> {
        let mut x = x.clone();
        for (layer, layer_state) in self.layers.iter().zip(state.layers.iter_mut()) {
            x = layer.forward_streaming(&x, layer_state)?;
        }
        Ok(x)
    }
}

struct TransformerLayer {
    self_attn: MultiheadAttention,
    norm1: LayerNorm,
    norm2: LayerNorm,
    linear1: Linear,
    linear2: Linear,
    layer_scale_1: Option<LayerScale>,
    layer_scale_2: Option<LayerScale>,
}

impl TransformerLayer {
    fn new(
        vb: VarBuilder,
        d_model: usize,
        num_heads: usize,
        hidden_scale: usize,
        context: Option<usize>,
        max_period: f64,
    ) -> CandleResult<Self> {
        let self_attn = MultiheadAttention::new(vb.pp("self_attn"), d_model, num_heads, context, max_period)?;
        let norm1 = LayerNorm::from_vb(vb.pp("norm1"), d_model, 1e-5, true)?;
        let norm2 = LayerNorm::from_vb(vb.pp("norm2"), d_model, 1e-5, true)?;
        let linear1 = candle_nn::linear_no_bias(d_model, d_model * hidden_scale, vb.pp("linear1"))?;
        let linear2 = candle_nn::linear_no_bias(d_model * hidden_scale, d_model, vb.pp("linear2"))?;

        let layer_scale_1 = if vb.contains_tensor("layer_scale_1.scale") {
            Some(LayerScale::new(vb.pp("layer_scale_1"), d_model)?)
        } else {
            None
        };
        let layer_scale_2 = if vb.contains_tensor("layer_scale_2.scale") {
            Some(LayerScale::new(vb.pp("layer_scale_2"), d_model)?)
        } else {
            None
        };

        Ok(Self {
            self_attn,
            norm1,
            norm2,
            linear1,
            linear2,
            layer_scale_1,
            layer_scale_2,
        })
    }

    fn forward(&self, x: &Tensor) -> CandleResult<Tensor> {
        let x_norm = self.norm1.forward(x)?;
        let mut update = self.self_attn.forward(&x_norm)?;
        if let Some(scale) = &self.layer_scale_1 {
            update = scale.forward(&update)?;
        }
        let x = x.add(&update)?;
        let x_norm = self.norm2.forward(&x)?;
        let mut ff = self.linear1.forward(&x_norm)?;
        ff = ff.gelu()?;
        ff = self.linear2.forward(&ff)?;
        if let Some(scale) = &self.layer_scale_2 {
            ff = scale.forward(&ff)?;
        }
        x.add(&ff)
    }

    fn forward_streaming(&self, x: &Tensor, state: &mut AttentionState) -> CandleResult<Tensor> {
        let x_norm = self.norm1.forward(x)?;
        let mut update = self.self_attn.forward_streaming(&x_norm, state)?;
        if let Some(scale) = &self.layer_scale_1 {
            update = scale.forward(&update)?;
        }
        let x = x.add(&update)?;
        let x_norm = self.norm2.forward(&x)?;
        let mut ff = self.linear1.forward(&x_norm)?;
        ff = ff.gelu()?;
        ff = self.linear2.forward(&ff)?;
        if let Some(scale) = &self.layer_scale_2 {
            ff = scale.forward(&ff)?;
        }
        x.add(&ff)
    }
}

struct MultiheadAttention {
    in_proj: Linear,
    out_proj: Linear,
    num_heads: usize,
    head_dim: usize,
    rope: RotaryEmbedding,
    context: Option<usize>,
}

impl MultiheadAttention {
    fn new(vb: VarBuilder, d_model: usize, num_heads: usize, context: Option<usize>, max_period: f64) -> CandleResult<Self> {
        let in_proj = candle_nn::linear_no_bias(d_model, 3 * d_model, vb.pp("in_proj"))?;
        let out_proj = candle_nn::linear_no_bias(d_model, d_model, vb.pp("out_proj"))?;
        let head_dim = d_model / num_heads;
        let rope = RotaryEmbedding::new(head_dim, max_period, in_proj.weight().device())?;
        Ok(Self {
            in_proj,
            out_proj,
            num_heads,
            head_dim,
            rope,
            context,
        })
    }

    fn forward(&self, x: &Tensor) -> CandleResult<Tensor> {
        let (b, t, _) = x.dims3()?;
        let projected = self.in_proj.forward(x)?;
        let packed = projected.reshape((b, t, 3, self.num_heads, self.head_dim))?;
        let parts = packed.chunk(3, 2)?;
        let mut q = parts[0].squeeze(2)?;
        let mut k = parts[1].squeeze(2)?;
        let v = parts[2].squeeze(2)?;

        let (rq, rk) = self.rope.apply(&q, &k)?;
        q = rq;
        k = rk;

        let q = q.transpose(1, 2)?.contiguous()?; // (b, h, t, d)
        let k = k.transpose(1, 2)?.contiguous()?;
        let v = v.transpose(1, 2)?.contiguous()?;

        let scale = (self.head_dim as f64).sqrt();
        let scale = Tensor::from_vec(vec![(1.0 / scale) as f32], (1,), x.device())?;
        let kt = k.transpose(2, 3)?.contiguous()?;
        let attn = q.matmul(&kt)?.broadcast_mul(&scale)?;

        let mask = causal_mask(t, self.context, x.device())?;
        let attn = attn.broadcast_add(&mask)?;
        let attn = candle_nn::ops::softmax_last_dim(&attn)?.contiguous()?;
        let ctx = attn.matmul(&v)?;
        let ctx = ctx.transpose(1, 2)?.reshape((b, t, self.num_heads * self.head_dim))?;
        self.out_proj.forward(&ctx)
    }

    fn forward_streaming(&self, x: &Tensor, state: &mut AttentionState) -> CandleResult<Tensor> {
        let (b, t, _) = x.dims3()?;
        let projected = self.in_proj.forward(x)?;
        let packed = projected.reshape((b, t, 3, self.num_heads, self.head_dim))?;
        let parts = packed.chunk(3, 2)?;
        let mut q = parts[0].squeeze(2)?;
        let mut k = parts[1].squeeze(2)?;
        let v = parts[2].squeeze(2)?;

        let past_len = state.past_len()?;
        let (rq, rk) = self.rope.apply_with_offset(&q, &k, past_len)?;
        q = rq;
        k = rk;

        let q = q.transpose(1, 2)?.contiguous()?; // (b, h, t, d)
        let k = k.transpose(1, 2)?.contiguous()?;
        let v = v.transpose(1, 2)?.contiguous()?;
        let (k_all, v_all) = state.append(&k, &v)?;
        let k_all = k_all.contiguous()?;
        let v_all = v_all.contiguous()?;

        let scale = (self.head_dim as f64).sqrt();
        let scale = Tensor::from_vec(vec![(1.0 / scale) as f32], (1,), x.device())?;
        let kt = k_all.transpose(2, 3)?.contiguous()?;
        let attn = q.matmul(&kt)?.broadcast_mul(&scale)?;

        let mask = causal_mask_with_offset(t, past_len, self.context, x.device())?;
        let attn = attn.broadcast_add(&mask)?;
        let attn = candle_nn::ops::softmax_last_dim(&attn)?.contiguous()?;
        let ctx = attn.matmul(&v_all)?;
        let ctx = ctx.transpose(1, 2)?.reshape((b, t, self.num_heads * self.head_dim))?;
        self.out_proj.forward(&ctx)
    }
}

struct RotaryEmbedding {
    freqs: Tensor,
}

impl RotaryEmbedding {
    fn new(head_dim: usize, max_period: f64, device: &Device) -> CandleResult<Self> {
        let half = head_dim / 2;
        let mut freqs = Vec::with_capacity(half);
        let scale = -((max_period.ln() as f32) * 2.0 / head_dim as f32);
        for i in 0..half {
            freqs.push((scale * i as f32).exp());
        }
        let freqs = Tensor::from_vec(freqs, (half,), device)?;
        Ok(Self { freqs })
    }

    fn apply(&self, q: &Tensor, k: &Tensor) -> CandleResult<(Tensor, Tensor)> {
        self.apply_with_offset(q, k, 0)
    }

    fn apply_with_offset(&self, q: &Tensor, k: &Tensor, offset: usize) -> CandleResult<(Tensor, Tensor)> {
        let (b, t, h, d) = q.dims4()?;
        let q = q.reshape((b, t, h, d / 2, 2))?;
        let k = k.reshape((b, t, h, d / 2, 2))?;
        let q_parts = q.chunk(2, D::Minus1)?;
        let k_parts = k.chunk(2, D::Minus1)?;
        let qr = &q_parts[0];
        let qi = &q_parts[1];
        let kr = &k_parts[0];
        let ki = &k_parts[1];

        let start = offset as f32;
        let end = (offset + t) as f32;
        let ts = Tensor::arange(start, end, q.device())?;
        let ts = ts.reshape((t, 1))?;
        let freqs = self.freqs.reshape((1, self.freqs.dims1()?))?;
        let angles = ts.broadcast_mul(&freqs)?;
        let cos = angles.cos()?;
        let sin = angles.sin()?;
        // Add trailing singleton dim for broadcasting against (b, t, h, d/2, 1).
        let cos = cos.reshape((1, t, 1, d / 2, 1))?;
        let sin = sin.reshape((1, t, 1, d / 2, 1))?;

        let qor = qr.broadcast_mul(&cos)?.broadcast_sub(&qi.broadcast_mul(&sin)?)?;
        let qoi = qr.broadcast_mul(&sin)?.broadcast_add(&qi.broadcast_mul(&cos)?)?;
        let kor = kr.broadcast_mul(&cos)?.broadcast_sub(&ki.broadcast_mul(&sin)?)?;
        let koi = kr.broadcast_mul(&sin)?.broadcast_add(&ki.broadcast_mul(&cos)?)?;

        let q = Tensor::stack(&[qor, qoi], D::Minus1)?.reshape((b, t, h, d))?;
        let k = Tensor::stack(&[kor, koi], D::Minus1)?.reshape((b, t, h, d))?;
        Ok((q, k))
    }
}

struct LayerScale {
    scale: Tensor,
}

impl LayerScale {
    fn new(vb: VarBuilder, channels: usize) -> CandleResult<Self> {
        let scale = vb.get((channels,), "scale")?;
        Ok(Self { scale })
    }

    fn forward(&self, x: &Tensor) -> CandleResult<Tensor> {
        x.broadcast_mul(&self.scale)
    }
}

struct LayerNorm {
    weight: Option<Tensor>,
    bias: Option<Tensor>,
    eps: f64,
}

impl LayerNorm {
    fn from_vb(vb: VarBuilder, dim: usize, eps: f64, affine: bool) -> CandleResult<Self> {
        let weight = if affine { Some(vb.get((dim,), "weight")?) } else { None };
        let bias = if affine { Some(vb.get((dim,), "bias")?) } else { None };
        Ok(Self { weight, bias, eps })
    }

    fn from_vb_no_affine(dim: usize, eps: f64, _vb: VarBuilder) -> CandleResult<Self> {
        Ok(Self {
            weight: None,
            bias: None,
            eps,
        })
    }

    fn forward(&self, x: &Tensor) -> CandleResult<Tensor> {
        let mean = x.mean_keepdim(D::Minus1)?;
        let var = x.broadcast_sub(&mean)?.sqr()?.mean_keepdim(D::Minus1)?;
        let x = x.broadcast_sub(&mean)?.broadcast_div(&(var + self.eps)?.sqrt()?)?;
        let x = match &self.weight {
            Some(weight) => x.broadcast_mul(weight)?,
            None => x,
        };
        let x = match &self.bias {
            Some(bias) => x.broadcast_add(bias)?,
            None => x,
        };
        Ok(x)
    }
}

struct RmsNorm {
    alpha: Tensor,
    eps: f64,
}

impl RmsNorm {
    fn from_vb(vb: VarBuilder, dim: usize, eps: f64) -> CandleResult<Self> {
        let alpha = vb.get((dim,), "alpha")?;
        Ok(Self { alpha, eps })
    }

    fn forward(&self, x: &Tensor) -> CandleResult<Tensor> {
        let mean = x.mean_keepdim(D::Minus1)?;
        let mean_sq = x.sqr()?.mean_keepdim(D::Minus1)?;
        let var = mean_sq.broadcast_sub(&mean.sqr()?)?;
        let denom = (var + self.eps)?.sqrt()?;
        x.broadcast_mul(&self.alpha)?.broadcast_div(&denom)
    }
}

struct StreamingConvTranspose1d {
    convtr: ConvTranspose1d,
    trim: usize,
}

struct ConvTranspose1dState {
    partial: Option<Tensor>,
}

impl StreamingConvTranspose1d {
    fn new(
        vb: VarBuilder,
        in_c: usize,
        out_c: usize,
        k_size: usize,
        stride: usize,
        groups: usize,
        bias: bool,
    ) -> CandleResult<Self> {
        let cfg = ConvTranspose1dConfig {
            padding: 0,
            output_padding: 0,
            stride,
            dilation: 1,
            groups,
        };
        let convtr = if bias {
            candle_nn::conv_transpose1d(in_c, out_c, k_size, cfg, vb)?
        } else {
            candle_nn::conv_transpose1d_no_bias(in_c, out_c, k_size, cfg, vb)?
        };
        let trim = k_size.saturating_sub(stride);
        Ok(Self { convtr, trim })
    }

    fn forward(&self, x: &Tensor) -> CandleResult<Tensor> {
        let y = x.apply(&self.convtr)?;
        if self.trim == 0 {
            return Ok(y);
        }
        let len = y.dim(D::Minus1)?;
        y.narrow(D::Minus1, 0, len - self.trim)
    }

    fn init_state(&self, batch_size: usize) -> CandleResult<ConvTranspose1dState> {
        if self.trim == 0 {
            return Ok(ConvTranspose1dState { partial: None });
        }
        let (_, out_per_group, _) = self.convtr.weight().dims3()?;
        let out_c = out_per_group * self.convtr.config().groups;
        let partial = Tensor::zeros(
            (batch_size, out_c, self.trim),
            self.convtr.weight().dtype(),
            self.convtr.weight().device(),
        )?;
        Ok(ConvTranspose1dState {
            partial: Some(partial),
        })
    }

    fn forward_streaming(
        &self,
        x: &Tensor,
        state: &mut ConvTranspose1dState,
    ) -> CandleResult<Tensor> {
        let mut y = x.apply(&self.convtr)?;
        if self.trim == 0 {
            return Ok(y);
        }
        let len = y.dim(D::Minus1)?;
        if let Some(partial) = &state.partial {
            let head = y.narrow(D::Minus1, 0, self.trim)?.add(partial)?;
            let tail = y.narrow(D::Minus1, self.trim, len - self.trim)?;
            y = Tensor::cat(&[head, tail], D::Minus1)?;
        }
        let mut next_partial = y.narrow(D::Minus1, len - self.trim, self.trim)?;
        if let Some(bias) = self.convtr.bias() {
            let bias = bias.reshape((1, bias.dims1()?, 1))?;
            next_partial = next_partial.broadcast_sub(&bias)?;
        }
        state.partial = Some(next_partial);
        y.narrow(D::Minus1, 0, len - self.trim)
    }
}

#[derive(Clone, Copy)]
enum PadMode {
    Constant,
    Replicate,
}

fn pad_mode_from_str(value: &str) -> CandleResult<PadMode> {
    match value {
        "constant" => Ok(PadMode::Constant),
        "replicate" => Ok(PadMode::Replicate),
        other => Err(candle::Error::Msg(format!(
            "unsupported pad_mode '{other}', expected constant or replicate"
        ))),
    }
}

struct StreamingConv1d {
    conv: Conv1d,
    pad_left: usize,
    pad_mode: PadMode,
}

struct Conv1dState {
    previous: Option<Tensor>,
    first: bool,
}

impl StreamingConv1d {
    fn new(
        vb: VarBuilder,
        in_c: usize,
        out_c: usize,
        k_size: usize,
        stride: usize,
        dilation: usize,
        pad_mode: PadMode,
        bias: bool,
    ) -> CandleResult<Self> {
        let cfg = Conv1dConfig {
            padding: 0,
            stride,
            dilation,
            groups: 1,
            cudnn_fwd_algo: None,
        };
        let conv = if bias {
            candle_nn::conv1d(in_c, out_c, k_size, cfg, vb)?
        } else {
            candle_nn::conv1d_no_bias(in_c, out_c, k_size, cfg, vb)?
        };
        let eff_k = (k_size - 1) * dilation + 1;
        let pad_left = eff_k.saturating_sub(stride);
        Ok(Self {
            conv,
            pad_left,
            pad_mode,
        })
    }

    fn forward(&self, x: &Tensor) -> CandleResult<Tensor> {
        let x = if self.pad_left > 0 {
            match self.pad_mode {
                PadMode::Constant => x.pad_with_zeros(D::Minus1, self.pad_left, 0)?,
                PadMode::Replicate => pad_replicate_left(x, self.pad_left)?,
            }
        } else {
            x.clone()
        };
        x.apply(&self.conv)
    }

    fn init_state(&self, batch_size: usize) -> CandleResult<Conv1dState> {
        if self.pad_left == 0 {
            return Ok(Conv1dState {
                previous: None,
                first: false,
            });
        }
        let (_, in_c, _) = self.conv.weight().dims3()?;
        let previous = Tensor::zeros(
            (batch_size, in_c, self.pad_left),
            self.conv.weight().dtype(),
            self.conv.weight().device(),
        )?;
        Ok(Conv1dState {
            previous: Some(previous),
            first: true,
        })
    }

    fn forward_streaming(&self, x: &Tensor, state: &mut Conv1dState) -> CandleResult<Tensor> {
        let (b, c, t) = x.dims3()?;
        let stride = self.conv.config().stride;
        if t % stride != 0 {
            return Err(candle::Error::Msg(format!(
                "streaming conv1d expects steps multiple of stride {stride}, got {t}"
            )));
        }
        let x_full = match &state.previous {
            Some(prev) if self.pad_left > 0 => {
                let prev_len = prev.dim(D::Minus1)?;
                if matches!(self.pad_mode, PadMode::Replicate) && state.first {
                    if t < prev_len {
                        return Err(candle::Error::Msg(format!(
                            "not enough samples to pad replicate: {t} < {prev_len}"
                        )));
                    }
                    let first = x.narrow(D::Minus1, 0, 1)?;
                    let pad_block = first.broadcast_as((b, c, prev_len))?;
                    Tensor::cat(&[pad_block, x.clone()], D::Minus1)?
                } else {
                    Tensor::cat(&[prev.clone(), x.clone()], D::Minus1)?
                }
            }
            _ => x.clone(),
        };
        let y = x_full.apply(&self.conv)?;
        if let Some(prev) = &state.previous {
            let prev_len = prev.dim(D::Minus1)?;
            if prev_len > 0 {
                let total_len = x_full.dim(D::Minus1)?;
                let next_prev = x_full.narrow(D::Minus1, total_len - prev_len, prev_len)?;
                state.previous = Some(next_prev);
            }
        }
        state.first = false;
        Ok(y)
    }
}

fn pad_replicate_left(x: &Tensor, pad: usize) -> CandleResult<Tensor> {
    if pad == 0 {
        return Ok(x.clone());
    }
    let (b, c, _) = x.dims3()?;
    let first = x.narrow(D::Minus1, 0, 1)?;
    let pad_block = first.broadcast_as((b, c, pad))?;
    Tensor::cat(&[pad_block, x.clone()], D::Minus1)
}

fn pad_for_conv1d(x: &Tensor, kernel_size: usize, stride: usize) -> CandleResult<Tensor> {
    let len = x.dim(D::Minus1)?;
    let extra = extra_padding_for_conv1d(len, kernel_size, stride, 0);
    if extra == 0 {
        return Ok(x.clone());
    }
    x.pad_with_zeros(D::Minus1, 0, extra)
}

fn extra_padding_for_conv1d(
    length: usize,
    kernel_size: usize,
    stride: usize,
    padding_total: usize,
) -> usize {
    let length = length as f64;
    let kernel_size = kernel_size as f64;
    let stride = stride as f64;
    let padding_total = padding_total as f64;
    let n_frames = (length - kernel_size + padding_total) / stride + 1.0;
    let ideal_length = (n_frames.ceil() - 1.0) * stride + (kernel_size - padding_total);
    let extra = ideal_length - length;
    if extra <= 0.0 {
        0
    } else {
        extra.round() as usize
    }
}

struct SeanetResnetBlock {
    conv1: StreamingConv1d,
    conv2: StreamingConv1d,
}

struct SeanetResnetState {
    conv1: Conv1dState,
    conv2: Conv1dState,
}

impl SeanetResnetBlock {
    fn new(
        vb: VarBuilder,
        dim: usize,
        residual_kernel_size: usize,
        dilation: usize,
        pad_mode: PadMode,
        compress: usize,
    ) -> CandleResult<Self> {
        if compress == 0 || dim % compress != 0 {
            return Err(candle::Error::Msg(
                "seanet compress must be a positive divisor of dimension".to_string(),
            ));
        }
        let hidden = dim / compress;
        let conv1 = StreamingConv1d::new(
            vb.pp("block").pp("1").pp("conv"),
            dim,
            hidden,
            residual_kernel_size,
            1,
            dilation,
            pad_mode,
            true,
        )?;
        let conv2 = StreamingConv1d::new(
            vb.pp("block").pp("3").pp("conv"),
            hidden,
            dim,
            1,
            1,
            1,
            pad_mode,
            true,
        )?;
        Ok(Self { conv1, conv2 })
    }

    fn forward(&self, x: &Tensor) -> CandleResult<Tensor> {
        let mut y = x.elu(1.0)?;
        y = self.conv1.forward(&y)?;
        y = y.elu(1.0)?;
        y = self.conv2.forward(&y)?;
        x.add(&y)
    }

    fn init_state(&self, batch_size: usize) -> CandleResult<SeanetResnetState> {
        Ok(SeanetResnetState {
            conv1: self.conv1.init_state(batch_size)?,
            conv2: self.conv2.init_state(batch_size)?,
        })
    }

    fn forward_streaming(&self, x: &Tensor, state: &mut SeanetResnetState) -> CandleResult<Tensor> {
        let mut y = x.elu(1.0)?;
        y = self.conv1.forward_streaming(&y, &mut state.conv1)?;
        y = y.elu(1.0)?;
        y = self.conv2.forward_streaming(&y, &mut state.conv2)?;
        x.add(&y)
    }
}

enum SeanetEncLayer {
    Conv1d(StreamingConv1d),
    Resnet(SeanetResnetBlock),
    Elu,
}

struct SeanetEncoder {
    layers: Vec<SeanetEncLayer>,
    hop_length: usize,
}

impl SeanetEncoder {
    fn new(vb: VarBuilder, cfg: &SeanetConfig) -> CandleResult<Self> {
        let channels = cfg.channels;
        let dimension = cfg.dimension;
        let n_filters = cfg.n_filters;
        let n_residual_layers = cfg.n_residual_layers;
        let ratios = cfg.ratios.clone();
        let kernel_size = cfg.kernel_size;
        let last_kernel_size = cfg.last_kernel_size;
        let residual_kernel_size = cfg.residual_kernel_size;
        let dilation_base = cfg.dilation_base;
        let pad_mode = pad_mode_from_str(&cfg.pad_mode)?;
        let compress = cfg.compress;

        let ratios_rev: Vec<usize> = ratios.iter().rev().copied().collect();
        let hop_length = ratios.iter().product();

        let mut layers = Vec::new();
        let mut idx = 0usize;
        let mut mult = 1usize;
        layers.push(SeanetEncLayer::Conv1d(StreamingConv1d::new(
            vb.pp("model").pp(&idx.to_string()).pp("conv"),
            channels,
            mult * n_filters,
            kernel_size,
            1,
            1,
            pad_mode,
            true,
        )?));
        idx += 1;

        for ratio in ratios_rev.iter() {
            for j in 0..n_residual_layers {
                let dilation = dilation_base.pow(j as u32);
                layers.push(SeanetEncLayer::Resnet(SeanetResnetBlock::new(
                    vb.pp("model").pp(&idx.to_string()),
                    mult * n_filters,
                    residual_kernel_size,
                    dilation,
                    pad_mode,
                    compress,
                )?));
                idx += 1;
            }
            layers.push(SeanetEncLayer::Elu);
            idx += 1;
            layers.push(SeanetEncLayer::Conv1d(StreamingConv1d::new(
                vb.pp("model").pp(&idx.to_string()).pp("conv"),
                mult * n_filters,
                mult * n_filters * 2,
                ratio * 2,
                *ratio,
                1,
                pad_mode,
                true,
            )?));
            mult *= 2;
            idx += 1;
        }

        layers.push(SeanetEncLayer::Elu);
        idx += 1;
        layers.push(SeanetEncLayer::Conv1d(StreamingConv1d::new(
            vb.pp("model").pp(&idx.to_string()).pp("conv"),
            mult * n_filters,
            dimension,
            last_kernel_size,
            1,
            1,
            pad_mode,
            true,
        )?));

        Ok(Self { layers, hop_length })
    }

    fn forward(&self, x: &Tensor) -> CandleResult<Tensor> {
        let mut x = x.clone();
        for layer in self.layers.iter() {
            x = match layer {
                SeanetEncLayer::Conv1d(layer) => layer.forward(&x)?,
                SeanetEncLayer::Resnet(layer) => layer.forward(&x)?,
                SeanetEncLayer::Elu => x.elu(1.0)?,
            };
        }
        Ok(x)
    }
}

enum SeanetLayer {
    Conv1d(StreamingConv1d),
    ConvTr(StreamingConvTranspose1d),
    Resnet(SeanetResnetBlock),
    Elu,
}

enum SeanetLayerState {
    Conv1d(Conv1dState),
    ConvTr(ConvTranspose1dState),
    Resnet(SeanetResnetState),
    Elu,
}

struct SeanetDecoderState {
    layers: Vec<SeanetLayerState>,
}

struct SeanetDecoder {
    layers: Vec<SeanetLayer>,
}

impl SeanetDecoder {
    fn new(vb: VarBuilder, cfg: &SeanetConfig) -> CandleResult<Self> {
        let dimension = cfg.dimension;
        let channels = cfg.channels;
        let n_filters = cfg.n_filters;
        let ratios = cfg.ratios.clone();
        let kernel_size = cfg.kernel_size;
        let last_kernel_size = cfg.last_kernel_size;
        let residual_kernel_size = cfg.residual_kernel_size;
        let dilation_base = cfg.dilation_base;
        let pad_mode = pad_mode_from_str(&cfg.pad_mode)?;
        let compress = cfg.compress;
        let mut layers = Vec::new();

        let mult = 1 << ratios.len();
        layers.push(SeanetLayer::Conv1d(StreamingConv1d::new(
            vb.pp("model").pp("0").pp("conv"),
            dimension,
            mult * n_filters,
            kernel_size,
            1,
            1,
            pad_mode,
            true,
        )?));

        let mut mult = mult;
        let mut idx = 1usize;
        for ratio in ratios.iter() {
            layers.push(SeanetLayer::Elu);
            idx += 1;
            layers.push(SeanetLayer::ConvTr(StreamingConvTranspose1d::new(
                vb.pp("model").pp(&idx.to_string()).pp("convtr"),
                mult * n_filters,
                mult * n_filters / 2,
                ratio * 2,
                *ratio,
                1,
                true,
            )?));
            idx += 1;
            for j in 0..cfg.n_residual_layers {
                let dilation = dilation_base.pow(j as u32);
                layers.push(SeanetLayer::Resnet(SeanetResnetBlock::new(
                    vb.pp("model").pp(&idx.to_string()),
                    mult * n_filters / 2,
                    residual_kernel_size,
                    dilation,
                    pad_mode,
                    compress,
                )?));
                idx += 1;
            }
            mult /= 2;
        }

        layers.push(SeanetLayer::Elu);
        idx += 1;
        layers.push(SeanetLayer::Conv1d(StreamingConv1d::new(
            vb.pp("model").pp(&idx.to_string()).pp("conv"),
            n_filters,
            channels,
            last_kernel_size,
            1,
            1,
            pad_mode,
            true,
        )?));

        Ok(Self { layers })
    }

    fn forward(&self, x: &Tensor) -> CandleResult<Tensor> {
        let mut x = x.clone();
        for layer in self.layers.iter() {
            x = match layer {
                SeanetLayer::Conv1d(layer) => layer.forward(&x)?,
                SeanetLayer::ConvTr(layer) => layer.forward(&x)?,
                SeanetLayer::Resnet(layer) => layer.forward(&x)?,
                SeanetLayer::Elu => x.elu(1.0)?,
            };
        }
        Ok(x)
    }

    fn init_state(&self, batch_size: usize) -> CandleResult<SeanetDecoderState> {
        let mut layers = Vec::with_capacity(self.layers.len());
        for layer in self.layers.iter() {
            let state = match layer {
                SeanetLayer::Conv1d(layer) => SeanetLayerState::Conv1d(layer.init_state(batch_size)?),
                SeanetLayer::ConvTr(layer) => SeanetLayerState::ConvTr(layer.init_state(batch_size)?),
                SeanetLayer::Resnet(layer) => SeanetLayerState::Resnet(layer.init_state(batch_size)?),
                SeanetLayer::Elu => SeanetLayerState::Elu,
            };
            layers.push(state);
        }
        Ok(SeanetDecoderState { layers })
    }

    fn forward_streaming(
        &self,
        x: &Tensor,
        state: &mut SeanetDecoderState,
    ) -> CandleResult<Tensor> {
        let mut x = x.clone();
        for (layer, layer_state) in self.layers.iter().zip(state.layers.iter_mut()) {
            x = match (layer, layer_state) {
                (SeanetLayer::Conv1d(layer), SeanetLayerState::Conv1d(state)) => {
                    layer.forward_streaming(&x, state)?
                }
                (SeanetLayer::ConvTr(layer), SeanetLayerState::ConvTr(state)) => {
                    layer.forward_streaming(&x, state)?
                }
                (SeanetLayer::Resnet(layer), SeanetLayerState::Resnet(state)) => {
                    layer.forward_streaming(&x, state)?
                }
                (SeanetLayer::Elu, SeanetLayerState::Elu) => x.elu(1.0)?,
                _ => {
                    return Err(candle::Error::Msg(
                        "seanet decoder state mismatch".to_string(),
                    ))
                }
            };
        }
        Ok(x)
    }
}

fn modulate(x: &Tensor, shift: &Tensor, scale: &Tensor) -> CandleResult<Tensor> {
    let one = Tensor::from_vec(vec![1f32], (1,), x.device())?;
    let scale = scale.broadcast_add(&one)?;
    x.broadcast_mul(&scale)?.broadcast_add(shift)
}

fn silu(x: &Tensor) -> CandleResult<Tensor> {
    candle_nn::ops::sigmoid(x)?.broadcast_mul(x)
}

fn split2(x: &Tensor) -> CandleResult<(Tensor, Tensor)> {
    let parts = x.chunk(2, D::Minus1)?;
    Ok((parts[0].clone(), parts[1].clone()))
}

fn split3(x: &Tensor) -> CandleResult<(Tensor, Tensor, Tensor)> {
    let parts = x.chunk(3, D::Minus1)?;
    Ok((parts[0].clone(), parts[1].clone(), parts[2].clone()))
}

fn sample_noise(
    device: &Device,
    shape: (usize, usize),
    std: f64,
    clamp: Option<f64>,
) -> CandleResult<Tensor> {
    match clamp {
        None => Tensor::randn(0f32, std as f32, shape, device),
        Some(limit) => {
            if limit <= 0.0 || std <= 0.0 {
                return Tensor::zeros(shape, DType::F32, device);
            }
            let count = shape.0 * shape.1;
            let mut rng = thread_rng();
            let normal = Normal::new(0.0f64, std)
                .map_err(|e| candle::Error::Msg(format!("invalid normal params: {e}")))?;
            let mut data = Vec::with_capacity(count);
            let lo = -limit;
            let hi = limit;
            while data.len() < count {
                let v = normal.sample(&mut rng);
                if v >= lo && v <= hi {
                    data.push(v as f32);
                }
            }
            Tensor::from_vec(data, shape, device)
        }
    }
}

fn lsd_decode(flow_net: &SimpleMlpAdaLn, c: &Tensor, x0: &Tensor, steps: usize) -> CandleResult<Tensor> {
    let mut current = x0.clone();
    for i in 0..steps {
        let s = i as f64 / steps as f64;
        let t = (i + 1) as f64 / steps as f64;
        let s = Tensor::from_vec(vec![s as f32], (1, 1), x0.device())?;
        let t = Tensor::from_vec(vec![t as f32], (1, 1), x0.device())?;
        let flow = flow_net.forward(c, &s, &t, &current)?;
        let step = Tensor::from_vec(vec![(1.0 / steps as f32)], (1,), x0.device())?;
        current = current.add(&flow.broadcast_mul(&step)?)?;
    }
    Ok(current)
}

fn causal_mask(t: usize, context: Option<usize>, device: &Device) -> CandleResult<Tensor> {
    let mut data = Vec::with_capacity(t * t);
    for i in 0..t {
        for j in 0..t {
            let allowed = if j > i {
                false
            } else if let Some(ctx) = context {
                i - j < ctx
            } else {
                true
            };
            if allowed {
                data.push(0f32);
            } else {
                data.push(-1e9f32);
            }
        }
    }
    let mask = Tensor::from_vec(data, (t, t), device)?;
    mask.reshape((1, 1, t, t))
}

fn causal_mask_with_offset(
    q_len: usize,
    past_len: usize,
    context: Option<usize>,
    device: &Device,
) -> CandleResult<Tensor> {
    let total_len = past_len + q_len;
    let mut data = Vec::with_capacity(q_len * total_len);
    for i in 0..q_len {
        let q_pos = past_len + i;
        for j in 0..total_len {
            let allowed = if j > q_pos {
                false
            } else if let Some(ctx) = context {
                q_pos - j < ctx
            } else {
                true
            };
            if allowed {
                data.push(0f32);
            } else {
                data.push(-1e9f32);
            }
        }
    }
    let mask = Tensor::from_vec(data, (q_len, total_len), device)?;
    mask.reshape((1, 1, q_len, total_len))
}
