use candle::Result;

#[cfg(feature = "symphonia")]
pub fn pcm_decode<P: AsRef<std::path::Path>>(path: P) -> Result<(Vec<f32>, u32)> {
    use symphonia::core::audio::{AudioBufferRef, Signal};
    use symphonia::core::codecs::{DecoderOptions, CODEC_TYPE_NULL};
    use symphonia::core::conv::FromSample;

    fn conv<T>(
        samples: &mut Vec<f32>,
        data: std::borrow::Cow<symphonia::core::audio::AudioBuffer<T>>,
    ) where
        T: symphonia::core::sample::Sample,
        f32: symphonia::core::conv::FromSample<T>,
    {
        samples.extend(data.chan(0).iter().map(|v| f32::from_sample(*v)))
    }

    let src = std::fs::File::open(path).map_err(candle::Error::wrap)?;
    let mss = symphonia::core::io::MediaSourceStream::new(Box::new(src), Default::default());
    let hint = symphonia::core::probe::Hint::new();

    let meta_opts: symphonia::core::meta::MetadataOptions = Default::default();
    let fmt_opts: symphonia::core::formats::FormatOptions = Default::default();

    let probed = symphonia::default::get_probe()
        .format(&hint, mss, &fmt_opts, &meta_opts)
        .map_err(candle::Error::wrap)?;
    let mut format = probed.format;

    let track = format
        .tracks()
        .iter()
        .find(|t| t.codec_params.codec != CODEC_TYPE_NULL)
        .ok_or_else(|| candle::Error::Msg("no supported audio tracks".to_string()))?;

    let dec_opts: DecoderOptions = Default::default();
    let mut decoder = symphonia::default::get_codecs()
        .make(&track.codec_params, &dec_opts)
        .map_err(|_| candle::Error::Msg("unsupported codec".to_string()))?;
    let track_id = track.id;
    let sample_rate = track.codec_params.sample_rate.unwrap_or(0);
    let mut pcm_data = Vec::new();

    while let Ok(packet) = format.next_packet() {
        while !format.metadata().is_latest() {
            format.metadata().pop();
        }

        if packet.track_id() != track_id {
            continue;
        }
        match decoder.decode(&packet).map_err(candle::Error::wrap)? {
            AudioBufferRef::F32(buf) => pcm_data.extend(buf.chan(0)),
            AudioBufferRef::U8(data) => conv(&mut pcm_data, data),
            AudioBufferRef::U16(data) => conv(&mut pcm_data, data),
            AudioBufferRef::U24(data) => conv(&mut pcm_data, data),
            AudioBufferRef::U32(data) => conv(&mut pcm_data, data),
            AudioBufferRef::S8(data) => conv(&mut pcm_data, data),
            AudioBufferRef::S16(data) => conv(&mut pcm_data, data),
            AudioBufferRef::S24(data) => conv(&mut pcm_data, data),
            AudioBufferRef::S32(data) => conv(&mut pcm_data, data),
            AudioBufferRef::F64(data) => conv(&mut pcm_data, data),
        }
    }
    Ok((pcm_data, sample_rate))
}

#[cfg(not(feature = "symphonia"))]
pub fn pcm_decode<P: AsRef<std::path::Path>>(_path: P) -> Result<(Vec<f32>, u32)> {
    candle::bail!("audio input requires the symphonia feature, try --features mimi")
}

#[cfg(feature = "rubato")]
pub fn resample(pcm_in: &[f32], sr_in: u32, sr_out: u32) -> Result<Vec<f32>> {
    use rubato::Resampler;

    let mut pcm_out =
        Vec::with_capacity((pcm_in.len() as f64 * sr_out as f64 / sr_in as f64) as usize + 1024);

    let mut resampler = rubato::FftFixedInOut::<f32>::new(sr_in as usize, sr_out as usize, 1024, 1)
        .map_err(candle::Error::wrap)?;
    let mut output_buffer = resampler.output_buffer_allocate(true);
    let mut pos_in = 0;
    while pos_in + resampler.input_frames_next() < pcm_in.len() {
        let (in_len, out_len) = resampler
            .process_into_buffer(&[&pcm_in[pos_in..]], &mut output_buffer, None)
            .map_err(candle::Error::wrap)?;
        pos_in += in_len;
        pcm_out.extend_from_slice(&output_buffer[0][..out_len]);
    }

    if pos_in < pcm_in.len() {
        let (_in_len, out_len) = resampler
            .process_partial_into_buffer(Some(&[&pcm_in[pos_in..]]), &mut output_buffer, None)
            .map_err(candle::Error::wrap)?;
        pcm_out.extend_from_slice(&output_buffer[0][..out_len]);
    }

    Ok(pcm_out)
}

#[cfg(not(feature = "rubato"))]
pub fn resample(_pcm_in: &[f32], _sr_in: u32, _sr_out: u32) -> Result<Vec<f32>> {
    candle::bail!("audio resampling requires the rubato feature, try --features mimi")
}
