use anyhow::{Error as E, Result};

use crate::models;
use crate::token_output_stream::TokenOutputStream;
use candle_core as candle;
use candle_core::{DType, Device, Tensor};
use candle_transformers::generation::LogitsProcessor;
use tokenizers::Tokenizer;

#[derive(Clone)]
pub enum Model {
    QuantizedMistral(models::quantized_mistral::QuantizedMistral),
    Mixtral(models::mixtral::Mixtral),
}

impl Model {
    pub fn forward(&mut self, input_ids: &Tensor, seqlen_offset: usize) -> candle::Result<Tensor> {
        match self {
            Model::QuantizedMistral(model) => model.model.forward(input_ids, seqlen_offset),
            Model::Mixtral(model) => model.model.forward(input_ids, seqlen_offset),
        }
    }
}

struct TextGeneration {
    model: Model,
    device: Device,
    tokenizer: TokenOutputStream,
    logits_processor: LogitsProcessor,
    repeat_penalty: f32,
    repeat_last_n: usize,
}

impl TextGeneration {
    #[allow(clippy::too_many_arguments)]
    fn new(
        model: Model,
        tokenizer: Tokenizer,
        seed: u64,
        temp: Option<f64>,
        top_p: Option<f64>,
        repeat_penalty: f32,
        repeat_last_n: usize,
        device: &Device,
    ) -> Self {
        let logits_processor = LogitsProcessor::new(seed, temp, top_p);
        Self {
            model,
            tokenizer: TokenOutputStream::new(tokenizer),
            logits_processor,
            repeat_penalty,
            repeat_last_n,
            device: device.clone(),
        }
    }

    fn next_token(&mut self, prompt: &str, max_len: usize) -> Result<Option<String>> {
        let mut tokens = self
            .tokenizer
            .tokenizer()
            .encode(prompt, true)
            .map_err(E::msg)?
            .get_ids()
            .to_vec();
        let eos_token = match self.tokenizer.get_token("</s>") {
            Some(token) => token,
            None => anyhow::bail!("cannot find the </s> token"),
        };

        let context_size = if self.tokenizer.generated_tokens().len() > 0 {
            1
        } else {
            tokens.len()
        };
        let start_pos = tokens.len().saturating_sub(context_size);
        let ctxt = &tokens[start_pos..];
        let input = Tensor::new(ctxt, &self.device)?.unsqueeze(0)?;
        let logits = self.model.forward(&input, start_pos)?;
        let logits = logits.squeeze(0)?.squeeze(0)?.to_dtype(DType::F32)?;
        let logits = if self.repeat_penalty == 1. {
            logits
        } else {
            let start_at = tokens.len().saturating_sub(self.repeat_last_n);
            candle_transformers::utils::apply_repeat_penalty(
                &logits,
                self.repeat_penalty,
                &tokens[start_at..],
            )?
        };

        let next_token = self.logits_processor.sample(&logits)?;
        tokens.push(next_token);
        if next_token == eos_token {
            return Ok(None);
        }

        Ok(self.tokenizer.next_token(next_token)?)
    }

    fn run(&mut self, prompt: &str, sample_len: usize) -> Result<String> {
        use std::io::Write;
        self.tokenizer.clear();
        let mut tokens = self
            .tokenizer
            .tokenizer()
            .encode(prompt, true)
            .map_err(E::msg)?
            .get_ids()
            .to_vec();

        let mut generated: String = String::new();

        let eos_token = match self.tokenizer.get_token("</s>") {
            Some(token) => token,
            None => anyhow::bail!("cannot find the </s> token"),
        };
        println!("decoding sample_len");
        for index in 0..sample_len {
            let context_size = if index > 0 { 1 } else { tokens.len() };
            let start_pos = tokens.len().saturating_sub(context_size);
            let ctxt = &tokens[start_pos..];
            let input = Tensor::new(ctxt, &self.device)?.unsqueeze(0)?;
            let logits = self.model.forward(&input, start_pos)?;
            let logits = logits.squeeze(0)?.squeeze(0)?.to_dtype(DType::F32)?;
            let logits = if self.repeat_penalty == 1. {
                logits
            } else {
                let start_at = tokens.len().saturating_sub(self.repeat_last_n);
                candle_transformers::utils::apply_repeat_penalty(
                    &logits,
                    self.repeat_penalty,
                    &tokens[start_at..],
                )?
            };

            let next_token = self.logits_processor.sample(&logits)?;
            tokens.push(next_token);
            if next_token == eos_token {
                break;
            }
            if let Some(t) = self.tokenizer.next_token(next_token)? {
                generated += &t;
                print!("{t}");
            }
        }
        let mut r = String::new();
        println!("decoding rest");
        if let Some(rest) = self.tokenizer.decode_rest().map_err(E::msg)? {
            r += &rest;
            generated += &rest;
        }
        println!("{r}");

        println!("");
        Ok(generated)
    }
}

impl Model {
    pub fn run(&self, prompt: &str) -> Result<String> {
        use tracing_chrome::ChromeLayerBuilder;
        use tracing_subscriber::prelude::*;

        let (tokenizer, params, device) = match self.clone() {
            Model::QuantizedMistral(m) => (m.tokenizer, m.params, m.device),
            Model::Mixtral(m) => (m.tokenizer, m.params, m.device),
        };

        let (chrome_layer, _guard) = ChromeLayerBuilder::new().build();
        tracing_subscriber::registry().with(chrome_layer).init();
        println!(
            "avx: {}, neon: {}, simd128: {}, f16c: {}",
            candle::utils::with_avx(),
            candle::utils::with_neon(),
            candle::utils::with_simd128(),
            candle::utils::with_f16c()
        );
        println!(
            "temp: {:.2} repeat-penalty: {:.2} repeat-last-n: {}",
            params.temperature.unwrap_or(0.),
            params.repeat_penalty,
            params.repeat_last_n
        );

        let mut pipeline = TextGeneration::new(
            self.clone(),
            tokenizer,
            params.seed,
            params.temperature,
            params.top_p,
            params.repeat_penalty,
            params.repeat_last_n,
            &device,
        );

        pipeline.run(prompt, params.sample_len)
    }
}
