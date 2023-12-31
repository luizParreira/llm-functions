use super::params::Params;
use anyhow::{Error as E, Result};
use candle_core::Device;
use candle_transformers::models::quantized_mistral::{Config, Model};
use hf_hub::{api::sync::Api, Repo, RepoType};
use tokenizers::Tokenizer;

#[derive(Clone)]
pub struct QuantizedMistral {
    pub model: Model,
    pub tokenizer: Tokenizer,
    pub device: Device,
    pub params: Params,
}

impl QuantizedMistral {
    pub fn load_from_hf(params: Params) -> Result<Self> {
        let api = Api::new()?;
        let repo = api.repo(Repo::with_revision(
            "lmz/candle-mistral".to_string(),
            RepoType::Model,
            "main".to_string(),
        ));

        let tokenizer_filename = repo.get("tokenizer.json")?;
        let filenames = vec![repo.get("model-q4k.gguf")?];
        let tokenizer = Tokenizer::from_file(tokenizer_filename).map_err(E::msg)?;
        let config = Config::config_7b_v0_1(params.use_flash_attn);
        let filename = &filenames[0];
        let vb = candle_transformers::quantized_var_builder::VarBuilder::from_gguf(filename)?;
        let model = Model::new(&config, vb)?;
        let device = Device::Cpu;

        Ok(Self {
            model,
            tokenizer,
            device,
            params,
        })
    }
}
