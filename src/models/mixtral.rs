use super::params::Params;
use crate::{device, load_statesensors};
use anyhow::{Error as E, Result};
use candle_core::{DType, Device};
use candle_nn::VarBuilder;
use candle_transformers::models::mixtral::{Config, Model};
use hf_hub::{api::sync::Api, Repo, RepoType};
use std::path::PathBuf;
use tokenizers::Tokenizer;

#[derive(Clone)]
pub struct Mixtral {
    pub model: Model,
    pub tokenizer: Tokenizer,
    pub device: Device,
    pub params: Params,
}

impl Mixtral {
    pub fn load_from_hf(
        hugging_face_model_id: &str,
        revision: &str,
        params: Params,
    ) -> Result<Self> {
        let api = Api::new()?;
        let repo = api.repo(Repo::with_revision(
            hugging_face_model_id.to_string(),
            RepoType::Model,
            revision.to_string(),
        ));
        let tokenizer_filename = repo.get("tokenizer.json")?;
        let filenames = load_statesensors::from_hf_hub(&repo, "model.safetensors.index.json")?;
        let tokenizer = Tokenizer::from_file(tokenizer_filename).map_err(E::msg)?;
        let config = Config::v0_1_8x7b(params.use_flash_attn);

        let device = device::choose(params.run_on_cpu)?;
        let dtype = if device.is_cuda() {
            DType::BF16
        } else {
            DType::F32
        };

        let vb = unsafe { VarBuilder::from_mmaped_safetensors(&filenames, dtype, &device)? };
        let model = Model::new(&config, vb)?;

        Ok(Self {
            model,
            tokenizer,
            device,
            params,
        })
    }

    pub fn load_from_file(model_path: &str, params: Params) -> anyhow::Result<Self> {
        let tokenizer_file = PathBuf::from(&format!("{}/tokenizer.json", model_path));
        let tokenizer = Tokenizer::from_file(tokenizer_file).map_err(E::msg)?;

        let config = Config::v0_1_8x7b(params.use_flash_attn);
        let device = device::choose(params.run_on_cpu)?;
        let dtype = if device.is_cuda() {
            DType::BF16
        } else {
            DType::F32
        };
        let weight_files =
            load_statesensors::from_file(model_path, "model.safetensors.index.json")?;
        let vb = unsafe { VarBuilder::from_mmaped_safetensors(&weight_files, dtype, &device)? };
        let model = Model::new(&config, vb)?;

        Ok(Self {
            model,
            tokenizer,
            device,
            params,
        })
    }
}
