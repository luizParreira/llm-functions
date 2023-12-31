use std::collections::HashMap;
use std::fmt::format;
use std::path::PathBuf;

use anyhow::{Error as E, Result};

use crate::functions::{self, Functions};
use crate::model::Model;
use crate::prompter::CompletionModelPrompter;
use crate::token_output_stream::TokenOutputStream;
use crate::{device, load_statesensors};
use candle_core as candle;
use candle_core::{DType, Device, Error, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::generation::LogitsProcessor;
use hf_hub::{api::sync::Api, Repo, RepoType};
use tokenizers::Tokenizer;

pub struct Generator {
    pub model: Model,
    pub functions: Functions,
    pub prompter: CompletionModelPrompter,
}

impl Generator {
    pub fn new(model: Model, functions: Functions, prompter: CompletionModelPrompter) -> Self {
        Self {
            model,
            functions,
            prompter,
        }
    }

    pub fn choose_function(&self, prompt: &str) -> Result<String> {
        let prompt = self.prompter.prompt(prompt, &self.functions)?;
        println!("{prompt}");
        self.model.run(&prompt)
    }
}
