use std::path::PathBuf;

use anyhow::{Error, Result};
use serde::Deserialize;

#[derive(Deserialize, Clone)]
pub struct Function {
    pub name: String,
    pub description: String,
    pub parameters: serde_json::Value,
}

pub struct Functions(Vec<Function>);

impl Functions {
    pub fn load_from_file(file: &str) -> Result<Self> {
        let json_file = PathBuf::from(file);
        let json_file = std::fs::File::open(json_file)?;

        Ok(Self(
            serde_json::from_reader(&json_file).map_err(Error::msg)?,
        ))
    }
}

impl Functions {
    pub fn functions(&self) -> Vec<Function> {
        return self.0.clone();
    }
}
