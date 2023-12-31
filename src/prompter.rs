use candle_nn::{func, Func};

use crate::functions::{Function, Functions};
use anyhow::{Error as E, Result};

pub struct CompletionModelPrompter;

impl CompletionModelPrompter {
    pub fn new() -> Self {
        Self {}
    }

    fn head() -> String {
        "\n\nAvailable functions:\n".to_string()
    }

    fn call_header() -> String {
        "\n\nFunction call: ".to_string()
    }

    fn prompt_for_function(function: Function) -> Result<String> {
        // header = (
        //     f"{function['name']} - {function['description']}"
        //     if "description" in function
        //     else function["name"]
        // )
        // schema = json.dumps(function["parameters"]["properties"], indent=4)
        // packed_schema = f"```jsonschema\n{schema}\n```"
        // return f"{header}\n{packed_schema}"

        let header = format!("{} - {}", function.name, function.description);
        let props_schema = serde_json::to_string_pretty(&function.parameters["properties"])?;
        let packed_schema = format!("```jsonschema\n{props_schema}\n```");

        return Ok(format!("{header}\n{packed_schema}"));
    }

    fn prompt_for_functions(functions: &Functions) -> Result<String> {
        Ok(functions
            .functions()
            .into_iter()
            .map(|f| Self::prompt_for_function(f))
            .collect::<Result<Vec<String>>>()?
            .join("\n\n"))
    }

    pub fn prompt(&self, prompt: &str, functions: &Functions) -> Result<String> {
        let functions = Self::prompt_for_functions(functions)?;
        let call_header = Self::call_header();
        let head = Self::head();

        Ok(format!("{prompt}{head}{functions}{call_header}"))
    }
}
