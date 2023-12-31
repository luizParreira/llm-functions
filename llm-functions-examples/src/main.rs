use llm_functions::{
    functions::Functions,
    generator, model,
    models::{params::Params, quantized_mistral},
    prompter::CompletionModelPrompter,
};

fn main() {
    // let weight_files: Vec<String> = (0..=7)
    //     .map(|n| format!("/Users/luizparreira/work/llm-models/mixtral/Mixtral-8x7B-Instruct-v0.1/consolidated.{n:0>2}.pt"))
    //     .collect();

    // let f = vec!["/Users/luizparreira/work/llm-models/mixtral/Mixtral-8x7B-Instruct-v0.1/model.safetensors.index.json".to_string()];

    // let tokenizer_file =
    //     "/Users/luizparreira/work/llm-models/mixtral/Mixtral-8x7B-Instruct-v0.1/tokenizer.json";
    // let generator = generator::HuggingFaceGenerator::new("mistralai/Mixtral-8x7B-v0.1");
    let qm = quantized_mistral::QuantizedMistral::load_from_hf(Params {
        temperature: Some(0.0),
        ..Default::default()
    });

    match qm {
        Ok(m) => {
            let m = model::Model::QuantizedMistral(m);
            let functions =
                Functions::load_from_file("/Users/luizparreira/work/llm-functions/schema.json")
                    .unwrap();
            let prompter = CompletionModelPrompter::new();
            let generator = generator::Generator::new(m, functions, prompter);
            let function = generator.choose_function("Comprar 10 reais de Bitcoin");
            println!("{function:?}")
        }
        Err(err) => println!("Error: {err:?}"),
    }

    // let generator = generator::LocalGenerator::new(
    //     "/Users/luizparreira/.cache/huggingface/hub/models--mistralai--Mixtral-8x7B-v0.1/snapshots/58301445dc1378584211722b7ebf8743ec4e192b",
    // );

    // let res = generator.run("Quantos planetas tem no sistema solar?");

    // println!("{res:?}");
}
