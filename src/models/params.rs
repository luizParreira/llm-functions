#[derive(Clone)]
pub struct Params {
    pub seed: u64,
    pub sample_len: usize,
    pub repeat_penalty: f32,
    pub repeat_last_n: usize,
    pub use_flash_attn: bool,
    pub run_on_cpu: bool,
    pub top_p: Option<f64>,
    pub temperature: Option<f64>,
}

impl Default for Params {
    fn default() -> Self {
        Self {
            seed: 299792458,
            sample_len: 100,
            repeat_penalty: 1.1,
            repeat_last_n: 64,
            use_flash_attn: false,
            run_on_cpu: true,
            top_p: None,
            temperature: None,
        }
    }
}
