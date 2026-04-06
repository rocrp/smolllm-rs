use smolllm;

#[tokio::main]
async fn main() -> Result<(), smolllm::Error> {
    env_logger::init();
    let _ = dotenvy::from_filename(
        dirs_next::home_dir()
            .expect("no home dir")
            .join(".env.smolllm"),
    );

    let response = smolllm::ask("What is Rust in one sentence?")
        .model("gemini/gemini-flash-lite-latest")
        .system_prompt("Be concise.")
        .await?;

    println!("Model: {}", response.model);
    println!("Response: {}", response.text);
    if !response.reasoning.is_empty() {
        println!("Reasoning: {}", response.reasoning);
    }
    println!(
        "Tokens: {} in / {} out, Duration: {:?}",
        response.usage.input_tokens, response.usage.output_tokens, response.usage.duration
    );

    Ok(())
}
