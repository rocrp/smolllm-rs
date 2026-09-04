use tokio_stream::StreamExt;

#[tokio::main]
async fn main() -> Result<(), smolllm::Error> {
    env_logger::init();
    let _ = dotenvy::from_filename(
        dirs_next::home_dir()
            .expect("no home dir")
            .join(".env.smolllm"),
    );

    let mut stream = smolllm::stream("Explain async/await in Rust in 3 sentences.")
        .model("gemini/gemini-flash-lite-latest")
        .system_prompt("Be concise and technical.")
        .await?;

    print!("Response: ");
    while let Some(chunk) = stream.next().await {
        let chunk = chunk?;
        print!("{}", chunk.content);
    }
    println!();

    if !stream.reasoning().is_empty() {
        println!("\nReasoning: {}", stream.reasoning());
    }

    println!("Requested: {}", stream.model());
    match stream.resolved_model() {
        Some(resolved) => println!("Server answered with: {resolved}"),
        None => println!("Server reported no model of its own"),
    }
    if stream.truncated() {
        println!("WARNING: the answer was cut short");
    }

    let usage = stream.usage();
    let approx = if usage.estimated { "~" } else { "" };
    println!(
        "Tokens: {approx}{} in / {approx}{} out, Duration: {:?}",
        usage.input_tokens, usage.output_tokens, usage.duration
    );

    Ok(())
}
