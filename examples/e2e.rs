use smolllm;
use tokio_stream::StreamExt;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::init();
    let _ = dotenvy::from_filename(
        dirs_next::home_dir()
            .expect("no home dir")
            .join(".env.smolllm"),
    );

    println!("=== Test 1: Basic ask ===");
    let resp = smolllm::ask("Say hello in 3 words")
        .model("gemini/gemini-flash-lite-latest")
        .await?;
    println!("  Response: {}", resp.text);
    assert!(!resp.text.is_empty());

    println!("\n=== Test 2: Streaming ===");
    let mut stream = smolllm::stream("Count from 1 to 5, one per line")
        .model("gemini/gemini-flash-lite-latest")
        .await?;
    print!("  Response: ");
    let mut full_content = String::new();
    while let Some(chunk) = stream.next().await {
        let chunk = chunk?;
        print!("{}", chunk.content);
        full_content.push_str(&chunk.content);
    }
    println!();
    assert!(!full_content.is_empty());

    println!("\n=== Test 3: With system prompt and handler ===");
    let resp = smolllm::ask("What is 2+2?")
        .model("gemini/gemini-flash-lite-latest")
        .system_prompt("You are a math tutor. Answer with just the number.")
        .handler(|chunk| {
            eprint!("{}", chunk.content);
        })
        .await?;
    eprintln!();
    println!("  Response: {}", resp.text);

    println!("\n=== Test 4: Model fallback (bad model, then good) ===");
    let resp = smolllm::ask("Say 'fallback works'")
        .model("gemini/nonexistent-model-xyz,gemini/gemini-flash-lite-latest")
        .await?;
    println!("  Response: {}", resp.text);
    println!("  Used model: {}", resp.model);

    println!("\n=== Test 5: Validate ===");
    match smolllm::validate("gemini/gemini-flash-lite-latest", None, None) {
        Ok(()) => println!("  Validation passed"),
        Err(e) => println!("  Validation failed: {}", e),
    }

    println!("\n=== Test 6: Temperature ===");
    let resp = smolllm::ask("Generate a random word")
        .model("gemini/gemini-flash-lite-latest")
        .temperature(1.5)
        .await?;
    println!("  Response: {}", resp.text);

    println!("\n=== Test 7: Remove backticks ===");
    let resp = smolllm::ask("Write a hello world in python. Only output the code, wrapped in backticks.")
        .model("gemini/gemini-flash-lite-latest")
        .remove_backticks()
        .await?;
    println!("  Response: {}", resp.text);
    assert!(!resp.text.starts_with("```"));

    println!("\n=== Test 8: Request hook ===");
    let resp = smolllm::ask("Say 'hook test'")
        .model("gemini/gemini-flash-lite-latest")
        .hook(|event| {
            println!("  Hook: model={} tokens={}", event.usage.model, event.usage.output_tokens);
        })
        .await?;
    println!("  Response: {}", resp.text);

    println!("\n=== Test 9: Message-based prompt ===");
    let messages = vec![
        smolllm::Message::system("You are a translator. Translate to French."),
        smolllm::Message::user("Hello, how are you?"),
    ];
    let resp = smolllm::ask(messages)
        .model("gemini/gemini-flash-lite-latest")
        .await?;
    println!("  Response: {}", resp.text);

    println!("\n=== All E2E tests passed! ===");
    Ok(())
}
