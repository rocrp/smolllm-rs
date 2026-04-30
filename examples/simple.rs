#[tokio::main]
async fn main() -> Result<(), smolllm::Error> {
    env_logger::init();
    let _ = dotenvy::from_filename(
        dirs_next::home_dir()
            .expect("no home dir")
            .join(".env.smolllm"),
    );

    let prompt = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "What is Rust in one sentence?".into());

    let response = smolllm::ask(&*prompt).await?;

    println!("{}", response.text);
    Ok(())
}
