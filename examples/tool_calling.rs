//! Tool calling: the caller runs the loop, the library never executes anything.
//!
//! Usage: `cargo run --example tool_calling -- [model ...]`

use serde_json::json;
use smolllm::{Message, Prompt, ToolCall};
use tokio_stream::StreamExt;

const QUESTION: &str =
    "What is the weather in Paris right now? You must call the get_weather tool to find out.";

fn tools() -> serde_json::Value {
    json!([{
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather for a city",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
        },
    }])
}

/// Whatever the tool actually does; the library never calls it for you.
fn dispatch(call: &ToolCall) -> String {
    println!(
        "  -> {}({})  [arguments are opaque JSON text]",
        call.function.name, call.function.arguments
    );
    json!({"city": "Paris", "temp_c": 18, "condition": "cloudy"}).to_string()
}

async fn run(model: &str) -> Result<(), Box<dyn std::error::Error>> {
    let extra = json!({"tools": tools(), "tool_choice": "auto"});

    let response = smolllm::ask(QUESTION)
        .model(model)
        .extra_body(extra.clone())?
        .await?;
    println!(
        "{model}: finish_reason={:?} tool_calls={}",
        response.finish_reason,
        response.tool_calls.len()
    );

    let Some(call) = response.tool_calls.first().cloned() else {
        println!("  model answered in prose: {}", response.text);
        return Ok(());
    };
    let result = dispatch(&call);

    // Replay the assistant turn and the tool result, then ask again.
    let replay = Prompt::Messages(vec![
        Message::user(QUESTION),
        Message::assistant_tool_calls(response.text, response.tool_calls),
        Message::tool(call.id, result),
    ]);
    let final_answer = smolllm::ask(replay)
        .model(model)
        .extra_body(extra.clone())?
        .await?;
    println!("  final: {}", final_answer.text);

    // Streaming: complete calls only, exposed after the stream ends.
    let mut stream = smolllm::stream(QUESTION)
        .model(model)
        .extra_body(extra)?
        .await?;
    let mut chunks = 0;
    while let Some(chunk) = stream.next().await {
        chunk?;
        chunks += 1;
    }
    println!(
        "  stream: chunks={chunks} finish_reason={:?} tool_calls={}",
        stream.finish_reason(),
        stream.tool_calls().len()
    );
    Ok(())
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let _ = dotenvy::from_filename(
        dirs_next::home_dir()
            .unwrap_or_default()
            .join(".env.smolllm"),
    );
    env_logger::init();

    let models: Vec<String> = std::env::args().skip(1).collect();
    let models = if models.is_empty() {
        vec!["deepseek/deepseek-v4-flash".to_string()]
    } else {
        models
    };
    for model in &models {
        if let Err(err) = run(model).await {
            println!("{model}: ERROR {err}");
        }
    }
    Ok(())
}
