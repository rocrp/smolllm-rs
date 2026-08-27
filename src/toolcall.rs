//! Tool calls: provider-issued requests to run a named function.
//!
//! Surfaced verbatim — the caller executes the call and replays the assistant
//! and tool messages. The library runs no agentic loop and never inspects,
//! validates or repairs the argument JSON.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

/// One tool call issued by the model.
#[derive(Debug, Clone, PartialEq, Eq, Default, Serialize, Deserialize)]
pub struct ToolCall {
    pub id: String,
    #[serde(rename = "type")]
    pub kind: String,
    pub function: ToolCallFunction,
    /// Provider keys the library does not model — Gemini's
    /// `extra_content.google.thought_signature`, for one, which the provider
    /// expects echoed back. Preserved so a replayed turn is lossless.
    #[serde(flatten, default, skip_serializing_if = "serde_json::Map::is_empty")]
    pub extra: serde_json::Map<String, serde_json::Value>,
}

/// The function a tool call names, with its arguments as opaque JSON text.
#[derive(Debug, Clone, PartialEq, Eq, Default, Serialize, Deserialize)]
pub struct ToolCallFunction {
    pub name: String,
    /// Opaque JSON text: never parsed, validated or repaired.
    pub arguments: String,
}

/// One streamed fragment of a tool call.
#[derive(Debug, Clone, Default, Deserialize)]
pub(crate) struct ToolCallDelta {
    #[serde(default)]
    pub index: Option<usize>,
    #[serde(default)]
    pub id: Option<String>,
    #[serde(rename = "type", default)]
    pub kind: Option<String>,
    #[serde(default)]
    pub function: Option<ToolCallFunctionDelta>,
    // `index` is consumed for slotting: a streaming artifact, and the assembled
    // list is ordered by it instead.
    #[serde(flatten, default)]
    pub extra: serde_json::Map<String, serde_json::Value>,
}

#[derive(Debug, Clone, Default, Deserialize)]
pub(crate) struct ToolCallFunctionDelta {
    #[serde(default)]
    pub name: Option<String>,
    #[serde(default)]
    pub arguments: Option<String>,
}

/// Reassembles streamed tool-call deltas into whole calls.
///
/// Providers stream a tool call across many frames: the first carries the id and
/// function name, later ones append fragments of the argument JSON. Fragments are
/// never pushed to consumers — a caller can only act on a complete call, so the
/// assembled list is exposed once the stream ends.
#[derive(Debug, Default)]
pub(crate) struct ToolCallAccumulator {
    slots: BTreeMap<usize, ToolCall>,
    last_slot: Option<usize>,
}

impl ToolCallAccumulator {
    pub(crate) fn feed(&mut self, deltas: Vec<ToolCallDelta>) {
        for delta in deltas {
            self.merge(delta);
        }
    }

    fn slot_for(&self, delta: &ToolCallDelta) -> usize {
        if let Some(index) = delta.index {
            return index;
        }
        // Providers that omit `index` start a new call whenever they send a
        // fresh `id`; everything else continues the call already in progress.
        let has_id = delta.id.as_ref().is_some_and(|id| !id.is_empty());
        match self.last_slot {
            Some(last) if !has_id => last,
            _ => self.slots.keys().next_back().map_or(0, |slot| slot + 1),
        }
    }

    fn merge(&mut self, delta: ToolCallDelta) {
        let slot = self.slot_for(&delta);
        self.last_slot = Some(slot);
        let call = self.slots.entry(slot).or_default();

        // Later frames repeat these as empty strings; keep the first real one.
        if let Some(id) = delta.id.filter(|id| !id.is_empty()) {
            call.id = id;
        }
        if let Some(kind) = delta.kind.filter(|kind| !kind.is_empty()) {
            call.kind = kind;
        }
        for (key, value) in delta.extra {
            call.extra.insert(key, value);
        }
        let Some(function) = delta.function else {
            return;
        };
        if let Some(name) = function.name.filter(|name| !name.is_empty()) {
            call.function.name = name;
        }
        if let Some(arguments) = function.arguments {
            call.function.arguments.push_str(&arguments);
        }
    }

    /// The assembled calls, ordered by their provider-assigned index.
    pub(crate) fn result(&self) -> Vec<ToolCall> {
        self.slots.values().cloned().collect()
    }
}
