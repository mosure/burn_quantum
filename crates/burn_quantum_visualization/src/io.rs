//! Bounded JSON interchange for native applications and the browser workbench.
use crate::model::VisualizationDocument;
use std::cell::RefCell;
mod preflight;

/// Maximum imported or exported JSON size. Detailed model budgets also apply.
pub const MAX_DOCUMENT_BYTES: usize = 16 * 1024 * 1024;
/// Maximum aggregate complex inner-product terms used when revalidating imported
/// dense unitaries: sum of gate dimension cubed. Applied before typed deserialization.
pub const MAX_UNITARY_VALIDATION_WORK: usize = 16_777_216;
/// Interchange errors retain their validation or filesystem context.
pub type IoResult<T> = Result<T, Box<dyn std::error::Error>>;

/// Parse an experiment only after applying the byte budget and scientific checks.
pub fn parse_document(text: &str) -> IoResult<VisualizationDocument> {
    if text.len() > MAX_DOCUMENT_BYTES {
        return Err("experiment JSON exceeds the 16 MiB input budget".into());
    }
    // Parsing generic JSON only traverses the bounded input. Deserializing core
    // Unitary values performs cubic validation, so inspect all budgets first.
    let value: serde_json::Value = serde_json::from_str(text)?;
    preflight::check(&value)?;
    let document: VisualizationDocument = serde_json::from_value(value)?;
    document.validate()?;
    Ok(document)
}

/// Validate and serialize a bounded experiment for storage or another viewer.
pub fn document_json(document: &VisualizationDocument) -> IoResult<String> {
    document.validate()?;
    let value = serde_json::to_value(document)?;
    preflight::check(&value)?;
    let text = serde_json::to_string_pretty(&value)?;
    if text.len() > MAX_DOCUMENT_BYTES {
        return Err("experiment JSON exceeds the 16 MiB export budget".into());
    }
    Ok(text)
}

/// Read a native experiment file with a bounded reader, including growing files.
#[cfg(not(target_arch = "wasm32"))]
pub fn read_input(path: impl AsRef<std::path::Path>) -> IoResult<VisualizationDocument> {
    use std::io::Read;
    let file = std::fs::File::open(path)?;
    let mut bytes = Vec::new();
    file.take(MAX_DOCUMENT_BYTES as u64 + 1)
        .read_to_end(&mut bytes)?;
    parse_document(std::str::from_utf8(&bytes)?)
}

/// Write a scientifically validated document to a native file.
#[cfg(not(target_arch = "wasm32"))]
pub fn write_export(
    path: impl AsRef<std::path::Path>,
    document: &VisualizationDocument,
) -> IoResult<()> {
    std::fs::write(path, document_json(document)?)?;
    Ok(())
}

thread_local! {
    static PENDING: RefCell<Option<VisualizationDocument>> = const { RefCell::new(None) };
    static CURRENT: RefCell<Option<VisualizationDocument>> = const { RefCell::new(None) };
}

/// Replace the single pending UI import. Validation happens before replacement.
pub fn queue_document(document: VisualizationDocument) -> IoResult<()> {
    document.validate()?;
    PENDING.with(|slot| *slot.borrow_mut() = Some(document));
    Ok(())
}

/// Consume the pending document on the workbench's main thread.
pub fn take_pending_document() -> Option<VisualizationDocument> {
    PENDING.with(|slot| slot.borrow_mut().take())
}

/// Record a snapshot only when the selected document changes, never every frame.
pub fn set_current_document(document: &VisualizationDocument) {
    CURRENT.with(|slot| *slot.borrow_mut() = Some(document.clone()));
}

/// Import JSON from the gallery without exposing renderer or simulation internals.
#[cfg(all(feature = "viewer", target_arch = "wasm32"))]
#[wasm_bindgen::prelude::wasm_bindgen]
pub fn load_document_json(text: &str) -> Result<(), wasm_bindgen::JsValue> {
    let document =
        parse_document(text).map_err(|e| wasm_bindgen::JsValue::from_str(&e.to_string()))?;
    queue_document(document).map_err(|e| wasm_bindgen::JsValue::from_str(&e.to_string()))
}

/// Export the selected trace; errors remain visible to the gallery caller.
#[cfg(all(feature = "viewer", target_arch = "wasm32"))]
#[wasm_bindgen::prelude::wasm_bindgen]
pub fn export_document_json() -> Result<String, wasm_bindgen::JsValue> {
    CURRENT.with(|slot| match slot.borrow().as_ref() {
        Some(document) => {
            document_json(document).map_err(|e| wasm_bindgen::JsValue::from_str(&e.to_string()))
        }
        None => Err(wasm_bindgen::JsValue::from_str(
            "No experiment is loaded yet",
        )),
    })
}
