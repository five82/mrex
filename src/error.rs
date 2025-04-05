// src/error.rs

use thiserror::Error;

#[derive(Error, Debug)]
pub enum MrexError {
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    #[error("FFmpeg/FFprobe command failed: {0}")]
    Command(String),

    #[error("Failed to parse command output: {0}")]
    Parse(String),

    #[error("JSON processing error: {0}")]
    Json(#[from] serde_json::Error),

    #[error("Plotting error: {0}")]
    Plot(String), // Placeholder, plotters might have its own error type

    #[error("Validation error: {0}")]
    Validation(String),

    #[error("Input error: {0}")]
    Input(String),
}

// Define a standard Result type for the crate
pub type Result<T> = std::result::Result<T, MrexError>;