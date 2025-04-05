// src/cli.rs

use clap::Parser;
use std::path::PathBuf;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
pub struct CliArgs {
    /// Original/reference video file
    #[arg(required = true)]
    pub reference: PathBuf,

    /// Encoded/processed video file to compare
    #[arg(required = true)]
    pub distorted: PathBuf,

    /// Optional prefix for output files (default: derived from distorted filename)
    #[arg(required = false)]
    pub output_prefix: Option<String>,

    /// Enable denoising of reference video (hqdn3d filter)
    #[arg(long)]
    pub denoise: bool,

    /// Specify output directory for results (default: current directory)
    #[arg(long, value_name = "DIR")]
    pub output_dir: Option<PathBuf>,

    /// Enable logging to file (e.g., mrex_YYYYMMDD_HHMMSS.log)
    #[arg(long)]
    pub log: bool,
}

pub fn parse_args() -> CliArgs {
    CliArgs::parse()
}