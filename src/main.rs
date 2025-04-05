mod error;
mod cli;
mod ffmpeg;
mod metrics;
mod plot;

use crate::cli::CliArgs;
use crate::error::{MrexError, Result}; // Use our custom Result
use log::{info, error, warn, LevelFilter};
use std::fs::{self};
use std::path::PathBuf;
use std::process::ExitCode;
use std::time::Instant;
use chrono::Local;
use fern; // Add missing import for logging

fn main() -> ExitCode {
    // Record start time early
    let start_time = Instant::now();

    // Parse arguments first to potentially setup logging based on them
    let args = cli::parse_args();

    // Setup logging (console and optional file)
    if let Err(e) = setup_logging(&args) {
        eprintln!("Error setting up logging: {}", e);
        return ExitCode::FAILURE;
    }

    info!("Starting mrex analysis...");
    info!("Arguments: {:?}", args);

    // Run the main application logic
    match run(args) {
        Ok(()) => {
            let duration = start_time.elapsed();
            info!("Analysis completed successfully in {:.2?}", duration);
            println!("Analysis completed successfully in {:.2?}", duration); // Also print to console
            ExitCode::SUCCESS
        }
        Err(e) => {
            let duration = start_time.elapsed();
            error!("Analysis failed after {:.2?}: {}", duration, e);
            eprintln!("Error: {}", e); // Also print to stderr for visibility
            ExitCode::FAILURE
        }
    }
}

/// Sets up logging to console and optionally to a file.
fn setup_logging(args: &CliArgs) -> std::result::Result<(), fern::InitError> {
    let base_config = fern::Dispatch::new()
        .format(|out, message, record| {
            out.finish(format_args!(
                "[{} {} {}] {}",
                Local::now().format("%Y-%m-%d %H:%M:%S"),
                record.level(),
                record.target(),
                message
            ))
        })
        .level(LevelFilter::Info) // Default level
        .level_for("mrex", LevelFilter::Debug); // More detailed logs for our crate

    let console_config = fern::Dispatch::new()
        .chain(std::io::stdout());

    let mut logger = base_config.chain(console_config);

    if args.log {
        let log_filename = format!("mrex_{}.log", Local::now().format("%Y%m%d_%H%M%S"));
        let log_path = args.output_dir.clone().unwrap_or_else(|| PathBuf::from(".")).join(log_filename);
        // Ensure output directory exists if specified for the log file
        if let Some(dir) = log_path.parent() {
             if !dir.exists() {
                 fs::create_dir_all(dir)?;
             }
        }
        let file_config = fern::Dispatch::new().chain(fern::log_file(log_path)?);
        logger = logger.chain(file_config);
        info!("Logging to file: {}", args.output_dir.clone().unwrap_or_else(|| PathBuf::from(".")).join(format!("mrex_{}.log", Local::now().format("%Y%m%d_%H%M%S"))).display());
    }

    logger.apply()?;
    Ok(())
}


/// Main application logic
fn run(args: CliArgs) -> Result<()> {
    info!("Reference: {}", args.reference.display());
    info!("Distorted: {}", args.distorted.display());

    // --- 1. Get Video Info & Validate Inputs ---
    let ref_info = ffmpeg::get_video_info(&args.reference)?;
    let dist_info = ffmpeg::get_video_info(&args.distorted)?;

    // Basic input validation (from original script)
    if ref_info.frame_count != dist_info.frame_count {
        return Err(MrexError::Input(format!(
            "Frame count mismatch! Reference: {}, Distorted: {}",
            ref_info.frame_count, dist_info.frame_count
        )));
    }
    // Compare FPS with a tolerance
    if (ref_info.fps - dist_info.fps).abs() > 0.01 {
         return Err(MrexError::Input(format!(
            "Frame rate mismatch! Reference: {:.3}, Distorted: {:.3}",
            ref_info.fps, dist_info.fps
        )));
    }
    // Check if distorted is larger than reference (likely wrong order)
    if dist_info.height > ref_info.height || dist_info.width > ref_info.width {
         warn!("Distorted video dimensions ({}x{}) are larger than reference ({}x{}).",
               dist_info.width, dist_info.height, ref_info.width, ref_info.height);
         warn!("This might indicate reference/distorted inputs are swapped.");
         // Consider making this an error? Original script errored only on height.
         // Let's keep it as a warning for now, but VMAF/XPSNR might fail later.
    }
    info!("Input validation passed.");


    // --- 2. Determine Output Paths ---
    let output_dir = args.output_dir.clone().unwrap_or_else(|| PathBuf::from("."));
    // Ensure output directory exists
    if !output_dir.exists() {
        info!("Creating output directory: {}", output_dir.display());
        fs::create_dir_all(&output_dir)
            .map_err(|e| MrexError::Io(e))?;
    }

    let prefix_str = args.output_prefix.clone().unwrap_or_else(|| {
        // Default prefix: distorted filename without extension
        args.distorted.file_stem().map_or_else(
            || "mrex_analysis".to_string(), // Fallback if no stem
            |stem| stem.to_string_lossy().to_string()
        )
    });
    let output_prefix = output_dir.join(prefix_str);
    info!("Using output prefix: {}", output_prefix.display());

    let vmaf_json_path = output_prefix.with_extension("json");
    let vmaf_plot_path = output_prefix.with_extension("vmaf.png");
    let xpsnr_plot_path = output_prefix.with_extension("xpsnr.png");


    // --- 3. Run Analyses ---
    let vmaf_json_path_returned = metrics::run_vmaf(&ref_info, &dist_info, &args, &output_prefix)?;
    // Ensure the returned path matches expected (should always be the case here)
    assert_eq!(vmaf_json_path_returned, vmaf_json_path);
    let xpsnr_frames = metrics::run_xpsnr(&ref_info, &dist_info, &args, &output_prefix)?;

    // --- 4. Merge Results ---
    let merged_result = metrics::merge_results(&vmaf_json_path, xpsnr_frames)?;

    // --- 5. Validate Results ---
    metrics::validate_results(&merged_result)?; // Call the validation function

    // --- 6. Generate Plots ---
    plot::generate_plot(&merged_result, "VMAF", &vmaf_plot_path)?;
    plot::generate_plot(&merged_result, "XPSNR", &xpsnr_plot_path)?;

    info!("Generated VMAF plot: {}", vmaf_plot_path.display());
    info!("Generated XPSNR plot: {}", xpsnr_plot_path.display());
    println!("Results saved:"); // User-facing confirmation
    println!("  JSON: {}", vmaf_json_path.display());
    println!("  VMAF Plot: {}", vmaf_plot_path.display());
    println!("  XPSNR Plot: {}", xpsnr_plot_path.display());


    Ok(())
}
