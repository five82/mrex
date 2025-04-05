// src/metrics.rs

use crate::cli::CliArgs;
use crate::error::{MrexError, Result};
use crate::ffmpeg::{run_ffmpeg, VideoInfo};
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::{Path, PathBuf};
use log::{debug, info, warn, error};

// --- Data Structures ---

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct FrameMetrics {
    #[serde(rename = "frameNum")]
    pub frame_num: u64,
    pub metrics: Metrics,
}

#[derive(Serialize, Deserialize, Debug, Clone, Default)]
pub struct Metrics {
    // Optional because XPSNR might not be present initially or fail
    #[serde(skip_serializing_if = "Option::is_none")]
    pub vmaf: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub psnr_y: Option<f64>, // VMAF JSON might include PSNR components
    #[serde(skip_serializing_if = "Option::is_none")]
    pub psnr_cb: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub psnr_cr: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ssim: Option<f64>, // VMAF JSON might include SSIM

    // XPSNR specific, also optional
    #[serde(skip_serializing_if = "Option::is_none")]
    pub xpsnr: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub xpsnr_y: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub xpsnr_u: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub xpsnr_v: Option<f64>,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct PooledMetrics {
    // Define pooled metrics structure based on VMAF JSON output
    // Example:
    pub vmaf: Option<MetricStats>,
    pub psnr_y: Option<MetricStats>,
    pub psnr_cb: Option<MetricStats>,
    pub psnr_cr: Option<MetricStats>,
    pub ssim: Option<MetricStats>,
    // Add XPSNR pooled metrics if needed, though usually calculated from frame data
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct MetricStats {
    pub min: f64,
    pub max: f64,
    pub mean: f64,
    #[serde(rename = "harmonic_mean", skip_serializing_if = "Option::is_none")]
    pub harmonic_mean: Option<f64>, // VMAF JSON has this
}


#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct AnalysisResult {
    pub frames: Vec<FrameMetrics>,
    #[serde(rename = "pooled_metrics", skip_serializing_if = "Option::is_none")]
    pub pooled_metrics: Option<PooledMetrics>, // VMAF JSON has this at top level
    #[serde(rename = "ffmpeg_log_path", skip_serializing_if = "Option::is_none")]
    pub ffmpeg_log_path: Option<String>, // VMAF JSON has this
    #[serde(rename = "vmaf_version", skip_serializing_if = "Option::is_none")]
    pub vmaf_version: Option<String>, // VMAF JSON has this
    // Add other top-level fields from VMAF JSON if needed
}

// --- Core Functions ---

/// Runs VMAF analysis using FFmpeg.
pub fn run_vmaf(
    ref_info: &VideoInfo,
    dist_info: &VideoInfo,
    args: &CliArgs,
    output_prefix: &Path,
) -> Result<PathBuf> {
    info!("Starting VMAF analysis...");
    let vmaf_json_path = output_prefix.with_extension("json");
    let model = select_vmaf_model(ref_info.height);
    let threads = num_cpus::get().to_string();

    // TODO: Construct the complex filter chain based on HDR, denoise, crop/scale
    let filter_chain = build_vmaf_filter(ref_info, dist_info, args, &vmaf_json_path, &model, &threads)?;

    let ffmpeg_args = vec![
        "-hide_banner".to_string(),
        "-loglevel".to_string(), "warning".to_string(), // Or adjust based on logging needs
        "-stats".to_string(),
        "-i".to_string(), dist_info.path.to_string_lossy().to_string(),
        "-i".to_string(), ref_info.path.to_string_lossy().to_string(),
        "-lavfi".to_string(), filter_chain,
        "-f".to_string(), "null".to_string(),
        "-".to_string(),
    ];

    // Use system ffmpeg for VMAF as per original script? Or assume ffmpeg in PATH has libvmaf?
    // For now, assume ffmpeg in PATH is sufficient.
    run_ffmpeg(&ffmpeg_args, "VMAF Calculation")?;

    if !vmaf_json_path.exists() {
        error!("VMAF JSON output file was not created: {}", vmaf_json_path.display());
        return Err(MrexError::Command("VMAF JSON output file was not created".to_string()));
    }

    info!("VMAF analysis complete. Output: {}", vmaf_json_path.display());
    Ok(vmaf_json_path)
}

/// Runs XPSNR analysis using FFmpeg.
pub fn run_xpsnr(
    ref_info: &VideoInfo,
    dist_info: &VideoInfo,
    args: &CliArgs,
    output_prefix: &Path,
) -> Result<Vec<FrameMetrics>> {
    info!("Starting XPSNR analysis...");
    let xpsnr_log_path = output_prefix.with_extension("xpsnr.log");
    let threads = num_cpus::get().to_string();

    // TODO: Construct the complex filter chain based on HDR, denoise, crop/scale
    // Ensure NO HDR->SDR conversion is applied here.
    let filter_chain = build_xpsnr_filter(ref_info, dist_info, args, &xpsnr_log_path, &threads)?;

    let ffmpeg_args = vec![
        "-hide_banner".to_string(),
        "-loglevel".to_string(), "warning".to_string(),
        "-stats".to_string(),
        "-threads".to_string(), threads.clone(),
        "-filter_threads".to_string(), threads.clone(),
        "-filter_complex_threads".to_string(), threads.clone(),
        "-i".to_string(), dist_info.path.to_string_lossy().to_string(),
        "-i".to_string(), ref_info.path.to_string_lossy().to_string(),
        "-lavfi".to_string(), filter_chain,
        "-f".to_string(), "null".to_string(),
        "-".to_string(),
    ];

    run_ffmpeg(&ffmpeg_args, "XPSNR Calculation")?;

    if !xpsnr_log_path.exists() {
        error!("XPSNR log file was not created: {}", xpsnr_log_path.display());
        // Don't necessarily error out, maybe just warn and return empty vec?
        // Original script errors out if gawk fails, let's stick to that for now.
        return Err(MrexError::Command("XPSNR log file was not created".to_string()));
    }

    // Parse the XPSNR log file
    let xpsnr_frames = parse_xpsnr_log(&xpsnr_log_path)?;

    // Clean up log file? Original script does.
    if let Err(e) = fs::remove_file(&xpsnr_log_path) {
        warn!("Failed to remove temporary XPSNR log file {}: {}", xpsnr_log_path.display(), e);
    }

    info!("XPSNR analysis complete. Parsed {} frames.", xpsnr_frames.len());
    Ok(xpsnr_frames)
}

/// Merges VMAF JSON data with parsed XPSNR frame data.
use std::collections::HashMap; // Add HashMap import

pub fn merge_results(
    vmaf_json_path: &Path,
    xpsnr_frames: Vec<FrameMetrics>,
) -> Result<AnalysisResult> {
    info!("Merging VMAF and XPSNR results...");
    let vmaf_content = fs::read_to_string(vmaf_json_path)
        .map_err(|e| MrexError::Io(e))?;

    let mut vmaf_result: AnalysisResult = serde_json::from_str(&vmaf_content)
        .map_err(|e| MrexError::Json(e))?;

    if xpsnr_frames.is_empty() {
        warn!("No XPSNR frames provided for merging.");
        // Proceed without merging XPSNR data
    } else {
        // Create a HashMap for efficient lookup of XPSNR frames by frame number
        let xpsnr_map: HashMap<u64, &Metrics> = xpsnr_frames
            .iter()
            .map(|frame| (frame.frame_num, &frame.metrics))
            .collect();

        info!(
            "Merging {} VMAF frames with {} XPSNR frames (VMAF subsampled: {})",
            vmaf_result.frames.len(),
            xpsnr_frames.len(),
            vmaf_result.frames.len() < xpsnr_frames.len()
        );

        let mut merged_count = 0;
        // Iterate through the VMAF frames (which might be subsampled)
        for vmaf_frame in &mut vmaf_result.frames {
            // Find the corresponding XPSNR frame using the HashMap
            if let Some(xpsnr_metrics) = xpsnr_map.get(&vmaf_frame.frame_num) {
                // Copy XPSNR metrics into the VMAF frame's metrics
                vmaf_frame.metrics.xpsnr = xpsnr_metrics.xpsnr;
                vmaf_frame.metrics.xpsnr_y = xpsnr_metrics.xpsnr_y;
                vmaf_frame.metrics.xpsnr_u = xpsnr_metrics.xpsnr_u;
                vmaf_frame.metrics.xpsnr_v = xpsnr_metrics.xpsnr_v;
                merged_count += 1;
            } else {
                // This case should ideally not happen if XPSNR runs on all frames
                // and VMAF frame numbers are a subset of XPSNR frame numbers.
                warn!(
                    "Could not find matching XPSNR data for VMAF frame {}",
                    vmaf_frame.frame_num
                );
            }
        }
        info!("Successfully merged XPSNR data for {} VMAF frames.", merged_count);
    }

    // Save the merged result back to the original JSON path
    let merged_content = serde_json::to_string_pretty(&vmaf_result)?;
    fs::write(vmaf_json_path, merged_content)?;

    info!("Successfully saved merged results to {}", vmaf_json_path.display());
    Ok(vmaf_result)
}


// --- Helper Functions ---

fn select_vmaf_model(height: u32) -> String {
    if height > 1080 {
        "version=vmaf_4k_v0.6.1".to_string()
    } else {
        "version=vmaf_v0.6.1".to_string()
    }
}

fn build_vmaf_filter(
    ref_info: &VideoInfo,
    dist_info: &VideoInfo,
    args: &CliArgs,
    vmaf_json_path: &Path,
    model: &str,
    threads: &str,
) -> Result<String> {
    debug!("Building VMAF filter chain...");
    debug!("Ref Info: {:?}", ref_info);
    debug!("Dist Info: {:?}", dist_info);
    debug!("Args: {:?}", args);

    const HDR_TO_SDR_FILTER: &str = "zscale=t=linear:npl=100,tonemap=tonemap=hable:desat=0,zscale=t=bt709:p=bt709:m=bt709:r=tv";
    const DENOISE_FILTER: &str = "hqdn3d=4:3:6:4"; // Default denoise params from script

    let mut ref_filters: Vec<String> = Vec::new();
    let mut dist_filters: Vec<String> = Vec::new();

    // --- Determine Crop ---
    // Crop reference if distorted is smaller and width matches (letterboxing)
    let crop_filter = if dist_info.height < ref_info.height && dist_info.width == ref_info.width {
        let height_diff = ref_info.height - dist_info.height;
        let crop_y = height_diff / 2;
        // Ensure crop dimensions are positive
        if dist_info.width > 0 && dist_info.height > 0 {
             info!("Applying crop filter to reference: {}x{} at y={}", dist_info.width, dist_info.height, crop_y);
             Some(format!("crop={}:{}:0:{}", dist_info.width, dist_info.height, crop_y))
        } else {
             warn!("Invalid dimensions for crop filter, skipping.");
             None
        }
    } else if dist_info.height != ref_info.height || dist_info.width != ref_info.width {
        // Scaling might be needed if dimensions differ in other ways, but VMAF typically requires matching dimensions.
        // The original script implicitly scaled the distorted video if no crop was applied.
        // Let's assume libvmaf handles scaling or requires pre-scaled input.
        // For now, we only implement the explicit crop for letterboxing.
        warn!("Dimensions mismatch (Ref: {}x{}, Dist: {}x{}) and not simple letterboxing. VMAF might fail if inputs aren't scaled.",
              ref_info.width, ref_info.height, dist_info.width, dist_info.height);
        None
    } else {
        None // Dimensions match
    };


    // --- Build Reference Filter Segment ---
    if ref_info.is_hdr {
        ref_filters.push(HDR_TO_SDR_FILTER.to_string());
    }
    if let Some(crop) = crop_filter {
        ref_filters.push(crop);
    }
    if args.denoise {
        ref_filters.push(DENOISE_FILTER.to_string());
    }

    let ref_filter_segment = if ref_filters.is_empty() {
        "null".to_string() // Use null filter if no processing needed
    } else {
        ref_filters.join(",")
    };

    // --- Build Distorted Filter Segment ---
    if dist_info.is_hdr {
        dist_filters.push(HDR_TO_SDR_FILTER.to_string());
    }

    let dist_filter_segment = if dist_filters.is_empty() {
        "null".to_string()
    } else {
        dist_filters.join(",")
    };

    // --- Construct Final Filtergraph ---
    // Ensure FPS matches reference
    let fps = ref_info.fps;
    // Escape JSON path for ffmpeg filter syntax if necessary (depends on OS/shell interpretation)
    // For simplicity, assume path is okay for now. Revisit if errors occur.
    let log_path_str = vmaf_json_path.to_string_lossy();

    // Note: Using [0:v] for distorted (first input) and [1:v] for reference (second input)
    // Define the processing segments without trailing commas
    let dist_proc = if dist_filter_segment == "null" { "".to_string() } else { dist_filter_segment };
    let ref_proc = if ref_filter_segment == "null" { "".to_string() } else { ref_filter_segment };

    // Conditionally add commas in the main format string
    let filter_graph = format!(
        "[0:v]{dist_proc}{dist_comma}fps={fps}:round=near[main2];[1:v]{ref_proc}{ref_comma}fps={fps}:round=near[main1];[main2][main1]libvmaf=model={model}:log_fmt=json:log_path='{log_path}':n_threads={threads}:n_subsample=8",
        dist_proc = dist_proc,
        dist_comma = if dist_proc.is_empty() { "" } else { "," }, // Add comma only if dist_proc has filters
        ref_proc = ref_proc,
        ref_comma = if ref_proc.is_empty() { "" } else { "," }, // Add comma only if ref_proc has filters
        fps = fps,
        model = model,
        log_path = log_path_str,
        threads = threads
    );

    info!("Constructed VMAF filter graph: {}", filter_graph);
    Ok(filter_graph)
}

fn build_xpsnr_filter(
    ref_info: &VideoInfo,
    dist_info: &VideoInfo,
    args: &CliArgs,
    xpsnr_log_path: &Path,
    _threads: &str, // threads arg not directly used by xpsnr filter itself, but passed to ffmpeg command
) -> Result<String> {
    debug!("Building XPSNR filter chain...");
    debug!("Ref Info: {:?}", ref_info);
    debug!("Dist Info: {:?}", dist_info);
    debug!("Args: {:?}", args);

    const DENOISE_FILTER: &str = "hqdn3d=4:3:6:4";

    let mut ref_filters: Vec<String> = Vec::new();
    let mut dist_filters: Vec<String> = Vec::new();

    // --- Determine Crop (Applied to Reference) ---
    let crop_filter = if dist_info.height < ref_info.height && dist_info.width == ref_info.width {
        let height_diff = ref_info.height - dist_info.height;
        let crop_y = height_diff / 2;
        if dist_info.width > 0 && dist_info.height > 0 {
             info!("Applying crop filter to reference for XPSNR: {}x{} at y={}", dist_info.width, dist_info.height, crop_y);
             Some(format!("crop={}:{}:0:{}", dist_info.width, dist_info.height, crop_y))
        } else {
             warn!("Invalid dimensions for crop filter, skipping.");
             None
        }
    } else if dist_info.height != ref_info.height || dist_info.width != ref_info.width {
         warn!("Dimensions mismatch (Ref: {}x{}, Dist: {}x{}) and not simple letterboxing. XPSNR might be inaccurate if inputs aren't scaled/aligned.",
              ref_info.width, ref_info.height, dist_info.width, dist_info.height);
        None
    } else {
        None
    };

    // --- Build Reference Filter Segment ---
    // NO HDR->SDR conversion for XPSNR reference processing
    if let Some(crop) = crop_filter {
        ref_filters.push(crop);
    }
    if args.denoise {
        ref_filters.push(DENOISE_FILTER.to_string());
    }

    let ref_filter_segment = if ref_filters.is_empty() {
        "null".to_string()
    } else {
        ref_filters.join(",")
    };

    // --- Build Distorted Filter Segment ---
    // NO HDR->SDR conversion for XPSNR distorted processing
    // Pixel format conversion IS needed based on distorted input HDR status
    let pixel_format = if dist_info.is_hdr {
        "format=yuv420p10le" // Use 10-bit for HDR input
    } else {
        "format=yuv420p"     // Use 8-bit for SDR input
    };
    dist_filters.push(pixel_format.to_string());

    let dist_filter_segment = dist_filters.join(","); // Will always have at least the format filter

    // --- Construct Final Filtergraph ---
    let fps = ref_info.fps;
    let log_path_str = xpsnr_log_path.to_string_lossy();

    // Note: Using [0:v] for distorted (first input) and [1:v] for reference (second input)
    // Define the processing segments without trailing commas
    let dist_proc = dist_filter_segment; // Always has at least format filter
    let ref_proc = if ref_filter_segment == "null" { "".to_string() } else { ref_filter_segment };

    // Conditionally add commas in the main format string
    let filter_graph = format!(
        "[0:v]{dist_proc}{dist_comma}fps={fps}:round=near[main2];[1:v]{ref_proc}{ref_comma}fps={fps}:round=near[main1];[main2][main1]xpsnr=stats_file='{log_path}'",
        dist_proc = dist_proc,
        dist_comma = ",", // dist_proc always has filters, so always add comma
        ref_proc = ref_proc,
        ref_comma = if ref_proc.is_empty() { "" } else { "," }, // Add comma only if ref_proc has filters
        fps = fps,
        log_path = log_path_str
    );

    info!("Constructed XPSNR filter graph: {}", filter_graph);
    Ok(filter_graph)
}


// Use lazy_static or once_cell for efficient regex compilation
use once_cell::sync::Lazy;
use regex::Regex;

static XPSNR_REGEX: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r"n:\s*(?P<n>\d+)\s+XPSNR\s+y:\s*(?P<y>inf|[0-9]+(?:\.[0-9]*)?)\s+XPSNR\s+u:\s*(?P<u>inf|[0-9]+(?:\.[0-9]*)?)\s+XPSNR\s+v:\s*(?P<v>inf|[0-9]+(?:\.[0-9]*)?)") // Match repeated XPSNR keyword
        .expect("Invalid XPSNR Regex")
});

fn parse_xpsnr_log(log_path: &Path) -> Result<Vec<FrameMetrics>> {
    info!("Parsing XPSNR log file: {}", log_path.display());
    let content = fs::read_to_string(log_path)?;
    let mut frames = Vec::new();

    for line in content.lines() {
        if let Some(caps) = XPSNR_REGEX.captures(line) {
            // Extract captures using names
            let frame_num_str = caps.name("n").ok_or_else(|| MrexError::Parse("Missing frame number in XPSNR log line".to_string()))?.as_str();
            let y_str = caps.name("y").ok_or_else(|| MrexError::Parse("Missing y value in XPSNR log line".to_string()))?.as_str();
            let u_str = caps.name("u").ok_or_else(|| MrexError::Parse("Missing u value in XPSNR log line".to_string()))?.as_str();
            let v_str = caps.name("v").ok_or_else(|| MrexError::Parse("Missing v value in XPSNR log line".to_string()))?.as_str();

            // Parse values
            let frame_num = frame_num_str.parse::<u64>()
                .map_err(|_| MrexError::Parse(format!("Invalid frame number: {}", frame_num_str)))?;

            // Helper to parse float or "inf"
            let parse_float_or_inf = |s: &str| -> Result<f64> {
                if s == "inf" {
                    Ok(f64::INFINITY)
                } else {
                    s.parse::<f64>().map_err(|_| MrexError::Parse(format!("Invalid float value: {}", s)))
                }
            };

            let y = parse_float_or_inf(y_str)?;
            let u = parse_float_or_inf(u_str)?;
            let v = parse_float_or_inf(v_str)?;

            // Skip frames with infinite values, as per original gawk script
            if y.is_infinite() || u.is_infinite() || v.is_infinite() {
                debug!("Skipping frame {} due to infinite XPSNR value(s)", frame_num);
                continue;
            }

            // Calculate weighted average XPSNR (6:1:1)
            let weighted_xpsnr = (6.0 * y + u + v) / 8.0;

            frames.push(FrameMetrics {
                frame_num,
                metrics: Metrics {
                    xpsnr: Some(weighted_xpsnr),
                    xpsnr_y: Some(y),
                    xpsnr_u: Some(u),
                    xpsnr_v: Some(v),
                    ..Default::default() // Initialize other metrics as None/Default
                },
            });
        } else if line.contains("XPSNR average") {
            // Skip the summary line explicitly
            continue;
        } else if line.contains("XPSNR") && line.contains("n:") {
             // Log lines that look like they should match but didn't
             warn!("Failed to parse potential XPSNR line: {}", line);
        }
    }

    if frames.is_empty() && !content.trim().is_empty() {
         // Check if content wasn't just whitespace before erroring
         error!("Failed to parse any valid frames from XPSNR log: {}", log_path.display());
         // Provide more context if possible
         let head = content.lines().take(5).collect::<Vec<_>>().join("\n");
         error!("Log head:\n{}", head);
         return Err(MrexError::Parse("No valid frames found in XPSNR log".to_string()));
    }

    info!("Parsed {} frames from XPSNR log", frames.len());
    Ok(frames)
}

// --- Validation ---

/// Validates the combined VMAF and XPSNR results based on common sense checks.
pub fn validate_results(result: &AnalysisResult) -> Result<()> {
    info!("Validating analysis results...");
    let total_frames = result.frames.len();
    if total_frames == 0 {
        return Err(MrexError::Validation("No frames found in the analysis result.".to_string()));
    }

    // --- VMAF Validation ---
    let vmaf_scores: Vec<f64> = result.frames.iter().filter_map(|f| f.metrics.vmaf).collect();
    if vmaf_scores.is_empty() {
        return Err(MrexError::Validation("No VMAF scores found in results.".to_string()));
    }

    let vmaf_count = vmaf_scores.len();
    if vmaf_count < total_frames {
        warn!("Missing VMAF scores for {} out of {} frames.", total_frames - vmaf_count, total_frames);
        // Decide if this should be an error? Original scripts didn't explicitly check this.
    }

    let vmaf_mean = vmaf_scores.iter().sum::<f64>() / vmaf_count as f64;
    let vmaf_max = vmaf_scores.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));

    // Use pooled metrics if available for comparison/consistency check?
    if let Some(pooled) = &result.pooled_metrics {
        if let Some(pooled_vmaf) = &pooled.vmaf {
             debug!("Pooled VMAF Mean: {:.2}, Max: {:.2}", pooled_vmaf.mean, pooled_vmaf.max);
             // Could add checks for consistency between frame data and pooled metrics
        }
    }

    // Check for suspiciously low VMAF scores (potential sync issues)
    // Thresholds from original scripts
    if vmaf_mean < 10.0 {
        return Err(MrexError::Validation(format!(
            "Mean VMAF score ({:.2}) is suspiciously low (< 10). Possible frame sync issue or invalid analysis.", vmaf_mean
        )));
    }
    if vmaf_max < 20.0 {
         return Err(MrexError::Validation(format!(
            "Max VMAF score ({:.2}) is suspiciously low (< 20). Possible frame sync issue or invalid analysis.", vmaf_max
        )));
    }
    info!("VMAF validation passed (Mean: {:.2}, Max: {:.2}).", vmaf_mean, vmaf_max);


    // --- XPSNR Validation (if present) ---
    let xpsnr_y_scores: Vec<f64> = result.frames.iter().filter_map(|f| f.metrics.xpsnr_y).collect();
    let xpsnr_u_scores: Vec<f64> = result.frames.iter().filter_map(|f| f.metrics.xpsnr_u).collect();
    let xpsnr_v_scores: Vec<f64> = result.frames.iter().filter_map(|f| f.metrics.xpsnr_v).collect();

    let xpsnr_count = xpsnr_y_scores.len(); // Assume Y component is representative

    if xpsnr_count > 0 {
        info!("Performing XPSNR validation ({} frames with scores)...", xpsnr_count);

        // Check if enough frames have XPSNR scores (e.g., > 90% of total)
        // Original script checked this before parsing, but we check after merge.
        if xpsnr_count < (total_frames * 9 / 10) {
             return Err(MrexError::Validation(format!(
                "Insufficient XPSNR frames ({}) compared to total frames ({}). Less than 90%.", xpsnr_count, total_frames
            )));
        }

        // Check if all U and V components are zero (original script check)
        let all_u_zero = xpsnr_u_scores.iter().all(|&u| u == 0.0);
        let all_v_zero = xpsnr_v_scores.iter().all(|&v| v == 0.0);
        if all_u_zero && all_v_zero && xpsnr_count == total_frames {
            // This might be valid for grayscale, but the original script flagged it.
            warn!("All XPSNR U and V components are zero. Input might be grayscale or there could be an issue.");
            // Decide if this should be an error? Let's keep it as a warning for now.
            // return Err(MrexError::Validation("All XPSNR U/V components are zero.".to_string()));
        }

        // Check Y component range (original script used 25-60 range check, but made it a warning)
        let xpsnr_y_mean = xpsnr_y_scores.iter().sum::<f64>() / xpsnr_count as f64;
        let min_threshold = 20.0; // Typical lower bound for reasonable quality
        let max_threshold = 60.0; // Typical upper bound

        if xpsnr_y_mean < min_threshold || xpsnr_y_mean > max_threshold {
             warn!(
                "Mean XPSNR Y-component ({:.2}) is outside the typical range ({}-{} dB). This might be normal for specific content.",
                xpsnr_y_mean, min_threshold, max_threshold
            );
             // Original script eventually treated this as non-fatal.
        } else {
             info!("XPSNR Y-component mean ({:.2}) is within typical range ({}-{} dB).", xpsnr_y_mean, min_threshold, max_threshold);
        }

        info!("XPSNR validation passed.");

    } else {
        info!("No XPSNR scores found in results, skipping XPSNR validation.");
    }


    info!("All validation checks passed.");
    Ok(())
}