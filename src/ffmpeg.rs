// src/ffmpeg.rs

use crate::error::{MrexError, Result};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use log::{debug, info, error};

#[derive(Debug, Clone)]
pub struct VideoInfo {
    pub path: PathBuf,
    pub width: u32,
    pub height: u32,
    pub frame_count: u64,
    pub fps: f64,
    pub is_hdr: bool,
    // Add other relevant fields like pixel format if needed
}

/// Runs ffprobe to get video metadata.
pub fn get_video_info(video_path: &Path) -> Result<VideoInfo> {
    info!("Probing video file: {}", video_path.display());
    if !video_path.exists() {
        return Err(MrexError::Input(format!(
            "Input video file not found: {}",
            video_path.display()
        )));
    }

    let output = Command::new("ffprobe")
        .args([
            "-v", "error",
            "-count_frames", // Explicitly count frames
            "-select_streams", "v:0",
            "-show_entries", "stream=width,height,nb_frames,nb_read_frames,r_frame_rate,color_space,color_transfer:format=nb_frames", // Request nb_read_frames too
            "-of", "json", // Easier parsing
            video_path.to_str().ok_or_else(|| MrexError::Input("Invalid video path".to_string()))?,
        ])
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .output()
        .map_err(|e| MrexError::Io(e))?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        error!("ffprobe failed for {}: {}", video_path.display(), stderr);
        return Err(MrexError::Command(format!(
            "ffprobe failed for {}: {}",
            video_path.display(), stderr
        )));
    }

    let stdout = String::from_utf8_lossy(&output.stdout);
    debug!("ffprobe output for {}: {}", video_path.display(), stdout);

    // Parse JSON output
    let json: serde_json::Value = serde_json::from_str(&stdout)
        .map_err(|e| MrexError::Parse(format!("Failed to parse ffprobe JSON: {}", e)))?;

    let stream = json["streams"].get(0).ok_or_else(|| MrexError::Parse("No video stream found in ffprobe output".to_string()))?;

    let width = stream["width"].as_u64().ok_or_else(|| MrexError::Parse("Missing width".to_string()))? as u32;
    let height = stream["height"].as_u64().ok_or_else(|| MrexError::Parse("Missing height".to_string()))? as u32;
    // Try getting frame count from stream[nb_frames], format[nb_frames], or stream[nb_read_frames] from the initial probe
    let frame_count_str_opt = stream["nb_frames"].as_str()
        .or_else(|| json["format"]["nb_frames"].as_str())
        .or_else(|| stream["nb_read_frames"].as_str()); // Fallback to nb_read_frames

    let frame_count = match frame_count_str_opt {
        Some(fc_str) => fc_str.parse::<u64>()
            .map_err(|e| MrexError::Parse(format!("Invalid frame count value from initial probe: {}", e)))?,
        None => {
            // If not found, run a separate ffprobe command to count frames
            info!("Frame count not found in initial probe, running count_frames probe for {}", video_path.display());
            let count_output = Command::new("ffprobe")
                .args([
                    "-v", "error",
                    "-count_frames",
                    "-select_streams", "v:0",
                    "-show_entries", "stream=nb_read_frames",
                    "-of", "csv=p=0", // Use CSV output for simplicity
                    video_path.to_str().ok_or_else(|| MrexError::Input("Invalid video path".to_string()))?,
                ])
                .stdout(Stdio::piped())
                .stderr(Stdio::piped())
                .output()
                .map_err(|e| MrexError::Io(e))?;

            if !count_output.status.success() {
                let stderr = String::from_utf8_lossy(&count_output.stderr);
                error!("ffprobe count_frames failed for {}: {}", video_path.display(), stderr);
                return Err(MrexError::Command(format!(
                    "ffprobe count_frames failed for {}: {}",
                    video_path.display(), stderr
                )));
            }

            let count_stdout = String::from_utf8_lossy(&count_output.stdout).trim().to_string();
            debug!("ffprobe count_frames output for {}: {}", video_path.display(), count_stdout);

            if count_stdout.is_empty() || count_stdout == "N/A" {
                 return Err(MrexError::Parse(format!("Failed to count frames for {}", video_path.display())));
            }

            count_stdout.parse::<u64>()
                 .map_err(|e| MrexError::Parse(format!("Invalid frame count value from count_frames probe: '{}', error: {}", count_stdout, e)))?
        }
    };

    let fps_str = stream["r_frame_rate"].as_str().ok_or_else(|| MrexError::Parse("Missing r_frame_rate".to_string()))?;
    let fps = parse_frame_rate(fps_str)?;

    let color_transfer = stream["color_transfer"].as_str().unwrap_or("unknown");
    let is_hdr = color_transfer == "smpte2084" || color_transfer == "arib-std-b67";

    info!("Detected Info for {}: {}x{} @ {} fps, {} frames, HDR={}",
          video_path.display(), width, height, fps, frame_count, is_hdr);

    Ok(VideoInfo {
        path: video_path.to_path_buf(),
        width,
        height,
        frame_count,
        fps,
        is_hdr,
    })
}

/// Parses frame rate string (e.g., "24000/1001") into f64.
fn parse_frame_rate(fps_str: &str) -> Result<f64> {
    if fps_str.contains('/') {
        let parts: Vec<&str> = fps_str.split('/').collect();
        if parts.len() == 2 {
            let num = parts[0].parse::<f64>().map_err(|_| MrexError::Parse(format!("Invalid FPS numerator: {}", parts[0])))?;
            let den = parts[1].parse::<f64>().map_err(|_| MrexError::Parse(format!("Invalid FPS denominator: {}", parts[1])))?;
            if den == 0.0 {
                Err(MrexError::Parse("FPS denominator cannot be zero".to_string()))
            } else {
                Ok(num / den)
            }
        } else {
            Err(MrexError::Parse(format!("Invalid FPS format: {}", fps_str)))
        }
    } else {
        fps_str.parse::<f64>().map_err(|_| MrexError::Parse(format!("Invalid FPS format: {}", fps_str)))
    }
}


/// Executes an FFmpeg command.
pub fn run_ffmpeg(args: &[String], description: &str) -> Result<()> {
    info!("Running FFmpeg for {}: ffmpeg {}", description, args.join(" "));

    let mut command = Command::new("ffmpeg");
    command.args(args);
    command.stdout(Stdio::piped()); // Capture stdout if needed, or inherit/null
    command.stderr(Stdio::piped()); // Capture stderr for logging/errors

    let start_time = std::time::Instant::now();
    let output = command.output().map_err(|e| MrexError::Io(e))?;
    let duration = start_time.elapsed();

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        error!("FFmpeg command failed for {} ({}ms): {}", description, duration.as_millis(), stderr);
        Err(MrexError::Command(format!(
            "FFmpeg {} failed: {}", description, stderr
        )))
    } else {
        let stderr = String::from_utf8_lossy(&output.stderr);
        debug!("FFmpeg stderr for {} ({}ms): {}", description, duration.as_millis(), stderr);
        info!("FFmpeg command successful for {} ({}ms)", description, duration.as_millis());
        Ok(())
    }
}