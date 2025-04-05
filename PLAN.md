# Plan: Rewriting VMAF/XPSNR Tool in Rust (`mrex`)

**Project Name:** `mrex`
**Binary Name:** `mrex`

**Core Goals:**

1.  Replicate the functionality of the `generate-vmaf.sh` script for calculating VMAF and XPSNR metrics using `ffmpeg`.
2.  Replicate the functionality of the `cli.py` script for plotting the generated VMAF and XPSNR scores.
3.  Adhere to the specified HDR handling:
    *   VMAF: Apply HDR-to-SDR tone mapping for HDR inputs.
    *   XPSNR: Do *not* apply HDR-to-SDR tone mapping.
4.  Implement in idiomatic Rust, focusing on safety, performance, and maintainability, following rules in `.clinerules`.
5.  Provide a command-line interface similar to the original tools, combining analysis and plotting.

**Implementation Steps:**

1.  **Project Setup:**
    *   Initialize a Rust binary project in the current directory: `cargo init` (User should run this, or I can use `execute_command`).
    *   Configure `Cargo.toml` with necessary metadata and dependencies (`clap`, `serde`, `serde_json`, `plotters`, `log`, `env_logger`/`tracing`, `thiserror`, `num_cpus`, potentially `regex`).

2.  **`.gitignore` Update:**
    *   Add standard Rust ignores (`/target`, `Cargo.lock` - if not committed) and temporary files to the existing `.gitignore`.

3.  **`README.md` Creation:**
    *   Create `README.md` with project title, description, prerequisites (`ffmpeg` with `libvmaf`), build instructions (`cargo build --release`), usage examples, and license info.

4.  **Define Core Modules & Structures:**
    *   Organize code into modules (e.g., `src/cli.rs`, `src/ffmpeg.rs`, `src/metrics.rs`, `src/plot.rs`, `src/error.rs`, `src/main.rs`).
    *   Define structs: `CliArgs` (clap), `VideoInfo` (metadata), `FrameMetrics`, `AnalysisResult`.
    *   Define custom error enum: `MrexError` (thiserror).

5.  **Implement CLI (`src/cli.rs`, `src/main.rs`):**
    *   Define `CliArgs` struct using `clap` to match combined options.
    *   Parse arguments in `main`.

6.  **FFmpeg/FFprobe Interaction (`src/ffmpeg.rs`):**
    *   `get_video_info(path: &Path) -> Result<VideoInfo, MrexError>`: Run `ffprobe`, parse output.
    *   `run_ffmpeg(args: &[String]) -> Result<(), MrexError>`: Execute `ffmpeg` command.

7.  **Filter Chain Construction:**
    *   Implement logic to build `lavfi` filter strings dynamically based on video info and options.
    *   Handle HDR-to-SDR mapping correctly for VMAF vs. XPSNR.
    *   Handle cropping/scaling.
    *   Use `num_cpus::get()` for threads.

8.  **Metric Calculation (`src/metrics.rs`):**
    *   `run_vmaf(...) -> Result<PathBuf, MrexError>`: Run VMAF `ffmpeg`, output JSON.
    *   `run_xpsnr(...) -> Result<Vec<FrameMetrics>, MrexError>`: Run XPSNR `ffmpeg`, parse text log, return frame data.
    *   `merge_results(...) -> Result<AnalysisResult, MrexError>`: Read VMAF JSON, merge XPSNR data, save combined JSON.

9.  **Validation:**
    *   `validate_results(result: &AnalysisResult) -> Result<(), MrexError>`: Implement checks mirroring original scripts.

10. **Plotting (`src/plot.rs`):**
    *   `generate_plot(...) -> Result<(), MrexError>`: Use `plotters` to create VMAF/XPSNR plots similar to original, save to file. Handle invalid/infinite values.

11. **Main Workflow (`src/main.rs`):**
    *   Orchestrate the steps: parse args, setup logging, get info, validate inputs, run analyses, merge, validate results, generate plots, handle errors, report time.

12. **Testing:**
    *   Unit tests for parsing, calculations, filter generation.
    *   Integration tests using sample videos.

13. **Documentation:**
    *   Add Rustdoc comments (`///`) throughout the code.