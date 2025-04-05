# mrex

A tool for calculating and plotting VMAF and XPSNR video quality metrics, implemented in Rust.

This tool aims to replicate the core functionality of the Python `vmaf-tools` scripts (`generate-vmaf` and `vmaf-plot`), focusing on VMAF and XPSNR calculation and plotting, with specific handling for HDR content as per the original requirements.

## Prerequisites

*   **Rust Toolchain:** Install from [rustup.rs](https://rustup.rs/).
*   **FFmpeg:** Must be installed and available in your system's PATH. Crucially, it needs to be compiled with `libvmaf` enabled. You can typically install this using a package manager like Homebrew (`brew install ffmpeg`) or build it from source.

## Building

Navigate to the project directory and run:

```bash
cargo build --release
```

The executable will be located at `target/release/mrex`.

## Usage

```bash
# Basic analysis and plot generation
./target/release/mrex <reference_video> <distorted_video> [output_prefix]

# Example
./target/release/mrex reference.mp4 distorted.mp4 my_analysis

# Options (Planned - based on original script)
# ./target/release/mrex --denoise --output-dir ./results reference.mp4 distorted.mp4 my_analysis_denoised
# ./target/release/mrex --log reference.mp4 distorted.mp4 my_analysis_logged
```

This will generate:

*   `<output_prefix>.json`: Combined VMAF and XPSNR scores per frame.
*   `<output_prefix>_vmaf_plot.png`: Plot of VMAF scores.
*   `<output_prefix>_xpsnr_plot.png`: Plot of XPSNR scores.

## License

This project is licensed under the terms specified in the [LICENSE](LICENSE) file.