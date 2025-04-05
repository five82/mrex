// src/plot.rs

use crate::error::{MrexError, Result};
use crate::metrics::AnalysisResult;
use plotters::prelude::*;
use plotters::style::full_palette::{ORANGE, GREEN}; // Import missing colors
use std::path::Path;
use log::{info, error, warn};
use quantiles::ckms::CKMS; // Use CKMS implementation from quantiles crate

const XPSNR_CAP_VALUE: f64 = 100.0; // Value to cap infinite XPSNR at

/// Generates a plot for a specific metric (VMAF or XPSNR).
pub fn generate_plot(
    result: &AnalysisResult,
    metric: &str, // "VMAF" or "XPSNR"
    output_path: &Path,
) -> Result<()> {
    info!("Generating {} plot: {}", metric, output_path.display());

    // --- 1. Extract and Prepare Data ---
    let scores: Vec<Option<f64>> = result.frames.iter().map(|frame| {
        match metric {
            "VMAF" => frame.metrics.vmaf,
            "XPSNR" => frame.metrics.xpsnr,
            _ => {
                warn!("Unsupported metric for plotting: {}", metric);
                None
            },
        }
    }).collect();

    let mut infinite_count = 0;
    let valid_scores: Vec<(u32, f64)> = scores.into_iter().enumerate()
        .filter_map(|(i, score_opt)| {
            score_opt.and_then(|score| {
                if metric == "XPSNR" {
                    let capped_score = if score.is_infinite() {
                        infinite_count += 1;
                        XPSNR_CAP_VALUE
                    } else {
                        score
                    };
                    // Keep scores > 0 after capping
                    if capped_score > 0.0 { Some((i as u32, capped_score)) } else { None }
                } else { // VMAF (assuming 0-100 range)
                    // Filter out <= 0 and potentially > 100 for VMAF
                    if score > 0.0 && score <= 100.0 { Some((i as u32, score)) } else { None }
                }
            })
        })
        .collect();

    if infinite_count > 0 {
        info!("Note: Capped {} infinite XPSNR values to {} dB for plotting.", infinite_count, XPSNR_CAP_VALUE);
    }

    if valid_scores.is_empty() {
        error!("No valid {} scores found to plot for {}", metric, output_path.display());
        return Err(MrexError::Plot(format!("No valid {} scores to plot", metric)));
    }

    let frame_indices: Vec<u32> = valid_scores.iter().map(|(i, _)| *i).collect();
    let mut score_values: Vec<f64> = valid_scores.iter().map(|(_, s)| *s).collect(); // Mutable for sorting

    // --- 2. Calculate Statistics ---
    let count = score_values.len();
    // Calculate mean manually
    let mean_val = if count > 0 { score_values.iter().sum::<f64>() / count as f64 } else { f64::NAN };

    // Sort the data first to easily get min/max and use indices for percentiles
    score_values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    // Use quantiles crate to get the *rank* (index) for percentiles
    let mut quantiles_data = CKMS::<f64>::new(0.001);
    for score in &score_values {
        quantiles_data.insert(*score);
    }

    // Helper to get value from sorted slice using rank from CKMS query
    let get_percentile_value = |quantile: f64| -> f64 {
        quantiles_data.query(quantile)
            .map(|(rank, _error_bound)| {
                // rank is usize index into the conceptual sorted stream
                // Clamp rank to valid indices of our sorted score_values Vec
                let index = (rank as usize).min(score_values.len().saturating_sub(1));
                score_values.get(index).copied().unwrap_or(f64::NAN)
            })
            .unwrap_or(f64::NAN)
    };

    let perc_1 = get_percentile_value(0.01);
    let perc_25 = get_percentile_value(0.25);
    let perc_75 = get_percentile_value(0.75);

    // Get min/max from the now sorted data
    let min_score = score_values.first().copied().unwrap_or(f64::NAN); // Already sorted
    let max_score = score_values.last().copied().unwrap_or(f64::NAN);

    // --- 3. Setup Plot ---
    // Adjust plot width slightly based on frame count? (Simpler: keep fixed for now)
    let root = BitMapBackend::new(output_path, (1280, 720)).into_drawing_area();
    root.fill(&WHITE).map_err(|e| MrexError::Plot(format!("Failed to fill plot background: {}", e)))?;

    // Determine plot range based on metric and data
    let (y_min, y_max) = match metric {
        "VMAF" => {
            // Start y-axis near 1st percentile, but not lower than e.g., 80 or 90 if perc_1 is high
            let lower_bound = perc_1.floor().max(0.0);
            (lower_bound.min(95.0), 100.5) // Ensure 100 is visible
        },
        "XPSNR" => {
            // Use min/max with some padding, potentially clamped
            let range_min = min_score.floor().max(20.0); // Don't usually go below 20dB
            let range_max = max_score.ceil().min(XPSNR_CAP_VALUE + 5.0); // Add padding, respect cap
            (range_min - 1.0, range_max + 1.0) // Add padding
        },
        _ => (min_score.floor(), max_score.ceil()), // Default fallback
    };
    let x_min = *frame_indices.first().unwrap_or(&0);
    let x_max = *frame_indices.last().unwrap_or(&count.try_into().unwrap_or(0)); // Use count if indices empty


    let mut chart = ChartBuilder::on(&root)
        .caption(format!("{} Scores ({} Valid Frames)", metric, count), ("sans-serif", 24).into_font())
        .margin(15) // Increased margin
        .x_label_area_size(50) // Increased label area size
        .y_label_area_size(70)
        .build_cartesian_2d(x_min..x_max, y_min..y_max)
        .map_err(|e| MrexError::Plot(format!("Failed to build chart: {}", e)))?;

    // --- 4. Configure Mesh and Grid ---
    let y_label_format = |y: &f64| format!("{:.1}", y); // Format y-axis labels
    let mut mesh = chart.configure_mesh();
    mesh.x_desc("Frame Number")
        .y_desc(format!("{} Score{}", metric, if metric == "XPSNR" { " (dB)" } else { "" }))
        .y_label_formatter(&y_label_format)
        .axis_desc_style(("sans-serif", 16))
        .label_style(("sans-serif", 14));

    // Custom grid lines based on metric
    match metric {
        "VMAF" => {
            mesh.y_max_light_lines(10) // More lines for 0-100 range
                .y_label_style(("sans-serif", 12).into_font().color(&BLACK.mix(0.6))) // Use IntoTextStyle
                .y_labels(10); // More labels
        },
        "XPSNR" => {
            mesh.y_max_light_lines(5) // Fewer lines for dB range
                .y_label_style(("sans-serif", 12).into_font().color(&BLACK.mix(0.6))) // Use IntoTextStyle
                .y_labels(5);
        },
        _ => {} // Default grid
    }
    mesh.draw().map_err(|e| MrexError::Plot(format!("Failed to draw mesh: {:?}", e)))?; // Use {:?} for error


    // --- 5. Draw Data Series and Stat Lines ---
    // Main score line
    chart.draw_series(LineSeries::new(
        valid_scores.iter().map(|(i, s)| (*i, *s)),
        BLUE.mix(0.8).stroke_width(1), // Pass color directly, not reference to ShapeStyle
    )).map_err(|e| MrexError::Plot(format!("Failed to draw main series: {:?}", e)))?
    .label(format!("{} Scores", metric))
    .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], BLUE.filled()));

    // Stat lines

    // Mean line
    chart.draw_series(LineSeries::new(vec![(x_min, mean_val), (x_max, mean_val)], get_stat_line_style(BLACK)))
         .map_err(|e| MrexError::Plot(format!("Failed to draw mean line: {:?}", e)))?
         .label(format!("Mean: {:.2}", mean_val)) // Use mean_val here
         .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], get_stat_line_style(BLACK)));
    // 1st Percentile line
    chart.draw_series(LineSeries::new(vec![(x_min, perc_1), (x_max, perc_1)], get_stat_line_style(RED)))
         .map_err(|e| MrexError::Plot(format!("Failed to draw 1% line: {:?}", e)))?
         .label(format!("1%:   {:.2}", perc_1))
         .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], get_stat_line_style(RED)));
    // 25th Percentile line
    chart.draw_series(LineSeries::new(vec![(x_min, perc_25), (x_max, perc_25)], get_stat_line_style(ORANGE)))
         .map_err(|e| MrexError::Plot(format!("Failed to draw 25% line: {:?}", e)))?
         .label(format!("25%: {:.2}", perc_25))
         .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], get_stat_line_style(ORANGE)));
    // 75th Percentile line
    chart.draw_series(LineSeries::new(vec![(x_min, perc_75), (x_max, perc_75)], get_stat_line_style(GREEN)))
         .map_err(|e| MrexError::Plot(format!("Failed to draw 75% line: {:?}", e)))?
         .label(format!("75%: {:.2}", perc_75))
         .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], get_stat_line_style(GREEN)));

    // --- 6. Configure Legend ---
    chart.configure_series_labels()
        .position(SeriesLabelPosition::LowerMiddle) // Position below plot
        .margin(10)
        .label_font(("sans-serif", 12))
        .background_style(WHITE.mix(0.8))
        .border_style(BLACK)
        .draw()
        .map_err(|e| MrexError::Plot(format!("Failed to draw legend: {:?}", e)))?;

    // --- 7. Finalize ---
    root.present().map_err(|e| MrexError::Plot(format!("Failed to save plot: {:?}", e)))?;
    info!("Successfully generated {} plot: {}", metric, output_path.display());

    Ok(())
}

/// Helper function to create a simple line style for statistics lines.
fn get_stat_line_style(color: RGBColor) -> ShapeStyle {
    ShapeStyle {
        color: color.to_rgba(),
        filled: false,
        stroke_width: 1,
    }
}