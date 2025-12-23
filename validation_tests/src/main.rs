//! Validation test runner for Slosh MPM.
//!
//! This binary runs all validation scenarios and exports results for comparison
//! with Taichi Elements.
//!
//! Usage:
//!   cargo run -p slosh_validation_tests --features "webgpu runtime render" -- [OPTIONS]
//!
//! Options:
//!   --scenario <NAME>    Run a specific scenario (or "all")
//!   --output-dir <PATH>  Output directory for results
//!   --render             Render simulation results after running (requires "render" feature)
//!   --render-only        Render existing results without running simulation

use slosh_validation::harness::*;
use slosh_validation::metrics::{generate_comparison_chart, ComparisonMetrics};
use slosh_validation::scenarios::{
    bouncing_ball::{bouncing_ball_scenario, BouncingBallParams},
    dam_break::{dam_break_scenario, DamBreakParams},
    elastic_beam::{elastic_beam_scenario, ElasticBeamParams},
    sand_column::{sand_column_scenario, SandColumnParams},
};
use std::path::PathBuf;

fn main() {
    let args: Vec<String> = std::env::args().collect();

    let mut scenario_filter = "all".to_string();
    let mut output_dir = PathBuf::from("validation_results");
    let mut render_after = false;
    let mut render_only = false;
    let mut render_files: Vec<PathBuf> = Vec::new();
    let mut compare_mode = false;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--scenario" => {
                i += 1;
                if i < args.len() {
                    scenario_filter = args[i].clone();
                }
            }
            "--output-dir" => {
                i += 1;
                if i < args.len() {
                    output_dir = PathBuf::from(&args[i]);
                }
            }
            "--render" => {
                render_after = true;
            }
            "--render-only" => {
                render_only = true;
            }
            "--compare" => {
                compare_mode = true;
            }
            "--file" => {
                i += 1;
                if i < args.len() {
                    render_files.push(PathBuf::from(&args[i]));
                    render_only = true; // --file implies render-only
                }
            }
            "--help" | "-h" => {
                print_help();
                return;
            }
            _ => {
                eprintln!("Unknown argument: {}", args[i]);
                print_help();
                return;
            }
        }
        i += 1;
    }

    // Handle compare mode
    if compare_mode {
        compare_results(&scenario_filter, &output_dir);
        return;
    }

    #[cfg(not(feature = "render"))]
    if render_after || render_only {
        eprintln!("Error: --render and --render-only require the 'render' feature.");
        eprintln!("Rerun with: cargo run -p slosh_validation_tests --features \"webgpu runtime render\" -- ...");
        return;
    }

    std::fs::create_dir_all(&output_dir).expect("Failed to create output directory");

    // Run Slosh simulations (unless --render-only)
    if !render_only {
        println!("\nRunning Slosh simulations...");
        pollster::block_on(run_slosh_simulations(&scenario_filter, &output_dir));
    }

    // Render results if requested
    #[cfg(feature = "render")]
    if render_after || render_only {
        render_results(&scenario_filter, &output_dir, &render_files);
    }
}

fn compare_results(scenario_filter: &str, output_dir: &PathBuf) {
    let scenarios = ["elastic_beam", "sand_column", "dam_break", "bouncing_ball"];

    let scenarios_to_compare: Vec<&str> = if scenario_filter == "all" {
        scenarios.to_vec()
    } else {
        vec![scenario_filter]
    };

    let mut all_metrics: Vec<ComparisonMetrics> = Vec::new();

    for scenario in &scenarios_to_compare {
        let slosh_path = output_dir.join(format!("{}_slosh.json", scenario));
        let genesis_path = output_dir.join(format!("{}_genesis.json", scenario));

        if !slosh_path.exists() {
            eprintln!("Skipping {}: Slosh results not found at {:?}", scenario, slosh_path);
            continue;
        }

        if !genesis_path.exists() {
            eprintln!("Skipping {}: Genesis results not found at {:?}", scenario, genesis_path);
            eprintln!("  Run: python validation_tests/genesis/{}.py", scenario);
            continue;
        }

        println!("\nComparing: {}", scenario);

        let slosh = match SimulationTrajectory::load_json(&slosh_path) {
            Ok(t) => t,
            Err(e) => {
                eprintln!("  Failed to load Slosh trajectory: {}", e);
                continue;
            }
        };

        let genesis = match SimulationTrajectory::load_json(&genesis_path) {
            Ok(t) => t,
            Err(e) => {
                eprintln!("  Failed to load Genesis trajectory: {}", e);
                continue;
            }
        };

        match ComparisonMetrics::compare(scenario, &slosh, &genesis) {
            Ok(metrics) => {
                println!("{}", metrics.report());

                // Generate comparison chart
                let comparison_dir = output_dir.join("comparison");
                std::fs::create_dir_all(&comparison_dir).ok();
                let chart_path = comparison_dir.join(format!("{}_comparison.png", scenario));
                match generate_comparison_chart(scenario, &slosh, &genesis, &metrics, &chart_path) {
                    Ok(()) => println!("Saved chart: {:?}", chart_path),
                    Err(e) => eprintln!("  Failed to generate chart: {}", e),
                }

                all_metrics.push(metrics);
            }
            Err(e) => {
                eprintln!("  Comparison failed: {}", e);
            }
        }
    }

    // Print overall summary
    if !all_metrics.is_empty() {
        println!("\n{}", "=".repeat(80));
        println!("OVERALL VALIDATION SUMMARY");
        println!("{}", "=".repeat(80));
        println!(
            "{:<25} {:<15} {:<15} {:<15}",
            "Scenario", "Mean Pos Err", "RMSE", "Final COM Err"
        );
        println!("{}", "-".repeat(80));
        for m in &all_metrics {
            println!(
                "{:<25} {:<15.6} {:<15.6} {:<15.6}",
                m.scenario, m.mean_position_error, m.rmse_position, m.final_com_error
            );
        }

        // Save summary JSON
        let comparison_dir = output_dir.join("comparison");
        std::fs::create_dir_all(&comparison_dir).ok();

        let summary = serde_json::json!({
            "scenarios": all_metrics.iter().map(|m| {
                serde_json::json!({
                    "name": m.scenario,
                    "max_position_error": m.max_position_error,
                    "mean_position_error": m.mean_position_error,
                    "rmse_position": m.rmse_position,
                    "final_com_error": m.final_com_error
                })
            }).collect::<Vec<_>>()
        });

        let summary_path = comparison_dir.join("summary.json");
        if let Err(e) = std::fs::write(&summary_path, serde_json::to_string_pretty(&summary).unwrap()) {
            eprintln!("\nFailed to save summary: {}", e);
        } else {
            println!("\nSaved summary: {:?}", summary_path);
        }
    }
}

#[cfg(feature = "render")]
fn render_results(scenario_filter: &str, output_dir: &PathBuf, render_files: &[PathBuf]) {
    use slosh_validation::viewer::{TrajectoryViewer, ViewerConfig};

    // If specific files were provided, use them directly
    let json_paths: Vec<PathBuf> = if !render_files.is_empty() {
        // Validate all files exist
        for path in render_files {
            if !path.exists() {
                eprintln!("File not found: {:?}", path);
                return;
            }
        }
        render_files.to_vec()
    } else {
        // Determine which scenario to render
        let scenario_to_render = if scenario_filter == "all" {
            // If "all", show a menu or pick the first available
            let scenarios = ["elastic_beam", "sand_column", "dam_break", "bouncing_ball", "twisting_cube", "rigid_coupling"];
            let mut found = None;
            for name in &scenarios {
                let json_path = output_dir.join(format!("{}.json", name));
                if json_path.exists() {
                    found = Some(name.to_string());
                    break;
                }
            }
            match found {
                Some(name) => name,
                None => {
                    eprintln!("No simulation results found in {:?}", output_dir);
                    eprintln!("Run a simulation first, or specify --scenario <name> or --file <path>");
                    return;
                }
            }
        } else {
            scenario_filter.to_string()
        };

        let path = output_dir.join(format!("{}_slosh.json", scenario_to_render));
        if !path.exists() {
            eprintln!("Results file not found: {:?}", path);
            eprintln!("Run the simulation first: cargo run -p slosh_validation_tests --features \"webgpu runtime\" -- --scenario {}", scenario_to_render);
            return;
        }
        vec![path]
    };

    if json_paths.len() == 1 {
        println!("\nLaunching viewer for: {:?}", json_paths[0]);
    } else {
        println!("\nLaunching viewer to compare {} trajectories:", json_paths.len());
        for path in &json_paths {
            println!("  - {:?}", path);
        }
    }

    let title = if json_paths.len() == 1 {
        json_paths[0]
            .file_stem()
            .map(|s| s.to_string_lossy().to_string())
            .unwrap_or_else(|| "Trajectory".to_string())
    } else {
        format!("Comparing {} trajectories", json_paths.len())
    };

    let config = ViewerConfig {
        title: format!("Slosh Validation: {}", title),
        ..Default::default()
    };

    let paths: Vec<&std::path::Path> = json_paths.iter().map(|p| p.as_path()).collect();
    match TrajectoryViewer::from_json_multi(&paths, config) {
        Ok(viewer) => viewer.run(),
        Err(e) => eprintln!("Failed to load trajectory: {}", e),
    }
}

async fn run_slosh_simulations(scenario_filter: &str, output_dir: &PathBuf) {
    let harness = match ValidationHarness::new().await {
        Ok(h) => h,
        Err(e) => {
            eprintln!("Failed to initialize GPU: {}", e);
            eprintln!("Make sure you have a compatible GPU and the correct features enabled.");
            return;
        }
    };

    std::fs::create_dir_all(&output_dir).expect("Failed to create slosh results directory");

    let scenarios: Vec<(&str, ScenarioConfig)> = vec![
        ("elastic_beam", elastic_beam_scenario(ElasticBeamParams::default())),
        ("sand_column", sand_column_scenario(SandColumnParams::default())),
        ("dam_break", dam_break_scenario(DamBreakParams::default())),
        ("bouncing_ball", bouncing_ball_scenario(BouncingBallParams::default())),
    ];

    for (name, config) in scenarios {
        if scenario_filter != "all" && scenario_filter != name {
            continue;
        }

        println!("\nRunning scenario: {}", name);
        println!("  Particles: {}", config.particles.len());
        println!("  Steps: {}", config.total_steps);

        match harness.run_scenario(config).await {
            Ok(trajectory) => {
                let json_path = output_dir.join(format!("{}_slosh.json", name));
                if let Err(e) = trajectory.export_json(&json_path) {
                    eprintln!("  Failed to export JSON: {}", e);
                } else {
                    println!("  Exported: {}", json_path.display());
                }

                let csv_dir = output_dir.join(format!("{}_slosh_csv", name));
                if let Err(e) = trajectory.export_csv(&csv_dir) {
                    eprintln!("  Failed to export CSV: {}", e);
                } else {
                    println!("  Exported CSV to: {}", csv_dir.display());
                }
            }
            Err(e) => {
                eprintln!("  Simulation failed: {}", e);
            }
        }
    }

    println!("\nSlosh simulations complete!");
}

fn print_help() {
    println!(
        r#"
Slosh MPM Validation Test Runner

Usage:
  cargo run -p slosh_validation_tests --features "webgpu runtime" -- [OPTIONS]
  cargo run -p slosh_validation_tests --features "webgpu runtime render" -- --render [OPTIONS]

Options:
  --scenario <NAME>     Run a specific scenario. Options:
                          all (default), elastic_beam, sand_column,
                          dam_break, bouncing_ball

  --output-dir <PATH>   Output directory for results (default: validation_results)

  --compare             Compare Slosh and Genesis results, computing error metrics.
                        Looks for <scenario>_slosh.json and <scenario>_genesis.json
                        in the output directory. Outputs a summary to comparison/summary.json.

  --render              Render simulation results after running
                        (requires "render" feature)

  --render-only         Render existing results without running simulation
                        (requires "render" feature)

  --file <PATH>         Render a specific JSON trajectory file (Slosh or Genesis)
                        (requires "render" feature, implies --render-only)
                        Can be specified multiple times to compare trajectories

  --help, -h            Show this help message

Examples:
  # Run all scenarios
  cargo run -p slosh_validation_tests --features "webgpu runtime"

  # Run only the elastic beam test
  cargo run -p slosh_validation_tests --features "webgpu runtime" -- --scenario elastic_beam

  # Run elastic beam and render results
  cargo run -p slosh_validation_tests --features "webgpu runtime render" -- --scenario elastic_beam --render

  # Compare Slosh vs Genesis results (compute error metrics)
  cargo run -p slosh_validation_tests -- --compare
  cargo run -p slosh_validation_tests -- --compare --scenario elastic_beam

  # Render existing Slosh results (no simulation)
  cargo run -p slosh_validation_tests --features "render" -- --render-only --scenario elastic_beam

  # Render a Genesis-generated JSON file
  cargo run -p slosh_validation_tests --features "render" -- --file results/elastic_beam_genesis.json

  # Visual comparison: overlay Slosh and Genesis particles (different colors)
  cargo run -p slosh_validation_tests --features "render" -- \
    --file validation_results/elastic_beam_slosh.json \
    --file validation_results/elastic_beam_genesis.json

Viewer Controls:
  Space           Play/Pause
  Left/Right      Step through snapshots
  R               Reset to beginning
  Mouse drag      Rotate camera
  Scroll          Zoom

Workflow:
  1. Run this tool to generate Slosh results
  2. Run the Genesis scripts: python validation_tests/genesis/<scenario>.py
  3. Compare results: --compare
  4. Visual comparison: use multiple --file arguments to overlay trajectories
"#
    );
}
