//! Metrics for comparing simulation trajectories.
//!
//! Provides quantitative measures for validating MPM simulations against reference data.

use crate::harness::{SimulationSnapshot, SimulationTrajectory};
use plotters::prelude::*;
use rstar::{RTree, RTreeObject, AABB, PointDistance};
use serde::{Deserialize, Serialize};
use std::path::Path;

/// A particle point for RTree spatial indexing.
#[derive(Clone, Copy)]
struct IndexedPoint {
    position: [f32; 3],
    index: usize,
}

impl RTreeObject for IndexedPoint {
    type Envelope = AABB<[f32; 3]>;

    fn envelope(&self) -> Self::Envelope {
        AABB::from_point(self.position)
    }
}

impl PointDistance for IndexedPoint {
    fn distance_2(&self, point: &[f32; 3]) -> f32 {
        let dx = self.position[0] - point[0];
        let dy = self.position[1] - point[1];
        let dz = self.position[2] - point[2];
        dx * dx + dy * dy + dz * dz
    }
}

/// Comparison metrics between two simulation trajectories.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ComparisonMetrics {
    /// Name of the test scenario.
    pub scenario: String,
    /// Maximum position error across all particles and timesteps (m).
    pub max_position_error: f32,
    /// Mean position error across all particles and timesteps (m).
    pub mean_position_error: f32,
    /// Root mean squared position error (m).
    pub rmse_position: f32,
    /// Per-timestep position errors (mean across particles).
    pub position_errors_over_time: Vec<f32>,
    /// Center of mass trajectory error.
    pub com_trajectory_error: f32,
    /// Final center of mass error.
    pub final_com_error: f32,
}

impl ComparisonMetrics {
    /// Compare two simulation trajectories and compute error metrics.
    ///
    /// Particles are matched by finding the closest particle in the initial snapshot,
    /// then tracking that matching throughout the simulation.
    pub fn compare(
        scenario: &str,
        reference: &SimulationTrajectory,
        test: &SimulationTrajectory,
    ) -> Result<Self, String> {
        if reference.snapshots.is_empty() || test.snapshots.is_empty() {
            return Err("Trajectories must have at least one snapshot".to_string());
        }

        // Warn about particle count mismatch
        let ref_particles = reference.snapshots[0].particles.len();
        let test_particles = test.snapshots[0].particles.len();
        if ref_particles != test_particles {
            eprintln!(
                "  Note: Particle count mismatch ({} vs {}), comparing {} particles",
                ref_particles,
                test_particles,
                ref_particles.min(test_particles)
            );
        }

        // Build particle matching from initial snapshot positions
        // This matches each reference particle to its closest test particle
        let initial_ref = &reference.snapshots[0];
        let initial_test = find_closest_snapshot(&test.snapshots, initial_ref.time);
        let matching = build_particle_matching(initial_ref, initial_test);

        // Determine the end time (earliest end of either simulation)
        let ref_end_time = reference.snapshots.last().map(|s| s.time).unwrap_or(0.0);
        let test_end_time = test.snapshots.last().map(|s| s.time).unwrap_or(0.0);
        let end_time = ref_end_time.min(test_end_time);

        // Match snapshots by closest time, stopping at the earliest end time
        let mut position_errors = Vec::new();
        let mut position_errors_over_time = Vec::new();

        for ref_snap in &reference.snapshots {
            // Stop if we've exceeded the end time of either simulation
            if ref_snap.time > end_time {
                break;
            }

            // Find closest test snapshot
            let test_snap = find_closest_snapshot(&test.snapshots, ref_snap.time);

            let pos_errors = compute_particle_errors(ref_snap, test_snap, &matching);

            if pos_errors.is_empty() {
                continue;
            }

            position_errors.extend(pos_errors.iter());

            // Mean position error for this timestep
            let mean_pos_err = pos_errors.iter().sum::<f32>() / pos_errors.len() as f32;
            position_errors_over_time.push(mean_pos_err);
        }

        if position_errors.is_empty() {
            return Err("No particles to compare".to_string());
        }

        let max_position_error = position_errors
            .iter()
            .cloned()
            .fold(0.0f32, f32::max);
        let mean_position_error = position_errors.iter().sum::<f32>() / position_errors.len() as f32;
        let rmse_position = (position_errors.iter().map(|e| e * e).sum::<f32>()
            / position_errors.len() as f32)
            .sqrt();

        // Center of mass trajectory error (only up to end_time)
        let mut com_errors: Vec<f32> = Vec::new();
        for ref_snap in &reference.snapshots {
            if ref_snap.time > end_time {
                break;
            }
            let test_snap = find_closest_snapshot(&test.snapshots, ref_snap.time);
            let ref_com = compute_com(ref_snap);
            let test_com = compute_com(test_snap);
            let dx = ref_com[0] - test_com[0];
            let dy = ref_com[1] - test_com[1];
            let dz = ref_com[2] - test_com[2];
            com_errors.push((dx * dx + dy * dy + dz * dz).sqrt());
        }

        let com_trajectory_error = if com_errors.is_empty() {
            0.0
        } else {
            com_errors.iter().sum::<f32>() / com_errors.len() as f32
        };
        let final_com_error = *com_errors.last().unwrap_or(&0.0);

        Ok(ComparisonMetrics {
            scenario: scenario.to_string(),
            max_position_error,
            mean_position_error,
            rmse_position,
            position_errors_over_time,
            com_trajectory_error,
            final_com_error,
        })
    }

    /// Check if the test passes with the given tolerance.
    pub fn passes(&self, position_tolerance: f32) -> bool {
        self.mean_position_error <= position_tolerance
    }

    /// Generate a human-readable report.
    pub fn report(&self) -> String {
        format!(
            r#"Validation Report: {}
================================
Position Metrics:
  Max Error:  {:.6} m
  Mean Error: {:.6} m
  RMSE:       {:.6} m

Center of Mass:
  Trajectory Error: {:.6} m
  Final Error:      {:.6} m
"#,
            self.scenario,
            self.max_position_error,
            self.mean_position_error,
            self.rmse_position,
            self.com_trajectory_error,
            self.final_com_error
        )
    }

    /// Export metrics to JSON.
    pub fn export_json(&self, path: &std::path::Path) -> Result<(), Box<dyn std::error::Error>> {
        let json = serde_json::to_string_pretty(self)?;
        std::fs::write(path, json)?;
        Ok(())
    }
}

fn find_closest_snapshot(snapshots: &[SimulationSnapshot], target_time: f32) -> &SimulationSnapshot {
    snapshots
        .iter()
        .min_by(|a, b| {
            let da = (a.time - target_time).abs();
            let db = (b.time - target_time).abs();
            da.partial_cmp(&db).unwrap()
        })
        .unwrap()
}

/// Build a particle matching between reference and test snapshots using an R-tree.
/// Returns indices mapping each reference particle to its closest test particle.
fn build_particle_matching(
    reference: &SimulationSnapshot,
    test: &SimulationSnapshot,
) -> Vec<usize> {
    if test.particles.is_empty() {
        return Vec::new();
    }

    // Build R-tree from test particles
    let points: Vec<IndexedPoint> = test
        .particles
        .iter()
        .enumerate()
        .map(|(i, p)| IndexedPoint {
            position: p.position,
            index: i,
        })
        .collect();
    let rtree = RTree::bulk_load(points);

    // Find nearest neighbor for each reference particle
    let mut matching = Vec::with_capacity(reference.particles.len());
    for ref_p in &reference.particles {
        if let Some(nearest) = rtree.nearest_neighbor(&ref_p.position) {
            matching.push(nearest.index);
        } else {
            matching.push(0); // Fallback, shouldn't happen if test is non-empty
        }
    }

    matching
}

fn compute_particle_errors(
    reference: &SimulationSnapshot,
    test: &SimulationSnapshot,
    matching: &[usize],
) -> Vec<f32> {
    if reference.particles.is_empty() || test.particles.is_empty() || matching.is_empty() {
        return Vec::new();
    }

    let mut pos_errors = Vec::with_capacity(matching.len());

    for (ref_idx, &test_idx) in matching.iter().enumerate() {
        let ref_p = &reference.particles[ref_idx];
        let test_p = &test.particles[test_idx];

        // Position error (Euclidean distance)
        let dx = ref_p.position[0] - test_p.position[0];
        let dy = ref_p.position[1] - test_p.position[1];
        let dz = ref_p.position[2] - test_p.position[2];
        pos_errors.push((dx * dx + dy * dy + dz * dz).sqrt());
    }

    pos_errors
}

fn compute_com(snapshot: &SimulationSnapshot) -> [f32; 3] {
    if snapshot.particles.is_empty() {
        return [0.0, 0.0, 0.0];
    }

    let n = snapshot.particles.len() as f32;
    let sum: [f32; 3] = snapshot.particles.iter().fold([0.0, 0.0, 0.0], |acc, p| {
        [
            acc[0] + p.position[0],
            acc[1] + p.position[1],
            acc[2] + p.position[2],
        ]
    });

    [sum[0] / n, sum[1] / n, sum[2] / n]
}

/// Compute the angle of repose from a settled granular pile.
/// Returns the angle in degrees.
pub fn compute_angle_of_repose(snapshot: &SimulationSnapshot) -> f32 {
    if snapshot.particles.is_empty() {
        return 0.0;
    }

    // Find the pile boundaries
    let min_y = snapshot
        .particles
        .iter()
        .map(|p| p.position[1])
        .fold(f32::MAX, f32::min);
    let max_y = snapshot
        .particles
        .iter()
        .map(|p| p.position[1])
        .fold(f32::MIN, f32::max);

    // Sample points at different heights and find horizontal extent
    let height = max_y - min_y;
    if height < 0.01 {
        return 0.0;
    }

    // Compute average radius at base
    let base_particles: Vec<_> = snapshot
        .particles
        .iter()
        .filter(|p| p.position[1] < min_y + height * 0.1)
        .collect();

    if base_particles.is_empty() {
        return 0.0;
    }

    let base_radius = base_particles
        .iter()
        .map(|p| {
            let x = p.position[0];
            let z = p.position[2];
            (x * x + z * z).sqrt()
        })
        .fold(0.0f32, f32::max);

    // Angle of repose = atan(height / base_radius)
    if base_radius > 0.01 {
        (height / base_radius).atan().to_degrees()
    } else {
        90.0
    }
}

/// Compute the runout distance of a collapsed column.
/// Returns the maximum horizontal distance from the original position.
pub fn compute_runout_distance(
    initial: &SimulationSnapshot,
    final_state: &SimulationSnapshot,
) -> f32 {
    if initial.particles.is_empty() || final_state.particles.is_empty() {
        return 0.0;
    }

    // Compute initial center of mass (horizontal)
    let initial_com = compute_com(initial);

    // Find maximum horizontal distance from initial COM
    final_state
        .particles
        .iter()
        .map(|p| {
            let dx = p.position[0] - initial_com[0];
            let dz = p.position[2] - initial_com[2];
            (dx * dx + dz * dz).sqrt()
        })
        .fold(0.0f32, f32::max)
}

/// Compute the maximum deflection of a beam (for cantilever tests).
pub fn compute_beam_deflection(
    initial: &SimulationSnapshot,
    current: &SimulationSnapshot,
    tip_region_fraction: f32,
) -> f32 {
    if initial.particles.is_empty() || current.particles.is_empty() {
        return 0.0;
    }

    // Find the rightmost particles (beam tip)
    let max_x = initial
        .particles
        .iter()
        .map(|p| p.position[0])
        .fold(f32::MIN, f32::max);
    let min_x = initial
        .particles
        .iter()
        .map(|p| p.position[0])
        .fold(f32::MAX, f32::min);

    let tip_threshold = max_x - (max_x - min_x) * tip_region_fraction;

    // Find tip particles in initial and current states
    let initial_tip_y: f32 = initial
        .particles
        .iter()
        .filter(|p| p.position[0] > tip_threshold)
        .map(|p| p.position[1])
        .sum::<f32>()
        / initial
            .particles
            .iter()
            .filter(|p| p.position[0] > tip_threshold)
            .count() as f32;

    let current_tip_y: f32 = current
        .particles
        .iter()
        .filter(|p| p.position[0] > tip_threshold)
        .map(|p| p.position[1])
        .sum::<f32>()
        / current
            .particles
            .iter()
            .filter(|p| p.position[0] > tip_threshold)
            .count()
            .max(1) as f32;

    (initial_tip_y - current_tip_y).abs()
}

/// Compute kinetic energy of the system.
pub fn compute_kinetic_energy(snapshot: &SimulationSnapshot, particle_mass: f32) -> f32 {
    snapshot
        .particles
        .iter()
        .map(|p| {
            let v2 = p.velocity[0].powi(2) + p.velocity[1].powi(2) + p.velocity[2].powi(2);
            0.5 * particle_mass * v2
        })
        .sum()
}

/// Compute potential energy of the system.
pub fn compute_potential_energy(
    snapshot: &SimulationSnapshot,
    particle_mass: f32,
    gravity: f32,
    reference_height: f32,
) -> f32 {
    snapshot
        .particles
        .iter()
        .map(|p| particle_mass * gravity * (p.position[1] - reference_height))
        .sum()
}

/// Generate comparison charts for two trajectories.
///
/// Creates a 4-panel PNG image with:
/// 1. Position error over time
/// 2. Center of mass Y trajectory comparison
/// 3. Final particle distribution (X-Y projection)
/// 4. Summary statistics
pub fn generate_comparison_chart(
    scenario: &str,
    slosh: &SimulationTrajectory,
    genesis: &SimulationTrajectory,
    metrics: &ComparisonMetrics,
    output_path: &Path,
) -> Result<(), Box<dyn std::error::Error>> {
    // Determine the end time (earliest end of either simulation)
    let slosh_end_time = slosh.snapshots.last().map(|s| s.time).unwrap_or(0.0);
    let genesis_end_time = genesis.snapshots.last().map(|s| s.time).unwrap_or(0.0);
    let end_time = slosh_end_time.min(genesis_end_time);

    let root = BitMapBackend::new(output_path, (1200, 1000)).into_drawing_area();
    root.fill(&WHITE)?;

    let areas = root.split_evenly((2, 2));

    // Panel 1: Position error over time
    draw_position_error_chart(&areas[0], slosh, metrics, end_time)?;

    // Panel 2: Center of mass Y trajectory
    draw_com_trajectory_chart(&areas[1], slosh, genesis, end_time)?;

    // Panel 3: Final particle distribution (X-Y) at end_time
    draw_particle_distribution(&areas[2], slosh, genesis, end_time)?;

    // Panel 4: Summary statistics
    draw_summary_panel(&areas[3], scenario, slosh, genesis, metrics, end_time)?;

    root.present()?;
    Ok(())
}

fn draw_position_error_chart(
    area: &DrawingArea<BitMapBackend, plotters::coord::Shift>,
    slosh: &SimulationTrajectory,
    metrics: &ComparisonMetrics,
    end_time: f32,
) -> Result<(), Box<dyn std::error::Error>> {
    let times: Vec<f32> = slosh
        .snapshots
        .iter()
        .take(metrics.position_errors_over_time.len())
        .filter(|s| s.time <= end_time)
        .map(|s| s.time)
        .collect();

    if times.is_empty() || metrics.position_errors_over_time.is_empty() {
        return Ok(());
    }

    let max_time = times.iter().cloned().fold(0.0f32, f32::max);
    let max_error = metrics
        .position_errors_over_time
        .iter()
        .cloned()
        .fold(0.0f32, f32::max)
        * 1.1; // Add 10% margin

    let mut chart = ChartBuilder::on(area)
        .caption("Position Error Over Time", ("sans-serif", 20))
        .margin(10)
        .x_label_area_size(40)
        .y_label_area_size(60)
        .build_cartesian_2d(0.0f32..max_time, 0.0f32..max_error)?;

    chart
        .configure_mesh()
        .x_desc("Time (s)")
        .y_desc("Mean Position Error (m)")
        .draw()?;

    let data: Vec<(f32, f32)> = times
        .iter()
        .zip(metrics.position_errors_over_time.iter())
        .map(|(&t, &e)| (t, e))
        .collect();

    chart.draw_series(LineSeries::new(data, &BLUE))?;

    Ok(())
}

fn draw_com_trajectory_chart(
    area: &DrawingArea<BitMapBackend, plotters::coord::Shift>,
    slosh: &SimulationTrajectory,
    genesis: &SimulationTrajectory,
    end_time: f32,
) -> Result<(), Box<dyn std::error::Error>> {
    let slosh_data: Vec<(f32, f32)> = slosh
        .snapshots
        .iter()
        .filter(|s| s.time <= end_time)
        .map(|s| (s.time, compute_com(s)[1]))
        .collect();

    let genesis_data: Vec<(f32, f32)> = genesis
        .snapshots
        .iter()
        .filter(|s| s.time <= end_time)
        .map(|s| (s.time, compute_com(s)[1]))
        .collect();

    if slosh_data.is_empty() && genesis_data.is_empty() {
        return Ok(());
    }

    let all_times: Vec<f32> = slosh_data
        .iter()
        .map(|(t, _)| *t)
        .chain(genesis_data.iter().map(|(t, _)| *t))
        .collect();
    let all_y: Vec<f32> = slosh_data
        .iter()
        .map(|(_, y)| *y)
        .chain(genesis_data.iter().map(|(_, y)| *y))
        .collect();

    let max_time = all_times.iter().cloned().fold(0.0f32, f32::max);
    let min_y = all_y.iter().cloned().fold(f32::MAX, f32::min);
    let max_y = all_y.iter().cloned().fold(f32::MIN, f32::max);
    let y_margin = (max_y - min_y) * 0.1;

    let mut chart = ChartBuilder::on(area)
        .caption("Center of Mass Height", ("sans-serif", 20))
        .margin(10)
        .x_label_area_size(40)
        .y_label_area_size(60)
        .build_cartesian_2d(0.0f32..max_time, (min_y - y_margin)..(max_y + y_margin))?;

    chart
        .configure_mesh()
        .x_desc("Time (s)")
        .y_desc("COM Y Position (m)")
        .draw()?;

    chart
        .draw_series(LineSeries::new(slosh_data, &BLUE))?
        .label("Slosh")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &BLUE));

    chart
        .draw_series(LineSeries::new(genesis_data, &RED))?
        .label("Genesis")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &RED));

    chart
        .configure_series_labels()
        .background_style(&WHITE.mix(0.8))
        .border_style(&BLACK)
        .draw()?;

    Ok(())
}

fn draw_particle_distribution(
    area: &DrawingArea<BitMapBackend, plotters::coord::Shift>,
    slosh: &SimulationTrajectory,
    genesis: &SimulationTrajectory,
    end_time: f32,
) -> Result<(), Box<dyn std::error::Error>> {
    // Get snapshot closest to end_time for each trajectory
    let slosh_final = slosh
        .snapshots
        .iter()
        .filter(|s| s.time <= end_time)
        .last();
    let genesis_final = genesis
        .snapshots
        .iter()
        .filter(|s| s.time <= end_time)
        .last();

    if slosh_final.is_none() && genesis_final.is_none() {
        return Ok(());
    }

    // Collect all positions to determine bounds
    let mut all_x: Vec<f32> = Vec::new();
    let mut all_y: Vec<f32> = Vec::new();

    if let Some(snap) = slosh_final {
        for p in &snap.particles {
            all_x.push(p.position[0]);
            all_y.push(p.position[1]);
        }
    }
    if let Some(snap) = genesis_final {
        for p in &snap.particles {
            all_x.push(p.position[0]);
            all_y.push(p.position[1]);
        }
    }

    if all_x.is_empty() {
        return Ok(());
    }

    let min_x = all_x.iter().cloned().fold(f32::MAX, f32::min);
    let max_x = all_x.iter().cloned().fold(f32::MIN, f32::max);
    let min_y = all_y.iter().cloned().fold(f32::MAX, f32::min);
    let max_y = all_y.iter().cloned().fold(f32::MIN, f32::max);

    // Use the same scale on both axes
    let x_range = max_x - min_x;
    let y_range = max_y - min_y;
    let max_range = x_range.max(y_range) * 1.1; // Add 10% margin

    let x_center = (min_x + max_x) / 2.0;
    let y_center = (min_y + max_y) / 2.0;

    let half_range = max_range / 2.0;

    let mut chart = ChartBuilder::on(area)
        .caption("Final Particle Distribution (X-Y)", ("sans-serif", 20))
        .margin(10)
        .x_label_area_size(40)
        .y_label_area_size(60)
        .build_cartesian_2d(
            (x_center - half_range)..(x_center + half_range),
            (y_center - half_range)..(y_center + half_range),
        )?;

    chart
        .configure_mesh()
        .x_desc("X (m)")
        .y_desc("Y (m)")
        .draw()?;

    // Draw Slosh particles
    if let Some(snap) = slosh_final {
        let points: Vec<(f32, f32)> = snap
            .particles
            .iter()
            .map(|p| (p.position[0], p.position[1]))
            .collect();
        chart
            .draw_series(
                points
                    .iter()
                    .map(|&(x, y)| Circle::new((x, y), 2, BLUE.mix(0.5).filled())),
            )?
            .label("Slosh")
            .legend(|(x, y)| Circle::new((x, y), 4, BLUE.filled()));
    }

    // Draw Genesis particles
    if let Some(snap) = genesis_final {
        let points: Vec<(f32, f32)> = snap
            .particles
            .iter()
            .map(|p| (p.position[0], p.position[1]))
            .collect();
        chart
            .draw_series(
                points
                    .iter()
                    .map(|&(x, y)| Circle::new((x, y), 2, RED.mix(0.5).filled())),
            )?
            .label("Genesis")
            .legend(|(x, y)| Circle::new((x, y), 4, RED.filled()));
    }

    chart
        .configure_series_labels()
        .background_style(&WHITE.mix(0.8))
        .border_style(&BLACK)
        .draw()?;

    Ok(())
}

fn draw_summary_panel(
    area: &DrawingArea<BitMapBackend, plotters::coord::Shift>,
    scenario: &str,
    slosh: &SimulationTrajectory,
    genesis: &SimulationTrajectory,
    metrics: &ComparisonMetrics,
    end_time: f32,
) -> Result<(), Box<dyn std::error::Error>> {
    area.fill(&WHITE)?;

    // Get snapshot closest to end_time for particle counts
    let slosh_final = slosh.snapshots.iter().filter(|s| s.time <= end_time).last();
    let genesis_final = genesis.snapshots.iter().filter(|s| s.time <= end_time).last();

    let slosh_particles = slosh_final.map(|s| s.particles.len()).unwrap_or(0);
    let genesis_particles = genesis_final.map(|s| s.particles.len()).unwrap_or(0);

    let lines = [
        format!("Validation: {}", scenario),
        String::new(),
        "Position Errors:".to_string(),
        format!("  Maximum: {:.6} m", metrics.max_position_error),
        format!("  Mean:    {:.6} m", metrics.mean_position_error),
        format!("  RMSE:    {:.6} m", metrics.rmse_position),
        String::new(),
        "Center of Mass:".to_string(),
        format!("  Mean Error:  {:.6} m", metrics.com_trajectory_error),
        format!("  Final Error: {:.6} m", metrics.final_com_error),
        String::new(),
        "Particle Count:".to_string(),
        format!("  Slosh:   {}", slosh_particles),
        format!("  Genesis: {}", genesis_particles),
        String::new(),
        format!("Compared up to: {:.2} s", end_time),
    ];

    let style = TextStyle::from(("monospace", 16)).color(&BLACK);
    let line_height = 20;
    let start_y = 30;
    let start_x = 30;

    for (i, line) in lines.iter().enumerate() {
        area.draw(&Text::new(
            line.clone(),
            (start_x, start_y + i as i32 * line_height),
            &style,
        ))?;
    }

    Ok(())
}
