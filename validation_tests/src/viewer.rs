//! Trajectory playback viewer using KISS3D.
//!
//! Provides visualization of simulation results stored in JSON trajectory files.
//! Supports rendering multiple trajectories simultaneously for visual comparison.

use crate::harness::SimulationTrajectory;
use kiss3d::camera::ArcBall;
use kiss3d::egui;
use kiss3d::light::Light;
use kiss3d::nalgebra::{Point3, Vector3};
use kiss3d::scene::SceneNode;
use kiss3d::window::Window;
use rapier3d::prelude::Aabb;
use std::path::Path;

/// Default color palette for distinguishing trajectories.
/// Colors are RGBA in [0, 1] range.
const TRAJECTORY_COLORS: &[[f32; 4]] = &[
    [0.4, 0.6, 1.0, 1.0], // Blue (original color)
    [1.0, 0.4, 0.3, 1.0], // Red
    [0.3, 0.9, 0.4, 1.0], // Green
    [1.0, 0.8, 0.2, 1.0], // Yellow
    [0.8, 0.4, 1.0, 1.0], // Purple
    [0.2, 0.9, 0.9, 1.0], // Cyan
    [1.0, 0.5, 0.0, 1.0], // Orange
    [1.0, 0.4, 0.7, 1.0], // Pink
];

/// Configuration for the trajectory viewer.
pub struct ViewerConfig {
    /// Window title
    pub title: String,
    /// Playback speed multiplier (1.0 = real-time)
    pub speed: f32,
    /// Whether to loop the playback
    pub loop_playback: bool,
    /// Particle render size multiplier
    pub particle_scale: f32,
    /// Background color
    pub background_color: (f32, f32, f32),
}

impl Default for ViewerConfig {
    fn default() -> Self {
        Self {
            title: "Slosh Validation Viewer".to_string(),
            speed: 1.0,
            loop_playback: true,
            particle_scale: 1.0,
            background_color: (0.15, 0.15, 0.2),
        }
    }
}

/// Trajectory playback viewer supporting multiple trajectories.
pub struct TrajectoryViewer {
    trajectories: Vec<SimulationTrajectory>,
    config: ViewerConfig,
}

impl TrajectoryViewer {
    /// Create a new viewer for the given trajectory.
    pub fn new(trajectory: SimulationTrajectory, config: ViewerConfig) -> Self {
        Self {
            trajectories: vec![trajectory],
            config,
        }
    }

    /// Create a new viewer for multiple trajectories.
    pub fn new_multi(trajectories: Vec<SimulationTrajectory>, config: ViewerConfig) -> Self {
        Self {
            trajectories,
            config,
        }
    }

    /// Load a trajectory from a JSON file.
    pub fn from_json(
        path: &Path,
        config: ViewerConfig,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let trajectory = SimulationTrajectory::load_json(path)?;
        Ok(Self::new(trajectory, config))
    }

    /// Load multiple trajectories from JSON files for comparison.
    pub fn from_json_multi(
        paths: &[&Path],
        config: ViewerConfig,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let trajectories: Result<Vec<_>, _> = paths
            .iter()
            .map(|p| SimulationTrajectory::load_json(p))
            .collect();
        Ok(Self::new_multi(trajectories?, config))
    }

    /// Run the viewer (blocking).
    pub fn run(self) {
        pollster::block_on(self.run_async());
    }

    /// Async version of run for use with kiss3d's event loop.
    pub async fn run_async(self) {
        if self.trajectories.is_empty() {
            eprintln!("No trajectories to display");
            return;
        }

        let mut window = Window::new(&self.config.title);
        window.set_background_color(
            self.config.background_color.0,
            self.config.background_color.1,
            self.config.background_color.2,
        );
        window.set_light(Light::StickToCamera);

        // Set up camera
        let eye = Point3::new(20.0, 15.0, 20.0);
        let at = Point3::new(5.0, 5.0, 0.0);
        let mut camera = ArcBall::new(eye, at);

        // Create particle spheres for each trajectory
        let mut particle_nodes: Vec<SceneNode> = Vec::new();
        let mut trajectory_info: Vec<(String, usize, [f32; 4])> = Vec::new();

        for (i, traj) in self.trajectories.iter().enumerate() {
            let particle_radius = traj.metadata.cell_width * 0.25 * self.config.particle_scale;
            let node = window.add_sphere(particle_radius);
            particle_nodes.push(node);

            let color = TRAJECTORY_COLORS[i % TRAJECTORY_COLORS.len()];
            trajectory_info.push((traj.name.clone(), traj.metadata.num_particles, color));
        }

        // Find the maximum number of snapshots across all trajectories
        let max_snapshots = self
            .trajectories
            .iter()
            .map(|t| t.snapshots.len())
            .max()
            .unwrap_or(0);

        // Playback state
        let mut current_snapshot_idx: usize = 0;
        let mut accumulated_time: f32 = 0.0;
        let mut paused = false;
        let mut last_time = std::time::Instant::now();
        let mut speed = self.config.speed;
        let loop_playback = self.config.loop_playback;

        // Get total simulation time from the longest trajectory
        let total_time = self
            .trajectories
            .iter()
            .filter_map(|t| t.snapshots.last().map(|s| s.time))
            .max_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap_or(0.0);

        // Determine time between snapshots from first trajectory
        let snapshot_dt = if self.trajectories[0].snapshots.len() > 1 {
            self.trajectories[0].dt * self.trajectories[0].snapshots[1].step as f32
        } else {
            self.trajectories[0].dt
        };

        println!("\nTrajectory Viewer Controls:");
        println!("  Space: Play/Pause");
        println!("  Left/Right arrows: Step through snapshots");
        println!("  R: Reset to beginning");
        println!("  Escape: Quit");
        println!(
            "\nLoaded {} trajectory/trajectories:",
            self.trajectories.len()
        );
        for (i, (name, num_particles, _color)) in trajectory_info.iter().enumerate() {
            let num_snapshots = self.trajectories[i].snapshots.len();
            println!(
                "  [{}] {} ({} particles, {} snapshots)",
                i + 1,
                name,
                num_particles,
                num_snapshots
            );
        }
        while !window.should_close() {
            // Handle keyboard input
            for event in window.events().iter() {
                use kiss3d::event::{Action, Key, WindowEvent};
                match event.value {
                    WindowEvent::Key(Key::Space, Action::Press, _) => {
                        paused = !paused;
                    }
                    WindowEvent::Key(Key::Left, Action::Press, _) => {
                        if current_snapshot_idx > 0 {
                            current_snapshot_idx -= 1;
                            accumulated_time = 0.0;
                        }
                    }
                    WindowEvent::Key(Key::Right, Action::Press, _) => {
                        if current_snapshot_idx < max_snapshots - 1 {
                            current_snapshot_idx += 1;
                            accumulated_time = 0.0;
                        }
                    }
                    WindowEvent::Key(Key::R, Action::Press, _) => {
                        current_snapshot_idx = 0;
                        accumulated_time = 0.0;
                    }
                    _ => {}
                }
            }

            // Update time
            let now = std::time::Instant::now();
            let dt = now.duration_since(last_time).as_secs_f32();
            last_time = now;

            if !paused {
                accumulated_time += dt * speed;

                // Advance to next snapshot if enough time has passed
                while accumulated_time >= snapshot_dt && current_snapshot_idx < max_snapshots - 1 {
                    accumulated_time -= snapshot_dt;
                    current_snapshot_idx += 1;
                }

                // Handle looping
                if current_snapshot_idx >= max_snapshots - 1 && loop_playback {
                    current_snapshot_idx = 0;
                    accumulated_time = 0.0;
                }
            }

            // Update particle positions for each trajectory
            let mut current_time_display = 0.0f32;
            for (traj_idx, traj) in self.trajectories.iter().enumerate() {
                // Clamp snapshot index for trajectories with fewer snapshots
                let snap_idx = current_snapshot_idx.min(traj.snapshots.len().saturating_sub(1));
                let snapshot = &traj.snapshots[snap_idx];
                let color = TRAJECTORY_COLORS[traj_idx % TRAJECTORY_COLORS.len()];

                let instance_data: Vec<_> = snapshot
                    .particles
                    .iter()
                    .map(|p| kiss3d::prelude::InstanceData {
                        position: kiss3d::nalgebra::Point3::from(p.position),
                        color,
                        ..Default::default()
                    })
                    .collect();
                particle_nodes[traj_idx].set_instances(&instance_data);

                // Use the time from the first trajectory for display
                if traj_idx == 0 {
                    current_time_display = snapshot.time;
                }
            }

            // Compute combined AABB and center of mass from all trajectories
            let mut all_particles: Vec<Point3<f32>> = Vec::new();
            for traj in &self.trajectories {
                let snap_idx = current_snapshot_idx.min(traj.snapshots.len().saturating_sub(1));
                let snapshot = &traj.snapshots[snap_idx];
                for p in &snapshot.particles {
                    all_particles.push(Point3::from(p.position));
                }
            }
            let particles_aabb = Aabb::from_points(all_particles.iter().cloned());
            let center_of_mass = if !all_particles.is_empty() {
                let sum: Vector3<f32> = all_particles.iter().map(|p| p.coords).sum();
                Point3::from(sum / all_particles.len() as f32)
            } else {
                Point3::origin()
            };
            // Draw UI
            let trajectory_info_clone = trajectory_info.clone();
            let num_trajectories = self.trajectories.len();
            window.draw_ui(|ctx| {
                egui::Window::new("Playback Controls").show(ctx, |ui| {
                    // Trajectory info - show each trajectory with its color
                    if num_trajectories == 1 {
                        ui.label(format!("Trajectory: {}", trajectory_info_clone[0].0));
                        ui.label(format!("Particles: {}", trajectory_info_clone[0].1));
                    } else {
                        ui.label(format!("Comparing {} trajectories:", num_trajectories));
                        for (_i, (name, num_particles, color)) in
                            trajectory_info_clone.iter().enumerate()
                        {
                            ui.horizontal(|ui| {
                                // Color indicator
                                let color32 = egui::Color32::from_rgba_unmultiplied(
                                    (color[0] * 255.0) as u8,
                                    (color[1] * 255.0) as u8,
                                    (color[2] * 255.0) as u8,
                                    255,
                                );
                                let (rect, _) = ui.allocate_exact_size(
                                    egui::vec2(12.0, 12.0),
                                    egui::Sense::hover(),
                                );
                                ui.painter().rect_filled(rect, 2.0, color32);
                                ui.label(format!("{} ({} particles)", name, num_particles));
                            });
                        }
                    }
                    ui.separator();

                    ui.label(format!("AABB mins: {:?}", particles_aabb.mins));
                    ui.label(format!("AABB maxs: {:?}", particles_aabb.maxs));
                    ui.label(format!(
                        "AABB sz: {:?}",
                        particles_aabb.half_extents() * 2.0
                    ));
                    ui.label(format!("COM: {:?}", center_of_mass));
                    ui.separator();

                    // Play/Pause button and step buttons
                    ui.horizontal(|ui| {
                        let play_pause_label = if paused { "▶ Play" } else { "⏸ Pause" };
                        if ui.button(play_pause_label).clicked() {
                            paused = !paused;
                        }
                        if ui.button("⏮ Reset").clicked() {
                            current_snapshot_idx = 0;
                            accumulated_time = 0.0;
                        }
                        if ui.button("◀ Prev").clicked() && current_snapshot_idx > 0 {
                            current_snapshot_idx -= 1;
                            accumulated_time = 0.0;
                        }
                        if ui.button("▶ Next").clicked() && current_snapshot_idx < max_snapshots - 1
                        {
                            current_snapshot_idx += 1;
                            accumulated_time = 0.0;
                        }
                    });

                    ui.separator();

                    // Timeline slider
                    ui.label(format!(
                        "Frame: {} / {} | Time: {:.2}s / {:.2}s",
                        current_snapshot_idx + 1,
                        max_snapshots,
                        current_time_display,
                        total_time
                    ));

                    // Slider for frame selection
                    let mut frame_f32 = current_snapshot_idx as f32;
                    let slider =
                        egui::Slider::new(&mut frame_f32, 0.0..=(max_snapshots - 1) as f32)
                            .text("Frame")
                            .show_value(false);
                    if ui.add(slider).changed() {
                        current_snapshot_idx = frame_f32 as usize;
                        accumulated_time = 0.0;
                    }

                    ui.separator();

                    // Speed control
                    ui.horizontal(|ui| {
                        ui.label("Speed:");
                        if ui.button("0.25x").clicked() {
                            speed = 0.25;
                        }
                        if ui.button("0.5x").clicked() {
                            speed = 0.5;
                        }
                        if ui.button("1x").clicked() {
                            speed = 1.0;
                        }
                        if ui.button("2x").clicked() {
                            speed = 2.0;
                        }
                        if ui.button("4x").clicked() {
                            speed = 4.0;
                        }
                    });
                    ui.add(egui::Slider::new(&mut speed, 0.1..=10.0).text("Speed"));

                    ui.separator();

                    // Status
                    let status = if paused { "⏸ PAUSED" } else { "▶ PLAYING" };
                    ui.label(format!("Status: {} at {:.1}x", status, speed));
                });
            });

            // Render
            window.render_with_camera(&mut camera).await;
        }
    }
}

/// Run the trajectory viewer for multiple trajectories simultaneously.
pub fn view_trajectories(
    paths: &[&Path],
    config: ViewerConfig,
) -> Result<(), Box<dyn std::error::Error>> {
    if paths.is_empty() {
        return Err("No trajectory files provided".into());
    }

    let viewer = TrajectoryViewer::from_json_multi(paths, config)?;
    viewer.run();
    Ok(())
}
