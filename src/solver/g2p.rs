//! Grid-to-Particle (G2P) transfer kernel.
//!
//! Interpolates grid velocities back to particles and updates particle velocity
//! gradients. This happens after grid forces have been applied.

use crate::grid::grid::{
    GpuActiveBlockHeader, GpuGrid, GpuGridHashMapEntry, GpuGridMetadata, GpuGridNode,
};
use crate::solver::{
    GpuBoundaryCondition, GpuMaterials, GpuParticleModelData, GpuParticles, GpuSimulationParams,
    ParticleDynamics, ParticlePosition, SimulationParams,
};
use nexus::dynamics::{GpuBodySet, GpuMassProperties, GpuVelocity};
use slang_hal::backend::Backend;
use slang_hal::function::GpuFunction;
use slang_hal::{Shader, ShaderArgs};
use stensor::tensor::{GpuScalar, GpuVector};

/// GPU compute kernel for Grid-to-Particle (G2P) velocity interpolation.
///
/// Samples grid velocities at particle positions using quadratic B-spline weights
/// and updates particle velocity gradients for deformation tracking (APIC method).
#[derive(Shader)]
#[shader(module = "slosh::solver::g2p")]
pub struct WgG2P<B: Backend> {
    /// Compiled G2P compute shader.
    pub g2p: GpuFunction<B>,
}

#[derive(ShaderArgs)]
struct G2PArgs<'a, B: Backend> {
    params: &'a GpuScalar<SimulationParams, B>,
    grid: &'a GpuScalar<GpuGridMetadata, B>,
    hmap_entries: &'a GpuVector<GpuGridHashMapEntry, B>,
    active_blocks: &'a GpuVector<GpuActiveBlockHeader, B>,
    nodes: &'a GpuVector<GpuGridNode, B>,
    sorted_particle_ids: &'a GpuVector<u32, B>,
    particles_pos: &'a GpuVector<ParticlePosition, B>,
    particles_dyn: &'a GpuVector<ParticleDynamics, B>,
    body_vels: &'a GpuVector<GpuVelocity, B>,
    body_mprops: &'a GpuVector<GpuMassProperties, B>,
    body_materials: &'a GpuVector<GpuBoundaryCondition, B>,
}

impl<B: Backend> WgG2P<B> {
    /// Launches the G2P kernel to update particle velocities from grid.
    ///
    /// # Arguments
    ///
    /// * `backend` - GPU backend for command recording
    /// * `pass` - Compute pass to record commands into
    /// * `sim_params` - Simulation parameters (timestep, gravity)
    /// * `grid` - Source grid to interpolate from
    /// * `particles` - Target particles to update
    /// * `bodies` - Rigid bodies for velocity blending near contacts
    pub fn launch<GpuModel: GpuParticleModelData>(
        &self,
        backend: &B,
        pass: &mut B::Pass,
        sim_params: &GpuSimulationParams<B>,
        grid: &GpuGrid<B>,
        particles: &GpuParticles<B, GpuModel>,
        bodies: &GpuBodySet<B>,
        body_materials: &GpuMaterials<B>,
    ) -> Result<(), B::Error> {
        let args = G2PArgs {
            params: &sim_params.params,
            grid: &grid.meta,
            hmap_entries: &grid.hmap_entries,
            active_blocks: &grid.active_blocks,
            nodes: &grid.nodes,
            sorted_particle_ids: particles.sorted_ids(),
            particles_pos: particles.positions(),
            particles_dyn: particles.dynamics(),
            body_vels: bodies.vels(),
            body_mprops: bodies.mprops(),
            body_materials: &body_materials.materials,
        };
        self.g2p.launch_indirect(
            backend,
            pass,
            &args,
            grid.indirect_n_g2p_p2g_groups.buffer(),
        )?;
        Ok(())
    }
}
