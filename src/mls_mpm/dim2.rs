//! 2D specialized implementation of the MLS-MPM cycle.

mod kernel;

use super::*;
pub(super) use kernel::clear_grid::clear_grid;
pub(super) use kernel::grid_to_particle::grid_to_particle;
pub(super) use kernel::particle_to_grid::particle_to_grid;
pub(super) use kernel::particle_to_grid_i32_atomic::{
    clear_grid_i32, i32_to_f32, particle_to_grid_i32_atomic,
};
pub(super) use kernel::update_grid::update_grid;

impl<'a, R: Runtime> MlsMpm<'a, R, 2> {
    /// Launches a single simulation step on the GPU.
    ///
    /// This automatically selects the appropriate pipeline (native f32 or scaled i32 atomics)
    /// based on the simulation configuration.
    pub fn launch(&self) -> Result<()> {
        if let Some(atomic) = &self.atomic_buffer {
            self.launch_i32_atomic(atomic)
        } else {
            self.launch_f32_atomic()
        }
    }

    /// Executes the 2D simulation cycle using native floating-point atomic operations.
    ///
    /// This is the most efficient pipeline but requires GPU hardware/API support for f32 atomics.
    pub fn launch_f32_atomic(&self) -> Result<()> {
        unsafe {
            // 1. Clear Grid: Reset mass and momentum buffers.
            dim2::clear_grid::launch_unchecked::<f32, R>(
                self.client,
                CubeCount::new_1d(self.grids.len().div_ceil(NUM_WORKGROUPS).try_into()?),
                CubeDim::new_1d(NUM_WORKGROUPS.try_into()?),
                self.grids.mass.as_array_arg(),
                self.grids.velocity[0].as_array_arg(),
                self.grids.velocity[1].as_array_arg(),
                (
                    ScalarArg::new(self.grids.dim()[0]),
                    ScalarArg::new(self.grids.dim()[1]),
                ),
            )?;

            // 2. Particle to Grid (P2G): Accumulate contributions from particles to neighbors.
            dim2::particle_to_grid::launch_unchecked::<f32, i32, R>(
                self.client,
                CubeCount::new_1d(self.particles.len().div_ceil(NUM_WORKGROUPS).try_into()?),
                CubeDim::new_1d(NUM_WORKGROUPS.try_into()?),
                self.particles.position[0].as_array_arg(),
                self.particles.position[1].as_array_arg(),
                self.particles.velocity[0].as_array_arg(),
                self.particles.velocity[1].as_array_arg(),
                self.particles.deformation_gradient[0][0].as_array_arg(),
                self.particles.deformation_gradient[0][1].as_array_arg(),
                self.particles.deformation_gradient[1][0].as_array_arg(),
                self.particles.deformation_gradient[1][1].as_array_arg(),
                self.particles.affine_velocity[0][0].as_array_arg(),
                self.particles.affine_velocity[0][1].as_array_arg(),
                self.particles.affine_velocity[1][0].as_array_arg(),
                self.particles.affine_velocity[1][1].as_array_arg(),
                self.particles.mass.as_array_arg(),
                self.particles.volume.as_array_arg(),
                self.particles.material_id.as_array_arg(),
                ScalarArg::new(self.particles.len()),
                self.grids.mass.as_array_arg(),
                self.grids.velocity[0].as_array_arg(),
                self.grids.velocity[1].as_array_arg(),
                (
                    ScalarArg::new(self.grids.dim()[0]),
                    ScalarArg::new(self.grids.dim()[1]),
                ),
                self.materials.kind.as_array_arg(),
                self.materials.property[1].as_array_arg(),
                self.materials.property[2].as_array_arg(),
                self.materials.property[3].as_array_arg(),
                ScalarArg::new(self.materials.len()),
                self.sim_param.as_launch_args(),
            )?;

            // 3. Update Grid: Apply forces and boundary conditions.
            dim2::update_grid::launch_unchecked::<f32, R>(
                self.client,
                CubeCount::new_1d(self.grids.len().div_ceil(NUM_WORKGROUPS).try_into()?),
                CubeDim::new_1d(NUM_WORKGROUPS.try_into()?),
                self.grids.mass.as_array_arg(),
                self.grids.velocity[0].as_array_arg(),
                self.grids.velocity[1].as_array_arg(),
                (
                    ScalarArg::new(self.grids.dim()[0]),
                    ScalarArg::new(self.grids.dim()[1]),
                ),
                self.grids.boundary_id.as_array_arg(),
                self.grids.acceleration[0].as_array_arg(),
                self.grids.acceleration[1].as_array_arg(),
                self.sim_param.as_launch_args(),
            )?;

            // 4. Grid to Particle (G2P): Interpolate velocity back and update particle states.
            dim2::grid_to_particle::launch_unchecked::<f32, i32, R>(
                self.client,
                CubeCount::new_1d(self.particles.len().div_ceil(NUM_WORKGROUPS).try_into()?),
                CubeDim::new_1d(NUM_WORKGROUPS.try_into()?),
                self.particles.position[0].as_array_arg(),
                self.particles.position[1].as_array_arg(),
                self.particles.velocity[0].as_array_arg(),
                self.particles.velocity[1].as_array_arg(),
                self.particles.deformation_gradient[0][0].as_array_arg(),
                self.particles.deformation_gradient[0][1].as_array_arg(),
                self.particles.deformation_gradient[1][0].as_array_arg(),
                self.particles.deformation_gradient[1][1].as_array_arg(),
                self.particles.affine_velocity[0][0].as_array_arg(),
                self.particles.affine_velocity[0][1].as_array_arg(),
                self.particles.affine_velocity[1][0].as_array_arg(),
                self.particles.affine_velocity[1][1].as_array_arg(),
                self.particles.material_id.as_array_arg(),
                ScalarArg::new(self.particles.len()),
                self.grids.velocity[0].as_array_arg(),
                self.grids.velocity[1].as_array_arg(),
                (
                    ScalarArg::new(self.grids.dim()[0]),
                    ScalarArg::new(self.grids.dim()[1]),
                ),
                self.materials.kind.as_array_arg(),
                ScalarArg::new(self.materials.len()),
                self.sim_param.as_launch_args(),
            )?;
        }

        Ok(())
    }

    /// Executes the 2D simulation cycle using scaled integer atomic operations.
    ///
    /// This provides a fallback for environments without native f32 atomic support.
    pub fn launch_i32_atomic(&self, atomic: &Atomic<'a, R, 2>) -> Result<()> {
        unsafe {
            // 1. Clear Grid: Reset floating-point and integer buffers.
            dim2::clear_grid::launch_unchecked::<f32, R>(
                self.client,
                CubeCount::new_1d(self.grids.len().div_ceil(NUM_WORKGROUPS).try_into()?),
                CubeDim::new_1d(NUM_WORKGROUPS.try_into()?),
                self.grids.mass.as_array_arg(),
                self.grids.velocity[0].as_array_arg(),
                self.grids.velocity[1].as_array_arg(),
                (
                    ScalarArg::new(self.grids.dim()[0]),
                    ScalarArg::new(self.grids.dim()[1]),
                ),
            )?;
            dim2::clear_grid_i32::launch_unchecked::<i32, R>(
                self.client,
                CubeCount::new_1d(self.grids.len().div_ceil(NUM_WORKGROUPS).try_into()?),
                CubeDim::new_1d(NUM_WORKGROUPS.try_into()?),
                atomic.grid_mass_buffer.as_array_arg(),
                atomic.grid_velocity_buffer[0].as_array_arg(),
                atomic.grid_velocity_buffer[1].as_array_arg(),
                (
                    ScalarArg::new(self.grids.dim()[0]),
                    ScalarArg::new(self.grids.dim()[1]),
                ),
            )?;

            // 2. Particle to Grid (P2G): Accumulate contributions using scaled integers.
            dim2::particle_to_grid_i32_atomic::launch_unchecked::<f32, i32, R>(
                self.client,
                CubeCount::new_1d(self.particles.len().div_ceil(NUM_WORKGROUPS).try_into()?),
                CubeDim::new_1d(NUM_WORKGROUPS.try_into()?),
                self.particles.position[0].as_array_arg(),
                self.particles.position[1].as_array_arg(),
                self.particles.velocity[0].as_array_arg(),
                self.particles.velocity[1].as_array_arg(),
                self.particles.deformation_gradient[0][0].as_array_arg(),
                self.particles.deformation_gradient[0][1].as_array_arg(),
                self.particles.deformation_gradient[1][0].as_array_arg(),
                self.particles.deformation_gradient[1][1].as_array_arg(),
                self.particles.affine_velocity[0][0].as_array_arg(),
                self.particles.affine_velocity[0][1].as_array_arg(),
                self.particles.affine_velocity[1][0].as_array_arg(),
                self.particles.affine_velocity[1][1].as_array_arg(),
                self.particles.mass.as_array_arg(),
                self.particles.volume.as_array_arg(),
                self.particles.material_id.as_array_arg(),
                ScalarArg::new(self.particles.len()),
                atomic.grid_mass_buffer.as_array_arg(),
                atomic.grid_velocity_buffer[0].as_array_arg(),
                atomic.grid_velocity_buffer[1].as_array_arg(),
                (
                    ScalarArg::new(self.grids.dim()[0]),
                    ScalarArg::new(self.grids.dim()[1]),
                ),
                self.materials.kind.as_array_arg(),
                self.materials.property[1].as_array_arg(),
                self.materials.property[2].as_array_arg(),
                self.materials.property[3].as_array_arg(),
                ScalarArg::new(self.materials.len()),
                self.sim_param.as_launch_args(),
                ScalarArg::new(atomic.atomic_scale),
            )?;

            // Convert accumulated fixed-point integers back to floating-point.
            dim2::i32_to_f32::launch_unchecked::<f32, i32, R>(
                self.client,
                CubeCount::new_1d(self.grids.len().div_ceil(NUM_WORKGROUPS).try_into()?),
                CubeDim::new_1d(NUM_WORKGROUPS.try_into()?),
                self.grids().mass.as_array_arg(),
                self.grids().velocity[0].as_array_arg(),
                self.grids().velocity[1].as_array_arg(),
                (
                    ScalarArg::new(self.grids.dim()[0]),
                    ScalarArg::new(self.grids.dim()[1]),
                ),
                atomic.grid_mass_buffer.as_array_arg(),
                atomic.grid_velocity_buffer[0].as_array_arg(),
                atomic.grid_velocity_buffer[1].as_array_arg(),
                ScalarArg::new(atomic.atomic_scale),
            )?;

            // 3. Update Grid
            dim2::update_grid::launch_unchecked::<f32, R>(
                self.client,
                CubeCount::new_1d(self.grids.len().div_ceil(NUM_WORKGROUPS).try_into()?),
                CubeDim::new_1d(NUM_WORKGROUPS.try_into()?),
                self.grids.mass.as_array_arg(),
                self.grids.velocity[0].as_array_arg(),
                self.grids.velocity[1].as_array_arg(),
                (
                    ScalarArg::new(self.grids.dim()[0]),
                    ScalarArg::new(self.grids.dim()[1]),
                ),
                self.grids.boundary_id.as_array_arg(),
                self.grids.acceleration[0].as_array_arg(),
                self.grids.acceleration[1].as_array_arg(),
                self.sim_param.as_launch_args(),
            )?;

            // 4. Grid to Particle
            dim2::grid_to_particle::launch_unchecked::<f32, i32, R>(
                self.client,
                CubeCount::new_1d(self.particles.len().div_ceil(NUM_WORKGROUPS).try_into()?),
                CubeDim::new_1d(NUM_WORKGROUPS.try_into()?),
                self.particles.position[0].as_array_arg(),
                self.particles.position[1].as_array_arg(),
                self.particles.velocity[0].as_array_arg(),
                self.particles.velocity[1].as_array_arg(),
                self.particles.deformation_gradient[0][0].as_array_arg(),
                self.particles.deformation_gradient[0][1].as_array_arg(),
                self.particles.deformation_gradient[1][0].as_array_arg(),
                self.particles.deformation_gradient[1][1].as_array_arg(),
                self.particles.affine_velocity[0][0].as_array_arg(),
                self.particles.affine_velocity[0][1].as_array_arg(),
                self.particles.affine_velocity[1][0].as_array_arg(),
                self.particles.affine_velocity[1][1].as_array_arg(),
                self.particles.material_id.as_array_arg(),
                ScalarArg::new(self.particles.len()),
                self.grids.velocity[0].as_array_arg(),
                self.grids.velocity[1].as_array_arg(),
                (
                    ScalarArg::new(self.grids.dim()[0]),
                    ScalarArg::new(self.grids.dim()[1]),
                ),
                self.materials.kind.as_array_arg(),
                ScalarArg::new(self.materials.len()),
                self.sim_param.as_launch_args(),
            )?;
        }

        Ok(())
    }
}
