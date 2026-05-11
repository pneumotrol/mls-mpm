//! GPU-backed storage for scaled integer atomic operations.

use crate::gpu::{GpuScalar, GpuVector};
use cubecl::prelude::*;

/// Parameters and buffers for scaled integer atomic operations.
///
/// This provides a fallback for environments (like some WebGPU implementations) that
/// do not support floating-point atomic additions during the Particle-to-Grid (P2G) phase.
#[derive(Clone)]
pub struct Atomic<'a, R: Runtime, const DIM: usize> {
    /// Scaling factor to convert floating-point values to fixed-point integers.
    pub(crate) atomic_scale: f32,
    /// Integer buffer for accumulated grid mass.
    pub(crate) grid_mass_buffer: GpuScalar<'a, R, i32>,
    /// Integer buffers for accumulated grid momentum (one per dimension).
    pub(crate) grid_velocity_buffer: GpuVector<'a, R, i32, DIM>,
}

impl<'a, R: Runtime, const DIM: usize> Atomic<'a, R, DIM> {
    /// Initializes new integer atomic buffers for a given grid dimension.
    pub(crate) fn new(
        client: &'a ComputeClient<R>,
        grid_dim: [usize; DIM],
        atomic_scale: f32,
    ) -> Self {
        let num_grids = grid_dim.iter().product();

        Self {
            atomic_scale,
            grid_mass_buffer: GpuScalar::new(client, num_grids),
            grid_velocity_buffer: GpuVector::new(client, num_grids),
        }
    }
}
