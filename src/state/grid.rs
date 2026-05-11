//! Background Eulerian grid state management on the GPU.

use crate::{
    gpu::{GpuScalar, GpuVector},
    interface::GridBuffer,
};
use cubecl::prelude::*;

/// A collection of background grid attributes stored on the GPU.
///
/// This structure manages the mass, velocity, and acceleration fields of the Eulerian grid,
/// as well as boundary condition identifiers.
#[derive(Clone)]
pub struct Grids<'a, R: Runtime, const DIM: usize> {
    /// Mass accumulated at each grid node.
    pub(crate) mass: GpuScalar<'a, R, f32>,
    /// Momentum (during P2G) or velocity (after Update Grid) at each grid node.
    pub(crate) velocity: GpuVector<'a, R, f32, DIM>,
    /// External acceleration (e.g., gravity) applied at each grid node.
    pub(crate) acceleration: GpuVector<'a, R, f32, DIM>,
    /// Boundary condition identifiers for each grid node.
    pub(crate) boundary_id: GpuScalar<'a, R, u32>,
    /// Dimensions of the grid.
    dim: [usize; DIM],
    /// Total number of grid nodes.
    len: usize,
}

impl<'a, R: Runtime, const DIM: usize> Grids<'a, R, DIM> {
    /// Initializes grid buffers with zeros on the GPU.
    pub(crate) fn new(client: &'a ComputeClient<R>, grid_dim: [usize; DIM]) -> Self {
        let num_grids = grid_dim.iter().product();

        Self {
            mass: GpuScalar::new(client, num_grids),
            velocity: GpuVector::new(client, num_grids),
            acceleration: GpuVector::new(client, num_grids),
            boundary_id: GpuScalar::new(client, num_grids),
            dim: grid_dim,
            len: num_grids,
        }
    }

    /// Returns the dimensions of the background grid.
    pub fn dim(&self) -> [usize; DIM] {
        self.dim
    }

    /// Returns the total number of grid nodes.
    pub fn len(&self) -> usize {
        self.len
    }

    /// Reads the mass field from the GPU back to a `GridBuffer`.
    pub fn read_mass(&self) -> GridBuffer<f32, DIM> {
        GridBuffer::from_vec(self.mass.read(), self.dim)
    }

    /// Reads the velocity field from the GPU back to an array of `GridBuffer`s.
    pub fn read_velocity(&self) -> [GridBuffer<f32, DIM>; DIM] {
        core::array::from_fn(|i| GridBuffer::from_vec(self.velocity[i].read(), self.dim))
    }

    /// Reads the acceleration field from the GPU back to an array of `GridBuffer`s.
    pub fn read_acceleration(&self) -> [GridBuffer<f32, DIM>; DIM] {
        core::array::from_fn(|i| GridBuffer::from_vec(self.acceleration[i].read(), self.dim))
    }

    /// Reads the boundary ID field from the GPU back to a `GridBuffer`.
    pub fn read_boundary_id(&self) -> GridBuffer<u32, DIM> {
        GridBuffer::from_vec(self.boundary_id.read(), self.dim)
    }

    /// Writes an acceleration field from a CPU array to the GPU buffers.
    pub fn write_acceleration(&mut self, acceleration: [GridBuffer<f32, DIM>; DIM]) {
        for (i, acceleration) in acceleration.iter().enumerate().take(DIM) {
            self.acceleration[i].write(acceleration);
        }
    }

    /// Writes a boundary ID field from a CPU `GridBuffer` to the GPU buffer.
    pub fn write_boundary_id(&mut self, boundary_id: GridBuffer<u32, DIM>) {
        self.boundary_id.write(&boundary_id);
    }
}
