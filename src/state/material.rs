//! Material property state management on the GPU.

use crate::gpu::{GpuScalar, GpuVector};
use cubecl::prelude::*;

/// A collection of material properties stored on the GPU.
///
/// This structure manages the material types and their corresponding physical parameters
/// (e.g., density, stiffness, viscosity) for all materials defined in the simulation.
#[derive(Clone)]
pub struct Materials<'a, R: Runtime, const NUM_PROPERTIES: usize> {
    /// Numeric identifiers for each material kind.
    pub(crate) kind: GpuScalar<'a, R, u32>,
    /// Physical properties for each material (e.g., density, elasticity).
    pub(crate) property: GpuVector<'a, R, f32, NUM_PROPERTIES>,
    /// Number of materials defined.
    len: usize,
}

impl<'a, R: Runtime, const NUM_PROPERTIES: usize> Materials<'a, R, NUM_PROPERTIES> {
    /// Initializes material buffers on the GPU.
    pub(crate) fn new(client: &'a ComputeClient<R>, num_materials: usize) -> Self {
        Self {
            kind: GpuScalar::new(client, num_materials),
            property: GpuVector::new(client, num_materials),
            len: num_materials,
        }
    }

    /// Returns the number of materials registered.
    pub fn len(&self) -> usize {
        self.len
    }
}
