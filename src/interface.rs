//! Public interface for defining and interacting with the simulation.
//!
//! This module provides the high-level descriptors and data structures used to
//! configure simulation states (particles, materials, grid) on the CPU before
//! synchronizing with the GPU.

mod grid;
mod material;
mod particle;

pub use grid::{BoundaryType, GridBuffer};
pub use material::{MaterialDescriptor, MaterialDict, MaterialKind};
pub use particle::{ParticleBuffer, ParticleDescriptor};

use bon::Builder;

/// High-level descriptor for initializing an `MlsMpm` simulation.
///
/// Contains all static configuration data required to allocate GPU buffers
/// and initialize physical states.
#[derive(Debug, Clone, Builder)]
pub struct MlsMpmDescriptor<const DIM: usize> {
    /// List of particles to be simulated.
    pub particles: Vec<ParticleDescriptor<DIM>>,
    /// Dictionary of materials used by the particles.
    pub materials: MaterialDict,
    /// Dimensions of the background grid (number of nodes in each dimension).
    pub grid_dim: [usize; DIM],
    /// Physical width of a single grid cell.
    pub grid_size: f32,
    /// Numerical time step size (delta t).
    pub time_step: f32,
    /// Optional scaling factor for integer atomic operations (P2G fallback).
    pub use_i32_atomic_with_scale: Option<f32>,
}
