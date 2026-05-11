//! Internal GPU state management for the simulation.
//!
//! This module defines the low-level structures that manage memory buffers on the GPU,
//! including particles, grid nodes, and material properties.

mod atomic;
mod grid;
mod material;
mod particle;

pub(crate) use atomic::Atomic;
pub(crate) use grid::Grids;
pub(crate) use material::Materials;
pub(crate) use particle::Particles;
