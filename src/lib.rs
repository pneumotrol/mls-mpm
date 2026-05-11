//! # MLS-MPM
//!
//! A high-performance Moving Least Squares Material Point Method (MLS-MPM) simulation library
//! using [CubeCL](https://cubecl.github.io/) for cross-platform GPU acceleration.
//!
//! ## Overview
//!
//! MLS-MPM is a hybrid Lagrangian-Eulerian method used for simulating various materials,
//! including liquids, sand, and elastic solids. It combines the strengths of particles (Lagrangian)
//! and grids (Eulerian) to handle large deformations and complex topological changes efficiently.
//!
//! This library leverages the GPU through CubeCL, supporting multiple backends like WGPU
//! (for Vulkan, Metal, DX12, WebGPU) and CUDA.
//!
//! ## Simulation Workflow
//!
//! Each simulation step executes the following pipeline:
//! 1. **Clear Grid**: Resets grid node mass and momentum buffers.
//! 2. **P2G (Particle to Grid)**: Accumulates particle mass and momentum onto the grid nodes.
//! 3. **Update Grid**: Normalizes grid momentum to velocity and applies boundary conditions.
//! 4. **G2P (Grid to Particle)**: Interpolates velocity back to particles and advances their positions.
//!
//! ## Quick Start
//!
//! ```rust
//! use mls_mpm::{
//!     MlsMpm, MlsMpmDescriptor, ParticleDescriptor, MaterialDict,
//!     MaterialDescriptor, MaterialKind, Result
//! };
//! use cubecl::prelude::*;
//!
//! fn main() -> Result<()> {
//!     // 1. Initialize GPU client
//!     let client = cubecl::wgpu::WgpuRuntime::client(&Default::default());
//!
//!     // 2. Setup materials
//!     let mut materials = MaterialDict::new();
//!     materials.register(
//!         MaterialDescriptor::builder()
//!             .name("water".into())
//!             .kind(MaterialKind::Fluid {
//!                 density: 1.0,
//!                 specific_heat_ratio: 1.0,
//!                 stiffness: 1000.0,
//!                 viscosity: 0.0,
//!             })
//!             .build(),
//!     )?;
//!
//!     // 3. Setup particles
//!     let particles = vec![
//!         ParticleDescriptor::builder()
//!             .position([0.5, 0.5])
//!             .velocity([1.0, 0.0])
//!             .material(materials.get("water")?)
//!             .build(),
//!     ];
//!
//!     // 4. Create simulation from descriptor
//!     let descriptor = MlsMpmDescriptor::builder()
//!         .particles(particles)
//!         .materials(materials)
//!         .grid_dim([32, 32])
//!         .grid_size(1.0 / 32.0)
//!         .time_step(0.0001)
//!         .build();
//!
//!     let mut sim = MlsMpm::from_descriptor(&client, descriptor);
//!
//!     // 5. Run the simulation loop
//!     sim.launch()?;
//!
//!     Ok(())
//! }
//! ```

mod gpu;
mod interface;
mod mls_mpm;
mod state;

pub use interface::{
    BoundaryType, MaterialDescriptor, MaterialDict, MaterialKind, MlsMpmDescriptor,
    ParticleDescriptor,
};
pub use mls_mpm::{MlsMpm, SimulationParameters};

/// Result type for simulation operations, using `color_eyre` for rich error reporting.
pub type Result<T, E = color_eyre::eyre::Report> = std::result::Result<T, E>;
