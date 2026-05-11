# MLS-MPM (Moving Least Squares Material Point Method)

A high-performance Moving Least Squares Material Point Method (MLS-MPM) simulation library written in Rust, leveraging [CubeCL](https://cubecl.github.io/) for cross-platform GPU acceleration.

## Simulation Results

### Explosion Example

![Particles]()
![Grids]()

### Gravity Example

![Particles]()
![Grids]()

## Overview

MLS-MPM is a hybrid Lagrangian-Eulerian method used for simulating materials like fluids, snow, and elastic bodies. It combines the strengths of particles (Lagrangian) and background grids (Eulerian) to handle large deformations and topological changes efficiently.

This project provides a 2D implementation that runs entirely on the GPU, minimizing CPU-GPU synchronization overhead and maximizing throughput.

- **Lagrangian Particles:** Represent the material mass, momentum, and deformation gradient.
- **Eulerian Grid:** Used for calculating internal forces, applying boundary conditions, and updating velocities.
- **Interpolation:** Quadratic B-spline interpolation ensures smooth data transfers between particles and the grid.

## Features

- **Cross-Platform GPU Support:** Uses CubeCL to target `wgpu` (Vulkan, Metal, DX12, WebGPU) or `cuda` backends.
- **Atomic Operations:** Optimized Particle-to-Grid (P2G) transfers using native `f32` atomics where supported.
- **Integer Atomic Fallback:** High compatibility for environments without native `f32` atomic support (e.g., certain WebGPU targets) via scaled `i32` fixed-point accumulation.
- **Boundary Conditions:** Supports `Sticky` (no-slip) and `Slip` boundary conditions at the grid level.
- **Multi-Material Support:** Efficient management of different material types (e.g., water, honey) within the same simulation.

## Installation

Add this to your `Cargo.toml`:

```toml
[dependencies]
mls-mpm = { git = "https://github.com/pneumotrol/mls-mpm" }
```

### Feature Flags

- `wgpu` (default): Runs on most platforms using Vulkan, Metal, DX12, or WebGPU.
- `cuda`: Optimized for NVIDIA GPUs.

## Quick Start

```rust
use mls_mpm::{
    MlsMpm, MlsMpmDescriptor, ParticleDescriptor, MaterialDict,
    MaterialDescriptor, MaterialKind, Result
};
use cubecl::prelude::*;

fn main() -> Result<()> {
    // 1. Initialize GPU client (e.g., WGPU)
    let client = cubecl::wgpu::WgpuRuntime::client(&Default::default());

    // 2. Setup materials
    let mut materials = MaterialDict::new();
    materials.register(
        MaterialDescriptor::builder()
            .name("water".into())
            .kind(MaterialKind::Fluid {
                density: 1.0,
                specific_heat_ratio: 1.0,
                stiffness: 1000.0,
                viscosity: 0.0,
            })
            .build(),
    )?;

    // 3. Setup particles
    let particles = vec![
        ParticleDescriptor::builder()
            .position([0.5, 0.5])
            .velocity([1.0, 0.0])
            .material(materials.get("water")?)
            .build(),
    ];

    // 4. Create simulation from descriptor
    let descriptor = MlsMpmDescriptor::builder()
        .particles(particles)
        .materials(materials)
        .grid_dim([32, 32])
        .grid_size(1.0 / 32.0)
        .time_step(0.0001)
        .build();

    let mut sim = MlsMpm::from_descriptor(&client, descriptor);

    // 5. Run the simulation loop
    for _ in 0..1000 {
        sim.launch()?;
    }

    // 6. Read results back to CPU
    let position = sim.particles().read_position();
    println!("Particle 0 position: {:?}", [position[0][0], position[1][0]]);

    Ok(())
}
```

## Running Examples

The project includes several examples demonstrating different physical scenarios.

```bash
# Run the minimal simulation with WGPU
cargo run --release --example minimal

# Run the gravity simulation with CUDA
cargo run --release --example gravity --features cuda --no-default-features

# Run the explosion simulation
cargo run --release --example explosion_integer

# Benchmark
cargo run --release --example benchmark
```

## License

This project is licensed under the MIT License.
