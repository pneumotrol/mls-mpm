//! Performance benchmark for the MLS-MPM simulation.
//!
//! This example runs a high-resolution simulation with 1 million particles
//! on a 1024x1024 grid to measure the GPU execution performance.
//! It does not perform any visualization to avoid I/O bottlenecks.
//!
//! The simulation distributes particles randomly across the entire grid
//! and executes 1000 steps to obtain a stable measurement of the
//! Particle-to-Grid (P2G) and Grid-to-Particle (G2P) kernel throughput.

use cubecl::prelude::*;
use indicatif::ProgressIterator;
use itertools::{Itertools, izip};
use mls_mpm::{
    MaterialDescriptor, MaterialDict, MaterialKind, MlsMpm, MlsMpmDescriptor, ParticleDescriptor,
    Result,
};
use rand::{RngExt, SeedableRng, rngs::StdRng};
use std::time::Instant;

fn main() -> Result<()> {
    // Initialize error handling
    color_eyre::install()?;

    // 1. Setup Compute Client
    // Select the backend based on enabled features (WGPU or CUDA).
    #[cfg(feature = "wgpu")]
    let client = cubecl::wgpu::WgpuRuntime::client(&Default::default());
    #[cfg(feature = "cuda")]
    let client = cubecl::cuda::CudaRuntime::client(&Default::default());

    // 2. Configure High-Resolution Simulation
    let mut materials = MaterialDict::new();
    materials.register(
        MaterialDescriptor::builder()
            .name("material_0".into())
            .kind(MaterialKind::Fluid {
                density: 1.0,
                specific_heat_ratio: 1.0,
                stiffness: 1000.0,
                viscosity: 0.0,
            })
            .build(),
    )?;

    let num_particles = 1 << 20; // 1,048,576 particles
    let grid_dim = [1024, 1024]; // 1,048,576 grid nodes
    let grid_size = 1.0;

    // Distribute particles randomly across the simulation domain.
    let mut rng = StdRng::seed_from_u64(57);
    let particles = (0..num_particles)
        .map(|_| {
            ParticleDescriptor::builder()
                .position([
                    rng.random_range(0.0..grid_dim[0] as f32) * grid_size,
                    rng.random_range(0.0..grid_dim[1] as f32) * grid_size,
                ])
                .material(materials.get("material_0").unwrap())
                .build()
        })
        .collect();

    // Create the simulation descriptor.
    let descriptor = MlsMpmDescriptor::builder()
        .particles(particles)
        .materials(materials)
        .grid_dim(grid_dim)
        .grid_size(grid_size)
        .time_step(0.0001)
        .build();

    // 3. Initialize Simulation
    // Allocates GPU buffers and synchronizes the initial state.
    let sim = MlsMpm::from_descriptor(&client, descriptor);

    // 4. Run Performance Benchmark
    println!("Launching performance benchmark (1M particles, 1024x1024 grid)...");
    let start = Instant::now();
    for _ in (0..1000).progress() {
        // Execute a single simulation step on the GPU.
        sim.launch()?;
    }
    let duration = start.elapsed();
    println!("Benchmark finished in {} us.", duration.as_micros());

    // 5. Verification
    // Read back a sample of the results to ensure the simulation executed correctly.
    println!("Sample results (first 10 particles):");
    let position = sim.particles().read_position();
    let velocity = sim.particles().read_velocity();
    for (i, (px, py, vx, vy)) in izip!(
        position[0].iter(),
        position[1].iter(),
        velocity[0].iter(),
        velocity[1].iter()
    )
    .enumerate()
    .take(10)
    {
        println!("[{i:>6}]: pos = ({px:8.2e}, {py:8.2e}), vel = ({vx:8.2e}, {vy:8.2e})");
    }

    println!("Sample results (first 10 grids):");
    let grid_mass = sim.grids().read_mass();
    let grid_velocity = sim.grids().read_velocity();
    for (i, (gm, gvx, gvy)) in izip!(
        grid_mass.iter(),
        grid_velocity[0].iter(),
        grid_velocity[1].iter()
    )
    .enumerate()
    .sorted_by(|a, b| a.1.0.partial_cmp(b.1.0).unwrap().reverse())
    .take(10)
    {
        println!("[{i:>6}]: grid_mass = {gm:8.2e}, grid_vel = ({gvx:8.2e}, {gvy:8.2e})");
    }

    Ok(())
}
