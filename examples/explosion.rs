//! Simulation of a central explosion (radial expansion) using MLS-MPM.
//!
//! This example demonstrates:
//! - Initializing particles in a circular/disk formation at the center of the grid.
//! - Setting up sticky boundary conditions to contain the expanding particles.
//! - Visualizing the resulting complex deformations and velocity fields.

mod plotter;

use cubecl::prelude::*;
use indicatif::ProgressIterator;
use mls_mpm::{
    BoundaryType, MaterialDescriptor, MaterialDict, MaterialKind, MlsMpm, MlsMpmDescriptor,
    ParticleDescriptor, Result,
};
use rand::{RngExt, SeedableRng, rngs::StdRng};

fn main() -> Result<()> {
    // Initialize error handling
    color_eyre::install()?;

    // 1. Setup Compute Client
    #[cfg(feature = "wgpu")]
    let client = cubecl::wgpu::WgpuRuntime::client(&Default::default());
    #[cfg(feature = "cuda")]
    let client = cubecl::cuda::CudaRuntime::client(&Default::default());

    // 2. Configure Simulation Parameters
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

    let num_particles = 1000;
    let grid_dim = [32, 32];
    let grid_size = 1.0 / 32.0;

    // Distribute particles in a disk shape at the center.
    let mut rng = StdRng::seed_from_u64(57);
    let particles = (0..num_particles)
        .map(|_| {
            let radius: f32 = rng.random_range(0.0..0.3);
            let angle: f32 = rng.random_range(0.0..2.0 * std::f32::consts::PI);
            ParticleDescriptor::builder()
                .position([radius * angle.cos() + 0.5, radius * angle.sin() + 0.5])
                .material(materials.get("material_0").unwrap())
                .build()
        })
        .collect();

    let descriptor = MlsMpmDescriptor::builder()
        .particles(particles)
        .materials(materials)
        .grid_dim(grid_dim)
        .grid_size(grid_size)
        .time_step(0.0001)
        .build();

    // 3. Initialize Simulation
    let mut sim = MlsMpm::from_descriptor(&client, descriptor);

    // 4. Define Boundary Conditions
    // We create a "sticky" wall around the edges of the grid to contain the expansion.
    let mut boundary_id = sim.grids().read_boundary_id();
    let grid_dim = sim.grids().dim();
    let wall = 3;
    for y in 0..grid_dim[1] {
        for x in 0..grid_dim[0] {
            if (x < wall || grid_dim[0] - wall <= x || y < wall || grid_dim[1] - wall <= y)
                && let Some(id) = boundary_id.at_mut([x, y])
            {
                *id = BoundaryType::Sticky as u32;
            }
        }
    }
    sim.grids_mut().write_boundary_id(boundary_id);

    // 5. Setup Visualization Output
    let filepath_base = "debug/explosion";
    std::fs::create_dir_all(format!("{filepath_base}/particles"))?;
    std::fs::create_dir_all(format!("{filepath_base}/grids"))?;

    // 6. Run Simulation Loop
    println!("Launching explosion simulation...");
    for i in (0..1000).progress() {
        // Visualization: Save plots every 10 steps.
        if i % 10 == 0 {
            let particle_position = sim.particles().read_position();
            let particle_velocity = sim.particles().read_velocity();
            let grid_mass = sim.grids().read_mass();
            let grid_velocity = sim.grids().read_velocity();
            let grid_acceleration = sim.grids().read_acceleration();
            let grid_boundary_id = sim.grids().read_boundary_id();

            plotter::plot_particles(
                &format!("{filepath_base}/particles/{i:04}.png"),
                (&particle_position[0], &particle_position[1]),
                (&particle_velocity[0], &particle_velocity[1]),
                sim.sim_param().grid_size,
                &sim.grids().dim(),
            )?;
            plotter::plot_grids(
                &format!("{filepath_base}/grids/{i:04}.png"),
                &grid_mass,
                (&grid_velocity[0], &grid_velocity[1]),
                &grid_boundary_id,
                (&grid_acceleration[0], &grid_acceleration[1]),
                sim.sim_param().grid_size,
                &sim.grids().dim(),
            )?;
        }

        // Execute a single simulation step on the GPU.
        sim.launch()?;
    }

    Ok(())
}
