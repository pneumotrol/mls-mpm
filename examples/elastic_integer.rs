//! Simulation of an elastic material falling under gravity using MLS-MPM with integer atomics.
//!
//! This example demonstrates:
//! - Initializing particles in a rotated square formation.
//! - Configuring an elastic (Neo-Hookean solid) material model.
//! - Applying a global acceleration field (gravity) to simulate deformation and bouncing.
//! - Utilizing integer atomics with scale configuration for the grid update.
//! - Reading back grid and particle states for visualization.

mod plotter;

use cubecl::prelude::*;
use indicatif::ProgressIterator;
use itertools::iproduct;
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
            .kind(MaterialKind::Elastic {
                density: 1.0,
                shear_modulus: 100.0,
                bulk_modulus: 5000.0,
            })
            .build(),
    )?;

    let num_particles = 1000;
    let grid_dim = [32, 32];
    let grid_size = 1.0 / 32.0;

    // Distribute particles randomly within a sub-region of the grid.
    let mut rng = StdRng::seed_from_u64(57);
    let mut particles = Vec::new();
    while particles.len() < num_particles {
        let x: f32 = rng.random_range(-1.0..1.0);
        let y: f32 = rng.random_range(-1.0..1.0);
        let theta = (30.0_f32).to_radians();
        let (x_rot, y_rot) = (
            theta.cos() * x + theta.sin() * y,
            -theta.sin() * x + theta.cos() * y,
        );
        let half_size = 0.15;
        if -half_size < x_rot && x_rot < half_size && -half_size < y_rot && y_rot < half_size {
            particles.push(
                ParticleDescriptor::builder()
                    .position([x + 0.5, y + 0.5])
                    .material(materials.get("material_0").unwrap())
                    .build(),
            );
        }
    }

    let descriptor = MlsMpmDescriptor::builder()
        .particles(particles)
        .materials(materials)
        .grid_dim(grid_dim)
        .grid_size(grid_size)
        .time_step(0.0001)
        .use_i32_atomic_with_scale(1e8)
        .build();

    // 3. Initialize Simulation
    let mut sim = MlsMpm::from_descriptor(&client, descriptor);

    // 4. Define Boundary Conditions
    // We create a "sticky" wall around the edges of the simulation domain.
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

    // 5. Define Acceleration Field (Gravity)
    // Apply a constant downward acceleration to all grid nodes.
    let mut acceleration = sim.grids().read_acceleration();
    let grid_dim = sim.grids().dim();
    for (i, j) in iproduct!(0..grid_dim[0], 0..grid_dim[1]) {
        if let Some(a) = acceleration[1].at_mut([i, j]) {
            *a = -500.0;
        }
    }
    sim.grids_mut().write_acceleration(acceleration);

    // 6. Setup Visualization Output
    let filepath_base = "debug/elastic_integer";
    std::fs::create_dir_all(format!("{filepath_base}/particles"))?;
    std::fs::create_dir_all(format!("{filepath_base}/grids"))?;

    // 7. Run Simulation Loop
    println!("Launching elastic simulation...");
    for i in (0..1000).progress() {
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
