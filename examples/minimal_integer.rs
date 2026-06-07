//! Minimal example of a Moving Least Squares Material Point Method (MLS-MPM) simulation using integer atomics.
//!
//! This example is identical to `minimal.rs` but demonstrates the use of scaled integer atomics
//! for the Particle-to-Grid (P2G) phase. This provides a fallback for environments that
//! do not support floating-point atomic additions.

mod plotter;

use cubecl::prelude::*;
use indicatif::ProgressIterator;
use mls_mpm::{
    BoundaryType, MaterialDescriptor, MaterialDict, MaterialKind, MlsMpm, MlsMpmDescriptor,
    ParticleDescriptor, Result,
};

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

    let grid_size = 1.0 / 16.0;
    let particles = vec![
        ParticleDescriptor::builder()
            .position([4.5 * grid_size, 4.5 * grid_size])
            .velocity([100.0, 50.0])
            .material(materials.get("material_0").unwrap())
            .build(),
    ];

    // Create descriptor with integer atomic fallback enabled.
    // Fixed-point scaling factor set to 1.0e8.
    let descriptor = MlsMpmDescriptor::builder()
        .particles(particles)
        .materials(materials)
        .grid_dim([32, 16])
        .grid_size(grid_size)
        .time_step(0.0001)
        .use_i32_atomic_with_scale(1e8)
        .build();

    // 3. Initialize Simulation
    let mut sim = MlsMpm::from_descriptor(&client, descriptor);

    // 4. Define Boundary Conditions
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
    let filepath_base = "debug/minimal_integer";
    std::fs::create_dir_all(format!("{filepath_base}/particles"))?;
    std::fs::create_dir_all(format!("{filepath_base}/grids"))?;

    // 6. Run Simulation Loop
    println!("Launching minimal simulation (integer atomics)...");
    for i in (0..100).progress() {
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
