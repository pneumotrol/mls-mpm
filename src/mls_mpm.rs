//! Core MLS-MPM simulation logic.
//!
//! This module defines the main `MlsMpm` structure, which orchestrates the simulation
//! by managing GPU states and launching kernels across multiple stages of the MPM cycle.

mod dim2;

use crate::{
    Result,
    interface::{MaterialKind, MlsMpmDescriptor},
    state::{Atomic, Grids, Materials, Particles},
};
use cubecl::prelude::*;

const NUM_WORKGROUPS: usize = 256;
const NUM_MATERIAL_PROPERTIES: usize = 10;

/// Parameters governing the simulation's numerical behavior.
#[derive(Debug, Clone, CubeType, CubeLaunch)]
pub struct SimulationParameters {
    /// Numerical time step size (delta t).
    pub dt: f32,
    /// Physical width of a single grid cell.
    pub grid_size: f32,
}

impl SimulationParameters {
    /// Converts parameters to CubeCL launch arguments.
    pub fn as_launch_args<R: Runtime>(&self) -> SimulationParametersLaunch<'_, R> {
        SimulationParametersLaunch::new(ScalarArg::new(self.dt), ScalarArg::new(self.grid_size))
    }
}

/// The main structure for the Moving Least Squares Material Point Method (MLS-MPM) simulation.
///
/// This struct manages the lifecycle of the simulation, including particle and grid data,
/// material properties, and the execution of GPU kernels.
#[derive(Clone)]
pub struct MlsMpm<'a, R: Runtime, const DIM: usize> {
    /// The CubeCL compute client.
    pub(crate) client: &'a ComputeClient<R>,
    /// Lagrangian particle data.
    pub(crate) particles: Particles<'a, R, DIM>,
    /// Eulerian grid data.
    pub(crate) grids: Grids<'a, R, DIM>,
    /// Material property data.
    pub(crate) materials: Materials<'a, R, NUM_MATERIAL_PROPERTIES>,
    /// Numerical simulation parameters.
    pub(crate) sim_param: SimulationParameters,
    /// Optional buffers for integer atomic operations (P2G fallback).
    pub(crate) atomic_buffer: Option<Atomic<'a, R, DIM>>,
}

impl<'a, R: Runtime, const DIM: usize> MlsMpm<'a, R, DIM> {
    /// Initializes a new simulation from an `MlsMpmDescriptor`.
    ///
    /// Allocates all necessary GPU buffers and synchronizes the initial state.
    pub fn from_descriptor(
        client: &'a ComputeClient<R>,
        descriptor: MlsMpmDescriptor<DIM>,
    ) -> Self {
        Self {
            client,
            particles: Self::crate_particle_buffer(client, &descriptor),
            grids: Self::create_grid_buffer(client, &descriptor),
            materials: Self::create_material_buffer(client, &descriptor),
            sim_param: SimulationParameters {
                dt: descriptor.time_step,
                grid_size: descriptor.grid_size,
            },
            atomic_buffer: Self::create_atomic_buffer(client, &descriptor),
        }
    }

    /// Returns a reference to the Lagrangian particle state.
    pub fn particles(&self) -> &Particles<'_, R, DIM> {
        &self.particles
    }

    /// Returns a reference to the Eulerian grid state.
    pub fn grids(&self) -> &Grids<'_, R, DIM> {
        &self.grids
    }

    /// Returns a reference to the material properties stored on the GPU.
    pub fn materials(&self) -> &Materials<'_, R, NUM_MATERIAL_PROPERTIES> {
        &self.materials
    }

    /// Returns a reference to the numerical simulation parameters.
    pub fn sim_param(&self) -> &SimulationParameters {
        &self.sim_param
    }

    /// Returns a mutable reference to the particle state.
    pub fn particles_mut(&mut self) -> &mut Particles<'a, R, DIM> {
        &mut self.particles
    }

    /// Returns a mutable reference to the grid state.
    pub fn grids_mut(&mut self) -> &mut Grids<'a, R, DIM> {
        &mut self.grids
    }

    /// Returns a mutable reference to the material properties.
    pub fn materials_mut(&mut self) -> &mut Materials<'a, R, NUM_MATERIAL_PROPERTIES> {
        &mut self.materials
    }

    /// Returns a mutable reference to the simulation parameters.
    pub fn sim_param_mut(&mut self) -> &mut SimulationParameters {
        &mut self.sim_param
    }

    /// Internal helper to initialize and synchronize particle buffers.
    fn crate_particle_buffer(
        client: &'a ComputeClient<R>,
        descriptor: &MlsMpmDescriptor<DIM>,
    ) -> Particles<'a, R, DIM> {
        let MlsMpmDescriptor {
            particles,
            materials,
            grid_size,
            ..
        } = descriptor;

        let len = particles.len();
        let mut position = vec![Vec::with_capacity(len); DIM];
        let mut velocity = vec![Vec::with_capacity(len); DIM];
        let mut deformation_gradient = vec![vec![Vec::with_capacity(len); DIM]; DIM];
        let mut affine_velocity = vec![vec![Vec::with_capacity(len); DIM]; DIM];
        let mut mass = Vec::with_capacity(len);
        let mut volume = Vec::with_capacity(len);
        let mut material_id = Vec::with_capacity(len);

        for particle in particles.iter() {
            let (id, kind) = &materials[&particle.material];
            let id: u32 = **id;

            let v = (0.5 * grid_size).powi(DIM as i32);
            let rho = match kind {
                MaterialKind::Fluid { density, .. } => density,
                MaterialKind::Elastic { density, .. } => density,
            };

            for i in 0..DIM {
                position[i].push(particle.position[i]);
                velocity[i].push(particle.velocity[i]);
                for j in 0..DIM {
                    deformation_gradient[i][j].push(particle.deformation_gradient[i][j]);
                    affine_velocity[i][j].push(0.0);
                }
            }
            mass.push(rho * v);
            volume.push(v);
            material_id.push(id);
        }

        let buffer = Particles::new(client, len);
        for i in 0..DIM {
            buffer.position[i].write(&position[i]);
            buffer.velocity[i].write(&velocity[i]);
            for j in 0..DIM {
                buffer.deformation_gradient[i][j].write(&deformation_gradient[i][j]);
                buffer.affine_velocity[i][j].write(&affine_velocity[i][j]);
            }
        }
        buffer.mass.write(&mass);
        buffer.volume.write(&volume);
        buffer.material_id.write(&material_id);

        buffer
    }

    /// Internal helper to initialize grid buffers.
    fn create_grid_buffer(
        client: &'a ComputeClient<R>,
        descriptor: &MlsMpmDescriptor<DIM>,
    ) -> Grids<'a, R, DIM> {
        let MlsMpmDescriptor { grid_dim, .. } = descriptor;

        Grids::new(client, *grid_dim)
    }

    /// Internal helper to initialize and synchronize material buffers.
    fn create_material_buffer(
        client: &'a ComputeClient<R>,
        descriptor: &MlsMpmDescriptor<DIM>,
    ) -> Materials<'a, R, NUM_MATERIAL_PROPERTIES> {
        let MlsMpmDescriptor { materials, .. } = descriptor;

        let len = materials.len();
        let mut material_kind = vec![0; len];
        let mut material_property = vec![vec![0.0; len]; NUM_MATERIAL_PROPERTIES];

        for (id, kind) in materials.values() {
            let id = **id as usize;
            match *kind {
                MaterialKind::Fluid {
                    density,
                    specific_heat_ratio,
                    stiffness,
                    viscosity,
                } => {
                    material_kind[id] = 0;
                    for (i, property) in material_property
                        .iter_mut()
                        .enumerate()
                        .take(NUM_MATERIAL_PROPERTIES)
                    {
                        property[id] = match i {
                            0 => density,
                            1 => specific_heat_ratio,
                            2 => stiffness,
                            3 => viscosity,
                            _ => 0.0,
                        }
                    }
                }
                MaterialKind::Elastic {
                    density,
                    shear_modulus,
                    bulk_modulus,
                } => {
                    material_kind[id] = 1;
                    for (i, property) in material_property
                        .iter_mut()
                        .enumerate()
                        .take(NUM_MATERIAL_PROPERTIES)
                    {
                        property[id] = match i {
                            0 => density,
                            1 => shear_modulus,
                            2 => bulk_modulus,
                            _ => 0.0,
                        }
                    }
                }
            }
        }

        let buffer = Materials::new(client, len);
        buffer.kind.write(&material_kind);
        for (i, property) in material_property
            .iter()
            .enumerate()
            .take(NUM_MATERIAL_PROPERTIES)
        {
            buffer.property[i].write(property);
        }

        buffer
    }

    /// Internal helper to initialize integer atomic buffers if requested.
    fn create_atomic_buffer(
        client: &'a ComputeClient<R>,
        descriptor: &MlsMpmDescriptor<DIM>,
    ) -> Option<Atomic<'a, R, DIM>> {
        let MlsMpmDescriptor {
            grid_dim,
            use_i32_atomic_with_scale,
            ..
        } = descriptor;

        if let Some(atomic_scale) = use_i32_atomic_with_scale {
            Some(Atomic::new(client, *grid_dim, *atomic_scale))
        } else {
            None
        }
    }
}
