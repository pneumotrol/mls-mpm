//! Lagrangian particle state management on the GPU.

use crate::{
    gpu::{GpuMatrix, GpuScalar, GpuVector},
    interface::ParticleBuffer,
};
use cubecl::prelude::*;

/// A collection of Lagrangian particle attributes stored on the GPU.
///
/// This structure manages the position, velocity, deformation gradient, and other
/// physical properties of the particles in the simulation.
#[derive(Clone)]
pub struct Particles<'a, R: Runtime, const DIM: usize> {
    /// Current position of particles.
    pub(crate) position: GpuVector<'a, R, f32, DIM>,
    /// Current velocity of particles.
    pub(crate) velocity: GpuVector<'a, R, f32, DIM>,
    /// Deformation gradient tensor (F) tracking material deformation.
    pub(crate) deformation_gradient: GpuMatrix<'a, R, f32, DIM>,
    /// Affine velocity (C) for Affine Particle-in-Cell (APIC) momentum tracking.
    pub(crate) affine_velocity: GpuMatrix<'a, R, f32, DIM>,
    /// Mass of each particle.
    pub(crate) mass: GpuScalar<'a, R, f32>,
    /// Initial volume of each particle.
    pub(crate) volume: GpuScalar<'a, R, f32>,
    /// Material identifier index for each particle.
    pub(crate) material_id: GpuScalar<'a, R, u32>,
    /// Total number of particles.
    len: usize,
}

impl<'a, R: Runtime, const DIM: usize> Particles<'a, R, DIM> {
    /// Initializes particle buffers with zeros on the GPU.
    pub(crate) fn new(client: &'a ComputeClient<R>, num_particles: usize) -> Self {
        Self {
            position: GpuVector::new(client, num_particles),
            velocity: GpuVector::new(client, num_particles),
            deformation_gradient: GpuMatrix::new(client, num_particles),
            affine_velocity: GpuMatrix::new(client, num_particles),
            mass: GpuScalar::new(client, num_particles),
            volume: GpuScalar::new(client, num_particles),
            material_id: GpuScalar::new(client, num_particles),
            len: num_particles,
        }
    }

    /// Returns the number of particles in the simulation.
    pub fn len(&self) -> usize {
        self.len
    }

    /// Reads particle positions from the GPU back to an array of `ParticleBuffer`s.
    pub fn read_position(&self) -> [ParticleBuffer<f32>; DIM] {
        core::array::from_fn(|i| ParticleBuffer::from_vec(self.position[i].read()))
    }

    /// Reads particle velocities from the GPU back to an array of `ParticleBuffer`s.
    pub fn read_velocity(&self) -> [ParticleBuffer<f32>; DIM] {
        core::array::from_fn(|i| ParticleBuffer::from_vec(self.velocity[i].read()))
    }

    /// Reads deformation gradients from the GPU back to a nested array of `ParticleBuffer`s.
    pub fn read_deformation_gradient(&self) -> [[ParticleBuffer<f32>; DIM]; DIM] {
        core::array::from_fn(|i| {
            core::array::from_fn(|j| {
                ParticleBuffer::from_vec(self.deformation_gradient[i][j].read())
            })
        })
    }

    /// Reads affine velocities from the GPU back to a nested array of `ParticleBuffer`s.
    pub fn read_affine_velocity(&self) -> [[ParticleBuffer<f32>; DIM]; DIM] {
        core::array::from_fn(|i| {
            core::array::from_fn(|j| ParticleBuffer::from_vec(self.affine_velocity[i][j].read()))
        })
    }

    /// Reads particle masses from the GPU back to a `ParticleBuffer`.
    pub fn read_mass(&self) -> ParticleBuffer<f32> {
        ParticleBuffer::from_vec(self.mass.read())
    }

    /// Reads initial particle volumes from the GPU back to a `ParticleBuffer`.
    pub fn read_volume(&self) -> ParticleBuffer<f32> {
        ParticleBuffer::from_vec(self.volume.read())
    }

    /// Reads particle material IDs from the GPU back to a `ParticleBuffer`.
    pub fn read_material_id(&self) -> ParticleBuffer<u32> {
        ParticleBuffer::from_vec(self.material_id.read())
    }
}
