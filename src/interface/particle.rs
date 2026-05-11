//! Particle-related data structures for the simulation interface.

use super::material::MaterialName;
use bon::Builder;
use std::ops::{Deref, DerefMut};

fn eye<const DIM: usize>() -> [[f32; DIM]; DIM] {
    core::array::from_fn(|i| core::array::from_fn(|j| if i == j { 1.0 } else { 0.0 }))
}

/// Configuration for a single particle's initial state.
#[derive(Debug, Clone, Builder)]
pub struct ParticleDescriptor<const DIM: usize> {
    /// Initial position of the particle.
    #[builder(default = [0.0; DIM])]
    pub(crate) position: [f32; DIM],
    /// Initial velocity of the particle.
    #[builder(default = [0.0; DIM])]
    pub(crate) velocity: [f32; DIM],
    /// Initial deformation gradient (identity by default).
    #[builder(default = eye())]
    pub(crate) deformation_gradient: [[f32; DIM]; DIM],
    /// Name of the material this particle belongs to.
    pub(crate) material: MaterialName,
}

/// A buffer for storing particle attributes on the CPU.
#[derive(Debug, Clone)]
pub struct ParticleBuffer<T> {
    value: Vec<T>,
}

impl<T> Deref for ParticleBuffer<T> {
    type Target = Vec<T>;

    fn deref(&self) -> &Self::Target {
        &self.value
    }
}

impl<T> DerefMut for ParticleBuffer<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.value
    }
}

impl<T> ParticleBuffer<T> {
    /// Creates a new `ParticleBuffer` from a raw vector.
    pub fn from_vec(value: Vec<T>) -> Self {
        Self { value }
    }

    /// Returns a reference to the element at the specified index.
    pub fn get(&self, pos: [usize; 1]) -> Option<&T> {
        let index = pos[0];
        self.value.get(index)
    }

    /// Returns a mutable reference to the element at the specified index.
    pub fn at_mut(&mut self, pos: [usize; 1]) -> Option<&mut T> {
        let index = pos[0];
        self.value.get_mut(index)
    }

    /// Returns the shape (length) of the buffer.
    pub fn shape(&self) -> [usize; 1] {
        [self.value.len()]
    }
}
