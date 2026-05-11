//! Grid-related data structures for the simulation interface.

use std::ops::{Deref, DerefMut};

/// Types of boundary conditions that can be applied to grid nodes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum BoundaryType {
    /// No boundary condition applied. Standard interior node.
    None = 0,
    /// Velocity is set to zero (no-slip/sticky boundary).
    Sticky = 1,
    /// Velocity in X is set to zero (slip in Y) on the left boundary.
    SlipLeft = 20,
    /// Velocity in X is set to zero (slip in Y) on the right boundary.
    SlipRight = 21,
    /// Velocity in Y is set to zero (slip in X) on the bottom boundary.
    SlipBottom = 22,
    /// Velocity in Y is set to zero (slip in X) on the top boundary.
    SlipTop = 23,
}

/// A multi-dimensional buffer for storing grid node attributes on the CPU.
///
/// Provides a convenient abstraction for 2D/3D grid data with automatic stride calculation.
#[derive(Debug, Clone)]
pub struct GridBuffer<T, const DIM: usize> {
    value: Vec<T>,
    shape: [usize; DIM],
    strides: [usize; DIM],
}

impl<T, const DIM: usize> Deref for GridBuffer<T, DIM> {
    type Target = Vec<T>;

    fn deref(&self) -> &Self::Target {
        &self.value
    }
}

impl<T, const DIM: usize> DerefMut for GridBuffer<T, DIM> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.value
    }
}

impl<T, const DIM: usize> GridBuffer<T, DIM> {
    /// Creates a new `GridBuffer` from a raw vector and shape.
    pub fn from_vec(value: Vec<T>, shape: [usize; DIM]) -> Self {
        let mut strides = [1; DIM];
        for i in 1..DIM {
            strides[i] = strides[i - 1] * shape[i - 1];
        }

        Self {
            value,
            shape,
            strides,
        }
    }

    /// Returns a reference to the element at the specified grid position.
    pub fn at(&self, pos: [usize; DIM]) -> Option<&T> {
        let index = pos
            .into_iter()
            .zip(self.strides)
            .take(DIM)
            .fold(0, |sum, (p, s)| sum + p * s);
        self.value.get(index)
    }

    /// Returns a mutable reference to the element at the specified grid position.
    pub fn at_mut(&mut self, pos: [usize; DIM]) -> Option<&mut T> {
        let index = pos
            .into_iter()
            .zip(self.strides)
            .take(DIM)
            .fold(0, |sum, (p, s)| sum + p * s);
        self.value.get_mut(index)
    }

    /// Returns the shape of the grid.
    pub fn shape(&self) -> [usize; DIM] {
        self.shape
    }
}
