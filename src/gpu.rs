//! GPU utility module for managing buffers and abstractions.

use cubecl::{CubeScalar, prelude::*, server::Handle};
use std::{
    marker::PhantomData,
    ops::{Deref, DerefMut},
};

const NUM_WORKGROUPS: usize = 256;

/// A generic GPU buffer for a specific runtime and type.
///
/// This structure provides a high-level API for creating, reading, and writing
/// data to the GPU using CubeCL's ComputeClient.
#[derive(Clone)]
pub(crate) struct GpuBuffer<'a, R: Runtime, T> {
    client: &'a ComputeClient<R>,
    handle: Handle,
    len: usize,
    _marker: PhantomData<(R, T)>,
}

impl<'a, R: Runtime, T: CubeScalar> GpuBuffer<'a, R, T> {
    /// Creates an empty buffer of a given length on the GPU.
    pub(crate) fn new(client: &'a ComputeClient<R>, len: usize) -> Self {
        Self {
            client,
            handle: client.empty(core::mem::size_of::<T>() * len),
            len,
            _marker: PhantomData,
        }
    }

    /// Reads data from the GPU buffer back into a `Vec` on the CPU.
    pub(crate) fn read(&self) -> Vec<T> {
        let result = self.client.read_one(self.handle.clone());
        bytemuck::cast_slice(&result).to_vec()
    }

    /// Overwrites the data in the GPU buffer with the provided CPU slice.
    ///
    /// # Panics
    ///
    /// Panics if the input data length does not match the buffer length.
    pub(crate) fn write(&self, data: &[T]) {
        assert_eq!(
            data.len(),
            self.len(),
            "Buffer length mismatch during write"
        );

        unsafe {
            // We use a simple copy kernel to write data to the buffer.
            // CubeCL's create_from_slice creates a temporary buffer, which we then copy from.
            let _ = write_buffer::launch::<f32, R>(
                self.client,
                CubeCount::new_1d(self.len.div_ceil(NUM_WORKGROUPS) as u32),
                CubeDim::new_1d(NUM_WORKGROUPS as u32),
                ArrayArg::from_raw_parts::<T>(
                    &self.client.create_from_slice(bytemuck::cast_slice(data)),
                    data.len(),
                    1,
                ),
                self.as_array_arg(),
            );
        }
    }

    /// Returns the number of elements in the buffer.
    pub(crate) fn len(&self) -> usize {
        self.len
    }

    /// Converts the buffer to a raw `ArrayArg` for use in CubeCL kernel launches.
    ///
    /// # Safety
    ///
    /// This method is unsafe as it exposes raw buffer handles. The caller must ensure
    /// the buffer is valid and the length is correct during the kernel execution.
    pub(crate) unsafe fn as_array_arg(&self) -> ArrayArg<'_, R> {
        unsafe { ArrayArg::from_raw_parts::<T>(&self.handle, self.len, 1) }
    }
}

/// A GPU-backed scalar or a single-component buffer.
#[derive(Clone)]
pub(crate) struct GpuScalar<'a, R: Runtime, T>(GpuBuffer<'a, R, T>);

impl<'a, R: Runtime, T> Deref for GpuScalar<'a, R, T> {
    type Target = GpuBuffer<'a, R, T>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<'a, R: Runtime, T> DerefMut for GpuScalar<'a, R, T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

impl<'a, R: Runtime, T: CubeScalar> GpuScalar<'a, R, T> {
    /// Initializes a new `GpuScalar` with an empty buffer.
    pub(crate) fn new(client: &'a ComputeClient<R>, len: usize) -> Self {
        Self(GpuBuffer::new(client, len))
    }
}

/// A GPU-backed vector of size `N`, where each component is a separate buffer.
#[derive(Clone)]
pub(crate) struct GpuVector<'a, R: Runtime, T, const N: usize>([GpuBuffer<'a, R, T>; N]);

impl<'a, R: Runtime, T, const N: usize> Deref for GpuVector<'a, R, T, N> {
    type Target = [GpuBuffer<'a, R, T>; N];

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<'a, R: Runtime, T, const N: usize> DerefMut for GpuVector<'a, R, T, N> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

impl<'a, R: Runtime, T: CubeScalar, const N: usize> GpuVector<'a, R, T, N> {
    /// Initializes a new `GpuVector` with empty buffers.
    pub(crate) fn new(client: &'a ComputeClient<R>, len: usize) -> Self {
        Self(core::array::from_fn(|_| GpuBuffer::new(client, len)))
    }
}

/// A GPU-backed matrix of size `NxN`, where each entry is a separate buffer.
#[derive(Clone)]
pub(crate) struct GpuMatrix<'a, R: Runtime, T, const N: usize>([[GpuBuffer<'a, R, T>; N]; N]);

impl<'a, R: Runtime, T, const N: usize> Deref for GpuMatrix<'a, R, T, N> {
    type Target = [[GpuBuffer<'a, R, T>; N]; N];

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<'a, R: Runtime, T, const N: usize> DerefMut for GpuMatrix<'a, R, T, N> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

impl<'a, R: Runtime, T: CubeScalar, const N: usize> GpuMatrix<'a, R, T, N> {
    /// Initializes a new `GpuMatrix` with empty buffers.
    pub(crate) fn new(client: &'a ComputeClient<R>, len: usize) -> Self {
        Self(core::array::from_fn(|_| {
            core::array::from_fn(|_| GpuBuffer::new(client, len))
        }))
    }
}

/// Internal kernel for writing CPU data to a GPU buffer.
#[cube(launch)]
pub(crate) fn write_buffer<F: Float>(src: &Array<Line<F>>, dst: &mut Array<Line<F>>) {
    let i = ABSOLUTE_POS;
    if i >= src.len() || i >= dst.len() {
        terminate!();
    }

    dst[i] = src[i];
}
