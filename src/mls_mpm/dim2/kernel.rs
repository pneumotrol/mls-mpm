pub(super) mod clear_grid;
pub(super) mod grid_to_particle;
pub(super) mod particle_to_grid;
pub(super) mod particle_to_grid_i32_atomic;
pub(super) mod update_grid;

use crate::mls_mpm::SimulationParameters;
use cubecl::prelude::*;

/// Checks if a 2D grid position is within the valid bounds of the grid.
#[cube]
fn is_grid_pos_valid<I: Int>(grid_pos: (I, I), grid_dim: (usize, usize)) -> bool {
    grid_pos.0 >= I::new(0)
        && grid_pos.1 >= I::new(0)
        && grid_pos.0 < I::cast_from(grid_dim.0)
        && grid_pos.1 < I::cast_from(grid_dim.1)
}

/// Converts a 2D grid position `(x, y)` to a flat 1D buffer index.
#[cube]
fn pos_to_idx<I: Int>(grid_pos: (I, I), grid_dim: (usize, usize)) -> usize {
    usize::cast_from(grid_pos.1) * grid_dim.0 + usize::cast_from(grid_pos.0)
}

/// Converts a flat 1D buffer index back to a 2D grid position `(x, y)`.
#[cube]
fn idx_to_pos(grid_index: usize, grid_dim: (usize, usize)) -> (i32, i32) {
    (
        (grid_index % grid_dim.0) as i32,
        (grid_index / grid_dim.0) as i32,
    )
}

#[cube]
fn bottom_left_of_3x3_grid<F: Float, I: Int>(position: (F, F), grid_size: F) -> (I, I) {
    let position_in_grid = (position.0 / grid_size, position.1 / grid_size);

    (
        I::cast_from(position_in_grid.0.floor()) - I::new(1),
        I::cast_from(position_in_grid.1.floor()) - I::new(1),
    )
}

#[cube]
fn quadratic_weights<F: Float>(position: (F, F), grid_size: F) -> ((F, F, F), (F, F, F)) {
    let position_in_grid = (position.0 / grid_size, position.1 / grid_size);

    let fx = (
        position_in_grid.0 - (position_in_grid.0.floor() + F::new(0.5)),
        position_in_grid.1 - (position_in_grid.1.floor() + F::new(0.5)),
    );

    (
        (
            F::new(0.50) * (fx.0 - F::new(0.50)) * (fx.0 - F::new(0.50)),
            F::new(0.75) - fx.0 * fx.0,
            F::new(0.50) * (fx.0 + F::new(0.50)) * (fx.0 + F::new(0.50)),
        ),
        (
            F::new(0.50) * (fx.1 - F::new(0.50)) * (fx.1 - F::new(0.50)),
            F::new(0.75) - fx.1 * fx.1,
            F::new(0.50) * (fx.1 + F::new(0.50)) * (fx.1 + F::new(0.50)),
        ),
    )
}

/// Calculates the distance vector from a particle's absolute position to the center of a grid cell.
///
/// This is useful for calculating affine momentum transfers in the MLS-MPM method.
#[cube]
fn particle_to_grid_center<F: Float, I: Int>(
    grid_pos: (I, I),
    position: (F, F),
    grid_size: F,
) -> (F, F) {
    let grid_center = (
        F::cast_from(grid_pos.0) * grid_size + F::new(0.5) * grid_size,
        F::cast_from(grid_pos.1) * grid_size + F::new(0.5) * grid_size,
    );
    (grid_center.0 - position.0, grid_center.1 - position.1)
}

/// Retrieves the combined 2D interpolation weight (wx * wy) for a given stencil offset.
///
/// # Arguments
///
/// * `offset` - Relative index within the 3x3 stencil (0, 1, or 2).
/// * `weight` - Precomputed B-spline weights for X and Y directions.
#[cube]
fn grid_weight<F: Float, I: Int>(offset: (I, I), weights: ((F, F, F), (F, F, F))) -> F {
    let wx = if offset.0 == I::new(0) {
        weights.0.0
    } else if offset.0 == I::new(1) {
        weights.0.1
    } else {
        weights.0.2
    };

    let wy = if offset.1 == I::new(0) {
        weights.1.0
    } else if offset.1 == I::new(1) {
        weights.1.1
    } else {
        weights.1.2
    };

    wx * wy
}
