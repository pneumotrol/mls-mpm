//! Kernel for clearing the grid state at the beginning of each simulation step.

use super::*;

/// Clears the mass and momentum buffers of the grid nodes.
///
/// This kernel prepares the Eulerian grid for the Particle-to-Grid (P2G) phase by
/// resetting all accumulation buffers to zero.
#[cube(launch_unchecked)]
pub(crate) fn clear_grid<F: Float>(
    // accumulated mass at grid nodes.
    g_mass: &mut Array<Line<F>>,
    // accumulated X-momentum at grid nodes.
    g_velocity_x: &mut Array<Line<F>>,
    // accumulated Y-momentum at grid nodes.
    g_velocity_y: &mut Array<Line<F>>,
    // dimensions of the background grid.
    dim_grids: (usize, usize),
) {
    let j = ABSOLUTE_POS;
    let num_grids = dim_grids.0 * dim_grids.1;
    if j >= num_grids {
        terminate!();
    }

    // Initialize grid mass and momentum buffers to zero.
    g_mass[j] = Line::new(F::new(0.0));
    g_velocity_x[j] = Line::new(F::new(0.0));
    g_velocity_y[j] = Line::new(F::new(0.0));
}
