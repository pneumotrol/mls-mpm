//! Kernel for updating the grid state and applying boundary conditions.

use super::*;

/// Updates grid node velocities by applying external forces and enforcing boundary conditions.
///
/// This kernel executes the Eulerian update phase:
/// 1. Normalizes momentum to velocity by dividing by node mass.
/// 2. Applies external acceleration fields (e.g., gravity).
/// 3. Handles Sticky or Slip boundary conditions based on node identifiers.
#[cube(launch_unchecked)]
pub(crate) fn update_grid<F: Float>(
    // Grids
    g_mass: &mut Array<F>,
    g_velocity_x: &mut Array<F>,
    g_velocity_y: &mut Array<F>,
    dim_grids: (usize, usize),
    // Grid environments
    e_boundary: &Array<u32>,
    e_acceleration_x: &Array<F>,
    e_acceleration_y: &Array<F>,
    // Simulation parameters
    sim_param: SimulationParameters,
) {
    let j = ABSOLUTE_POS;
    let num_grids = dim_grids.0 * dim_grids.1;
    if j >= num_grids {
        terminate!();
    }

    // Fetch accumulated grid properties.
    let m = g_mass[j];
    let mut v = (g_velocity_x[j], g_velocity_y[j]);

    // Fetch environmental forces and boundary data.
    let a = (e_acceleration_x[j], e_acceleration_y[j]);
    let boundary_type = e_boundary[j];
    let dt = F::cast_from(sim_param.dt);

    // Normalize momentum to velocity and apply external accelerations.
    if m > F::cast_from(f32::EPSILON) {
        v.0 /= m;
        v.1 /= m;

        v.0 += a.0 * dt;
        v.1 += a.1 * dt;
    } else {
        v.0 = F::new(0.0);
        v.1 = F::new(0.0);
    }

    // Apply boundary conditions (0: None, 1: Sticky, 20-23: Slip).
    match boundary_type {
        0_u32 => {
            // BoundaryType::None
        }
        1_u32 => {
            // BoundaryType::Sticky - Set all velocity components to zero.
            v.0 = F::new(0.0);
            v.1 = F::new(0.0);
        }
        20_u32 | 21_u32 => {
            // BoundaryType::SlipLeft or SlipRight - Zero out normal (X) velocity.
            v.0 = F::new(0.0);
        }
        22_u32 | 23_u32 => {
            // BoundaryType::SlipBottom or SlipTop - Zero out normal (Y) velocity.
            v.1 = F::new(0.0);
        }
        _ => {}
    }

    // Write updated velocity back to the grid.
    g_velocity_x[j] = v.0;
    g_velocity_y[j] = v.1;
}
