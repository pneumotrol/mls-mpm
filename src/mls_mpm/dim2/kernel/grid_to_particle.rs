//! Kernel for transferring grid data back to particles (G2P) and updating their positions.

use super::*;

/// Performs the Grid-to-Particle (G2P) transfer and advects particles.
///
/// This kernel implements the reverse mapping from the Eulerian grid to Lagrangian particles.
/// It performs the following operations:
/// 1. Interpolates node velocities back to particle positions.
/// 2. Updates the deformation gradient (F) based on the local velocity gradient.
/// 3. Advects particle positions using explicit Euler integration.
/// 4. Handles material-specific constitutive updates (e.g., fluid volume preservation).
#[cube(launch_unchecked)]
pub(crate) fn grid_to_particle<F: Float, I: Int>(
    // Particles
    p_position_x: &mut Array<F>,
    p_position_y: &mut Array<F>,
    p_velocity_x: &mut Array<F>,
    p_velocity_y: &mut Array<F>,
    p_deformation_gradient_xx: &mut Array<F>,
    p_deformation_gradient_xy: &mut Array<F>,
    p_deformation_gradient_yx: &mut Array<F>,
    p_deformation_gradient_yy: &mut Array<F>,
    p_affine_velocity_xx: &mut Array<F>,
    p_affine_velocity_xy: &mut Array<F>,
    p_affine_velocity_yx: &mut Array<F>,
    p_affine_velocity_yy: &mut Array<F>,
    p_material_id: &Array<u32>,
    num_particles: usize,
    // Grids
    g_velocity_x: &Array<F>,
    g_velocity_y: &Array<F>,
    dim_grids: (usize, usize),
    // Materials
    m_kind: &Array<u32>,
    num_materials: usize,
    // Simulation parameters
    sim_param: SimulationParameters,
) {
    let i = ABSOLUTE_POS;
    if i >= num_particles {
        terminate!();
    }

    // Fetch particle state.
    let x = (p_position_x[i], p_position_y[i]);
    let f = (
        (p_deformation_gradient_xx[i], p_deformation_gradient_xy[i]),
        (p_deformation_gradient_yx[i], p_deformation_gradient_yy[i]),
    );
    let material_id = usize::cast_from(p_material_id[i]);

    // Fetch simulation parameters.
    let dt = F::cast_from(sim_param.dt);
    let grid_size = F::cast_from(sim_param.grid_size);

    // Fetch material kind.
    if material_id >= num_materials {
        terminate!();
    }
    let material_kind = m_kind[material_id];

    // Compute interpolation weights.
    let grid_origin = bottom_left_of_3x3_grid::<F, I>(x, grid_size);
    let weights = quadratic_weights::<F>(x, grid_size);
    let d_inv = F::new(4.0) / (grid_size * grid_size);

    // Interpolate velocity and compute the affine velocity matrix (C) from neighboring grid nodes.
    let mut dxdt = (F::new(0.0), F::new(0.0));
    let mut c = ((F::new(0.0), F::new(0.0)), (F::new(0.0), F::new(0.0)));

    for gy in 0..3_u32 {
        for gx in 0..3_u32 {
            let offset = (I::cast_from(gx), I::cast_from(gy));
            let grid_pos = (grid_origin.0 + offset.0, grid_origin.1 + offset.1);

            if is_grid_pos_valid::<I>(grid_pos, dim_grids) {
                let j = pos_to_idx::<I>(grid_pos, dim_grids);
                let r_p2g = particle_to_grid_center::<F, I>(grid_pos, x, grid_size);
                let w = grid_weight::<F, I>(offset, weights);

                let weighted_velocity = (w * g_velocity_x[j], w * g_velocity_y[j]);
                dxdt.0 += weighted_velocity.0;
                dxdt.1 += weighted_velocity.1;
                c.0.0 += weighted_velocity.0 * r_p2g.0 * d_inv;
                c.0.1 += weighted_velocity.0 * r_p2g.1 * d_inv;
                c.1.0 += weighted_velocity.1 * r_p2g.0 * d_inv;
                c.1.1 += weighted_velocity.1 * r_p2g.1 * d_inv;
            }
        }
    }

    // Update the deformation gradient (F_new = (I + C * dt) * F_old).
    let mut f_new = (
        (
            (F::new(1.0) + c.0.0 * dt) * f.0.0 + (c.0.1 * dt) * f.1.0,
            (F::new(1.0) + c.0.0 * dt) * f.0.1 + (c.0.1 * dt) * f.1.1,
        ),
        (
            (c.1.0 * dt) * f.0.0 + (F::new(1.0) + c.1.1 * dt) * f.1.0,
            (c.1.0 * dt) * f.0.1 + (F::new(1.0) + c.1.1 * dt) * f.1.1,
        ),
    );

    // Apply material-specific deformation updates.
    match material_kind {
        0 => {
            // Fluid: Preserve volume by enforcing only the determinant (J) change.
            let jacobian = f_new.0.0 * f_new.1.1 - f_new.0.1 * f_new.1.0;
            f_new.0.0 = F::new(1.0);
            f_new.0.1 = F::new(0.0);
            f_new.1.0 = F::new(0.0);
            f_new.1.1 = jacobian;
        }
        1 => {
            // Elastic: Purely elastic Neo-Hookean solid. Keep F_new as is.
        }
        _ => {
            terminate!();
        }
    }

    // Advect particle position and store updated state.
    p_position_x[i] = x.0 + dxdt.0 * dt;
    p_position_y[i] = x.1 + dxdt.1 * dt;
    p_velocity_x[i] = dxdt.0;
    p_velocity_y[i] = dxdt.1;
    p_deformation_gradient_xx[i] = f_new.0.0;
    p_deformation_gradient_xy[i] = f_new.0.1;
    p_deformation_gradient_yx[i] = f_new.1.0;
    p_deformation_gradient_yy[i] = f_new.1.1;
    p_affine_velocity_xx[i] = c.0.0;
    p_affine_velocity_xy[i] = c.0.1;
    p_affine_velocity_yx[i] = c.1.0;
    p_affine_velocity_yy[i] = c.1.1;
}
