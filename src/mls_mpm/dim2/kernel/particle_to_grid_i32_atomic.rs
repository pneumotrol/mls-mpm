//! Kernel for clearing and accumulating particle data onto the grid using integer atomics.

use super::*;

/// Resets integer grid buffers to zero.
#[cube(launch_unchecked)]
pub(crate) fn clear_grid_i32<I: Int>(
    g_mass: &mut Array<Line<I>>,
    g_velocity_x: &mut Array<Line<I>>,
    g_velocity_y: &mut Array<Line<I>>,
    dim_grids: (usize, usize),
) {
    let j = ABSOLUTE_POS;
    let num_grids = dim_grids.0 * dim_grids.1;
    if j >= num_grids {
        terminate!();
    }

    g_mass[j] = Line::new(I::new(0));
    g_velocity_x[j] = Line::new(I::new(0));
    g_velocity_y[j] = Line::new(I::new(0));
}

/// Performs the Particle-to-Grid (P2G) transfer using scaled `i32` atomic operations.
///
/// This provides a fallback for hardware that does not support native `f32` atomic additions.
#[cube(launch_unchecked)]
pub(crate) fn particle_to_grid_i32_atomic<F: Float, I: Int>(
    // Particles
    p_position_x: &Array<F>,
    p_position_y: &Array<F>,
    p_velocity_x: &Array<F>,
    p_velocity_y: &Array<F>,
    p_deformation_gradient_xx: &Array<F>,
    p_deformation_gradient_xy: &Array<F>,
    p_deformation_gradient_yx: &Array<F>,
    p_deformation_gradient_yy: &Array<F>,
    p_affine_velocity_xx: &Array<F>,
    p_affine_velocity_xy: &Array<F>,
    p_affine_velocity_yx: &Array<F>,
    p_affine_velocity_yy: &Array<F>,
    p_mass: &Array<F>,
    p_volume: &Array<F>,
    p_material_id: &Array<u32>,
    num_particles: usize,
    // Grids (Integer atomics)
    g_mass: &mut Array<Atomic<I>>,
    g_velocity_x: &mut Array<Atomic<I>>,
    g_velocity_y: &mut Array<Atomic<I>>,
    dim_grids: (usize, usize),
    // Particle physical parameters
    m_kind: &Array<u32>,
    m_param_01: &Array<F>,
    m_param_02: &Array<F>,
    m_param_03: &Array<F>,
    num_materials: usize,
    // Simulation parameters
    sim_param: SimulationParameters,
    atomic_scale: f32,
) {
    let i = ABSOLUTE_POS;
    if i >= num_particles {
        terminate!();
    }

    // Fetch particle state.
    let x = (p_position_x[i], p_position_y[i]);
    let dxdt = (p_velocity_x[i], p_velocity_y[i]);
    let f = (
        (p_deformation_gradient_xx[i], p_deformation_gradient_xy[i]),
        (p_deformation_gradient_yx[i], p_deformation_gradient_yy[i]),
    );
    let c = (
        (p_affine_velocity_xx[i], p_affine_velocity_xy[i]),
        (p_affine_velocity_yx[i], p_affine_velocity_yy[i]),
    );
    let m = p_mass[i];
    let v = p_volume[i];
    let material_id = usize::cast_from(p_material_id[i]);

    // Fetch simulation parameters.
    let dt = F::cast_from(sim_param.dt);
    let grid_size = F::cast_from(sim_param.grid_size);

    // Fetch material kind.
    if material_id >= num_materials {
        terminate!();
    }
    let material_kind = m_kind[material_id];

    // Compute interpolation parameters.
    let grid_origin = bottom_left_of_3x3_grid::<F, I>(x, grid_size);
    let weights = quadratic_weights::<F>(x, grid_size);
    let d_inv = F::new(4.0) / (grid_size * grid_size);
    let jacobian = f.0.0 * f.1.1 - f.0.1 * f.1.0;

    let mut stress = ((F::new(0.0), F::new(0.0)), (F::new(0.0), F::new(0.0)));
    match material_kind {
        0 => {
            let k = m_param_01[material_id];
            let e = m_param_02[material_id];
            let mu = m_param_03[material_id];

            let pressure = e * ((-k * jacobian.ln()).exp() - F::new(1.0));

            stress.0.0 = pressure + mu * (c.0.0 + c.0.0);
            stress.0.1 = mu * (c.0.1 + c.1.0);
            stress.1.0 = mu * (c.1.0 + c.0.1);
            stress.1.1 = pressure + mu * (c.1.1 + c.1.1);
        }
        1 => {
            let mu = m_param_01[material_id];
            let lambda = m_param_02[material_id];
            let j_inv = F::new(1.0) / jacobian;

            let b_xx = f.0.0 * f.0.0 + f.0.1 * f.0.1;
            let b_xy = f.0.0 * f.1.0 + f.0.1 * f.1.1;
            let b_yx = b_xy;
            let b_yy = f.1.0 * f.1.0 + f.1.1 * f.1.1;

            let vol_stress = (lambda * jacobian.ln()) * j_inv;

            stress.0.0 = -((mu * j_inv) * (b_xx - F::new(1.0)) + vol_stress);
            stress.0.1 = -((mu * j_inv) * b_xy);
            stress.1.0 = -((mu * j_inv) * b_yx);
            stress.1.1 = -((mu * j_inv) * b_yy + vol_stress);
        }
        _ => {}
    }

    // Compute Affine momentum matrix.
    let affine = (
        (
            m * c.0.0 + stress.0.0 * v * d_inv * dt,
            m * c.0.1 + stress.0.1 * v * d_inv * dt,
        ),
        (
            m * c.1.0 + stress.1.0 * v * d_inv * dt,
            m * c.1.1 + stress.1.1 * v * d_inv * dt,
        ),
    );

    // Scatter contributions to neighbors using fixed-point integer scaling.
    for gy in 0..3_usize {
        for gx in 0..3_usize {
            let offset = (I::cast_from(gx), I::cast_from(gy));
            let grid_pos = (grid_origin.0 + offset.0, grid_origin.1 + offset.1);

            if is_grid_pos_valid::<I>(grid_pos, dim_grids) {
                let j = pos_to_idx::<I>(grid_pos, dim_grids);
                let r_p2g = particle_to_grid_center::<F, I>(grid_pos, x, grid_size);
                let w = grid_weight::<F, I>(offset, weights);

                let mass_contrib = w * m;
                let momentum_contrib = (
                    w * (m * dxdt.0 + affine.0.0 * r_p2g.0 + affine.0.1 * r_p2g.1),
                    w * (m * dxdt.1 + affine.1.0 * r_p2g.0 + affine.1.1 * r_p2g.1),
                );

                let scale = F::cast_from(atomic_scale);
                g_mass[j].fetch_add(I::cast_from(mass_contrib * scale));
                g_velocity_x[j].fetch_add(I::cast_from(momentum_contrib.0 * scale));
                g_velocity_y[j].fetch_add(I::cast_from(momentum_contrib.1 * scale));
            }
        }
    }
}

/// Converts accumulated fixed-point integer values back to floating point.
#[cube(launch_unchecked)]
pub(crate) fn i32_to_f32<F: Float, I: Int>(
    g_mass: &mut Array<F>,
    g_velocity_x: &mut Array<F>,
    g_velocity_y: &mut Array<F>,
    dim_grids: (usize, usize),
    g_mass_buffer: &Array<I>,
    g_velocity_x_buffer: &Array<I>,
    g_velocity_y_buffer: &Array<I>,
    atomic_scale: f32,
) {
    let j = ABSOLUTE_POS;
    let num_grids = dim_grids.0 * dim_grids.1;
    if j >= num_grids {
        terminate!();
    }

    let scale = F::cast_from(atomic_scale);
    g_mass[j] = F::cast_from(g_mass_buffer[j]) / scale;
    g_velocity_x[j] = F::cast_from(g_velocity_x_buffer[j]) / scale;
    g_velocity_y[j] = F::cast_from(g_velocity_y_buffer[j]) / scale;
}
