use color_eyre::Result;
use itertools::izip;
use plotters::prelude::*;

pub fn plot_particles(
    filename: &str,
    (px, py): (&[f32], &[f32]),
    (vx, vy): (&[f32], &[f32]),
    grid_size: f32,
    dim_grids: &[usize; 2],
) -> Result<()> {
    let (xmax, ymax) = {
        (
            dim_grids[0] as f32 * grid_size,
            dim_grids[1] as f32 * grid_size,
        )
    };
    let (width, height) = {
        let base_size = 800.0;
        let max = if xmax > ymax { xmax } else { ymax };
        (
            (xmax / max * base_size) as u32,
            (ymax / max * base_size) as u32,
        )
    };

    // Plot setup
    let root = BitMapBackend::new(filename, (width, height)).into_drawing_area();
    root.fill(&WHITE)?;
    let mut chart = ChartBuilder::on(&root)
        .caption(filename, ("sans-selif", 30).into_font())
        .margin(10)
        .x_label_area_size(30)
        .y_label_area_size(30)
        .build_cartesian_2d(0.0..xmax, 0.0..ymax)?;
    chart.configure_mesh().draw()?;

    // Plot particles
    chart.draw_series(izip!(px, py).map(|(&px, &py)| Circle::new((px, py), 5, RED.filled())))?;

    // Plot particle velocities
    let v_norm_max = izip!(vx, vy)
        .map(|(&px, &py)| clamp(norm(px, py)))
        .reduce(f32::max)
        .unwrap();
    chart.draw_series(izip!(px, py, vx, vy).map(|(&px, &py, &vx, &vy)| {
        let (vx, vy) = (vx / v_norm_max * grid_size, vy / v_norm_max * grid_size);
        PathElement::new([(px, py), (px + vx, py + vy)], BLUE)
    }))?;

    Ok(())
}

pub fn plot_grids(
    filename: &str,
    gm: &[f32],
    (gvx, gvy): (&[f32], &[f32]),
    boundary: &[u32],
    (fvx, fvy): (&[f32], &[f32]),
    grid_size: f32,
    dim_grids: &[usize; 2],
) -> Result<()> {
    let (xmax, ymax) = {
        (
            dim_grids[0] as f32 * grid_size,
            dim_grids[1] as f32 * grid_size,
        )
    };
    let (width, height) = {
        let base_size = 800.0;
        let max = if xmax > ymax { xmax } else { ymax };
        (
            (xmax / max * base_size) as u32,
            (ymax / max * base_size) as u32,
        )
    };

    // Plot setup
    let root = BitMapBackend::new(filename, (width, height)).into_drawing_area();
    root.fill(&WHITE)?;
    let mut chart = ChartBuilder::on(&root)
        .caption(filename, ("sans-selif", 30).into_font())
        .margin(10)
        .x_label_area_size(30)
        .y_label_area_size(30)
        .build_cartesian_2d(0.0..xmax, 0.0..ymax)?;
    chart.configure_mesh().draw()?;

    // Fill grids with mass (density)
    let gm_max = gm.iter().map(|&x| clamp(x)).reduce(f32::max).unwrap();
    chart.draw_series(gm.iter().enumerate().map(|(i, &gm)| {
        let (gx, gy) = idx_to_pos(i, dim_grids, grid_size);
        Rectangle::new(
            [
                (gx - 0.5 * grid_size, gy - 0.5 * grid_size),
                (gx + 0.5 * grid_size, gy + 0.5 * grid_size),
            ],
            RGBAColor(0, 0, 0, (gm / gm_max).into()).filled(),
        )
    }))?;
    chart.draw_series(boundary.iter().enumerate().map(|(i, &boundary)| {
        let (gx, gy) = idx_to_pos(i, dim_grids, grid_size);
        // Apply colors based on BoundaryType (defined in types.rs)
        let color = match boundary {
            0 => {
                // BoundaryType::None
                RGBAColor(0, 0, 0, 0.0)
            }
            1 => {
                // BoundaryType::Sticky
                RGBAColor(255, 0, 0, 0.5)
            }
            20..=23 => {
                // BoundaryType::SlipLeft, SlipRight, SlipBottom, or SlipTop
                RGBAColor(0, 0, 255, 0.5)
            }
            _ => RGBAColor(0, 0, 0, 0.0),
        };
        Rectangle::new(
            [
                (gx - 0.5 * grid_size, gy - 0.5 * grid_size),
                (gx + 0.5 * grid_size, gy + 0.5 * grid_size),
            ],
            color.filled(),
        )
    }))?;

    // Plot grid velocities
    let gv_norm_max = izip!(gvx, gvy)
        .map(|(&gvx, &gvy)| clamp(norm(gvx, gvy)))
        .reduce(f32::max)
        .unwrap();
    chart.draw_series(izip!(gvx, gvy).enumerate().map(|(i, (&gvx, &gvy))| {
        let (gx, gy) = idx_to_pos(i, dim_grids, grid_size);
        let (gvx, gvy) = (gvx / gv_norm_max * grid_size, gvy / gv_norm_max * grid_size);
        PathElement::new([(gx, gy), (gx + gvx, gy + gvy)], BLUE)
    }))?;

    // Plot environment forces
    let fv_norm_max = izip!(fvx, fvy)
        .map(|(&fvx, &fvy)| clamp(norm(fvx, fvy)))
        .reduce(f32::max)
        .unwrap();
    chart.draw_series(izip!(fvx, fvy).enumerate().map(|(i, (&fvx, &fvy))| {
        let (fx, fy) = idx_to_pos(i, dim_grids, grid_size);
        let (fvx, fvy) = (fvx / fv_norm_max * grid_size, fvy / fv_norm_max * grid_size);
        PathElement::new([(fx, fy), (fx + fvx, fy + fvy)], RED)
    }))?;

    Ok(())
}

fn idx_to_pos(i: usize, dim_grids: &[usize; 2], grid_size: f32) -> (f32, f32) {
    (
        ((i % dim_grids[0]) as f32 + 0.5) * grid_size,
        ((i / dim_grids[0]) as f32 + 0.5) * grid_size,
    )
}

fn norm(x: f32, y: f32) -> f32 {
    (x * x + y * y).sqrt()
}

fn clamp(x: f32) -> f32 {
    x.clamp(f32::EPSILON, f32::MAX)
}
