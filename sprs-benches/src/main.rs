use sprs_rand::rand_csr_std;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let shapes_and_densities = [
        ((15, 25), vec![0.1]),
        (
            (1500, 2500),
            vec![1e-4, 5e-4, 1e-3, 2e-3, 5e-3, 8e-3, 1e-2, 2e-2, 5e-2],
        ),
        (
            (15000, 25000),
            vec![1e-5, 2e-5, 5e-5, 1e-4, 2e-4, 5e-4, 1e-3, 2e-3, 3e-3, 5e-3],
        ),
        ((150000, 25000), vec![1e-7, 1e-6, 1e-5, 1e-4]),
        ((150000, 250000), vec![1e-7, 1e-6, 1e-5, 1e-4]),
    ];

    for (shape, densities) in &shapes_and_densities {
        let mut times = Vec::with_capacity(densities.len());
        let mut times_old = Vec::with_capacity(densities.len());
        let mut nnzs = Vec::with_capacity(densities.len());
        let mut res_densities = Vec::with_capacity(densities.len());
        let mut workspace = vec![0.; shape.0];
        for &density in densities {
            println!("Generating matrices");
            let now = std::time::Instant::now();
            let m1 = rand_csr_std(*shape, density);
            let m2 = rand_csr_std((shape.1, shape.0), density);
            let elapsed = now.elapsed().as_millis();
            println!("Generating matrices took {}ms", elapsed);

            let now = std::time::Instant::now();
            let prod = &m1 * &m2;
            let elapsed = now.elapsed().as_millis();
            println!(
                "New product of shape ({}, {}) and density {} done in {}ms",
                shape.0, shape.1, density, elapsed,
            );
            times.push(elapsed);
            let now = std::time::Instant::now();
            let old_res = sprs::prod::csr_mul_csr(&m1, &m2, &mut workspace);
            let elapsed = now.elapsed().as_millis();
            println!(
                "Old product of shape ({}, {}) and density {} done in {}ms",
                shape.0, shape.1, density, elapsed,
            );
            times_old.push(elapsed);
            assert_eq!(prod, old_res);
            nnzs.push(prod.nnz());
            res_densities.push(prod.density());
        }
        println!("Results for shape: ({}, {})", shape.0, shape.1);
        println!("Product nnzs: {:?}", nnzs);
        println!("Product densities: {:?}", res_densities);
        println!("Product times: {:?}", times);
        println!("Product times (old): {:?}", times_old);

        // plot
        {
            use plotters::prelude::*;
            let title = format!("sparse_mult_perf_{}_{}.png", shape.0, shape.1);
            let res = (640, 480);
            let root = BitMapBackend::new(&title, res).into_drawing_area();
            root.fill(&WHITE)?;
            let max_density = *res_densities.last().unwrap_or(&1.) as f32;
            let max_time =
                *std::cmp::max(times.iter().max(), times_old.iter().max())
                    .unwrap_or(&1) as f32;
            let mut chart = ChartBuilder::on(&root)
                .caption("Time vs density", ("sans-serif", 50).into_font())
                .margin(5)
                .x_label_area_size(30)
                .y_label_area_size(50)
                .build_ranged(0f32..max_density, 0f32..max_time)?;

            chart.configure_mesh().draw()?;

            chart
                .draw_series(LineSeries::new(
                    res_densities
                        .iter()
                        .map(|d| *d as f32)
                        .zip(times.iter().map(|t| *t as f32)),
                    &RED,
                ))?
                .label("SMMP")
                .legend(|(x, y)| {
                    PathElement::new(vec![(x, y), (x + 20, y)], &RED)
                });

            chart
                .draw_series(LineSeries::new(
                    res_densities
                        .iter()
                        .map(|d| *d as f32)
                        .zip(times_old.iter().map(|t| *t as f32)),
                    &BLUE,
                ))?
                .label("Old prod")
                .legend(|(x, y)| {
                    PathElement::new(vec![(x, y), (x + 20, y)], &BLUE)
                });

            chart
                .configure_series_labels()
                .background_style(&WHITE.mix(0.8))
                .border_style(&BLACK)
                .draw()?;
        }
    }

    Ok(())
}
