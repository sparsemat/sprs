#[cfg(feature = "nightly")]
use pyo3::{
    prelude::*,
    types::{IntoPyDict, PyModule},
};
use sprs_rand::rand_csr_std;

#[cfg(feature = "nightly")]
fn scipy_mat<'a>(
    scipy_sparse: &'a PyModule,
    py: &Python,
    mat: &sprs::CsMat<f64>,
) -> Result<&'a PyAny, String> {
    scipy_sparse
        .call(
            "csr_matrix",
            ((
                mat.data().to_vec(),
                mat.indices().to_vec(),
                mat.indptr().to_vec(),
            ),),
            Some([("shape", mat.shape())].into_py_dict(*py)),
        )
        .map_err(|e| {
            let res = format!("Python error: {:?}", e);
            e.print_and_set_sys_last_vars(*py);
            res
        })
}

extern "C" {

    fn prod_nnz(
        a_rows: usize,
        a_cols: usize,
        b_cols: usize,
        a_indptr: *const isize,
        a_indices: *const isize,
        a_data: *const f64,
        b_indptr: *const isize,
        b_indices: *const isize,
        b_data: *const f64,
    ) -> usize;

}

fn eigen_prod(a: sprs::CsMatView<f64>, b: sprs::CsMatView<f64>) -> usize {
    let (a_rows, a_cols) = a.shape();
    let (b_rows, b_cols) = b.shape();
    assert_eq!(a_cols, b_rows);
    assert!(a.is_csr());
    assert!(a.rows() <= isize::MAX as usize);
    assert!(a.indptr()[a.rows()] <= isize::MAX as usize);
    assert!(b.is_csr());
    assert!(b.rows() <= isize::MAX as usize);
    assert!(b.indptr()[b.rows()] <= isize::MAX as usize);
    let a_indptr = a.indptr().as_ptr() as *const isize;
    let a_indices = a.indices().as_ptr() as *const isize;
    let a_data = a.data().as_ptr();
    let b_indptr = b.indptr().as_ptr() as *const isize;
    let b_indices = b.indices().as_ptr() as *const isize;
    let b_data = b.data().as_ptr();
    // Safety: sprs guarantees the validity of these pointers, and our wrapping
    // around Eigen respects the CSR format invariants. The safety thus relies
    // on the correctness of Eigen, which is well tested. The reinterpretation
    // of the index data is safe as the two types have the same size and accept
    // all bit patterns.
    // Correctness: relying on sprs guarantees on the CSR structure, we can
    // guarantee that no index data is greater than `isize::MAX`, thus the
    // reinterpretation of the data will not produce negative values.
    unsafe {
        prod_nnz(
            a_rows, a_cols, b_cols, a_indptr, a_indices, a_data, b_indptr,
            b_indices, b_data,
        )
    }
}

fn bench_densities() -> Result<(), Box<dyn std::error::Error>> {
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

    #[cfg(feature = "nightly")]
    let gil = Python::acquire_gil();
    #[cfg(feature = "nightly")]
    let py = gil.python();
    #[cfg(feature = "nightly")]
    let scipy_sparse = PyModule::import(py, "scipy.sparse").map_err(|e| {
        let res = format!("Python error: {:?}", e);
        e.print_and_set_sys_last_vars(py);
        res
    })?;

    for (shape, densities) in &shapes_and_densities {
        let mut times = Vec::with_capacity(densities.len());
        let mut times_old = Vec::with_capacity(densities.len());
        #[cfg(feature = "nightly")]
        let mut times_py = Vec::with_capacity(densities.len());
        let mut times_eigen = Vec::with_capacity(densities.len());
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

            // bench scipy as well
            #[cfg(feature = "nightly")]
            {
                let m1_py = scipy_mat(scipy_sparse, &py, &m1)?;
                let m2_py = scipy_mat(scipy_sparse, &py, &m2)?;
                let now = std::time::Instant::now();
                let _prod_py = py
                    .eval(
                        "m1 * m2",
                        Some([("m1", m1_py), ("m2", m2_py)].into_py_dict(py)),
                        None,
                    )
                    .map_err(|e| {
                        let res = format!("Python error: {:?}", e);
                        e.print_and_set_sys_last_vars(py);
                        res
                    })?;
                let elapsed = now.elapsed().as_millis();
                println!(
                    "Scipy product of shape ({}, {}) and density {} done in {}ms",
                    shape.0, shape.1, density, elapsed,
                );
                times_py.push(elapsed);
            }

            // bench eigen
            let now = std::time::Instant::now();
            let _nnz = eigen_prod(m1.view(), m2.view());
            let elapsed = now.elapsed().as_millis();
            println!(
                "Eigen product of shape ({}, {}) and density {} done in {}ms",
                shape.0, shape.1, density, elapsed,
            );
            times_eigen.push(elapsed);
        }
        println!("Results for shape: ({}, {})", shape.0, shape.1);
        println!("Product nnzs: {:?}", nnzs);
        println!("Product densities: {:?}", res_densities);
        println!("Product times: {:?}", times);
        println!("Product times (old): {:?}", times_old);
        #[cfg(feature = "nightly")]
        println!("Product times (scipy): {:?}", times_py);
        println!("Product times (eigen): {:?}", times_eigen);

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

            #[cfg(feature = "nightly")]
            chart
                .draw_series(LineSeries::new(
                    res_densities
                        .iter()
                        .map(|d| *d as f32)
                        .zip(times_py.iter().map(|t| *t as f32)),
                    &GREEN,
                ))?
                .label("Scipy")
                .legend(|(x, y)| {
                    PathElement::new(vec![(x, y), (x + 20, y)], &GREEN)
                });

            chart
                .draw_series(LineSeries::new(
                    res_densities
                        .iter()
                        .map(|d| *d as f32)
                        .zip(times_eigen.iter().map(|t| *t as f32)),
                    &CYAN,
                ))?
                .label("Eigen")
                .legend(|(x, y)| {
                    PathElement::new(vec![(x, y), (x + 20, y)], &CYAN)
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

fn main() -> Result<(), Box<dyn std::error::Error>> {
    bench_densities()
}
