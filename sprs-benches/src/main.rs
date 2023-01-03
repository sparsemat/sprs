use pyo3::{
    prelude::*,
    types::{IntoPyDict, PyModule},
};
use sprs::smmp;
use sprs_rand::rand_csr_std;

fn scipy_mat<'a>(
    scipy_sparse: &'a PyModule,
    py: Python,
    mat: &sprs::CsMat<f64>,
) -> Result<&'a PyAny, String> {
    let indptr = mat.indptr().to_proper().to_vec();
    scipy_sparse
        .call_method(
            "csr_matrix",
            ((mat.data().to_vec(), mat.indices().to_vec(), indptr),),
            Some([("shape", mat.shape())].into_py_dict(py)),
        )
        .map_err(|e| {
            let res = format!("Python error: {e:?}");
            e.print_and_set_sys_last_vars(py);
            res
        })
}

#[cfg(feature = "eigen")]
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

#[cfg(feature = "eigen")]
fn eigen_prod(a: sprs::CsMatView<f64>, b: sprs::CsMatView<f64>) -> usize {
    use std::convert::TryFrom;
    let (a_rows, a_cols) = a.shape();
    let (b_rows, b_cols) = b.shape();
    assert_eq!(a_cols, b_rows);
    assert!(a.is_csr());
    assert!(isize::try_from(a.rows()).is_ok());
    assert!(isize::try_from(a.indptr().nnz()).is_ok());
    assert!(b.is_csr());
    assert!(isize::try_from(b.rows()).is_ok());
    assert!(isize::try_from(b.indptr().nnz()).is_ok());
    let a_indptr_proper = a.proper_indptr();
    let a_indices = a.indices().as_ptr() as *const isize;
    let a_data = a.data().as_ptr();
    let b_indptr_proper = b.proper_indptr();
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
            a_rows,
            a_cols,
            b_cols,
            a_indptr_proper.as_ptr() as *const isize,
            a_indices,
            a_data,
            b_indptr_proper.as_ptr() as *const isize,
            b_indices,
            b_data,
        )
    }
}

#[derive(Default)]
struct BenchSpec {
    shape: (usize, usize),
    densities: Vec<f64>,
    #[allow(dead_code)] // this variable is only present with eigen's feature
    forbid_eigen: bool,
    shapes: Vec<(usize, usize)>, // will trigger shape benchmark
    nnz_over_rows: usize,        // used to compute density in shape benchmark
    bench_filename: String,      // used to name in shape benchmark
}

fn bench_densities() -> Result<(), Box<dyn std::error::Error>> {
    let bench_specs = [
        BenchSpec {
            shape: (1500, 2500),
            densities: vec![
                1e-4, 5e-4, 1e-3, 2e-3, 5e-3, 8e-3, 1e-2, 2e-2, 5e-2,
            ],
            ..Default::default()
        },
        BenchSpec {
            shape: (15_000, 25_000),
            densities: vec![
                1e-5, 2e-5, 5e-5, 1e-4, 2e-4,
                5e-4, //1e-3, 2e-3, 3e-3, 5e-3,
            ],
            ..Default::default()
        },
        BenchSpec {
            shape: (15_0000, 25_000),
            densities: vec![1e-7, 1e-6, 1e-5, 1e-4],
            forbid_eigen: true,
            ..Default::default()
        },
        BenchSpec {
            shape: (15_0000, 25_0000),
            densities: vec![1e-7, 1e-6, 1e-5, 1e-4],
            forbid_eigen: true,
            ..Default::default()
        },
        BenchSpec {
            shapes: vec![
                (1500, 1500),
                (2500, 2500),
                (3500, 3500),
                (15000, 15000),
                (25000, 25000),
                (35000, 35000),
                (45000, 45000),
                (55000, 55000),
            ],
            nnz_over_rows: 4,
            bench_filename: "sparse_mult_perf_by_shape.png".to_string(),
            ..Default::default()
        },
        BenchSpec {
            shapes: vec![
                (1500, 1500),
                (3500, 3500),
                (7500, 7500),
                (15_000, 15_000),
                (35_000, 35_000),
                (75_000, 75_000),
                (150_000, 150_000),
                (350_000, 350_000),
                (750_000, 750_000),
                (1_500_000, 1_500_000),
                (2_500_000, 2_500_000),
            ],
            forbid_eigen: true,
            nnz_over_rows: 4,
            bench_filename: "sparse_mult_perf_by_shape_no_old.png".to_string(),
            ..Default::default()
        },
    ];

    let gil = Python::acquire_gil();
    let py = gil.python();
    let scipy_sparse = PyModule::import(py, "scipy.sparse").map_err(|e| {
        let res = format!("Python error: {e:?}");
        e.print_and_set_sys_last_vars(py);
        res
    })?;

    for spec in &bench_specs {
        let shape = spec.shape;

        let is_shape_bench = !spec.shapes.is_empty();
        let densities = if is_shape_bench {
            spec.shapes
                .iter()
                .map(|(_rows, cols)| {
                    (spec.nnz_over_rows as f64) / (*cols as f64)
                })
                .collect()
        } else {
            spec.densities.clone()
        };
        let shapes = if is_shape_bench {
            spec.shapes.clone()
        } else {
            std::iter::repeat(shape).take(densities.len()).collect()
        };

        let mut times = Vec::with_capacity(densities.len());
        let mut times_autothread = Vec::with_capacity(densities.len());
        let mut times_2threads = Vec::with_capacity(densities.len());
        let mut times_4threads = Vec::with_capacity(densities.len());
        let mut times_py = Vec::with_capacity(densities.len());
        #[cfg(feature = "eigen")]
        let mut times_eigen = Vec::with_capacity(densities.len());
        let mut nnzs = Vec::with_capacity(densities.len());
        let mut res_densities = Vec::with_capacity(densities.len());
        for (density, shape) in densities.iter().zip(shapes.iter()) {
            let density = *density;
            let shape = *shape;
            println!("Generating matrices");
            let now = std::time::Instant::now();
            let m1 = rand_csr_std(shape, density);
            let m2 = rand_csr_std((shape.1, shape.0), density);
            let elapsed = now.elapsed().as_millis();
            println!("Generating matrices took {elapsed}ms");

            smmp::set_thread_threading_strategy(
                smmp::ThreadingStrategy::Fixed(1),
            );
            let now = std::time::Instant::now();
            let prod = &m1 * &m2;
            let elapsed = now.elapsed().as_millis();
            println!(
                "sprs product of shape ({}, {}) and density {} done in {}ms",
                shape.0, shape.1, density, elapsed,
            );
            times.push(elapsed);

            smmp::set_thread_threading_strategy(
                smmp::ThreadingStrategy::Fixed(2),
            );
            let now = std::time::Instant::now();
            let prod_ = &m1 * &m2;
            let elapsed = now.elapsed().as_millis();
            println!(
                "sprs product (2 threads) of shape ({}, {}) and density {} done in {}ms",
                shape.0, shape.1, density, elapsed,
            );
            assert_eq!(prod, prod_);
            times_2threads.push(elapsed);

            smmp::set_thread_threading_strategy(
                smmp::ThreadingStrategy::Fixed(4),
            );
            let now = std::time::Instant::now();
            let prod_ = &m1 * &m2;
            let elapsed = now.elapsed().as_millis();
            println!(
                "sprs product (4 threads) of shape ({}, {}) and density {} done in {}ms",
                shape.0, shape.1, density, elapsed,
            );
            assert_eq!(prod, prod_);
            times_4threads.push(elapsed);

            smmp::set_thread_threading_strategy(
                smmp::ThreadingStrategy::Automatic,
            );
            let now = std::time::Instant::now();
            let prod_ = &m1 * &m2;
            let elapsed = now.elapsed().as_millis();
            println!(
                "sprs product (auto thread) of shape ({}, {}) and density {} done in {}ms",
                shape.0, shape.1, density, elapsed,
            );
            assert_eq!(prod, prod_);
            times_autothread.push(elapsed);

            nnzs.push(prod.nnz());
            res_densities.push(prod.density());

            // bench scipy as well
            {
                let m1_py = scipy_mat(scipy_sparse, py, &m1)?;
                let m2_py = scipy_mat(scipy_sparse, py, &m2)?;
                let now = std::time::Instant::now();
                let _prod_py = py
                    .eval(
                        "m1 * m2",
                        Some([("m1", m1_py), ("m2", m2_py)].into_py_dict(py)),
                        None,
                    )
                    .map_err(|e| {
                        let res = format!("Python error: {e:?}");
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
            #[cfg(feature = "eigen")]
            {
                if !spec.forbid_eigen {
                    let now = std::time::Instant::now();
                    let _nnz = eigen_prod(m1.view(), m2.view());
                    let elapsed = now.elapsed().as_millis();
                    println!(
                        "Eigen product of shape ({}, {}) and density {} done in {}ms",
                        shape.0, shape.1, density, elapsed,
                    );
                    times_eigen.push(elapsed);
                }
            }
        }
        println!("Results for shape: ({}, {})", shape.0, shape.1);
        println!("Product nnzs: {nnzs:?}");
        println!("Product densities: {res_densities:?}");
        println!("Product times (sprs): {times:?}");
        println!("Product times (sprs, 2 threads): {times_2threads:?}");
        println!("Product times (sprs, 4 threads): {times_4threads:?}");
        println!("Product times (sprs, auto threads): {times_autothread:?}");
        println!("Product times (scipy): {times_py:?}");
        #[cfg(feature = "eigen")]
        println!("Product times (eigen): {times_eigen:?}");

        // plot
        {
            use plotters::prelude::*;
            let title = if is_shape_bench {
                spec.bench_filename.clone()
            } else {
                format!("sparse_mult_perf_{}_{}.png", shape.0, shape.1)
            };
            let res = (640, 480);
            let root = BitMapBackend::new(&title, res).into_drawing_area();
            root.fill(&WHITE)?;
            let max_density = *res_densities.last().unwrap_or(&1.) as f32;
            let max_shape =
                *shapes.iter().map(|(rows, _)| rows).max().unwrap_or(&1) as f32;
            let max_absciss = if is_shape_bench {
                max_shape
            } else {
                max_density
            };
            let caption = if is_shape_bench {
                "Time vs shape"
            } else {
                "Time vs density"
            };
            let max_time =
                *std::cmp::max(times.iter().max(), times_2threads.iter().max())
                    .unwrap_or(&1);
            let max_time = std::cmp::max(
                max_time,
                *times_4threads.iter().max().unwrap_or(&1),
            );
            let max_time = std::cmp::max(
                max_time,
                *times_autothread.iter().max().unwrap_or(&1),
            );
            #[cfg(feature = "eigen")]
            let max_time = std::cmp::max(
                max_time,
                *times_eigen.iter().max().unwrap_or(&1),
            );
            let max_time =
                std::cmp::max(max_time, *times_py.iter().max().unwrap_or(&1));
            let max_time = max_time as f32;
            let mut chart = ChartBuilder::on(&root)
                .caption(caption, ("sans-serif", 50).into_font())
                .margin(5)
                .x_label_area_size(30)
                .y_label_area_size(50)
                .build_ranged(0_f32..max_absciss, 0_f32..max_time)?;

            let abscisses = if is_shape_bench {
                shapes.iter().map(|(rows, _)| *rows as f64).collect()
            } else {
                res_densities
            };

            chart.configure_mesh().draw()?;

            chart
                .draw_series(LineSeries::new(
                    abscisses
                        .iter()
                        .map(|d| *d as f32)
                        .zip(times.iter().map(|t| *t as f32)),
                    &MAGENTA,
                ))?
                .label("sprs (1T)")
                .legend(|(x, y)| {
                    PathElement::new(vec![(x, y), (x + 20, y)], &MAGENTA)
                });

            chart
                .draw_series(LineSeries::new(
                    abscisses
                        .iter()
                        .map(|d| *d as f32)
                        .zip(times_2threads.iter().map(|t| *t as f32)),
                    &BLACK,
                ))?
                .label("sprs (2T)")
                .legend(|(x, y)| {
                    PathElement::new(vec![(x, y), (x + 20, y)], &BLACK)
                });

            chart
                .draw_series(LineSeries::new(
                    abscisses
                        .iter()
                        .map(|d| *d as f32)
                        .zip(times_4threads.iter().map(|t| *t as f32)),
                    &YELLOW,
                ))?
                .label("sprs (4T)")
                .legend(|(x, y)| {
                    PathElement::new(vec![(x, y), (x + 20, y)], &YELLOW)
                });

            chart
                .draw_series(LineSeries::new(
                    abscisses
                        .iter()
                        .map(|d| *d as f32)
                        .zip(times_autothread.iter().map(|t| *t as f32)),
                    &BLUE,
                ))?
                .label("sprs (auto thread)")
                .legend(|(x, y)| {
                    PathElement::new(vec![(x, y), (x + 20, y)], &BLUE)
                });

            chart
                .draw_series(LineSeries::new(
                    abscisses
                        .iter()
                        .map(|d| *d as f32)
                        .zip(times_py.iter().map(|t| *t as f32)),
                    &GREEN,
                ))?
                .label("Scipy")
                .legend(|(x, y)| {
                    PathElement::new(vec![(x, y), (x + 20, y)], &GREEN)
                });

            #[cfg(feature = "eigen")]
            {
                if !spec.forbid_eigen {
                    chart
                        .draw_series(LineSeries::new(
                            abscisses
                                .iter()
                                .map(|d| *d as f32)
                                .zip(times_eigen.iter().map(|t| *t as f32)),
                            &CYAN,
                        ))?
                        .label("Eigen")
                        .legend(|(x, y)| {
                            PathElement::new(vec![(x, y), (x + 20, y)], &CYAN)
                        });
                }
            }

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
