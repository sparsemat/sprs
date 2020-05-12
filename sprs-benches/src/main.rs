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

#[derive(Default)]
struct BenchSpec {
    shape: (usize, usize),
    densities: Vec<f64>,
    forbid_old: bool,
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
            shape: (15000, 25000),
            densities: vec![
                1e-5, 2e-5, 5e-5, 1e-4, 2e-4, 5e-4, 1e-3, 2e-3, 3e-3, 5e-3,
            ],
            ..Default::default()
        },
        BenchSpec {
            shape: (150000, 25000),
            densities: vec![1e-7, 1e-6, 1e-5, 1e-4],
            forbid_old: true,
            ..Default::default()
        },
        BenchSpec {
            shape: (150000, 250000),
            densities: vec![1e-7, 1e-6, 1e-5, 1e-4],
            forbid_old: true,
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
                (15000, 15000),
                (35000, 35000),
                (75000, 75000),
                (150000, 150000),
                (350000, 350000),
                (750000, 750000),
                (1500000, 1500000),
                (2500000, 2500000),
            ],
            forbid_old: true,
            nnz_over_rows: 4,
            bench_filename: "sparse_mult_perf_by_shape_no_old.png".to_string(),
            ..Default::default()
        },
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

    for spec in &bench_specs {
        let shape = spec.shape;

        let is_shape_bench = spec.shapes.len() != 0;
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
        let mut times_old = Vec::with_capacity(densities.len());
        #[cfg(feature = "nightly")]
        let mut times_py = Vec::with_capacity(densities.len());
        #[cfg(feature = "eigen")]
        let mut times_eigen = Vec::with_capacity(densities.len());
        let mut nnzs = Vec::with_capacity(densities.len());
        let mut res_densities = Vec::with_capacity(densities.len());
        for (density, shape) in densities.iter().zip(shapes.iter()) {
            let mut workspace = vec![0.; shape.0];
            let density = *density;
            let shape = *shape;
            println!("Generating matrices");
            let now = std::time::Instant::now();
            let m1 = rand_csr_std(shape, density);
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

            if !spec.forbid_old {
                let now = std::time::Instant::now();
                let old_res = sprs::prod::csr_mul_csr(&m1, &m2, &mut workspace);
                let elapsed = now.elapsed().as_millis();
                println!(
                    "Old product of shape ({}, {}) and density {} done in {}ms",
                    shape.0, shape.1, density, elapsed,
                );
                times_old.push(elapsed);
                assert_eq!(prod, old_res);
            }

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
        println!("Product nnzs: {:?}", nnzs);
        println!("Product densities: {:?}", res_densities);
        println!("Product times: {:?}", times);
        println!("Product times (old): {:?}", times_old);
        #[cfg(feature = "nightly")]
        println!("Product times (scipy): {:?}", times_py);
        #[cfg(feature = "eigen")]
        println!("Product times (eigen): {:?}", times_eigen);

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
                *std::cmp::max(times.iter().max(), times_old.iter().max())
                    .unwrap_or(&1);
            #[cfg(feature = "eigen")]
            let max_time = std::cmp::max(
                max_time,
                *times_eigen.iter().max().unwrap_or(&1),
            );
            #[cfg(feature = "nightly")]
            let max_time =
                std::cmp::max(max_time, *times_py.iter().max().unwrap_or(&1));
            let max_time = max_time as f32;
            let mut chart = ChartBuilder::on(&root)
                .caption(caption, ("sans-serif", 50).into_font())
                .margin(5)
                .x_label_area_size(30)
                .y_label_area_size(50)
                .build_ranged(0f32..max_absciss, 0f32..max_time)?;

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
                    &RED,
                ))?
                .label("sprs (new)")
                .legend(|(x, y)| {
                    PathElement::new(vec![(x, y), (x + 20, y)], &RED)
                });

            if !spec.forbid_old {
                chart
                    .draw_series(LineSeries::new(
                        abscisses
                            .iter()
                            .map(|d| *d as f32)
                            .zip(times_old.iter().map(|t| *t as f32)),
                        &BLUE,
                    ))?
                    .label("sprs (old)")
                    .legend(|(x, y)| {
                        PathElement::new(vec![(x, y), (x + 20, y)], &BLUE)
                    });
            }

            #[cfg(feature = "nightly")]
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
