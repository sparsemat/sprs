fn main() -> Result<(), Box<dyn std::error::Error>> {
    sprs_benches::bench_prod::bench_densities()
}
