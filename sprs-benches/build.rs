fn main() {
    cc::Build::new()
        .cpp(true)
        .include("/usr/include/eigen3")
        .file("src/eigen.cpp")
        .compile("eigen");
}
