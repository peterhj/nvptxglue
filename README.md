`nvptxglue` is a [`bindgen`](https://github.com/rust-lang/rust-bindgen)-like
utility for building Rust code and generating glue for `*-nvidia-cuda` targets.
Like `bindgen`, `nvptxglue` is used in build scripts to generate Rust bindings.
Unlike `bindgen`, `nvptxglue` effectively cross-compiles a Rust subcrate for a
`*-nvidia-cuda` target (using the NVPTX backend,
[`xargo`](https://github.com/japaric/xargo), and `nvcc`), while the generated
glue bindings are from Rust to Rust (using
[`syn`](https://github.com/dtolnay/syn)).
