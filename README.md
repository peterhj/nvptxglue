(WIP so expect breakage and rough corners)

`nvptxglue` is a [`bindgen`](https://github.com/rust-lang/rust-bindgen)-like
utility for building Rust code and generating glue for `*-nvidia-cuda` targets.
Like `bindgen`, `nvptxglue` is used in build scripts to generate Rust bindings.
Unlike `bindgen`, `nvptxglue` effectively cross-compiles a Rust subcrate for a
`*-nvidia-cuda` target (using combinations of: the NVPTX backend,
[`xargo`](https://github.com/japaric/xargo), and `nvcc`), while the generated
glue bindings are from Rust to Rust (using
[`syn`](https://github.com/dtolnay/syn)).

## Related projects

Like [`ptx-builder`](https://github.com/denzp/rust-ptx-linker), `nvptxglue` is
meant to be used in build scripts to facilitate the use of NVPTX-targeted
Rust crates in the host crate. However unlike `ptx-builder`, `nvptxglue` aims to
enable finer grained control of the CUDA compilation lifecycle, e.g. generating
only PTX from Rust and making the PTX available in a host crate, or replicating
the "native" CUDA toolchain, based on nvcc, to compile _fat binaries_ that
support multiple real or virtual GPU architectures.

[`ptx-linker`](https://github.com/denzp/rust-ptx-linker) uses LLVM to get around
some limitations of rustc's NVPTX target support. `nvptxglue` can use
`ptx-linker` during the "xargo phase" (todo).

[RustaCUDA](https://github.com/bheisler/rustacuda) is a high-level interface for
safely launching CUDA kernels via the driver API in Rust. `nvptxglue` can
generate RustaCUDA-compatible Rust bindings: what is needed is an implementation
of the `Glue` trait to emit the generated Rust code to an arbitrary writer
(todo; I reserved a `RustacudaGlue` type for this).

## Examples

Basic example using the builtin `CudaFfiGlue` writer
(see the
[build script](https://github.com/peterhj/nvptxglue-example/blob/master/build.rs)
as well):

    git clone https://github.com/peterhj/nvptxglue
    git clone https://github.com/peterhj/nvptxglue-example
    cd nvptxglue-example
    cat build.rs
    LIBRARY_PATH=/usr/local/cuda/lib64 cargo build --release   # or wherever your cuda libs are
