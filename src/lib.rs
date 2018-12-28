extern crate quote;
extern crate syn;
extern crate walkdir;

use quote::{ToTokens};
use walkdir::{WalkDir};

use std::env;
use std::fs::{self, File};
use std::io::{Read, Write, BufWriter, Result as IoResult};
use std::path::{PathBuf};
use std::process::{Command};
use std::str::{from_utf8};

pub mod prelude {
  pub use crate::{
    GlueFun,
    GlueSpec,
    Glue,
    CudaFfiGlue,
    Cc::*,
    Cc,
    Gencode,
    Phase,
    CompilerOutput,
  };
}

pub struct GlueFun {
  pub name: String,
  pub args: Vec<(String, String)>,
  pub ret:  Option<String>,
  pub coop: bool,
}

pub struct GlueSpec {
  pub output:   CompilerOutput,
  pub funs:     Vec<GlueFun>,
}

impl GlueSpec {
  pub fn write_bindings_to_file<G: Glue, P: Into<PathBuf>>(self, glue: G, output_path: P) -> Result<GlueSpec, ()> {
    let output_file = File::create(output_path.into()).unwrap();
    let mut output_writer = BufWriter::new(output_file);
    match glue.write_bindings(&self, &mut output_writer) {
      Err(_) => Err(()),
      Ok(_) => Ok(self),
    }
  }
}

pub trait Glue {
  fn write_bindings(&self, spec: &GlueSpec, writer: &mut dyn Write) -> IoResult<()>;
}

#[derive(Default)]
pub struct CudaFfiGlue {
}

impl Glue for CudaFfiGlue {
  fn write_bindings(&self, spec: &GlueSpec, writer: &mut dyn Write) -> IoResult<()> {
    let fatbin_path = match &spec.output {
      &CompilerOutput::Fatbin(ref path) => path.to_str().unwrap(),
      _ => panic!(),
    };
    writeln!(writer, "use cuda_ffi_types::cuda::{{CUfunction, CUmodule, CUstream}};")?;
    writeln!(writer, "use cuda::ffi::{{cuLaunchKernel, cuModuleGetFunction, cuModuleLoadFatBinary}};")?;
    writeln!(writer, "use std::cell::{{RefCell}};")?;
    writeln!(writer, "use std::ffi::{{CString}};")?;
    writeln!(writer, "use std::os::raw::{{c_uint, c_void}};")?;
    writeln!(writer, "use std::ptr::{{null_mut}};")?;
    writeln!(writer, "")?;
    writeln!(writer, "static _THIS_FATBIN_IMAGE: &'static [u8] = include_bytes!(\"{}\");", fatbin_path)?;
    writeln!(writer, "")?;
    writeln!(writer, "thread_local! {{")?;
    writeln!(writer, "  static _THIS_HMOD: RefCell<_ThisHmod> = RefCell::new(_ThisHmod::default());")?;
    writeln!(writer, "}}")?;
    writeln!(writer, "")?;
    //writeln!(writer, "#[derive(Default)]")?;
    writeln!(writer, "struct _ThisHmod {{")?;
    writeln!(writer, "  hmod_ptr: CUmodule,")?;
    for fun in spec.funs.iter() {
      writeln!(writer, "  _{}: CUfunction,", fun.name)?;
    }
    writeln!(writer, "}}")?;
    writeln!(writer, "")?;
    writeln!(writer, "impl Default for _ThisHmod {{")?;
    writeln!(writer, "  fn default() -> _ThisHmod {{")?;
    writeln!(writer, "    _ThisHmod{{")?;
    writeln!(writer, "      hmod_ptr: null_mut(),")?;
    for fun in spec.funs.iter() {
      writeln!(writer, "      _{}: null_mut(),", fun.name)?;
    }
    writeln!(writer, "    }}")?;
    writeln!(writer, "  }}")?;
    writeln!(writer, "}}")?;
    writeln!(writer, "")?;
    writeln!(writer, "#[inline]")?;
    writeln!(writer, "fn _pack_kparam<'a, T: Copy + 'static>(x: &'a T) -> *mut c_void {{")?;
    writeln!(writer, "  x as *const T as *mut c_void")?;
    writeln!(writer, "}}")?;
    writeln!(writer, "")?;
    for fun in spec.funs.iter() {
      write!(writer, "pub unsafe fn {}<_KShape: Into<((u32, u32, u32), (u32, u32, u32), u32)>>(", fun.name)?;
      for &(ref arg_name, ref arg_ty) in fun.args.iter() {
        write!(writer, "{}: {}, ", arg_name, arg_ty)?;
      }
      write!(writer, "_kshape: _KShape, _stream_ptr: CUstream")?;
      match &fun.ret {
        &None => writeln!(writer, ") {{")?,
        &Some(ref ret_ty) => writeln!(writer, ") -> {} {{", ret_ty)?,
      }
      writeln!(writer, "  let _kshape = _kshape.into();")?;
      writeln!(writer, "  let mut _kparams: [*mut c_void; {}] = [null_mut(); {}];", fun.args.len(), fun.args.len())?;
      for (arg_rank, &(ref arg_name, _)) in fun.args.iter().enumerate() {
        writeln!(writer, "  _kparams[{}] = _pack_kparam(&{});", arg_rank, arg_name)?;
      }
      writeln!(writer, "  let this_func_ptr = _THIS_HMOD.with(|this_hmod| {{")?;
      writeln!(writer, "    let mut this_hmod = this_hmod.borrow_mut();")?;
      writeln!(writer, "    if this_hmod._{}.is_null() {{", fun.name)?;
      writeln!(writer, "      if this_hmod.hmod_ptr.is_null() {{")?;
      writeln!(writer, "        let mut hmod_ptr: CUmodule = null_mut();")?;
      writeln!(writer, "        cuModuleLoadFatBinary(&mut hmod_ptr as *mut CUmodule, _THIS_FATBIN_IMAGE.as_ptr() as *const c_void);")?;
      writeln!(writer, "        assert!(!hmod_ptr.is_null());")?;
      writeln!(writer, "        this_hmod.hmod_ptr = hmod_ptr;")?;
      writeln!(writer, "      }}")?;
      writeln!(writer, "      let func_name = CString::new(\"{}\").unwrap();", fun.name)?;
      writeln!(writer, "      let mut func_ptr: CUfunction = null_mut();")?;
      writeln!(writer, "      cuModuleGetFunction(&mut func_ptr as *mut CUfunction, this_hmod.hmod_ptr, func_name.as_ptr());")?;
      writeln!(writer, "      assert!(!func_ptr.is_null());")?;
      writeln!(writer, "      this_hmod._{} = func_ptr;", fun.name)?;
      writeln!(writer, "    }}")?;
      writeln!(writer, "    this_hmod._{}", fun.name)?;
      writeln!(writer, "  }});")?;
      writeln!(writer, "  cuLaunchKernel(")?;
      writeln!(writer, "      this_func_ptr,")?;
      writeln!(writer, "      (_kshape.0).0 as c_uint, (_kshape.0).1 as c_uint, (_kshape.0).2 as c_uint,")?;
      writeln!(writer, "      (_kshape.1).0 as c_uint, (_kshape.1).1 as c_uint, (_kshape.1).2 as c_uint,")?;
      writeln!(writer, "      _kshape.2 as c_uint,")?;
      writeln!(writer, "      _stream_ptr,")?;
      writeln!(writer, "      (&_kparams as &[*mut c_void]).as_ptr() as *mut *mut c_void,")?;
      writeln!(writer, "      null_mut());")?;
      writeln!(writer, "}}")?;
      writeln!(writer, "")?;
    }
    Ok(())
  }
}

#[derive(Default)]
pub struct RustacudaGlue {
}

impl Glue for RustacudaGlue {
  fn write_bindings(&self, _spec: &GlueSpec, _writer: &mut dyn Write) -> IoResult<()> {
    unimplemented!();
  }
}

fn build_nvptx_target_json(target_cpu: &str) -> String {
  format!("{{
  \"arch\": \"nvptx64\"
, \"cpu\": \"{}\"
, \"data-layout\": \"e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v32:32:32-v64:64:64-v128:128:128-n16:32:64\"
, \"linker\": false
, \"linker-flavor\": \"ld\"
, \"llvm-target\": \"nvptx64-nvidia-cuda\"
, \"max-atomic-width\": 0
, \"obj-is-bitcode\": true
, \"os\": \"cuda\"
, \"panic-strategy\": \"abort\"
, \"target-c-int-width\": \"32\"
, \"target-endian\": \"little\"
, \"target-pointer-width\": \"64\"
}}\n", target_cpu)
}

#[allow(non_camel_case_types)]
#[derive(Debug)]
pub enum Cc {
  Cc_2_0,
  Cc_2_1,
  Cc_3_0,
  Cc_3_5,
  Cc_3_7,
  Cc_5_0,
  Cc_5_2,
  Cc_6_0,
  Cc_6_1,
  Cc_7_0,
  Cc_7_5,
}

impl Cc {
  pub fn compute_str(&self) -> &'static str {
    match self {
      &Cc::Cc_2_0 => "compute_20",
      &Cc::Cc_2_1 => "compute_21",
      &Cc::Cc_3_0 => "compute_30",
      &Cc::Cc_3_5 => "compute_35",
      &Cc::Cc_3_7 => "compute_37",
      &Cc::Cc_5_0 => "compute_50",
      &Cc::Cc_5_2 => "compute_52",
      &Cc::Cc_6_0 => "compute_60",
      &Cc::Cc_6_1 => "compute_61",
      &Cc::Cc_7_0 => "compute_70",
      &Cc::Cc_7_5 => "compute_75",
    }
  }

  pub fn sm_str(&self) -> &'static str {
    match self {
      &Cc::Cc_2_0 => "sm_20",
      &Cc::Cc_2_1 => "sm_21",
      &Cc::Cc_3_0 => "sm_30",
      &Cc::Cc_3_5 => "sm_35",
      &Cc::Cc_3_7 => "sm_37",
      &Cc::Cc_5_0 => "sm_50",
      &Cc::Cc_5_2 => "sm_52",
      &Cc::Cc_6_0 => "sm_60",
      &Cc::Cc_6_1 => "sm_61",
      &Cc::Cc_7_0 => "sm_70",
      &Cc::Cc_7_5 => "sm_75",
    }
  }
}

#[derive(Debug)]
pub enum Gencode {
  Ptx(Cc),
  Gpu(Cc),
}

impl Gencode {
  pub fn llvm_target_cpu_str(&self) -> &'static str {
    match self {
      &Gencode::Ptx(ref cc) => {
        cc.sm_str()
      }
      &Gencode::Gpu(ref cc) => {
        cc.sm_str()
      }
    }
  }

  pub fn flags(&self) -> Vec<String> {
    match self {
      &Gencode::Ptx(ref cc) => {
        vec!["-gencode".to_string(), format!("arch={},code={}", cc.compute_str(), cc.compute_str())]
      }
      &Gencode::Gpu(ref cc) => {
        vec!["-gencode".to_string(), format!("arch={},code={}", cc.compute_str(), cc.sm_str())]
      }
    }
  }
}

#[derive(Debug)]
pub enum Phase {
  Ptx,
  Cubin,
  Fatbin,
}

#[derive(Debug)]
pub enum CompilerOutput {
  Ptx(PathBuf),
  Cubin(PathBuf),
  Fatbin(PathBuf),
}

struct Kernel {
  name: String,
  coop: bool,
}

pub struct Builder {
  root_path:        PathBuf,
  output_path:      PathBuf,
  subcrate_path:    Option<PathBuf>,
  ptx_linker:       bool,
  gencodes:         Vec<Gencode>,
  kernels:          Vec<Kernel>,
}

impl Default for Builder {
  fn default() -> Builder {
    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    Builder{
      root_path:        manifest_dir,
      output_path:      out_dir,
      subcrate_path:    None,
      ptx_linker:       false,
      gencodes:         Vec::new(),
      kernels:          Vec::new(),
    }
  }
}

impl Builder {
  pub fn crate_dir<P: Into<PathBuf>>(mut self, subcrate_path: P) -> Builder {
    self.subcrate_path = Some(subcrate_path.into());
    self
  }

  pub fn ptx_linker(mut self, enabled: bool) -> Builder {
    self.ptx_linker = enabled;
    self
  }

  pub fn gencode(mut self, opt: Gencode) -> Builder {
    self.gencodes.push(opt);
    self
  }

  pub fn whitelist_kernel<S: Into<String>>(mut self, kernel_name: S) -> Builder {
    self.kernels.push(Kernel{
      name: kernel_name.into(),
      coop: false,
    });
    self
  }

  pub fn whitelist_coop_kernel<S: Into<String>>(mut self, kernel_name: S) -> Builder {
    self.kernels.push(Kernel{
      name: kernel_name.into(),
      coop: true,
    });
    self
  }

  fn run_xargo(&self) {
    let mut xargo_flags: Vec<String> = Vec::new();

    println!("DEBUG: Builder::generate(): gencodes: {:?}", self.gencodes);
    assert_eq!(self.gencodes.len(), 1,
        "todo: compiling multiple ptx files into a fatbin");
    let target_json = build_nvptx_target_json(self.gencodes[0].llvm_target_cpu_str());
    println!("DEBUG: target json: {}", target_json);
    {
      let mut target_json_file = File::create(self.output_path.join(format!("nvptx64-nvidia-cuda.json"))).unwrap();
      target_json_file.write_all(target_json.as_bytes()).unwrap();
    }

    xargo_flags.push("rustc".to_string());
    //xargo_flags.push("--manifest-path".to_string());
    //xargo_flags.push(self.subcrate_path.clone().unwrap().join("Cargo.toml").as_os_str().to_str().unwrap().to_string());
    // FIXME: GPU arch-specific target path.
    xargo_flags.push("--target".to_string());
    xargo_flags.push(format!("nvptx64-nvidia-cuda"));
    //xargo_flags.push("-C".to_string());
    //xargo_flags.push("target-cpu=sm_61".to_string());
    xargo_flags.push("--release".to_string());
    xargo_flags.push("--".to_string());
    xargo_flags.push("--emit=asm".to_string());

    println!("DEBUG: target path: {:?}", self.output_path.clone());
    match Command::new("xargo")
      //.current_dir(self.output_path.clone())
      .current_dir(self.subcrate_path.clone().unwrap())
      .env("RUST_TARGET_PATH", self.output_path.clone())
      //.env("RUSTFLAGS", "-C target-cpu=sm_61")
      .args(xargo_flags)
      .output()
    {
      Err(_) => panic!("failed to get xargo output"),
      Ok(output) => {
        if !output.status.success() {
          println!("FATAL: xargo not successful");
          println!();
          println!("### BEGIN XARGO STDOUT ###");
          print!("{}", from_utf8(&output.stdout).unwrap());
          println!("### END XARGO STDOUT ###");
          println!();
          println!("### BEGIN XARGO STDERR ###");
          print!("{}", from_utf8(&output.stderr).unwrap());
          println!("### END XARGO STDERR ###");
          panic!();
        }
      }
    }
  }

  fn run_xargo_with_ptx_linker(&self) {
    unimplemented!();
  }

  fn run_compiler(&self, phase: Phase) -> CompilerOutput {
    match self.ptx_linker {
      false => self.run_xargo(),
      true  => self.run_xargo_with_ptx_linker(),
    }

    let mut ptx_paths = Vec::new();
    let mut fatbin_paths = Vec::new();
    for e in WalkDir::new(self.subcrate_path.clone().unwrap().join("target").join("nvptx64-nvidia-cuda").join("release").join("deps")) {
      let e = e.unwrap();
      if e.path().extension().and_then(|s| s.to_str()) == Some("s") {
        let asm_path = e.path();
        println!("DEBUG: maybe emitted ptx path: {:?}", asm_path);
        let mut ptx_path = asm_path.to_owned();
        ptx_path.set_extension("ptx");
        match fs::copy(asm_path, &ptx_path) {
          Err(_) => panic!("failed to copy .s file to .ptx file"),
          Ok(_) => {}
        }
        let mut fatbin_path = self.output_path.join(ptx_path.file_name().unwrap());
        fatbin_path.set_extension("fatbin");
        ptx_paths.push(ptx_path);
        fatbin_paths.push(fatbin_path);
      }
    }
    // TODO: support for multiple compilation units.
    assert_eq!(ptx_paths.len(), 1);

    let gencode_flags: Vec<String> = self.gencodes.iter().flat_map(|gencode| gencode.flags()).collect();
    println!("DEBUG: gencode flags: {:?}", gencode_flags);

    match phase {
      Phase::Ptx => {
        CompilerOutput::Ptx(fs::canonicalize(&ptx_paths[0]).unwrap())
      }
      Phase::Cubin => {
        unimplemented!();
      }
      Phase::Fatbin => {
        match Command::new("/usr/local/cuda/bin/nvcc")
          .current_dir(self.output_path.clone())
          .arg("-O3")
          .arg("-fatbin")
          .args(gencode_flags)
          .arg(fs::canonicalize(&ptx_paths[0]).unwrap())
          .output()
        {
          Err(_) => panic!("failed to get nvcc output"),
          Ok(output) => {
            if !output.status.success() {
              println!("FATAL: nvcc not successful");
              println!();
              println!("### BEGIN NVCC STDOUT ###");
              print!("{}", from_utf8(&output.stdout).unwrap());
              println!("### END NVCC STDOUT ###");
              println!();
              println!("### BEGIN NVCC STDERR ###");
              print!("{}", from_utf8(&output.stderr).unwrap());
              println!("### END NVCC STDERR ###");
              panic!();
            }
          }
        }
        CompilerOutput::Fatbin(fatbin_paths[0].clone())
      }
    }
  }

  pub fn compile(self, phase: Phase) -> Result<GlueSpec, ()> {
    for entry in WalkDir::new(self.subcrate_path.clone().unwrap().to_str().unwrap()) {
      let entry = entry.unwrap();
      println!("cargo:rerun-if-changed={}", entry.path().display());
    }

    let output = self.run_compiler(phase);

    // FIXME: instead of using WalkDir, can read the dependency file.
    let mut rs_paths = Vec::new();
    for e in WalkDir::new(self.subcrate_path.clone().unwrap()) {
      let e = e.unwrap();
      if e.path().extension().and_then(|s| s.to_str()) == Some("rs") {
        let rs_path = e.path();
        rs_paths.push(rs_path.to_owned());
      }
    }

    let mut glue_funs = Vec::new();
    for rs_path in rs_paths.iter() {
      let mut rs_file = File::open(rs_path).unwrap();
      let mut src_buf = String::new();
      match rs_file.read_to_string(&mut src_buf) {
        Err(_) => panic!("failed to read rust source file"),
        Ok(_) => {}
      }
      let syntax = match syn::parse_file(&src_buf) {
        Err(_) => panic!("failed to parse rust source file"),
        Ok(syntax) => syntax,
      };
      for tl_item in syntax.items.iter() {
        match tl_item {
          &syn::Item::Fn(ref fun_item) => {
            match &fun_item.abi {
              &Some(ref abi) => {
                if abi.name.as_ref().map(|s| s.value()) == Some("ptx-kernel".to_string()) {
                  // FIXME: crosscheck with whitelisted kernel set.
                  let name = fun_item.ident.to_string();
                  let mut args = Vec::new();
                  for arg in fun_item.decl.inputs.iter() {
                    match arg {
                      &syn::FnArg::Captured(ref arg) => {
                        let arg_name = match &arg.pat {
                          &syn::Pat::Ident(ref pat) => {
                            pat.ident.to_string()
                          }
                          _ => panic!("unsupported ptx-kernel function argument pattern"),
                        };
                        let arg_ty = arg.ty.clone().into_token_stream().to_string();
                        args.push((arg_name, arg_ty));
                      }
                      _ => panic!("unsupported ptx-kernel function argument"),
                    }
                  }
                  let ret = match &fun_item.decl.output {
                    &syn::ReturnType::Default => {
                      None
                    }
                    &syn::ReturnType::Type(_, ref ret_ty) => {
                      Some(ret_ty.clone().into_token_stream().to_string())
                    }
                  };
                  glue_funs.push(GlueFun{
                    name,
                    args,
                    ret,
                    coop:   false,
                  });
                }
              }
              _ => {}
            }
          }
          _ => {}
        }
      }
    }

    Ok(GlueSpec{
      output:   output,
      funs:     glue_funs,
    })
  }
}
