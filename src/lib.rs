extern crate syn;
extern crate walkdir;

use walkdir::{WalkDir};

use std::env;
use std::fs::{self, File};
use std::io::{Write};
use std::path::{PathBuf};
use std::process::{Command};
use std::str::{from_utf8};

pub mod prelude {
  pub use crate::{
    Cc::*,
    Cc,
    Gencode,
  };
}

fn build_nvptx_target_json(target_arch: &str) -> String {
  format!("{{
  \"arch\": \"nvptx64\"
, \"cpu\": \"{}\"
, \"data-layout\": \"e-i64:64-v16:16-v32:32-n16:32:64\"
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
}}\n", target_arch)
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
  Sass(Cc),
}

impl Gencode {
  pub fn target_arch_str(&self) -> &'static str {
    match self {
      &Gencode::Ptx(ref cc) => {
        cc.compute_str()
      }
      &Gencode::Sass(ref cc) => {
        cc.sm_str()
      }
    }
  }

  pub fn flags(&self) -> Vec<String> {
    match self {
      &Gencode::Ptx(ref cc) => {
        vec!["-gencode".to_string(), format!("arch={},code={}", cc.compute_str(), cc.compute_str())]
      }
      &Gencode::Sass(ref cc) => {
        vec!["-gencode".to_string(), format!("arch={},code={}", cc.compute_str(), cc.sm_str())]
      }
    }
  }
}

pub struct Builder {
  root_path:        PathBuf,
  output_path:      PathBuf,
  subcrate_path:    Option<PathBuf>,
  gencodes:         Vec<Gencode>,
  kernels:          Vec<String>,
}

impl Default for Builder {
  fn default() -> Builder {
    // TODO
    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    Builder{
      root_path:        manifest_dir,
      output_path:      out_dir,
      subcrate_path:    None,
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

  pub fn gencode(mut self, opt: Gencode) -> Builder {
    self.gencodes.push(opt);
    self
  }

  pub fn whitelist_kernel<S: Into<String>>(mut self, kernel: S) -> Builder {
    self.kernels.push(kernel.into());
    self
  }

  pub fn generate(self) -> Result<Glue, ()> {
    // TODO

    for entry in WalkDir::new(self.subcrate_path.clone().unwrap().to_str().unwrap()) {
      let entry = entry.unwrap();
      println!("cargo:rerun-if-changed={}", entry.path().display());
    }

    let mut xargo_flags: Vec<String> = Vec::new();
    //let mut nvcc_flags: Vec<String> = Vec::new();

    println!("DEBUG: Builder::generate(): gencodes: {:?}", self.gencodes);
    assert_eq!(self.gencodes.len(), 1);
    let target_json = build_nvptx_target_json(self.gencodes[0].target_arch_str());
    println!("DEBUG:   target json: {}", target_json);
    {
      //let mut target_json_file = File::create(self.output_path.join(format!("nvptx64-nvidia-cuda-{}.json", self.gencodes[0].target_arch_str()))).unwrap();
      let mut target_json_file = File::create(self.output_path.join(format!("nvptx64-nvidia-cuda.json"))).unwrap();
      target_json_file.write_all(target_json.as_bytes()).unwrap();
    }

    xargo_flags.push("rustc".to_string());
    //xargo_flags.push("--manifest-path".to_string());
    //xargo_flags.push(self.subcrate_path.clone().unwrap().join("Cargo.toml").as_os_str().to_str().unwrap().to_string());
    xargo_flags.push("--target".to_string());
    //xargo_flags.push(format!("nvptx64-nvidia-cuda-{}", self.gencodes[0].target_arch_str()));
    xargo_flags.push(format!("nvptx64-nvidia-cuda"));
    //xargo_flags.push("-C".to_string());
    //xargo_flags.push("target-cpu=sm_61".to_string());
    xargo_flags.push("--release".to_string());
    xargo_flags.push("--".to_string());
    xargo_flags.push("--emit=asm".to_string());

    let mut gencode_flags: Vec<String> = self.gencodes.iter().flat_map(|gencode| gencode.flags()).collect();
    println!("DEBUG:   gencode flags: {:?}", gencode_flags);
    //nvcc_flags.append(&mut gencode_flags);

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

    let mut ptx_paths = Vec::new();
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
        ptx_paths.push(ptx_path);
      }
    }
    assert_eq!(ptx_paths.len(), 1);

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

    Ok(Glue{})
  }
}

pub struct Glue {
}

impl Glue {
  pub fn write_to_file<P: Into<PathBuf>>(self, output_path: P) -> Result<(), ()> {
    // TODO
    //unimplemented!();
    Ok(())
  }
}
