extern crate syn;

use std::path::{PathBuf};
use std::process::{Command};

pub mod prelude {
  pub use crate::{
    Cc::*,
    Gencode,
  };
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
  subcrate_path:    Option<PathBuf>,
  gencodes:         Vec<Gencode>,
}

impl Default for Builder {
  fn default() -> Builder {
    // TODO
    Builder{
      root_path:        PathBuf::from(""), // FIXME
      subcrate_path:    None,
      gencodes:         Vec::new(),
    }
  }
}

impl Builder {
  pub fn gencode(mut self, opt: Gencode) -> Builder {
    self.gencodes.push(opt);
    self
  }

  pub fn whitelist_kernel(mut self, name: String) -> Builder {
    // TODO
    //unimplemented!();
    self
  }

  pub fn generate(self) -> Result<Glue, ()> {
    // TODO
    let mut nvcc_flags: Vec<String> = Vec::new();
    println!("DEBUG: Builder::generate(): gencodes: {:?}", self.gencodes);
    let gencode_flags: Vec<String> = self.gencodes.iter().flat_map(|gencode| gencode.flags()).collect();
    println!("DEBUG:   gencode flags: {:?}", gencode_flags);
    //unimplemented!();
    Ok(Glue{})
  }
}

pub struct Glue {
}

impl Glue {
  pub fn write_to_file(self) -> Result<(), ()> {
    // TODO
    //unimplemented!();
    Ok(())
  }
}
