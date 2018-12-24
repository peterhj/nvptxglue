extern crate syn;

pub use crate::Cc::*;

use std::path::{PathBuf};
use std::process::{Command};

#[allow(non_camel_case_types)]
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

pub enum Gencode {
  Ptx(Cc),
  Sass(Cc),
}

pub struct Builder {
  root_path:        PathBuf,
  subcrate_path:    Option<PathBuf>,
  gencode_opts:     Vec<Gencode>,
}

impl Default for Builder {
  fn default() -> Builder {
    // TODO
    unimplemented!();
  }
}

impl Builder {
  pub fn gencode(mut self, opt: Gencode) -> Builder {
    self.gencode_opts.push(opt);
    self
  }

  pub fn whitelist_kernel(mut self, name: String) -> Builder {
    // TODO
    //unimplemented!();
    self
  }

  pub fn generate(self) -> Result<Glue, ()> {
    // TODO
    unimplemented!();
  }
}

pub struct Glue {
}

impl Glue {
  pub fn write_to_file(self) -> Result<(), ()> {
    // TODO
    unimplemented!();
  }
}
