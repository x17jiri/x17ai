// Copyright 2024 Jiri Bobek. All rights reserved.
// License: GPL 3.0 or later. See LICENSE.txt for details.

use crate::{Expr, Tensor};
use std::fmt;
use std::rc::Rc;

pub trait Device {
	fn name(&self) -> &str;

	fn eval(self: Rc<Self>, expr: Rc<Expr>, dotfile: Option<&str>) -> Rc<Tensor>;

	fn owns(&self, tensor: &Tensor) -> bool;

	fn format(
		&self,
		f: &mut fmt::Formatter,
		tensor: &Tensor,
		off: usize,
		len: usize,
		stride: isize,
	) -> fmt::Result;
}
