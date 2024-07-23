// Copyright 2024 Jiri Bobek. All rights reserved.
// License: GPL 3.0 or later. See LICENSE.txt for details.

use crate::*;
use std::fmt;
use std::rc::Rc;

pub trait Device {
	fn name(&self) -> &str;

	fn new_buffer(self: Rc<Self>, size_bytes: usize) -> Rc<dyn Buffer>;
}
