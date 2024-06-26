// Copyright 2024 Jiri Bobek. All rights reserved.
// License: GPL 3.0 or later. See LICENSE.txt for details.

use crate::buffer::BufferPtr;
use crate::shape::Shape;
use crate::device::VMT;

#[derive(Clone, Debug)]
pub struct Tensor {
    buf: BufferPtr,
    vmt: *const VMT,
    shape: Shape,
}

impl Tensor {
    pub fn zero_(&self) {
        let vmt = unsafe { &*self.vmt };
        let f = vmt.zero_;
        f(vmt, self.buf.ptr, &self.shape);
    }
}