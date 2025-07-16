//------------------------------------------------------------------------------
//
// Copyright 2025 Jiri Bobek. All rights reserved.
// License: GPL 3.0 or later. See LICENSE.txt for details.
//
//------------------------------------------------------------------------------

use std::sync::{Arc, OnceLock, RwLock};

use super::kernel_builder::KernelData;

//--------------------------------------------------------------------------------------------------

pub struct KernelRegistry {
	pub(crate) kernels: Vec<Arc<KernelData>>,
}

impl KernelRegistry {
	pub fn instance() -> &'static RwLock<Self> {
		static instance: OnceLock<RwLock<KernelRegistry>> = OnceLock::new();
		instance.get_or_init(|| RwLock::new(Self { kernels: Vec::new() }))
	}

	pub(crate) fn add_kernel(
		&mut self,
		build_kernel: impl FnOnce(usize) -> Arc<KernelData>,
	) -> Arc<KernelData> {
		let id = self.kernels.len();
		let kernel = build_kernel(id);
		self.kernels.push(kernel.clone());
		kernel
	}
}

//--------------------------------------------------------------------------------------------------
