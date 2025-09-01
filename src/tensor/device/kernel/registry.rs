//------------------------------------------------------------------------------
//
// Copyright 2025 Jiri Bobek. All rights reserved.
// License: GPL 3.0 or later. See LICENSE.txt for details.
//
//------------------------------------------------------------------------------

use std::hint::{cold_path, likely};
use std::sync::{Arc, OnceLock, RwLock};

use const_siphasher::sip::SipHasher13;
use hashbrown::HashTable;

use crate::tensor::device::kernel::runner::KernelData;

//--------------------------------------------------------------------------------------------------

pub struct KernelMapEntry {
	pub key_hash: u64,
	pub value: Arc<KernelData>,
}

pub struct KernelMap {
	map: HashTable<KernelMapEntry>,
}

impl Default for KernelMap {
	fn default() -> Self {
		Self::new()
	}
}

impl KernelMap {
	pub fn new() -> Self {
		Self { map: HashTable::new() }
	}

	pub const fn hash_key(key: &[u64]) -> u64 {
		let mut key_hasher =
			SipHasher13::new_with_keys(3141_5926_5358_9793_u64, 2384_6264_3383_2795_u64);
		let mut i = 0;
		while i < key.len() {
			key_hasher.write_u64(key[i]);
			i += 1;
		}
		key_hasher.finish()
	}

	#[inline(never)]
	pub fn find(&self, key: &[u64], key_hash: u64) -> Option<&KernelMapEntry> {
		self.map.find(key_hash, |item| {
			item.key_hash == key_hash && likely(item.value.key.as_ref() == key)
		})
	}

	#[inline(never)]
	pub fn insert_unique(&mut self, key_hash: u64, value: Arc<KernelData>) -> &KernelMapEntry {
		self.map
			.insert_unique(key_hash, KernelMapEntry { key_hash, value }, |entry| entry.key_hash)
			.into_mut()
	}
}

//--------------------------------------------------------------------------------------------------

pub struct KernelRegistry {
	kernels: Vec<Arc<KernelData>>,
	map: KernelMap,
}

impl KernelRegistry {
	pub fn instance() -> Arc<RwLock<Self>> {
		static instance: OnceLock<Arc<RwLock<KernelRegistry>>> = OnceLock::new();
		instance
			.get_or_init(|| {
				Arc::new(RwLock::new(Self {
					kernels: Vec::new(),
					map: KernelMap::new(),
				}))
			})
			.clone()
	}

	pub(crate) fn add_kernel(
		&mut self,
		key: &[u64],
		key_hash: u64,
		build_kernel: impl FnOnce(usize) -> Arc<KernelData>,
	) -> Arc<KernelData> {
		if let Some(entry) = self.map.find(key, key_hash) {
			cold_path();
			return entry.value.clone();
		}
		let id = self.kernels.len();
		let kernel = build_kernel(id);
		self.kernels.push(kernel.clone());
		self.map.insert_unique(key_hash, kernel.clone());
		kernel
	}
}

//--------------------------------------------------------------------------------------------------
