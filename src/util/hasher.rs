//------------------------------------------------------------------------------
//
// Copyright 2025 Jiri Bobek. All rights reserved.
// License: GPL 3.0 or later. See LICENSE.txt for details.
//
//------------------------------------------------------------------------------

use const_siphasher::sip::SipHasher13;

//--------------------------------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
#[repr(transparent)]
pub struct HashWord(u64);

impl HashWord {
	pub const fn zero() -> Self {
		Self(0)
	}

	pub const fn get_byte(slice: &[Self], index: usize) -> u8 {
		let bytes = slice[index / 8].0.to_ne_bytes();
		bytes[index % 8]
	}

	pub const fn set_byte(slice: &mut [Self], index: usize, byte: u8) {
		let mut bytes = slice[index / 8].0.to_ne_bytes();
		bytes[index % 8] = byte;
		slice[index / 8].0 = u64::from_ne_bytes(bytes);
	}
}

pub const HASH_WORD_SIZE: usize = std::mem::size_of::<HashWord>();
pub const HASH_WORD_ALIGN: usize = std::mem::align_of::<HashWord>();

//--------------------------------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, Default)]
pub struct Hasher(SipHasher13);

impl Hasher {
	pub fn new(key0: u64, key1: u64) -> Self {
		Self(SipHasher13::new_with_keys(key0, key1))
	}

	pub fn write(&mut self, words: &[HashWord]) {
		for &word in words {
			self.0.write_u64(word.0);
		}
	}

	pub fn finish(self) -> u64 {
		self.0.finish()
	}
}

//--------------------------------------------------------------------------------------------------

pub struct RandomState {
	pub key0: u64,
	pub key1: u64,
}

impl RandomState {
	pub fn new() -> Self {
		// TODO - could be randomized
		Self {
			key0: 3141_5926_5358_9793_u64,
			key1: 2384_6264_3383_2795_u64,
		}
	}

	pub fn build_hasher(&self) -> Hasher {
		Hasher::new(self.key0, self.key1)
	}

	pub fn hash_one(&self, x: &[HashWord]) -> u64 {
		let mut hasher = self.build_hasher();
		hasher.write(x);
		hasher.finish()
	}
}

//--------------------------------------------------------------------------------------------------
