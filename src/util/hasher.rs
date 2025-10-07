//------------------------------------------------------------------------------
//
// Copyright 2025 Jiri Bobek. All rights reserved.
// License: GPL 3.0 or later. See LICENSE.txt for details.
//
//------------------------------------------------------------------------------

use const_siphasher::sip::SipHasher13;

//--------------------------------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, Default)]
#[repr(transparent)]
pub struct HashWord(u64);

impl HashWord {
	pub const fn zero() -> Self {
		Self(0)
	}

	pub const fn get_byte(&self, index: usize) -> u8 {
		let bytes = self.0.to_ne_bytes();
		bytes[index % 8]
	}

	pub const fn set_byte(&mut self, index: usize, byte: u8) {
		let mut bytes = self.0.to_ne_bytes();
		bytes[index % 8] = byte;
		self.0 = u64::from_ne_bytes(bytes);
	}
}

#[repr(transparent)]
pub struct HashWordSlice<'a> {
	words: &'a mut [HashWord],
}

impl<'a> HashWordSlice<'a> {
	pub const fn new(words: &'a mut [HashWord]) -> Self {
		Self { words }
	}

	pub const fn get_byte(&self, index: usize) -> u8 {
		self.words[index / 8].get_byte(index % 8)
	}

	pub const fn set_byte(&mut self, index: usize, byte: u8) {
		self.words[index / 8].set_byte(index % 8, byte)
	}
}

pub const HASH_WORD_SIZE: usize = std::mem::size_of::<HashWord>();
pub const HASH_WORD_ALIGN: usize = std::mem::align_of::<HashWord>();

#[derive(Debug, Clone, Copy, Default)]
pub struct Hasher(SipHasher13);

impl Hasher {
	pub fn new() -> Self {
		// TODO - could be randomized
		Self(SipHasher13::new_with_keys(3141_5926_5358_9793_u64, 2384_6264_3383_2795_u64))
	}

	pub fn write(&mut self, words: &[HashWord]) {
		for &word in words {
			self.0.write_u64(word);
		}
	}

	pub fn finish(self) -> u64 {
		self.0.finish()
	}
}

//--------------------------------------------------------------------------------------------------
