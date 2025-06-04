//------------------------------------------------------------------------------
//
// Copyright 2025 Jiri Bobek. All rights reserved.
// License: GPL 3.0 or later. See LICENSE.txt for details.
//
//------------------------------------------------------------------------------

use log;
use std::intrinsics::unlikely;

// State initialization constant ("expand 32-byte k")
const CONST: [u32; 4] = [0x_6170_7865, 0x_3320_646e, 0x_7962_2d32, 0x_6b20_6574];

pub const STATE_WORDS: usize = 16;

//pub const BLOCK_SIZE: usize = 64; // bytes

pub struct Rng {
	state: [u32; STATE_WORDS],
	block: [u32; STATE_WORDS],
	pos: usize,
}

impl Rng {
	pub fn new_default() -> Self {
		#[rustfmt::skip]
		Self::new_with_seed(&[
			0x_0a69_ee79, 0x_fb23_8e49, 0x_9bf9_a072, 0x_00da_bd56,
			0x_0420_fb57, 0x_7d06_2de2, 0x_2b40_4131, 0x_4ed7_e569,
		])
	}

	pub fn new_with_seed(key: &[u32; 8]) -> Self {
		// just some arbitrary constants
		let iv = [0x_1ada_b14a, 0x_4c3d_51fd];
		Rng {
			state: #[rustfmt::skip] [
				CONST[0], CONST[1], CONST[2], CONST[3],
				key[0],   key[1],   key[2],   key[3],
				key[4],   key[5],   key[6],   key[7],
				0,        0,        iv[0],    iv[1],
			],
			block: [0; STATE_WORDS],
			pos: 0,
		}
	}

	// generates a block of random numbers
	pub fn get_block(&mut self) -> [u32; STATE_WORDS] {
		let mut result = self.state;

		// do 4 double rounds, i.e. 8 rounds
		for _ in 0..4 {
			Self::quarter_round(0, 4, 8, 12, &mut result);
			Self::quarter_round(1, 5, 9, 13, &mut result);
			Self::quarter_round(2, 6, 10, 14, &mut result);
			Self::quarter_round(3, 7, 11, 15, &mut result);

			Self::quarter_round(0, 5, 10, 15, &mut result);
			Self::quarter_round(1, 6, 11, 12, &mut result);
			Self::quarter_round(2, 7, 8, 13, &mut result);
			Self::quarter_round(3, 4, 9, 14, &mut result);
		}

		// add original state
		#[allow(clippy::needless_range_loop)]
		for i in 0..STATE_WORDS {
			result[i] = result[i].wrapping_add(self.state[i]);
		}

		// increment counter
		let (t, c) = self.state[12].overflowing_add(1);
		self.state[12] = t;
		self.state[13] = self.state[13].wrapping_add(u32::from(c));

		result
	}

	// internal function used by get_block()
	#[inline(always)]
	fn quarter_round(a: usize, b: usize, c: usize, d: usize, state: &mut [u32; STATE_WORDS]) {
		state[a] = state[a].wrapping_add(state[b]);
		state[d] ^= state[a];
		state[d] = state[d].rotate_left(16);

		state[c] = state[c].wrapping_add(state[d]);
		state[b] ^= state[c];
		state[b] = state[b].rotate_left(12);

		state[a] = state[a].wrapping_add(state[b]);
		state[d] ^= state[a];
		state[d] = state[d].rotate_left(8);

		state[c] = state[c].wrapping_add(state[d]);
		state[b] ^= state[c];
		state[b] = state[b].rotate_left(7);
	}

	/// Generates a random u32 with uniform distribution
	pub fn get_u32(&mut self) -> u32 {
		if self.pos == 0 {
			self.block = self.get_block();
			self.pos = STATE_WORDS;
		}

		self.pos -= 1;
		self.block[self.pos]
	}

	/// Generates a float in the range [0.0, 1.0) with uniform distribution
	pub fn get_uniform(&mut self) -> f64 {
		let lo = u64::from(self.get_u32());
		let hi = u64::from(self.get_u32());
		let val = (hi << 32) | lo;

		(val as f64) * (1.0 / (4294967296.0 * 4294967296.0))
	}

	/// Generates a float with normal distribution with mean 0 and variance 1.
	/// The generated values are guaranteed to be in the range (-10.0, 10.0)
	pub fn get_normal(&mut self) -> f64 {
		let x = 1.0 - self.get_uniform(); // (0.0, 1.0]
		let y = self.get_uniform(); // [0.0, 1.0)

		// box mueller transform
		let r = (-2.0 * x.ln()).sqrt();
		let theta = std::f64::consts::TAU * y;
		let z0 = r * theta.cos();
		let z1 = r * theta.sin();

		// Combine z0 and z1 into a single value
		let result = z0 * z1;

		if unlikely(result.abs() >= 10.0) {
			log::warn!("Rng::get_normal(): clamping {} to (-10.0, 10.0)", result);
			return 0.0;
		}

		result
	}
}
