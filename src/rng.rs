//------------------------------------------------------------------------------
//
// Copyright 2025 Jiri Bobek. All rights reserved.
// License: GPL 3.0 or later. See LICENSE.txt for details.
//
//------------------------------------------------------------------------------

use log;
use std::hint::cold_path;

use crate::tensor::device::cpu::math::FromToF64;

// State initialization constant ("expand 32-byte k")
const CONST: [u32; 4] = [0x_6170_7865, 0x_3320_646e, 0x_7962_2d32, 0x_6b20_6574];

const STATE_WORDS: usize = 16;

pub struct Rng {
	state: [u32; STATE_WORDS],
}

impl Default for Rng {
	fn default() -> Self {
		#[rustfmt::skip]
		Self::new(&[
			0x0a, 0x69, 0xee, 0x79, 0xfb, 0x23, 0x8e, 0x49,
			0x9b, 0xf9, 0xa0, 0x72, 0x00, 0xda, 0xbd, 0x56,
			0x04, 0x20, 0xfb, 0x57, 0x7d, 0x06, 0x2d, 0xe2,
			0x2b, 0x40, 0x41, 0x31, 0x4e, 0xd7, 0xe5, 0x69,
			0x1a, 0xda, 0xb1, 0x4a, 0x4c, 0x3d, 0x51, 0xfd,
			0x5c, 0x3f, 0x2a, 0x7e, 0x1f, 0x2b, 0x6b, 0x8c,
		])
	}
}

#[allow(clippy::indexing_slicing)]
impl Rng {
	pub fn new(seed: &[u8; 48]) -> Self {
		let C0 = CONST[0];
		let C1 = CONST[1];
		let C2 = CONST[2];
		let C3 = CONST[3];
		let k0 = u32::from_le_bytes([seed[0], seed[1], seed[2], seed[3]]);
		let k1 = u32::from_le_bytes([seed[4], seed[5], seed[6], seed[7]]);
		let k2 = u32::from_le_bytes([seed[8], seed[9], seed[10], seed[11]]);
		let k3 = u32::from_le_bytes([seed[12], seed[13], seed[14], seed[15]]);
		let k4 = u32::from_le_bytes([seed[16], seed[17], seed[18], seed[19]]);
		let k5 = u32::from_le_bytes([seed[20], seed[21], seed[22], seed[23]]);
		let k6 = u32::from_le_bytes([seed[24], seed[25], seed[26], seed[27]]);
		let k7 = u32::from_le_bytes([seed[28], seed[29], seed[30], seed[31]]);
		let v0 = u32::from_le_bytes([seed[32], seed[33], seed[34], seed[35]]);
		let v1 = u32::from_le_bytes([seed[36], seed[37], seed[38], seed[39]]);
		let v2 = u32::from_le_bytes([seed[40], seed[41], seed[42], seed[43]]);
		let v3 = u32::from_le_bytes([seed[44], seed[45], seed[46], seed[47]]);
		Self {
			state: #[rustfmt::skip] [
				C0, C1, C2, C3,
				k0, k1, k2, k3,
				k4, k5, k6, k7,
				v0, v1, v2, v3,
			],
		}
	}

	// generates a block of random numbers
	#[inline(never)]
	fn get_block(&mut self) -> [u32; STATE_WORDS] {
		let mut result = self.state;

		// do 7 double rounds, i.e. 14 rounds
		for _ in 0..7 {
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

	/// Generates a float with normal distribution with mean 0 and variance 1.
	/// The generated values are guaranteed to be in the range (-10.0, 10.0)
	fn get_normal_clamped(&mut self) -> f64 {
		let block: [u32; 16] = self.get_block();
		let uniform: [f64; 16] = block.map(|v| {
			let v: f64 = v.into();
			v * (1.0 / 4_294_967_296.0)
		});

		let mut normal: [f64; 16] = uniform;
		for i in 0..8 {
			let x = 1.0 - normal[2 * i]; // (0.0, 1.0]
			let y = normal[2 * i + 1]; // [0.0, 1.0)

			// box mueller transform
			let r = (-2.0 * x.ln()).sqrt();
			let theta = std::f64::consts::TAU * y;
			let z0 = r * theta.cos();
			let z1 = r * theta.sin();

			normal[2 * i] = z0;
			normal[2 * i + 1] = z1;
		}

		// Combine the floats into a single value
		#[allow(clippy::suboptimal_flops)]
		#[rustfmt::skip]
		let result =
			(
				(
					((normal[ 0] * normal[ 4]) * (normal[ 8] * normal[12]))
					+
					((normal[ 1] * normal[ 5]) * (normal[ 9] * normal[13]))
				) + (
					((normal[ 2] * normal[ 6]) * (normal[10] * normal[14]))
					+
					((normal[ 3] * normal[ 7]) * (normal[11] * normal[15]))
				)
			) * 0.5;

		if result.abs() > 10.0 {
			cold_path();
			log::warn!("Rng::get_normal(): clamping {result} to (-10.0, 10.0)");
			return 0.0;
		}

		result
	}

	pub fn randn<T: FromToF64>(&mut self, out: &mut [T]) {
		for v in out.iter_mut() {
			*v = T::from_f64(self.get_normal_clamped());
		}
	}
}
