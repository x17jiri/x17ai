// Copyright 2024 Jiri Bobek. All rights reserved.
// License: GPL 3.0 or later. See LICENSE.txt for details.

// State initialization constant ("expand 32-byte k")
const CONSTANTS: [u32; 4] = [0x_6170_7865, 0x_3320_646e, 0x_7962_2d32, 0x_6b20_6574];

pub const STATE_WORDS: usize = 16;

pub const BLOCK_SIZE: usize = 64; // bytes

pub struct Rng {
	state: [u32; STATE_WORDS],
	block: [u32; STATE_WORDS],
	pos: usize,
	saved_norm: Option<f64>,
}

impl Rng {
	pub fn new(key: &[u32; 8], iv: &[u32; 2]) -> Self {
		#[rustfmt::skip]
		Rng {
			state: [
				CONSTANTS[0], CONSTANTS[1], CONSTANTS[2], CONSTANTS[3],
				key[0],       key[1],       key[2],       key[3],
				key[4],       key[5],       key[6],       key[7],
				0,            0,            iv[0],        iv[1],
			],
			block: [0; STATE_WORDS],
			pos: 0,
			saved_norm: None,
		}
	}

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
		for i in 0..STATE_WORDS {
			result[i] = result[i].wrapping_add(self.state[i]);
		}

		// increment counter
		let (t, c) = self.state[12].overflowing_add(1);
		self.state[12] = t;
		self.state[13] = self.state[13].wrapping_add(c as u32);

		result
	}

	pub fn get_u32(&mut self) -> u32 {
		if self.pos == 0 {
			self.block = self.get_block();
			self.pos = STATE_WORDS;
		}

		self.pos -= 1;
		self.block[self.pos]
	}

	// converts u32 to a float in the range [0.0, 1.0)
	pub fn uniform(val: u32) -> f64 {
		let mantissa = (val as u64) << (52 - 32);
		let exp = (1023 as u64) << 52;

		f64::from_bits(mantissa | exp) - 1.0
	}

	// converts u32 to a float in the range (-1.0, 1.0)
	pub fn signed_uniform(val: u32) -> f64 {
		let val = val as u64;

		let sign = (val & 0x_8000_0000) << 32;
		let mantissa = val << (52 - 31);
		let exp = (1023 as u64) << 52;
		let val = sign | mantissa | exp;

		// At this point, the value is in the range (-2.0, -1.0] union [1.0, 2.0)
		// If val < 0, result = val + 1 = val - sign(val)
		// If val > 0, result = val - 1 = val - sign(val)
		let one = (1023 as u64) << 52;
		let sign = sign | one;

		let val = f64::from_bits(val);
		let sign = f64::from_bits(sign);

		val - sign
	}

	pub fn get_normal(&mut self) -> f64 {
		if let Some(val) = self.saved_norm {
			self.saved_norm = None;
			return val;
		}

		let eps = 1.0 / 4294967296.0;
		loop {
			let x = Self::signed_uniform(self.get_u32());
			let y = Self::signed_uniform(self.get_u32());
			let r2 = (x * x) + (y * y);
			if r2 < eps || r2 > 1.0 {
				continue;
			}

			let mult = (-2.0 * r2.ln() / r2).sqrt();
			let z0 = x * mult;
			let z1 = y * mult;

			self.saved_norm = Some(z1);
			return z0;
		}
	}

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
}
