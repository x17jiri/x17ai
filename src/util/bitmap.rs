//------------------------------------------------------------------------------
//
// Copyright 2025 Jiri Bobek. All rights reserved.
// License: GPL 3.0 or later. See LICENSE.txt for details.
//
//------------------------------------------------------------------------------

use thin_vec::{ThinVec, thin_vec};

//--------------------------------------------------------------------------------------------------

pub struct Bitmap {
	pub rows: usize,
	pub words_per_row: usize,
	pub data: Vec<usize>,
}

impl Bitmap {
	const WORD_BITS: usize = usize::BITS as usize;

	pub fn new(rows: usize, cols: usize) -> Self {
		let words_per_row = (cols + Self::WORD_BITS - 1) / Self::WORD_BITS;
		Self {
			rows,
			words_per_row,
			data: vec![0; rows * words_per_row],
		}
	}

	pub fn resize(&mut self, rows: usize, cols: usize) {
		self.words_per_row = (cols + Self::WORD_BITS - 1) / Self::WORD_BITS;
		self.rows = rows;
		let words = rows * self.words_per_row;
		unsafe {
			self.data.set_len(0);
			self.data.reserve(words);
			let buffer = self.data.spare_capacity_mut();
			for word in buffer.get_unchecked_mut(0..words) {
				word.write(0);
			}
			self.data.set_len(words);
		};
	}

	pub fn row(&self, row: usize) -> &[usize] {
		let offset = row * self.words_per_row;
		&self.data[offset..offset + self.words_per_row]
	}

	pub fn row_mut(&mut self, row: usize) -> &mut [usize] {
		let offset = row * self.words_per_row;
		&mut self.data[offset..offset + self.words_per_row]
	}

	pub fn row_raw(&self, row: usize) -> *const usize {
		let offset = row * self.words_per_row;
		self.data[offset..offset + self.words_per_row].as_ptr()
	}

	pub fn row_raw_mut(&mut self, row: usize) -> *mut usize {
		let offset = row * self.words_per_row;
		self.data[offset..offset + self.words_per_row].as_mut_ptr()
	}

	pub fn copy_row(&mut self, src_row: usize, dst_row: usize) {
		let src_row = self.row_raw(src_row);
		let dst_row = self.row_raw_mut(dst_row);
		if src_row == dst_row {
			return;
		}
		unsafe {
			std::ptr::copy_nonoverlapping(src_row, dst_row, self.words_per_row);
		}
	}

	pub fn set_bit(&mut self, row: usize, col: usize) {
		let word_index = col / Self::WORD_BITS;
		let bit_index = col % Self::WORD_BITS;
		self.data[row * self.words_per_row + word_index] |= 1 << bit_index;
	}

	pub fn union(&mut self, dst_row: usize, src_row1: usize, src_row2: usize) {
		let dst_row = self.row_raw_mut(dst_row);
		let src_row1 = self.row_raw(src_row1);
		let src_row2 = self.row_raw(src_row2);
		unsafe {
			for i in 0..self.words_per_row {
				dst_row.add(i).write(src_row1.add(i).read() | src_row2.add(i).read());
			}
		}
	}
}

//--------------------------------------------------------------------------------------------------
