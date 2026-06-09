//------------------------------------------------------------------------------
//
// Copyright 2025 Jiri Bobek. All rights reserved.
// License: GPL 3.0 or later. See LICENSE.txt for details.
//
//------------------------------------------------------------------------------

use crate::util::index_vec::{IndexTrait, IndexVec};

//--------------------------------------------------------------------------------------------------

#[derive(Clone)]
pub struct Bitmap {
	pub rows: usize,
	pub words_per_row: usize,
	pub data: Vec<usize>,
}

impl Default for Bitmap {
	fn default() -> Self {
		Self {
			rows: 0,
			words_per_row: 0,
			data: Vec::new(),
		}
	}
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

	pub fn clear_and_resize(&mut self, rows: usize, cols: usize) {
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

	pub fn copy_row(&mut self, dst_row: usize, src_row: usize) {
		let src_row = self.row_raw(src_row);
		let dst_row = self.row_raw_mut(dst_row);
		if src_row == dst_row {
			return;
		}
		unsafe {
			std::ptr::copy_nonoverlapping(src_row, dst_row, self.words_per_row);
		}
	}

	pub fn get_bit(&self, row: usize, col: usize) -> bool {
		let word_index = col / Self::WORD_BITS;
		let bit_index = col % Self::WORD_BITS;
		let word = self.data[row * self.words_per_row + word_index];
		let mask = 1 << bit_index;
		(word & mask) != 0
	}

	pub fn set_bit(&mut self, row: usize, col: usize) -> bool {
		let word_index = col / Self::WORD_BITS;
		let bit_index = col % Self::WORD_BITS;
		let word = &mut self.data[row * self.words_per_row + word_index];
		let mask = 1 << bit_index;
		let was_set = (*word & mask) != 0;
		*word |= mask;
		was_set
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

	pub fn and_not(&mut self, dst_row: usize, src_row1: usize, src_row2: usize) {
		let dst_row = self.row_raw_mut(dst_row);
		let src_row1 = self.row_raw(src_row1);
		let src_row2 = self.row_raw(src_row2);
		unsafe {
			for i in 0..self.words_per_row {
				dst_row.add(i).write(src_row1.add(i).read() & !src_row2.add(i).read());
			}
		}
	}

	pub fn have_common_bits(&self, row_a: usize, row_b: usize) -> bool {
		let a = self.row_raw(row_a);
		let b = self.row_raw(row_b);
		unsafe {
			for i in 0..self.words_per_row {
				let a_word = a.add(i).read();
				let b_word = b.add(i).read();
				if (a_word & b_word) != 0 {
					return true;
				}
			}
		}
		false
	}
}

//--------------------------------------------------------------------------------------------------

#[derive(Clone, Default)]
pub struct IndexBitmap<Index: IndexTrait> {
	pub raw: Bitmap,
	_marker: std::marker::PhantomData<Index>,
}

impl<Index: IndexTrait> IndexBitmap<Index> {
	pub fn new() -> Self {
		Self {
			raw: Bitmap::new(0, 0),
			_marker: std::marker::PhantomData,
		}
	}

	pub fn clear_and_resize<T>(&mut self, rows_model: &IndexVec<Index, T>, cols: usize) {
		self.raw.clear_and_resize(rows_model.raw.len(), cols);
	}

	pub fn row(&self, row: Index) -> &[usize] {
		self.raw.row(row.to_raw())
	}

	pub fn copy_row(&mut self, dst_row: Index, src_row: Index) {
		self.raw.copy_row(dst_row.to_raw(), src_row.to_raw());
	}

	pub fn get_bit(&self, row: Index, col: usize) -> bool {
		self.raw.get_bit(row.to_raw(), col)
	}

	pub fn set_bit(&mut self, row: Index, col: usize) -> bool {
		self.raw.set_bit(row.to_raw(), col)
	}

	pub fn union(&mut self, dst_row: Index, src_row1: Index, src_row2: Index) {
		self.raw.union(dst_row.to_raw(), src_row1.to_raw(), src_row2.to_raw());
	}

	pub fn and_not(&mut self, dst_row: Index, src_row1: Index, src_row2: Index) {
		self.raw.and_not(dst_row.to_raw(), src_row1.to_raw(), src_row2.to_raw());
	}

	pub fn have_common_bits(&self, row_a: Index, row_b: Index) -> bool {
		self.raw.have_common_bits(row_a.to_raw(), row_b.to_raw())
	}
}

//--------------------------------------------------------------------------------------------------
