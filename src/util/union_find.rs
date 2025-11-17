//------------------------------------------------------------------------------
//
// Copyright 2025 Jiri Bobek. All rights reserved.
// License: GPL 3.0 or later. See LICENSE.txt for details.
//
//------------------------------------------------------------------------------

//--------------------------------------------------------------------------------------------------

#[derive(Clone)]
pub struct UnionFind {
	link_parent: Vec<isize>,
}

#[allow(clippy::indexing_slicing)]
impl UnionFind {
	pub fn new(size: usize) -> Self {
		Self { link_parent: vec![-1; size] }
	}

	#[inline]
	pub fn size(&self) -> usize {
		self.link_parent.len()
	}

	pub fn union(&mut self, key0: usize, key1: usize) -> usize {
		let (parent0, neg_rank0) = self.__find(key0);
		let (parent1, neg_rank1) = self.__find(key1);
		if parent0 == parent1 {
			return parent0;
		}

		if neg_rank0 == neg_rank1 {
			self.link_parent[parent0] = neg_rank0 - 1;
			self.link_parent[parent1] = parent0 as isize;
		} else {
			let (parent0, parent1) =
				if neg_rank0 < neg_rank1 { (parent0, parent1) } else { (parent1, parent0) };
			self.link_parent[parent1] = parent0 as isize;
		}
		parent0
	}

	#[inline]
	fn __find(&mut self, key: usize) -> (usize, isize) {
		let mut key: usize = key;
		let mut parent: isize = self.link_parent[key];
		if parent < 0 {
			let neg_rank = parent;
			return (key, neg_rank);
		}
		loop {
			let grand_parent = self.link_parent[parent as usize];
			if grand_parent < 0 {
				let neg_rank = grand_parent;
				return (parent as usize, neg_rank);
			}
			self.link_parent[key] = grand_parent;
			key = parent as usize;
			parent = grand_parent;
		}
	}

	pub fn compact_ids(mut self) -> (Vec<usize>, usize) {
		let mut sets = 0;
		for i in 0..self.link_parent.len() {
			let mut key = i;
			let mut parent = self.link_parent[key];
			while key >= i && parent != key {
				let grand_parent = self.link_parent[parent];
				self.link_parent[key] = i;
				key = parent;
				parent = grand_parent;
			}
			if key < i {
				self.link_parent[i] = self.link_parent[key];
			} else {
				self.link_parent[key] = i;
				self.link_parent[i] = sets;
				sets += 1;
			}
		}
		(self.link_parent, sets)
	}
}
