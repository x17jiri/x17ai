//------------------------------------------------------------------------------
//
// Copyright 2025 Jiri Bobek. All rights reserved.
// License: GPL 3.0 or later. See LICENSE.txt for details.
//
//------------------------------------------------------------------------------

//--------------------------------------------------------------------------------------------------

use std::hint::{likely, unreachable_unchecked};

pub trait Hash {
	fn hash(&self) -> usize;
}

#[repr(C)]
pub struct Bucket<T: Hash + Eq> {
	hash: usize,
	key: T,
}

pub struct HashSet<T: Hash + Eq> {
	buckets: *mut Bucket<T>,
	elems: usize,
	mask: usize,
}

impl<T: Hash + Eq> Default for HashSet<T> {
	fn default() -> Self {
		Self::new()
	}
}

// TODO - Drop

static FAKE_BUCKET: isize = -1;

impl<T: Hash + Eq> HashSet<T> {
	pub fn new() -> Self {
		Self {
			buckets: (&raw const FAKE_BUCKET) as *mut Bucket<T>,
			elems: 0,
			mask: 0,
		}
	}

	#[allow(clippy::cast_possible_wrap)]
	pub fn find<Q>(
		&self,
		hash: usize,
		mut eq: impl FnMut(&T) -> bool,
	) -> (Option<&Bucket<T>>, usize)
	where
		Q: Hash,
		T: PartialEq<Q>,
	{
		let mask = self.mask;
		let buckets = self.buckets;

		let hash = hash & (isize::MAX as usize);
		for dist in 0.. {
			let i = (hash + dist) & mask;
			let bucket_hash = unsafe { *(buckets.add(i) as *const usize) };

			if bucket_hash == hash {
				let bucket = unsafe { buckets.add(i).as_ref_unchecked() };
				if likely(eq(&bucket.key)) {
					return (Some(bucket), i);
				}
			}

			let bucket_dist = (i - bucket_hash) & mask;
			if (bucket_hash as isize) < 0 || bucket_dist < dist {
				return (None, i);
			}
		}
		unsafe { unreachable_unchecked() };
	}
}

//--------------------------------------------------------------------------------------------------
