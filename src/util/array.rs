//------------------------------------------------------------------------------
//
// Copyright 2025 Jiri Bobek. All rights reserved.
// License: GPL 3.0 or later. See LICENSE.txt for details.
//
//------------------------------------------------------------------------------

use std::mem::MaybeUninit;
/*
pub fn map_owned<const N: usize, T, U>(array: [T; N], mut f: impl FnMut(usize, T) -> U) -> [U; N] {
	let mut u = [const { MaybeUninit::uninit() }; N];
	for i in 0..N {
		u[i].write(f(i, array[i]));
	}
	unsafe { MaybeUninit::array_assume_init(u) }
}
*/
pub fn map_borrowed<const N: usize, T, U>(
	array: &[T; N], mut f: impl FnMut(usize, &T) -> U,
) -> [U; N] {
	let mut u = [const { MaybeUninit::uninit() }; N];
	for i in 0..N {
		u[i].write(f(i, &array[i]));
	}
	unsafe { MaybeUninit::array_assume_init(u) }
}
