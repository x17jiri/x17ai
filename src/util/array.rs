//------------------------------------------------------------------------------
//
// Copyright 2025 Jiri Bobek. All rights reserved.
// License: GPL 3.0 or later. See LICENSE.txt for details.
//
//------------------------------------------------------------------------------

use std::mem::MaybeUninit;

pub fn map_into<const N: usize, T, U>(array: [T; N], mut f: impl FnMut(usize, T) -> U) -> [U; N] {
	let mut u = [const { MaybeUninit::uninit() }; N];
	for (i, t) in array.into_iter().enumerate() {
		u[i].write(f(i, t));
	}
	unsafe { MaybeUninit::array_assume_init(u) }
}

pub fn try_map_into<const N: usize, T, U, E>(
	array: [T; N],
	mut f: impl FnMut(usize, T) -> Result<U, E>,
) -> Result<[U; N], E> {
	let mut u = [const { MaybeUninit::uninit() }; N];
	for (i, t) in array.into_iter().enumerate() {
		u[i].write(f(i, t)?);
	}
	Ok(unsafe { MaybeUninit::array_assume_init(u) })
}

pub fn map<const N: usize, T, U>(array: &[T; N], mut f: impl FnMut(usize, &T) -> U) -> [U; N] {
	let mut u = [const { MaybeUninit::uninit() }; N];
	for i in 0..N {
		u[i].write(f(i, &array[i]));
	}
	unsafe { MaybeUninit::array_assume_init(u) }
}

pub fn map_mut<const N: usize, T, U>(
	array: &mut [T; N],
	mut f: impl FnMut(usize, &mut T) -> U,
) -> [U; N] {
	let mut u = [const { MaybeUninit::uninit() }; N];
	for i in 0..N {
		u[i].write(f(i, &mut array[i]));
	}
	unsafe { MaybeUninit::array_assume_init(u) }
}

pub fn try_map<const N: usize, T, U, E>(
	array: &[T; N],
	mut f: impl FnMut(usize, &T) -> Result<U, E>,
) -> Result<[U; N], E> {
	let mut u = [const { MaybeUninit::uninit() }; N];
	for i in 0..N {
		u[i].write(f(i, &array[i])?);
	}
	Ok(unsafe { MaybeUninit::array_assume_init(u) })
}

pub fn try_from_iter<const N: usize, T, I>(iter: I) -> Option<[T; N]>
where
	I: IntoIterator<Item = T, IntoIter: ExactSizeIterator + Sized>,
{
	let mut iter = iter.into_iter();
	if iter.len() != N {
		return None;
	}
	let mut array = [const { MaybeUninit::uninit() }; N];
	for a in array.iter_mut() {
		a.write(iter.next().unwrap());
	}
	Some(unsafe { MaybeUninit::array_assume_init(array) })
}

pub fn concat_arrays<T, const A: usize, const B: usize>(a: [T; A], b: [T; B]) -> [T; A + B] {
	let mut c = [const { MaybeUninit::uninit() }; A + B];
	for (i, t) in a.into_iter().enumerate() {
		c[i].write(t);
	}
	for (i, t) in b.into_iter().enumerate() {
		c[A + i].write(t);
	}
	unsafe { MaybeUninit::array_assume_init(c) }
}

pub fn try_map_backward<const N: usize, T, U, E>(
	array: &[T; N],
	mut f: impl FnMut(usize, &T) -> Result<U, E>,
) -> Result<[U; N], E> {
	let mut u = [const { MaybeUninit::uninit() }; N];
	for i in (0..N).rev() {
		u[i].write(f(i, &array[i])?);
	}
	Ok(unsafe { MaybeUninit::array_assume_init(u) })
}
