//------------------------------------------------------------------------------
//
// Copyright 2025 Jiri Bobek. All rights reserved.
// License: GPL 3.0 or later. See LICENSE.txt for details.
//
//------------------------------------------------------------------------------

use crate::tensor::generic::map::ND;
use crate::tensor::generic::{self};
use crate::util::UnwrapInfallible;

pub unsafe fn zip_elems<
	T: Copy,
	const O: usize, // number of outputs
	const C: usize, // number of contiguous inputs
	const B: usize, // number of broadcasted inputs
>(
	o: [generic::Tensor<ND<2>, &mut [T]>; O], //
	c: [generic::Tensor<ND<2>, &[T]>; C],
	b: [generic::Tensor<ND<2>, &[T]>; B],
	mut f: impl FnMut([&mut T; O], [T; C], [T; B]),
) {
	let shape = if let Some(t) = o.first() {
		t.nd_shape().unwrap_infallible()
	} else if let Some(t) = c.first() {
		t.nd_shape().unwrap_infallible()
	} else if let Some(t) = b.first() {
		t.nd_shape().unwrap_infallible()
	} else {
		return;
	};

	debug_assert!(o.iter().all(|t| t.ensure_safe().is_ok()));
	debug_assert!(c.iter().all(|t| t.ensure_safe().is_ok()));
	debug_assert!(b.iter().all(|t| t.ensure_safe().is_ok()));

	debug_assert!(o.iter().all(|t| t.nd_shape().unwrap_infallible() == shape));
	debug_assert!(c.iter().all(|t| t.nd_shape().unwrap_infallible() == shape));
	debug_assert!(b.iter().all(|t| t.nd_shape().unwrap_infallible() == shape));

	debug_assert!(o.iter().all(|t| t.map().dims[1].is_contiguous()));
	debug_assert!(c.iter().all(|t| t.map().dims[1].is_contiguous()));
	debug_assert!(b.iter().all(|t| t.map().dims[1].is_broadcasted()));

	let o = o.map(|t| {
		let (map, buf) = t.into_parts();
		(buf.as_mut_ptr().add(map.offset), map.dims[0].stride)
	});
	let c = c.map(|t| {
		let (map, buf) = t.into_parts();
		(buf.as_ptr().add(map.offset), map.dims[0].stride)
	});
	let b = b.map(|t| {
		let (map, buf) = t.into_parts();
		(buf.as_ptr().add(map.offset), map.dims[0].stride)
	});

	for j in 0..shape[0] {
		let o = o.map(|(ptr, stride)| ptr.add(j * stride));
		let c = c.map(|(ptr, stride)| ptr.add(j * stride));
		let b = b.map(|(ptr, stride)| ptr.add(j * stride));
		for i in 0..shape[1] {
			let o = o.map(|ptr| ptr.add(i).as_mut().unwrap_unchecked());
			let c = c.map(|ptr| ptr.add(i).read());
			let b = b.map(|ptr| ptr.read());
			f(o, c, b);
		}
	}
}

pub unsafe fn zip_vecs<T: Copy, const O: usize, const C: usize>(
	o: [generic::Tensor<ND<2>, &mut [T]>; O],
	c: [generic::Tensor<ND<2>, &[T]>; C],
	mut f: impl FnMut([&mut [T]; O], [&[T]; C]),
) {
	let shape = if let Some(t) = o.first() {
		t.nd_shape().unwrap_infallible()
	} else if let Some(t) = c.first() {
		t.nd_shape().unwrap_infallible()
	} else {
		return;
	};

	debug_assert!(o.iter().all(|t| t.ensure_safe().is_ok()));
	debug_assert!(c.iter().all(|t| t.ensure_safe().is_ok()));

	debug_assert!(o.iter().all(|t| t.nd_shape().unwrap_infallible() == shape));
	debug_assert!(c.iter().all(|t| t.nd_shape().unwrap_infallible() == shape));

	let o = o.map(|t| {
		let (map, buf) = t.into_parts();
		(buf.as_mut_ptr().add(map.offset), map.dims[0].stride)
	});
	let c = c.map(|t| {
		let (map, buf) = t.into_parts();
		(buf.as_ptr().add(map.offset), map.dims[0].stride)
	});

	for j in 0..shape[0] {
		let o =
			o.map(|(ptr, stride)| std::slice::from_raw_parts_mut(ptr.add(j * stride), shape[1]));
		let c = c.map(|(ptr, stride)| std::slice::from_raw_parts(ptr.add(j * stride), shape[1]));
		f(o, c);
	}
}

#[allow(clippy::many_single_char_names)]
pub unsafe fn zip_vec_reduce<T: Copy, const C: usize>(
	r: generic::Tensor<ND<2>, &mut [T]>,
	c: [generic::Tensor<ND<2>, &[T]>; C],
	mut f: impl FnMut(&mut T, [&[T]; C]),
) {
	let shape = c.first().unwrap().nd_shape().unwrap_infallible();

	debug_assert!(r.ensure_safe().is_ok());
	debug_assert!(c.iter().all(|t| t.ensure_safe().is_ok()));

	debug_assert!(r.nd_shape().unwrap_infallible() == [shape[0], 1]);
	debug_assert!(c.iter().all(|t| t.nd_shape().unwrap_infallible() == shape));

	let r = {
		let (map, buf) = r.into_parts();
		(buf.as_mut_ptr().add(map.offset), map.dims[0].stride)
	};
	let c = c.map(|t| {
		let (map, buf) = t.into_parts();
		(buf.as_ptr().add(map.offset), map.dims[0].stride)
	});

	for j in 0..shape[0] {
		let r = std::slice::from_raw_parts_mut(r.0.add(j * r.1), 1);
		let t = c.map(|(ptr, stride)| std::slice::from_raw_parts(ptr.add(j * stride), shape[1]));
		f(r.first_mut().unwrap_unchecked(), t);
	}
}
