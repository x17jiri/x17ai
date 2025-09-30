//------------------------------------------------------------------------------
//
// Copyright 2025 Jiri Bobek. All rights reserved.
// License: GPL 3.0 or later. See LICENSE.txt for details.
//
//------------------------------------------------------------------------------

use std::mem::MaybeUninit;

//--------------------------------------------------------------------------------------------------

#[repr(C)]
pub struct Derived<Base, D: ?Sized, T: ?Sized> {
	vmt: MaybeUninit<<D as std::ptr::Pointee>::Metadata>,
	base: Base,
	value: T,
}

/*
#[repr(C)]
pub struct RcInner<D: ?Sized, T: ?Sized> {
	rc: Cell<usize>,
	metadata: MaybeUninit<<D as std::ptr::Pointee>::Metadata>,
	value: T,
}

pub struct ThinRc<D: ?Sized, T: ?Sized> {
	inner: NonNull<ThinRcInner<T, ()>>,
	phantom: std::marker::PhantomData<D>,
}

impl<D: ?Sized, T> ThinRc<D, T>
where
	T: Unsize<D>,
{
	pub fn new(value: T) -> Self {
		let inner = Box::new(RcInner::<D, T> {
			rc: Cell::new(1),
			metadata: MaybeUninit::uninit(),
			value,
		});
		Self {
			inner: Box::leak(inner).into(),
			phantom: std::marker::PhantomData,
		}
	}
}

impl<D: ?Sized, T: ?Sized> ThinRc<D, T> {
	pub fn from_inner(inner: &RcInner<T>) -> Self {
		inner.rc.set(inner.rc.get() + 1);
		Rc { inner: inner.into() }
	}

	pub fn inner(&self) -> &RcInner<T> {
		unsafe { self.inner.as_ref() }
	}

	pub fn is_unique(&self) -> bool {
		self.inner().rc.get() == 1
	}

	#[inline(never)]
	fn __drop(&mut self) {
		unsafe { std::ptr::drop_in_place(self.inner.as_mut()) };
	}
}

impl<T: ?Sized> Clone for Rc<T> {
	#[inline]
	fn clone(&self) -> Self {
		Rc::from_inner(self.inner())
	}
}

impl<T: ?Sized> Drop for Rc<T> {
	#[inline]
	fn drop(&mut self) {
		let inner = self.inner();
		inner.rc.set(inner.rc.get() - 1);
		if inner.rc.get() == 0 {
			Self::__drop(self);
		}
	}
}

impl<T: ?Sized> std::ops::Deref for Rc<T> {
	type Target = T;

	#[inline]
	fn deref(&self) -> &Self::Target {
		&self.inner().value
	}
}

impl<T: ?Sized> !std::marker::Send for Rc<T> {}
impl<T: ?Sized> !std::marker::Sync for Rc<T> {}
impl<T: ?Sized + Unsize<U>, U: ?Sized> CoerceUnsized<Rc<U>> for Rc<T> {}
impl<T: ?Sized + Unsize<U>, U: ?Sized> DispatchFromDyn<Rc<U>> for Rc<T> {}
*/
//--------------------------------------------------------------------------------------------------
