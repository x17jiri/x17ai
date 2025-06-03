use std::cell::Cell;
use std::ptr::NonNull;

#[repr(C)]
pub struct RcInner<T> {
	rc: Cell<usize>,
	value: T,
}

pub struct Rc<T> {
	inner: NonNull<RcInner<T>>,
}

impl<T> Rc<T> {
	pub fn new(value: T) -> Self {
		Rc {
			inner: Box::leak(Box::new(RcInner { rc: Cell::new(1), value })).into(),
		}
	}

	pub fn from_inner(inner: &RcInner<T>) -> Self {
		inner.rc.set(inner.rc.get() + 1);
		Rc { inner: NonNull::from(inner) }
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

impl<T> Clone for Rc<T> {
	#[inline]
	fn clone(&self) -> Self {
		Rc::from_inner(self.inner())
	}
}

impl<T> Drop for Rc<T> {
	#[inline]
	fn drop(&mut self) {
		let inner = self.inner();
		inner.rc.set(inner.rc.get() - 1);
		if inner.rc.get() == 0 {
			Self::__drop(self);
		}
	}
}

impl<T> !std::marker::Send for Rc<T> {}
impl<T> !std::marker::Sync for Rc<T> {}

impl<T> std::ops::Deref for Rc<T> {
	type Target = T;

	#[inline]
	fn deref(&self) -> &Self::Target {
		&self.inner().value
	}
}
