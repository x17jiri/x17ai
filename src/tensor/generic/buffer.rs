pub trait Buffer: Clone {
	type Ref: Buffer;

	fn as_ref(&self) -> Self::Ref;
}

impl<T> Buffer for &[T] {
	type Ref = Self;

	fn as_ref(&self) -> Self::Ref {
		*self
	}
}
