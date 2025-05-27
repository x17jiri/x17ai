// Copyright 2025 Jiri Bobek. All rights reserved.
// License: GPL 3.0 or later. See LICENSE.txt for details.

#[derive(Clone, Copy)]
pub struct Indent(usize);

impl Indent {
	pub fn new(val: usize) -> Self {
		Self(val)
	}
}

// operator +
impl std::ops::Add<usize> for Indent {
	type Output = Self;

	fn add(self, rhs: usize) -> Self::Output {
		Self(self.0 + rhs)
	}
}

// operator +=
impl std::ops::AddAssign<usize> for Indent {
	fn add_assign(&mut self, rhs: usize) {
		self.0 += rhs;
	}
}

// operator -
impl std::ops::Sub<usize> for Indent {
	type Output = Self;

	fn sub(self, rhs: usize) -> Self::Output {
		Self(self.0 - rhs)
	}
}

// operator -=
impl std::ops::SubAssign<usize> for Indent {
	fn sub_assign(&mut self, rhs: usize) {
		self.0 -= rhs;
	}
}

// std::fmt::Display for Indent
impl std::fmt::Display for Indent {
	fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
		for _ in 0..self.0 {
			write!(f, "\t")?;
		}
		Ok(())
	}
}
