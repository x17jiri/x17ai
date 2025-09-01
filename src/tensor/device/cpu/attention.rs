//------------------------------------------------------------------------------
//
// Copyright 2025 Jiri Bobek. All rights reserved.
// License: GPL 3.0 or later. See LICENSE.txt for details.
//
//------------------------------------------------------------------------------

use std::cell::Cell;
use std::ops::RangeFull;

use crate::tensor::device::cpu::math;
use crate::util::FromToF64;

//--------------------------------------------------------------------------------------------------

#[derive(Clone, Copy)]
pub struct View2D<'a, T> {
	pub data: &'a [Cell<T>],
	pub cols: usize,
}

impl<'a, T> View2D<'a, T> {
	pub fn item(&self, row: usize, col: usize) -> &'a Cell<T> {
		debug_assert!(col < self.cols);
		let index = row * self.cols + col;
		&self.data[index]
	}

	pub fn slice(&self, head: usize, _: RangeFull) -> &'a [Cell<T>] {
		let begin = head * self.cols;
		let end = begin + self.cols;
		&self.data[begin..end]
	}
}

//--------------------------------------------------------------------------------------------------

#[derive(Clone, Copy)]
pub struct View3D<'a, T> {
	pub data: &'a [Cell<T>],
	pub seq_len: usize,
	pub seq_stride: usize,
	pub head_shift: usize,
	pub heads: usize,
	pub features: usize,
}

impl<'a, T> View3D<'a, T> {
	pub fn slice(&self, input: usize, head: usize, _: RangeFull) -> &'a [Cell<T>] {
		let head = head >> self.head_shift;
		let begin = input * self.seq_stride + head * self.features;
		let end = begin + self.features;
		&self.data[begin..end]
	}

	pub fn sub_sequence(&self, range: Range<usize>) -> Self {
		let data_begin = range.start * self.seq_stride;
		let data_end = range.end * self.seq_stride;
		let seq_len = range.end.saturating_sub(range.start);
		Self {
			data: &self.data[data_begin..data_end],
			seq_len,
			..*self
		}
	}
}

//--------------------------------------------------------------------------------------------------

fn attention_tile<T: Copy + FromToF64, const FIRST: bool>(
	acc: View3D<f64>, // [output, head, vo_feature]

	q: View3D<T>, // [output, head, qk_feature]
	k: View3D<T>, // [input, head, qk_feature]
	v: View3D<T>, // [input, head, vo_feature]

	// `acc`, `prev_m` and `prev_l` will be initialized when processing the first tile.
	prev_m: View2D<f64>, // [output, head]
	prev_l: View2D<f64>, // [output, head]

	// Scratch space for storing scores. It doesn't need to be initialized.
	// On GPU, its shape will be [output, head, input]. However, we process outputs
	// sequentially, so we don't need separate space for each output.
	scores: View2D<f64>, // [head, input]
) {
	let O = q.seq_len;
	let I = k.seq_len;
	let H = q.heads;
	let VO = v.features;
	for j in 0..O {
		for i in 0..I {
			for h in 0..H {
				let q = q.slice(j, h, ..);
				let k = k.slice(i, h, ..);
				scores.item(h, i).set(math::dot(q, k));
				// scores[h][i].set(math::dot(q, k))
			}
		}
		if FIRST {
			for h in 0..H {
				let scores = scores.slice(h, ..);
				let scores = &scores[..I]; // TODO
				let (first_m, first_l) = math::softmax_part1_(scores);

				//let S: Vec<f64> = scores.iter().map(|s| s.get() / first_l).collect();
				//println!("j = {}, h = {}, scores = {:.4?}", j, h, S.as_slice());

				prev_m.item(j, h).set(first_m);
				prev_l.item(j, h).set(first_l);

				let acc = acc.slice(j, h, ..);
				for i in 0..1 {
					let v = v.slice(i, h, ..);
					let score = scores[i].get().to_f64();
					for f in 0..VO {
						let v = v[f].get().to_f64();
						acc[f].set(score * v);
					}
				}
				for i in 1..I {
					let v = v.slice(i, h, ..);
					let score = scores[i].get().to_f64();
					for f in 0..VO {
						let v = v[f].get().to_f64();
						acc[f].set(acc[f].get() + score * v);
					}
				}
			}
		} else {
			for h in 0..H {
				let scores = scores.slice(h, ..);
				let scores = &scores[..I]; // TODO
				let (new_m, new_l) = math::softmax_part1_(scores);

				//let S: Vec<f64> = scores.iter().map(|s| s.get() / new_l).collect();
				//println!("j = {}, h = {}, ..scores = {:.4?}", j, h, S.as_slice());

				let prev_m = prev_m.item(j, h);
				let m = new_m.max(prev_m.get());

				let prev_weight = (prev_m.get() - m).exp();
				let new_weight = (new_m - m).exp();
				prev_m.set(m);

				let prev_l = prev_l.item(j, h);
				prev_l.set(prev_l.get() * prev_weight + new_l * new_weight);

				let acc = acc.slice(j, h, ..);
				for i in 0..1 {
					let v = v.slice(i, h, ..);
					let score = scores[i].get().to_f64() * new_weight;
					for f in 0..VO {
						let v = v[f].get().to_f64();
						acc[f].set(acc[f].get() * prev_weight + score * v);
					}
				}
				for i in 1..I {
					let v = v.slice(i, h, ..);
					let score = scores[i].get().to_f64() * new_weight;
					for f in 0..VO {
						let v = v[f].get().to_f64();
						acc[f].set(acc[f].get() + score * v);
					}
				}
			}
		}
	}
}

fn attention_finish<T: Copy + FromToF64>(
	dst: View3D<T>,      // [output, head, vo_feature]
	acc: View3D<f64>,    // [output, head, vo_feature]
	prev_l: View2D<f64>, // [output, head]
) {
	let O = dst.seq_len;
	let H = dst.heads;
	for j in 0..O {
		for h in 0..H {
			let norm = prev_l.item(j, h).get();
			let o_slice = acc.slice(j, h, ..);
			let dst_slice = dst.slice(j, h, ..);
			math::softmax_part2(o_slice, norm, dst_slice);
		}
	}
}

#[inline(never)]
fn attention_f<T: Copy + HasDType + FromToF64>(
	&self,
	o: &SliceSet,
	q: &SliceSet,
	k: &SliceSet,
	v: &SliceSet,
	params: &AttentionParams,
) {
	let o = self.cast_slice_set::<T>(o);
	let q = self.cast_slice_set::<T>(q);
	let k = self.cast_slice_set::<T>(k);
	let v = self.cast_slice_set::<T>(v);

	const Bq: usize = 64; // Number of outputs processed in one tile.
	const Bkv: usize = 13; // Number of inputs processed in one tile.
	let H = params.heads;

	let o_size = Bq * H * params.v_features;
	let prev_l_size = Bq * H;
	let prev_m_size = Bq * H;
	let scores_size = H * Bkv;

	let o_off = 0;
	let o_end = o_off + o_size;

	let prev_m_off = o_end;
	let prev_m_end = prev_m_off + prev_m_size;

	let prev_l_off = prev_m_end;
	let prev_l_end = prev_l_off + prev_l_size;

	let scores_off = prev_l_end;
	let scores_end = scores_off + scores_size;

	let mem_size = scores_end;

	let scratch_space = vec![Default::default(); mem_size];

	let acc = View3D {
		data: &scratch_space[o_off..o_end],
		seq_len: Bq,
		seq_stride: H * params.v_features,
		head_shift: 0,
		heads: H,
		features: params.v_features,
	};
	let prev_m = View2D {
		data: &scratch_space[prev_m_off..prev_m_end],
		cols: H,
	};
	let prev_l = View2D {
		data: &scratch_space[prev_l_off..prev_l_end],
		cols: H,
	};
	let scores = View2D {
		data: &scratch_space[scores_off..scores_end],
		cols: Bkv,
	};

	let q = View3D {
		data: q.buffer,
		seq_len: q.count,
		seq_stride: q.stride,
		head_shift: 0,
		heads: H,
		features: params.qk_features,
	};
	let k = View3D {
		data: k.buffer,
		seq_len: k.count,
		seq_stride: k.stride,
		head_shift: params.k_shift,
		heads: H >> params.k_shift,
		features: params.qk_features,
	};
	let v = View3D {
		data: v.buffer,
		seq_len: v.count,
		seq_stride: v.stride,
		head_shift: params.v_shift,
		heads: H >> params.v_shift,
		features: params.v_features,
	};
	let o = View3D {
		data: o.buffer,
		seq_len: o.count,
		seq_stride: o.stride,
		head_shift: 0,
		heads: H,
		features: params.v_features,
	};

	// TODO masking, scale

	let seq_len = o.seq_len;
	for j in (0..seq_len).step_by(Bq) {
		let je = (j + Bq).min(seq_len);
		let q = q.sub_sequence(j..je);

		for i in (0..seq_len).step_by(Bkv) {
			let ie = (i + Bkv).min(seq_len);
			let k = k.sub_sequence(i..ie);
			let v = v.sub_sequence(i..ie);

			// First tile will initialize `acc`, `prev_m`, `prev_l`
			if i == 0 {
				Self::attention_tile::<T, true>(acc, q, k, v, prev_m, prev_l, scores);
			} else {
				Self::attention_tile::<T, false>(acc, q, k, v, prev_m, prev_l, scores);
			}
		}

		let o = o.sub_sequence(j..je);
		Self::attention_finish::<T>(o, acc, prev_l);
	}
}

//--------------------------------------------------------------------------------------------------
