//! Cache-blocked, register-tiled GEMM driver (issue #121).
//!
//! The classic Goto/BLIS loop nest, written ONCE and shared by every GEMM
//! entry point in this crate (the canonical-implementation rule — the
//! scalar `gemm_f32` export, the SIMD128 `gemm_f32_simd128`
//! export and the native cdylib all run this exact driver; they differ only
//! in the micro-kernel plugged into it and in where the packing buffers
//! live):
//!
//! ```text
//! for jc in 0..n step NC            // B column block  -> packed B (KC x NC) stays in L2
//!   for pc in 0..k step KC          // shared-dim block
//!     pack B[pc.., jc..] into NR-wide column panels
//!     for ic in 0..m step MC        // A row block     -> packed A (MC x KC) stays in L2
//!       pack A[ic.., pc..] into MR-tall row panels
//!       for each NR panel, for each MR panel:
//!         micro-kernel: MR x NR register tile += A panel (MR x KC) @ B panel (KC x NR)
//!         write the tile back through `out`'s strides (alpha/beta, edge clipping)
//! ```
//!
//! Packing is what makes the ABI's arbitrary (row, col) strides free: the
//! micro-kernel only ever sees contiguous, zero-padded panels, so a
//! transposed or otherwise strided operand costs one O(n^2) gather per
//! block instead of a strided load in the O(n^3) inner loop. Edge tiles
//! (m % MR, n % NR) are handled by zero-padding the packed panels and
//! clipping at write-back, so the micro-kernels have no remainder loops.
//!
//! **Everything here is panic-free raw-pointer code, by design.** The
//! SIMD128 build is a second wasm module that shares the scalar module's
//! linear memory (`--import-memory`, see lib.rs's `simd` module doc), and
//! instantiating it re-writes its own `.rodata` segment over the scalar
//! module's at the same addresses. That is only harmless while the two
//! modules' data segments are byte-identical — a new panic location (a
//! bounds-checked slice index, an `unwrap`) compiled into the SIMD-only
//! code would add rodata the scalar build doesn't have. `Kernels.load()`
//! verifies the invariant at runtime (snapshot + compare, SIMD disabled
//! and memory restored on mismatch), and the SIMD entry point additionally
//! never touches the heap: its packing buffers live on the wasm shadow
//! stack, because the SIMD module's own copy of the allocator would
//! otherwise hand out memory the scalar module's allocator also owns.
//!
//! Numerics: each output element is still a plain f32 dot product, but
//! accumulated in KC-sized partial sums (`out += alpha * partial` per
//! K-block) rather than one left-to-right sum, so results differ from the
//! old naive kernel in the last few ulps — well inside the NumPy oracle's
//! `matmul:f32` tolerance (rtol 1e-4). No FMA is used anywhere (WASM
//! SIMD128 has none; relaxed-SIMD's `f32x4.relaxed_madd` is
//! implementation-defined fused-or-not, which would make results
//! engine-dependent), so the scalar and SIMD128 paths perform the same
//! multiply-then-add per element in the same order and agree bit-for-bit.

/// Shared-dimension block. A 4x8 micro-kernel's A panel (MR*KC f32 = 4 KiB)
/// plus B panel (KC*NR f32 = 8 KiB) fits comfortably in L1.
pub(crate) const KC: usize = 256;
/// Row block of A kept packed across a whole B block (MC*KC f32 = 64 KiB).
pub(crate) const MC: usize = 64;
/// Column block of B kept packed across all row blocks (KC*NC f32 = 256 KiB).
pub(crate) const NC: usize = 256;

/// Size (in f32 elements) of the packed-A scratch buffer every entry point
/// must provide (only the wasm entry points use the fixed maximum; native
/// sizes its heap scratch to the problem).
#[cfg_attr(not(target_arch = "wasm32"), allow(dead_code))]
pub(crate) const APACK_LEN: usize = MC * KC;
/// Size (in f32 elements) of the packed-B scratch buffer.
#[cfg_attr(not(target_arch = "wasm32"), allow(dead_code))]
pub(crate) const BPACK_LEN: usize = KC * NC;

/// Largest MR*NR tile any micro-kernel in this crate produces.
const MAX_TILE: usize = 64;

/// A read-only strided matrix view (offsets/strides in elements).
#[derive(Clone, Copy)]
pub(crate) struct MatRef {
    pub ptr: *const f32,
    pub offset: isize,
    pub row_stride: isize,
    pub col_stride: isize,
}

/// A writable strided matrix view.
#[derive(Clone, Copy)]
pub(crate) struct MatMut {
    pub ptr: *mut f32,
    pub offset: isize,
    pub row_stride: isize,
    pub col_stride: isize,
}

impl MatRef {
    #[inline(always)]
    unsafe fn at(&self, i: usize, j: usize) -> f32 {
        *self
            .ptr
            .offset(self.offset + i as isize * self.row_stride + j as isize * self.col_stride)
    }
}

impl MatMut {
    #[inline(always)]
    unsafe fn slot(&self, i: usize, j: usize) -> *mut f32 {
        self.ptr
            .offset(self.offset + i as isize * self.row_stride + j as isize * self.col_stride)
    }
}

/// A register-tiled micro-kernel: computes the full MR x NR product of one
/// packed A panel (`kc` columns of MR rows, row index fastest) and one
/// packed B panel (`kc` rows of NR columns, column index fastest), and
/// STORES (not accumulates) it row-major into `tile` (MR*NR f32).
pub(crate) trait MicroKernel {
    const MR: usize;
    const NR: usize;
    /// # Safety
    /// `a` must point to `kc * MR` readable f32, `b` to `kc * NR`, and
    /// `tile` to `MR * NR` writable f32.
    unsafe fn run(kc: usize, a: *const f32, b: *const f32, tile: *mut f32);
}

/// Portable scalar micro-kernel, 4x8. Written so LLVM keeps the 32
/// accumulators in registers (fully unrolled over the fixed-size array);
/// on native targets it autovectorizes (NEON/SSE), on the scalar wasm
/// build it is 32 f32 locals.
pub(crate) struct ScalarKernel;

impl MicroKernel for ScalarKernel {
    const MR: usize = 4;
    const NR: usize = 8;

    #[inline(always)]
    unsafe fn run(kc: usize, a: *const f32, b: *const f32, tile: *mut f32) {
        let mut acc = [[0.0f32; 8]; 4];
        let mut ap = a;
        let mut bp = b;
        for _ in 0..kc {
            let mut bv = [0.0f32; 8];
            for (j, v) in bv.iter_mut().enumerate() {
                *v = *bp.add(j);
            }
            for (i, row) in acc.iter_mut().enumerate() {
                let av = *ap.add(i);
                for (c, bj) in row.iter_mut().zip(bv.iter()) {
                    *c += av * *bj;
                }
            }
            ap = ap.add(4);
            bp = bp.add(8);
        }
        for (i, row) in acc.iter().enumerate() {
            for (j, v) in row.iter().enumerate() {
                *tile.add(i * 8 + j) = *v;
            }
        }
    }
}

/// Pack `A[row0 .. row0+mc, col0 .. col0+kc]` into ceil(mc/MR) panels, each
/// `kc` groups of MR consecutive row values; rows past `mc` are zero.
#[inline(always)]
unsafe fn pack_a(
    mr: usize,
    a: MatRef,
    row0: usize,
    col0: usize,
    mc: usize,
    kc: usize,
    dst: *mut f32,
) {
    let mut d = dst;
    let mut ir = 0;
    while ir < mc {
        let rows = if mc - ir < mr { mc - ir } else { mr };
        for p in 0..kc {
            for i in 0..mr {
                *d.add(i) = if i < rows {
                    a.at(row0 + ir + i, col0 + p)
                } else {
                    0.0
                };
            }
            d = d.add(mr);
        }
        ir += mr;
    }
}

/// Pack `B[row0 .. row0+kc, col0 .. col0+nc]` into ceil(nc/NR) panels, each
/// `kc` groups of NR consecutive column values; columns past `nc` are zero.
#[inline(always)]
unsafe fn pack_b(
    nr: usize,
    b: MatRef,
    row0: usize,
    col0: usize,
    kc: usize,
    nc: usize,
    dst: *mut f32,
) {
    let mut d = dst;
    let mut jr = 0;
    while jr < nc {
        let cols = if nc - jr < nr { nc - jr } else { nr };
        if cols == nr && b.col_stride == 1 {
            // Fast path for the common row-major B: each group is NR
            // contiguous source floats.
            for p in 0..kc {
                let src = b
                    .ptr
                    .offset(b.offset + (row0 + p) as isize * b.row_stride + (col0 + jr) as isize);
                core::ptr::copy_nonoverlapping(src, d, nr);
                d = d.add(nr);
            }
        } else {
            for p in 0..kc {
                for j in 0..nr {
                    *d.add(j) = if j < cols {
                        b.at(row0 + p, col0 + jr + j)
                    } else {
                        0.0
                    };
                }
                d = d.add(nr);
            }
        }
        jr += nr;
    }
}

/// `out[i, j] = alpha * partial + (first ? beta * out : out)` for the valid
/// `rows x cols` corner of an MR x NR tile. `beta == 0` never READS `out`
/// (so an uninitialized/NaN destination can't leak into the result — the
/// same contract the naive kernel had).
#[inline(always)]
#[allow(clippy::too_many_arguments)]
unsafe fn write_tile(
    out: MatMut,
    row0: usize,
    col0: usize,
    rows: usize,
    cols: usize,
    nr: usize,
    tile: *const f32,
    alpha: f32,
    beta: f32,
    first: bool,
) {
    for i in 0..rows {
        for j in 0..cols {
            let slot = out.slot(row0 + i, col0 + j);
            let v = alpha * *tile.add(i * nr + j);
            *slot = if first {
                if beta != 0.0 {
                    v + beta * *slot
                } else {
                    v
                }
            } else {
                v + *slot
            };
        }
    }
}

/// `out = alpha * A @ B + beta * out` with A (m x k), B (k x n), out (m x n),
/// all strided. `apack`/`bpack` are caller-provided scratch of at least
/// `APACK_LEN`/`BPACK_LEN` f32 (see the module doc for why the caller owns
/// them). `out` must not overlap `a`/`b`.
///
/// # Safety
/// Every (ptr, offset, strides) triple must describe in-bounds storage for
/// its matrix shape; scratch pointers must be valid for the lengths above.
#[inline]
#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn gemm_blocked<K: MicroKernel>(
    a: MatRef,
    b: MatRef,
    out: MatMut,
    m: usize,
    n: usize,
    k: usize,
    alpha: f32,
    beta: f32,
    apack: *mut f32,
    bpack: *mut f32,
) {
    if m == 0 || n == 0 {
        return;
    }
    if k == 0 {
        // Empty inner dimension: A@B is all zeros, so out = beta * out.
        for i in 0..m {
            for j in 0..n {
                let slot = out.slot(i, j);
                *slot = if beta != 0.0 { beta * *slot } else { 0.0 };
            }
        }
        return;
    }
    // Compile-time check that the kernel's tile fits the stack buffer below.
    const { assert!(K::MR * K::NR <= MAX_TILE) };
    let mut tile = [0.0f32; MAX_TILE];
    let mut jc = 0;
    while jc < n {
        let nc = if n - jc < NC { n - jc } else { NC };
        let mut pc = 0;
        while pc < k {
            let kc = if k - pc < KC { k - pc } else { KC };
            pack_b(K::NR, b, pc, jc, kc, nc, bpack);
            let mut ic = 0;
            while ic < m {
                let mc = if m - ic < MC { m - ic } else { MC };
                pack_a(K::MR, a, ic, pc, mc, kc, apack);
                let mut jr = 0;
                while jr < nc {
                    let cols = if nc - jr < K::NR { nc - jr } else { K::NR };
                    let bp = bpack.add((jr / K::NR) * kc * K::NR);
                    let mut ir = 0;
                    while ir < mc {
                        let rows = if mc - ir < K::MR { mc - ir } else { K::MR };
                        let ap = apack.add((ir / K::MR) * kc * K::MR);
                        K::run(kc, ap, bp, tile.as_mut_ptr());
                        write_tile(
                            out,
                            ic + ir,
                            jc + jr,
                            rows,
                            cols,
                            K::NR,
                            tile.as_ptr(),
                            alpha,
                            beta,
                            pc == 0,
                        );
                        ir += K::MR;
                    }
                    jr += K::NR;
                }
                ic += MC;
            }
            pc += KC;
        }
        jc += NC;
    }
}
