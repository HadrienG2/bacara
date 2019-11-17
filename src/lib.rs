//! This crate provides a Concurrent Bitmap Allocator, which you can use for
//! your dynamic memory allocation needs in those real-time threads where the
//! system memory allocator should not be used.
//!
//! The implementation is both thread-safe and real-time-friendly: after its
//! RT-unsafe initialization stage, you can use this allocator to allocate and
//! liberate memory from multiple RT threads, and none of them will acquire a
//! lock or call into any RT-unsafe operating system facility in the process.
//!
//! # Bitmap allocation primer
//!
//! A bitmap allocator is a general-purpose memory allocator: it allows
//! allocating variable-sized buffers from its backing store, and later on
//! deallocating them individually and in any order.
//!
//! This allocation algorithm works by dividing the buffer of memory that it is
//! managing (which we'll call **backing store**) into evenly sized **blocks**,
//! and tracking which blocks are in use using an array of bits, a **bitmap**.
//!
//! Allocation is done by scanning the bitmap for a suitably large hole
//! (continuous sequence of zeroes), filling that hole with ones, and mapping 
//! the hole's index in the bitmap into a pointer within the backing store.
//! Deallocation is done by mapping back from the user-provided pointer to a
//! range of indices within the bitmap and resetting those bits to zero.
//!
//! The **block size** is the most important tuning parameter of a bitmap
//! allocator, and should be chosen wisely:
//!
//! - Because the allocation overhead and bitmap size are proportional to the
//!   number of blocks managed by the allocator, the CPU and memory overhead of
//!   a bitmap allocator will grow as its block size shrinks. From this
//!   perspective, using the highest block size you can get away with is best.
//! - But since allocations are tracked with block granularity, higher block
//!   sizes mean less efficient use of the backing store, as the allocator is
//!   more likely to allocate more memory than the client needs.
//!
//! Furthermore, pratical implementations of bitmap allocation on modern
//! non-bit-addressable hardware will reach their peak CPU efficiency when
//! processing allocation requests whose size is an implementation-defined
//! multiple of the block size, which we will refer to as a **superblock**.
//! Depending on your requirements, you may want to tune superblock size rather
//! than block size, which is why our API will allow you to do both.
//!
//! You should tune your (super)block size based on the full range of envisioned
//! allocation workloads, and even consider instantiating multiple allocators
//! with different block sizes if your allocation patterns vary widely, because
//! a block size that is a good compromise for a given allocation pattern may be
//! a less ideal choice for another allocation pattern.
//!
//! # Example
//!
//! FIXME: Oh yes I do need those, but API must be done first ;)

use std::{
    mem::{self, MaybeUninit},
    ptr::NonNull,
    sync::atomic::AtomicUsize,
};



pub struct Allocator {
    backing_store: NonNull<[MaybeUninit<u8>]>,

    usage_bitmap: Box<[AtomicUsize]>,

    // Provides a way to clearly tell the compiler that bs is a power of 2 so
    // that it optimizes our integer divides and modulos. Use block_size()
    // and friends in actual code.
    block_size_shift: usize,
}

impl Allocator {
    // Computed properties
    //
    // These three fns must be inlined as it's super-important that the compiler
    // realizes that it can use a power-of-2 fast path for divisions and modulos

    #[inline(always)]
    const fn block_size(&self) -> usize {
        1 << self.block_size_shift
    }

    #[inline(always)]
    const fn blocks_per_superblock() -> usize {
        mem::size_of::<usize>() * 8
    }

    #[inline(always)]
    const fn superblock_size(&self) -> usize {
        self.block_size() * Self::blocks_per_superblock()
    }

    // TODO: Design some nicer API that can work with either block size or
    //       superblock size and does not have the "many usize params" smell,
    //       probably some kind of AllocatorBuilder
    //
    // All parameters are in bytes, with the following constraints:
    // - align is a global alignment that all allocations uphold, must be
    //   nonzero and a power of 2 (std::alloc::Layout constraint)
    // - block_size must be a multiple of alignment (so that all blocks, and
    //   thus all allocations, are aligned) and a power of 2 (for fast divide)
    // - capacity must not be zero (for system allocator) and will be rounded up
    //   to next multiple of superblock size (to avoid complicating alg with
    //   weird bitmap end), and that must not overflow or else we'll error out.
    pub fn new(_align: usize, _block_size: usize, _capacity: usize) -> Self {
        unimplemented!()
    }

    // TODO: Add some Box-ish abstraction that auto-deallocates and auto-derefs
    //       as it's safe in such a lifetime-based API
    // TODO: Basically mirror all questions from alloc_unbound
    // NOTE: Could actually be based on alloc_unbound and just wrap its
    //       Box-ish output into a lifetime-safety layer
    pub fn alloc_bound<'s>(&'s self, _size: usize) -> Option<&'s mut [MaybeUninit<u8>]> {
        unimplemented!()
    }

    // TODO: Add some Box-ish abstraction that auto-deallocates, but do not make
    //       it auto-deref as it's an unsafe operation (no check that pointer
    //       outlives backing store of allocator)
    // TODO: Clarify safety contract of output pointer
    // TODO: Support overaligned allocations? Accept std::alloc::Layout?
    // NOTE: Should not call it alloc to leave API headroom for GlobalAlloc impl
    pub fn alloc_unbound(&self, _size: usize) -> Option<NonNull<[MaybeUninit<u8>]>> {
        unimplemented!()
    }

    // TODO: Add realloc_bound and realloc_unbound APIs.

    // TODO: Should not be called by the user but by RAII thingies, and
    //       therefore should remain a private API, see above
    // NOTE: Unlike system allocator, we don't need full layout, only size
    // NOTE: Should not be called dealloc, to leave API headroom for
    //       implementing GlobalAlloc in the future
    // NOTE: Unsafe because pointer must come from this allocator...
    unsafe fn free(_ptr: NonNull<[MaybeUninit<u8>]>) {
        unimplemented!()
    }
}

impl Drop for Allocator {
    // TODO: Deallocate backing store, check bitmap = 0 in debug builds
    fn drop(&mut self) {
        unimplemented!()
    }
}

// TODO: Implement GlobalAlloc trait? Not of immediate use until new can be made
//       const fn, but may want to keep the door open.


#[cfg(test)]
mod tests {
    #[test]
    fn it_works() {
        assert_eq!(2 + 2, 4);
    }

    // TODO: Tests tests test absolutely everything as this code is going to be
    //       super-extremely tricky
}

// TODO: Benchmark at various block sizes and show a graph on README
