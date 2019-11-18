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

pub mod builder;

use std::{
    alloc::{self, Layout},
    mem::{self, MaybeUninit},
    ptr::NonNull,
    sync::atomic::{self, AtomicUsize, Ordering},
};


/// A thread-safe bitmap allocator
#[derive(Debug)]
pub struct Allocator {
    /// Beginning of the backing store from which we'll be allocating memory
    ///
    /// Guaranteed by `AllocatorBuilder` to contain an integer number of
    /// superblocks, each of which maps into one `AtomicUsize` in usage_bitmap.
    backing_store_start: NonNull<MaybeUninit<u8>>,

    /// Bitmap tracking usage of the backing store's storage blocks
    ///
    /// The backing store is divided into blocks of `Self::block_size()` bytes,
    /// and each bit of this bitmap tracks whether a block is allocated or free,
    /// in backing store order.
    ///
    /// Because modern CPUs are not bit-addressable, we must manipulate our bits
    /// in bunches, via unsigned integers. This leads to the creation of a new
    /// artificial storage granularity, tracked by a full integer-sized bunch of
    /// bits in the bitmap, which we call a superblock.
    ///
    /// We use usize as our unsigned integer type because that's the one which
    /// is most likely to have native atomic operations on a given CPU arch. But
    /// this may lead to very big superblocks on future CPU archs, so we might
    /// need to reconsider this decision in the future and use e.g. `AtomicU32`.
    usage_bitmap: Box<[AtomicUsize]>,

    /// Bitshift-based representation of the block size
    ///
    /// This odd value storage is not a space optimization, but a way to tell
    /// the compiler's optimizer that the block size has to be a power of two so
    /// that it optimizes our integer divisions and remainders. Please use
    /// methods like block_size() to query the block size.
    block_size_shift: u8,

    /// Backing store alignment (and thus storage block alignment)
    ///
    /// We must keep track of this because Rust's allocator API will expect us
    /// to give back this information upon deallocating the backing store.
    alignment: usize,
}

impl Allocator {
    /// Start building an allocator
    ///
    /// See the `AllocatorBuilder` documentation for more details on the
    /// subsequent allocator configuration process.
    pub const fn new() -> builder::AllocatorBuilder {
        builder::AllocatorBuilder::new()
    }

    /// Allocator constructor proper, without invariant checking
    ///
    /// This method mostly exists as an implementation detail of
    /// `AllocatorBuilder`, and there is no plan to make it public at the moment
    /// since I cannot think of a single reason to do so. You're not really
    /// building Allocators in a tight loop, are you?
    ///
    /// # Safety
    ///
    /// The block_align, block_size and capacity parameters may be assumed to
    /// uphold all the preconditions listed as "must" bullet points in the
    /// corresponding `AllocatorBuilder` struct members' documentation.
    pub(crate) unsafe fn new_unchecked(block_align: usize,
                                       block_size: usize,
                                       capacity: usize) -> Self {
        // Allocate the backing store
        //
        // This is safe because we've checked all preconditions of `Layout`
        // and `alloc()` during the `AllocatorBuilder` construction process,
        // including the fact that capacity is not zero.
        let backing_store_layout =
            Layout::from_size_align(capacity, block_align)
                   .expect("All Layout preconditions should have been checked");
        let backing_store_start =
            NonNull::new(alloc::alloc(backing_store_layout))
                    .unwrap_or_else(|| alloc::handle_alloc_error(backing_store_layout))
                    .cast::<MaybeUninit<u8>>();

        // Build the usage-tracking bitmap
        let superblock_size = block_size * Self::blocks_per_superblock();
        let usage_bitmap = std::iter::repeat(0)
                                     .map(AtomicUsize::new)
                                     .take(capacity / superblock_size)
                                     .collect::<Vec<_>>()
                                     .into_boxed_slice();

        // Build and return the allocator struct
        Allocator {
            backing_store_start,
            usage_bitmap,
            block_size_shift: block_size.trailing_zeros() as u8,
            alignment: block_align,
        }
    }

    /// Implementation-defined number of storage blocks per superblock
    ///
    /// This is the multiplicative factor between a bitmap allocator's block
    /// size and its superblock size. That quantity is machine-dependent and
    /// subjected to change in future versions of this crate, so please always
    /// call this function instead of relying on past results from it.
    //
    // NOTE: This function must be inlined as it's super-important that the
    //       compiler knows that its output is a power of 2 for fast div/rem.
    #[inline(always)]
    pub const fn blocks_per_superblock() -> usize {
        mem::size_of::<usize>() * 8
    }

    /// Block size of this allocator (in bytes)
    ///
    /// This is the granularity at which the allocator's internal bitmap tracks
    /// which regions of the backing store are used and unused.
    //
    // NOTE: This function must be inlined as it's super-important that the
    //       compiler knows that its output is a power of 2 for fast div/rem.
    #[inline(always)]
    const fn block_size(&self) -> usize {
        1 << (self.block_size_shift as usize)
    }

    /// Superblock size of this allocator (in bytes)
    ///
    /// This is the allocation granularity at which this allocator should
    /// exhibit optimal CPU performance.
    //
    // NOTE: This function must be inlined as it's super-important that the
    //       compiler knows that its output is a power of 2 for fast div/rem.
    #[inline(always)]
    const fn superblock_size(&self) -> usize {
        self.block_size() * Self::blocks_per_superblock()
    }

    /// Size of this allocator's backing store (in bytes)
    ///
    /// This is the maximal amount of memory that may be allocated from this
    /// allocator, assuming no memory waste due to unused block bytes and no
    /// fragmentation issues.
    fn capacity(&self) -> usize {
        self.usage_bitmap.len() * self.superblock_size()
    }

    /// Safely allocate a memory buffer, with auto-liberation
    ///
    /// This is a safe interface on top of the raw memory allocation facility
    /// provided by `alloc_unbound` and `dealloc_unbound`. It leverages the Rust
    /// borrow checker to ensure at compile time that no allocated buffer will
    /// outlive its Allocator parent.
    ///
    /// The price to pay for this higher-level interface is that allocated
    /// buffer objects will be bigger, and that it's harder to store both an
    /// `Allocator` and some buffers allocated from it within a single struct
    /// because that would make it a self-referential struct.
    //
    // FIXME: Add a Box-ish abstraction that auto-deallocates and auto-derefs,
    //        made safe by the use of a lifetime-based API.
    //
    // TODO: As alloc_unbound matures, adapt alloc_bound accordingly.
    pub fn alloc_bound<'s>(&'s self, _size: usize) -> Option<&'s mut [MaybeUninit<u8>]> {
        unimplemented!()
    }

    /// Allocate a memory buffer, without borrow checking
    ///
    /// Returns a pointer to the allocated slice if successful, or `None` if the
    /// allocator is not able to satisfy this request because not enough
    /// contiguous storage is available in its backing store.
    ///
    /// This function is not unsafe per se, in the sense that no undefined
    /// behavior can occur as a direct result of calling it. However, you should
    /// really make sure that the output buffer pointer is passed back to
    /// `Allocator::dealloc_unbound()` before the allocator is dropped.
    ///
    /// Otherwise, the pointer will be invalidated, and dereferencing it after
    /// that _will_ unleash undefined behavior.
    //
    // TODO: Prepare support for overaligned allocations and `GlobalAlloc` impl
    //       by accepting `std::alloc::Layout`. Initially, we can just return
    //       None when the requested alignment is higher than self.alignment.
    pub fn alloc_unbound(&self, size: usize) -> Option<NonNull<[MaybeUninit<u8>]>> {
        // Handle the zero-sized edge case
        if size == 0 {
            return Some(
                // This is safe because...
                // - The backing store pointer is obviously valid for 0 elements
                // - It has the minimal alignment we promise to always offer
                // - Lifetimes don't matter as we're building a raw pointer
                // - We won't overflow isize with a zero-length slice
                // - &mut aliasing is not an issue for zero-sized slices.
                unsafe { std::slice::from_raw_parts_mut(
                    self.backing_store_start.as_ptr(),
                    0
                ) }.into()
            );
        }

        // TODO: Now handle serious allocations
        unimplemented!()
    }

    // TODO: Add a realloc_unbound() API, and a matching realloc() method to the
    //       higher-level buffer object returned by alloc_bound()

    /// Deallocate a buffer that was previously allocated via `alloc_unbound`
    ///
    /// # Safety
    ///
    /// `ptr` must denote a block of memory currently allocated via this
    /// allocator, i.e. it must have been generated by `alloc_unbound` without
    /// further tampering and it should not have already been deallocated.
    ///
    /// `ptr` will be dangling after calling this function, and should neither
    /// be dereferenced nor passed to `dealloc_unbound` again.
    pub unsafe fn dealloc_unbound(&self, ptr: NonNull<[MaybeUninit<u8>]>) {
        // TODO: Try to deduplicate this code.

        // In debug builds, check that the input pointer does come from our
        // backing store, with all the properties that one would expect.
        let ptr_start = ptr.cast::<MaybeUninit<u8>>().as_ptr();
        let store_start = self.backing_store_start.as_ptr();
        debug_assert!(ptr_start >= store_start,
                      "Deallocated ptr starts before backing store");
        let ptr_offset = (ptr_start as usize) - (store_start as usize);
        debug_assert!(ptr_offset < self.capacity(),
                      "Deallocated ptr starts after backing store");
        debug_assert_eq!(ptr_offset % self.block_size(), 0,
                         "Deallocated ptr doesn't start on a block boundary");
        let ptr_len = ptr.as_ref().len();
        debug_assert!(ptr_len < self.capacity() - ptr_offset,
                      "Deallocated ptr overflows backing store");
        debug_assert_eq!(ptr_len % self.block_size(), 0,
                         "Deallocated ptr doesn't stop on a block boundary");

        // Do not do anything beyond that for zero-sized allocations, by
        // definition they have no associated storage block to be freed
        if ptr_len == 0 { return; }

        // Make sure that the subsequent writes to the allocation bitmap are
        // ordered after any previous access to the buffer by the current
        // thread, to avoid data races with other threads concurrently
        // reallocating and filling the memory that we are liberating.
        atomic::fence(Ordering::Release);

        // Switch to block coordinates as that's what our bitmap speaks
        let mut start_block_idx = ptr_offset / self.block_size();
        let end_block_idx = start_block_idx + (ptr_len / self.block_size());

        // Does the buffer starts in the middle of a superblock?
        let local_start_idx = start_block_idx % Self::blocks_per_superblock();
        if local_start_idx != 0 {
            // If so, determine how many buffer blocks fall into that first
            // superblock, bearing in mind it may not be used through the end...
            let num_head_blocks =
                (Self::blocks_per_superblock() - local_start_idx)
                    .min(end_block_idx - start_block_idx);

            // ...compute the corresponding bit pattern to be zeroed out...
            let allocation_mask =
                ((1 << num_head_blocks) - 1) << local_start_idx;

            // ...and clear those bits. As a reminder, required memory ordering
            // on bitmap operations is enforced by the Release fence above.
            let head_superblock_idx =
                start_block_idx / Self::blocks_per_superblock();
            let old_bits = self.usage_bitmap[head_superblock_idx]
                               .fetch_and(!allocation_mask, Ordering::Relaxed);

            // In debug builds, make sure that the corresponding blocks were
            // indeed marked as allocated.
            debug_assert_eq!(
                old_bits & allocation_mask, allocation_mask,
                "Deallocated a head block which was marked as free"
            );

            // ...and now we can move forward in the buffer deallocation process
            start_block_idx += num_head_blocks;

            // If we're done, we should stop now, as the subsequent logic cannot
            // handle the case of stopping in the middle of the head superblock.
            if start_block_idx == end_block_idx { return; }
        }

        // If control reached this point, start_block_idx is now at the
        // beginning of a superblock. We can thus switch to superblock-wise
        // deallocation, which is both simpler and much faster, until we reach
        // the end of the last fully allocated superblock.
        let first_superblock_idx =
            start_block_idx / Self::blocks_per_superblock();
        let tail_superblock_idx =
            end_block_idx / Self::blocks_per_superblock();
        for superblock in &self.usage_bitmap[first_superblock_idx..tail_superblock_idx] {
            // The general idea is to just reset every full superblock to 0,
            // which means no allocated data. But in debug builds, we also
            // check that they were set to an all-ones bit pattern before, as
            // expected, otherwise some double free or corruption occured.
            if cfg!(debug_assertions) {
                debug_assert_eq!(
                    superblock.swap(0, Ordering::Relaxed), std::usize::MAX,
                    "Deallocated a superblock which was marked partially free"
                );
            } else {
                superblock.store(0, Ordering::Relaxed);
            }
        }

        // Are we done yet?
        start_block_idx = tail_superblock_idx * Self::blocks_per_superblock();
        if start_block_idx == end_block_idx { return; }

        // Otherwise, our buffer ends in the middle of a superblock
        let num_tail_blocks = end_block_idx - start_block_idx;

        // Compute the bit pattern to be zeroed out in that superblock...
        let allocation_mask = (1 << num_tail_blocks) - 1;

        // ...and clear those bits.
        let old_bits = self.usage_bitmap[tail_superblock_idx]
                           .fetch_and(!allocation_mask, Ordering::Relaxed);

        // In debug builds, make sure that the corresponding blocks were
        // indeed marked as allocated, as we did before.
        debug_assert_eq!(old_bits & allocation_mask, allocation_mask,
                         "Deallocated a tail block which was marked as free");
    }
}

impl Drop for Allocator {
    fn drop(&mut self) {
        // Make sure that no storage blocks were allocated, as the corresponding
        // pointers will become dangling when the allocator is dropped.
        //
        // We don't need any particular memory ordering constraint because our
        // &mut self allows us to assume exclusive access to the `Allocator`
        // management structures, without any fear that other threads might be
        // concurrently calling allocation and deallocation methods.
        debug_assert!(
            self.usage_bitmap.iter()
                             .all(|bits| bits.load(Ordering::Relaxed) == 0),
            "Allocator was dropped while there were still live allocations"
        );

        // Deallocate the backing store. This is safe because...
        // - An allocator is always created with a backing store allocation
        // - Only Drop, which happens at most once, can liberate that allocation
        // - The layout matches that used in `Allocator::new_unchecked()`
        let backing_store_layout =
            Layout::from_size_align(self.capacity(), self.alignment)
                   .expect("All Layout preconditions were checked by builder");
        unsafe { alloc::dealloc(self.backing_store_start.cast::<u8>().as_ptr(),
                                backing_store_layout); }
    }
}

// TODO: Implement GlobalAlloc trait? Will require lazy_static/OnceCell until
//       AllocatorBuilder can be made const fn, but people might still find it
//       useful for certain use cases...
//
//       To allow this, do not use alloc(), dealloc(), alloc_zeroed() and
//       realloc() in our main API, or if we do make them look like GlobalAlloc.


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
