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

mod bitmap;
mod builder;
mod holes;
mod transaction;

use crate::{
    bitmap::AtomicSuperblockBitmap,
    holes::{Hole, HoleSearch},
    transaction::AllocTransaction,
};

#[cfg(test)]
use require_unsafe_in_body::require_unsafe_in_bodies;

use std::{
    alloc::{self, Layout},
    mem::MaybeUninit,
    ptr::NonNull,
    sync::atomic::{self, Ordering},
};


// Re-export allocator builder at the crate root
pub use builder::Builder;

// Re-export allocation transaction for other components
pub(crate) use crate::bitmap::SuperblockBitmap;

/// Number of blocks in a superblock
///
/// This is what's publicly exposed as Allocator::BLOCKS_PER_SUPERBLOCK, but
/// it's also internally exposed as a module-level const so that it's shorter
/// and can be brought into scope with "use".
pub(crate) const BLOCKS_PER_SUPERBLOCK: usize =
    std::mem::size_of::<SuperblockBitmap>() * 8;


/// A thread-safe bitmap allocator
#[derive(Debug)]
pub struct Allocator {
    /// Beginning of the backing store from which we'll be allocating memory
    ///
    /// Guaranteed by `Builder` to contain an integer number of superblocks,
    /// each of which maps into one `AtomicUsize` in usage_bitmap.
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
    usage_bitmap: Box<[AtomicSuperblockBitmap]>,

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

// `require_unsafe_in_bodies` is only enabled on development builds because it
// requires procedural macros and that adds a lot to crate build times.
#[cfg_attr(test, require_unsafe_in_bodies)]
impl Allocator {
    /// Start building an allocator
    ///
    /// See the `Builder` documentation for more details on the subsequent
    /// allocator configuration process.
    pub const fn new() -> Builder {
        Builder::new()
    }

    /// Allocator constructor proper, without invariant checking
    ///
    /// This method mostly exists as an implementation detail of `Builder`, and
    /// there is no plan to make it public at the moment since I cannot think of
    /// a single reason to do so. You're not really building Allocators in a
    /// tight loop, are you?
    ///
    /// # Safety
    ///
    /// The block_align, block_size and capacity parameters may be assumed to
    /// uphold all the preconditions listed as "must" bullet points in the
    /// corresponding `Builder` struct members' documentation, either in this
    /// constructor or other methods of Allocator.
    #[cfg_attr(not(test), allow(unused_unsafe))]
    pub(crate) unsafe fn new_unchecked(block_align: usize,
                                       block_size: usize,
                                       capacity: usize) -> Self {
        // Allocate the backing store
        //
        // This is safe because we've checked all preconditions of `Layout`
        // and `alloc()` during the `Builder` construction process, including
        // the fact that capacity is not zero which is the one thing that makes
        // `alloc::alloc()` unsafe.
        let backing_store_layout =
            Layout::from_size_align(capacity, block_align)
                   .expect("All Layout preconditions should have been checked");
        let backing_store_start =
            NonNull::new(unsafe { alloc::alloc(backing_store_layout) })
                    .unwrap_or_else(|| {
                        alloc::handle_alloc_error(backing_store_layout)
                    })
                    .cast::<MaybeUninit<u8>>();

        // Build the usage-tracking bitmap
        let superblock_size = block_size * BLOCKS_PER_SUPERBLOCK;
        let usage_bitmap = std::iter::repeat(SuperblockBitmap::EMPTY)
                                     .map(AtomicSuperblockBitmap::new)
                                     .take(capacity / superblock_size)
                                     .collect::<Box<[_]>>();

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
    pub const BLOCKS_PER_SUPERBLOCK: usize = BLOCKS_PER_SUPERBLOCK;

    /// Block alignment of this allocator (in bytes)
    ///
    /// Every block managed by the allocator is guaranteed to have this
    /// memory alignment "for free".
    pub const fn block_alignment(&self) -> usize {
        self.alignment
    }

    /// Block size of this allocator (in bytes)
    ///
    /// This is the granularity at which the allocator's internal bitmap tracks
    /// which regions of the backing store are used and unused.
    pub const fn block_size(&self) -> usize {
        1 << (self.block_size_shift as u32)
    }

    /// Superblock size of this allocator (in bytes)
    ///
    /// This is the allocation granularity at which this allocator should
    /// exhibit optimal CPU performance.
    pub const fn superblock_size(&self) -> usize {
        self.block_size() * BLOCKS_PER_SUPERBLOCK
    }

    /// Size of this allocator's backing store (in bytes)
    ///
    /// This is the maximal amount of memory that may be allocated from this
    /// allocator, assuming no memory waste due to unused block bytes and no
    /// fragmentation issues.
    pub fn capacity(&self) -> usize {
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
        // Detect and report unrealistic requests in debug builds
        debug_assert!(size < self.capacity(),
                      "Requested size is above allocator capacity");

        // Handle the zero-sized edge case
        //
        // TODO: We'll need to revise this if we ever allow overaligned allocs.
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

        // Convert requested size to a number of requested blocks
        let num_blocks = div_round_up(size, self.block_size());

        // Start looking for suitable free memory "holes" in the usage bitmap
        //
        // TODO: Always starting at the beginning isn't nice because we keep
        //       hammering the same first superblocks while latter ones are
        //       most likely to be free. Consider adding a state variable to
        //       pick up at the superblock which we left off. It's just a
        //       perf hint so atomic loads/stores should be enough.
        //
        //       We could go one step further and store both a superblock
        //       index and a block index within that variable, via some
        //       bit-packing trick : |superblock_idx|block_idx|. But this
        //       puts an implementation limitation on the amount of
        //       superblocks which we can track and in the end we're still
        //       hitting the same superblock, so it may not be worthwhile.
        //
        //       We should update this state as quickly as possible, so that
        //       other threads move away from the hole that we're looking
        //       at, but not too often as otherwise the state variable would
        //       be hammered too hard.
        //
        //       Random index selection isn't a good strategy here as it
        //       will lead to fragmentation, which will make large
        //       allocations fail all the time.
        //
        //       Generally speaking, this shouldn't be investigated before
        //       we have benchmarks, first because the perf tradeoffs are
        //       subtle and second because it will split the bitmap in two
        //       halves that join between the end of the second half and the
        //       beginning of the first half, and the complexity must be
        //       justified by some proven substantial perf benefit.
        let (mut hole_search, hole_opt) = HoleSearch::new(
            num_blocks,
            self.usage_bitmap.iter()
                             .map(|asb| asb.load(Ordering::Relaxed))
        );
        let mut hole = hole_opt?;

        // Try to allocate the current hole, retry on failure.
        let first_block_idx = 'alloc_attempts: loop {
            match hole {
                // All blocks are in a single superblock, no transaction needed
                Hole::SingleSuperblock { superblock_idx,
                                         first_block_subidx } => {
                    // ...so we just compute the mask and try to allocate
                    let mask = SuperblockBitmap::new_mask(first_block_subidx,
                                                          num_blocks as u32);
                    match self.try_alloc_blocks(superblock_idx, mask)
                    {
                        // We managed to allocate this hole
                        Ok(()) => {
                            let first_block_idx =
                                superblock_idx * BLOCKS_PER_SUPERBLOCK
                                    + (first_block_subidx as usize);
                            break 'alloc_attempts first_block_idx;
                        },

                        // We failed to allocate this hole, but we got an
                        // updated view of the bit pattern for this superblock
                        Err(observed_bitmap) => {
                            hole = hole_search.retry(superblock_idx,
                                                     observed_bitmap)?;
                            continue 'alloc_attempts;
                        }
                    }
                },

                // Blocks span more than one superblock, need a transaction
                Hole::MultipleSuperblocks { body_start_idx,
                                            mut num_head_blocks } => {
                    // Given the number of head blocks, we can find all other
                    // parameters of the active transaction.
                    let other_blocks = num_blocks - num_head_blocks as usize;
                    let num_body_superblocks =
                        other_blocks / BLOCKS_PER_SUPERBLOCK;
                    let mut num_tail_blocks =
                        (other_blocks % BLOCKS_PER_SUPERBLOCK) as u32;

                    // Try to allocate the body of the transaction
                    let mut transaction = match AllocTransaction::with_body(
                        self,
                        body_start_idx,
                        num_body_superblocks
                    ) {
                        // On success, bubble up the transaction object
                        Ok(transaction) => transaction,

                        // On failure, send back what went wrong
                        Err((bad_superblock_idx, observed_bitmap)) => {
                            hole = hole_search.retry(bad_superblock_idx,
                                                     observed_bitmap)?;
                            continue 'alloc_attempts;
                        },
                    };

                    // Try to allocate the head of the hole (if any)
                    while num_head_blocks > 0 {
                        if let Err(observed_head_blocks) =
                            transaction.try_alloc_head(num_head_blocks)
                        {
                            // On head allocation failure, try to "move the hole
                            // forward", pushing more blocks to the tail.
                            num_tail_blocks +=
                                num_head_blocks - observed_head_blocks;
                            num_head_blocks = observed_head_blocks;
                        }
                    }

                    // If needed, allocate one more body superblock.
                    // This can happen as a result of moving the hole forward:
                    //     |0011|1111|1110|0000| -> |0000|1111|1111|1000|
                    if num_tail_blocks >= BLOCKS_PER_SUPERBLOCK as u32 {
                        if let Err(observed_bitmap) =
                            transaction.try_extend_body()
                        {
                            hole = hole_search.retry(transaction.body_end_idx(),
                                                     observed_bitmap)?;
                            continue 'alloc_attempts;
                        } else {
                            num_tail_blocks -= BLOCKS_PER_SUPERBLOCK as u32;
                        }
                    }

                    // Try to allocate the tail of the hole (if any)
                    if num_tail_blocks > 0 {
                        if let Err(observed_bitmap) =
                            transaction.try_alloc_tail(num_tail_blocks)
                        {
                            hole = hole_search.retry(transaction.body_end_idx(),
                                                     observed_bitmap)?;
                            continue 'alloc_attempts;
                        }
                    }

                    // We managed to allocate everything! Isn't that right?
                    debug_assert_eq!(transaction.num_blocks(), num_blocks,
                                     "Allocated an incorrect number of blocks");

                    // Commit the transaction and get the first block index
                    let first_block_idx = transaction.commit();
                    break 'alloc_attempts first_block_idx;
                },
            }
        };

        // Make sure that the previous reads from the allocation bitmap are
        // ordered before any subsequent access to the buffer by the current
        // thread, to avoid data races with the thread that deallocated the
        // memory that we are in the process of allocating.
        atomic::fence(Ordering::Acquire);

        // Translate our allocation's first block index into an actual memory
        // address within the allocator's backing store. This is safe because...
        //
        // - The pointer has to be in bounds, since we got the block coordinates
        //   from the usage bitmap and by construction, there are no
        //   out-of-bounds blocks in the bitmap (remember that we rounded the
        //   requested allocation size to a multiple of the superblock size).
        // - By construction, backing store capacity cannot be above
        //   `isize::MAX`, so in-bounds offsets shouldn't go above that limit.
        // - Since we're targeting an allocation that we got from the system
        //   allocator, the address computation shouldn't overflow usize and
        //   wrap around.
        let target_start = unsafe {
            self.backing_store_start
                .as_ptr()
                .add(first_block_idx * self.block_size())
        };

        // Add requested length (_not_ actual allocation length) to turn this
        // start-of-allocation pointer into an allocated slice. This is safe
        // to do because...
        //
        // - If our allocation algorithm is correct, no one else currently holds
        //   a slice to this particular subset of the backing store.
        // - If our allocation algorithm is correct, "size" bytes are in bounds
        // - Lifetimes don't matter as we'll just turn this into a pointer
        // - The backing store pointer cannot be null because the constructor
        //   aborts if allocation fails by returning a null pointer.
        // - There is no alignment problem as we're building a NonNull<u8>
        // - "size" cannot overflow isize because the backing store capacity is
        //   not allowed to do so.
        let target_slice = unsafe {
            std::slice::from_raw_parts_mut(target_start, size)
        };

        // Finally, we can build and return the output pointer
        NonNull::new(target_slice as *mut _)
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
    #[cfg_attr(not(test), allow(unused_unsafe))]
    pub unsafe fn dealloc_unbound(&self, ptr: NonNull<[MaybeUninit<u8>]>) {
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

        // Check pointer length as well. This is safe because we requested a
        // valid pointer as part of this function's safety preconditions.
        let ptr_len = unsafe { ptr.as_ref() }.len();
        debug_assert!(ptr_len < self.capacity() - ptr_offset,
                      "Deallocated ptr overflows backing store");
        // NOTE: ptr_len may not be a multiple of the block size because we
        //       allow users to under-allocate blocks

        // Do not do anything beyond that for zero-sized allocations, by
        // definition they have no associated storage block to be freed
        if ptr_len == 0 { return; }

        // Make sure that the subsequent writes to the allocation bitmap are
        // ordered after any previous access to the buffer by the current
        // thread, to avoid data races with other threads concurrently
        // reallocating and filling the memory that we are liberating.
        atomic::fence(Ordering::Release);

        // Switch to block coordinates as that's what our bitmap speaks
        let mut block_idx = ptr_offset / self.block_size();
        let end_block_idx = block_idx + div_round_up(ptr_len,
                                                     self.block_size());

        // Does our first block fall in the middle of a superblock?
        let local_start_idx = (block_idx % BLOCKS_PER_SUPERBLOCK) as u32;
        if local_start_idx != 0 {
            // Compute index of that superblock
            let superblock_idx = block_idx / BLOCKS_PER_SUPERBLOCK;

            // Compute how many blocks are allocated within the superblock,
            // bearing in mind that the buffer may end there
            let local_len =
                (BLOCKS_PER_SUPERBLOCK - local_start_idx as usize)
                    .min(end_block_idx - block_idx) as u32;

            // Deallocate leading buffer blocks in this first superblock
            self.dealloc_blocks(
                superblock_idx,
                SuperblockBitmap::new_mask(local_start_idx, local_len)
            );

            // Advance block pointer, stop if all blocks were liberated
            block_idx += local_len as usize;
            if block_idx == end_block_idx { return; }
        }

        // If control reached this point, block_idx is now at the start of a
        // superblock, so we can switch to faster superblock-wise deallocation.
        // Deallocate all superblocks until the one where end_block_idx resides.
        let start_superblock_idx = block_idx / BLOCKS_PER_SUPERBLOCK;
        let end_superblock_idx = end_block_idx / BLOCKS_PER_SUPERBLOCK;
        for superblock_idx in start_superblock_idx..end_superblock_idx {
            self.dealloc_superblock(superblock_idx);
        }

        // Advance block pointer, stop if all blocks were liberated
        block_idx = end_superblock_idx * BLOCKS_PER_SUPERBLOCK;
        if block_idx == end_block_idx { return; }

        // Deallocate trailing buffer blocks in the last superblock
        let remaining_len = (end_block_idx - block_idx) as u32;
        self.dealloc_blocks(end_superblock_idx,
                            SuperblockBitmap::new_tail_mask(remaining_len));
    }

    /// Try to atomically allocate a full superblock
    ///
    /// Returns observed superblock allocation bitfield on failure.
    ///
    /// This operation has `Relaxed` memory ordering and must be followed by an
    /// `Acquire` memory barrier in order to avoid allocation being reordered
    /// after usage of the memory block by the compiler or CPU.
    pub(crate) fn try_alloc_superblock(
        &self,
        superblock_idx: usize
    ) -> Result<(), SuperblockBitmap> {
        debug_assert!(superblock_idx < self.usage_bitmap.len(),
                      "Superblock index is out of bitmap range");
        self.usage_bitmap[superblock_idx]
            .try_alloc_all(Ordering::Relaxed, Ordering::Relaxed)
    }

    /// Try to atomically allocate a subset of the blocks within a superblock
    ///
    /// Returns observed superblock allocation bitfield on failure.
    ///
    /// This operation has `Relaxed` memory ordering and must be followed by an
    /// `Acquire` memory barrier in order to avoid allocation being reordered
    /// after usage of the memory block by the compiler or CPU.
    pub(crate) fn try_alloc_blocks(
        &self,
        superblock_idx: usize,
        mask: SuperblockBitmap
    ) -> Result<(), SuperblockBitmap> {
        debug_assert!(superblock_idx < self.usage_bitmap.len(),
                      "Superblock index is out of bitmap range");
        self.usage_bitmap[superblock_idx].try_alloc_mask(mask,
                                                         Ordering::Relaxed,
                                                         Ordering::Relaxed)
    }

    /// Atomically deallocate a full superblock
    ///
    /// This operation has `Relaxed` memory ordering and must be preceded by a
    /// `Release` memory barrier in order to avoid deallocation being reordered
    /// before usage of the memory block by the compiler or CPU.
    pub(crate) fn dealloc_superblock(&self, superblock_idx: usize) {
        debug_assert!(superblock_idx < self.usage_bitmap.len(),
                      "Superblock index is out of bitmap range");
        self.usage_bitmap[superblock_idx].dealloc_all(Ordering::Relaxed);
    }

    /// Atomically deallocate a subset of the blocks within a superblock
    ///
    /// This operation has `Relaxed` memory ordering and must be preceded by a
    /// `Release` memory barrier in order to avoid deallocation being reordered
    /// before usage of the memory block by the compiler or CPU.
    pub(crate) fn dealloc_blocks(&self,
                                 superblock_idx: usize,
                                 mask: SuperblockBitmap) {
        debug_assert!(superblock_idx < self.usage_bitmap.len(),
                      "Superblock index is out of bitmap range");
        self.usage_bitmap[superblock_idx].dealloc_mask(mask, Ordering::Relaxed);
    }
}

impl Drop for Allocator {
    fn drop(&mut self) {
        // Make sure that no storage blocks were allocated, as the corresponding
        // pointers will become dangling when the allocator is dropped.
        debug_assert!(
            self.usage_bitmap
                .iter_mut()
                .all(|bits| *bits.get_mut() == SuperblockBitmap::EMPTY),
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
//       Builder can be made const fn, but people might still find it useful for
//       certain use cases...
//
//       To allow this, I must not use the names alloc(), dealloc(),
//       alloc_zeroed() and realloc() in the inherente Allocator API.
//
//       GlobalAlloc methods must not unwind. Since pretty much everything can
//       panic in Rust and I love assertions, I'll probably want to use
//       catch_unwind, then return a null pointer if possible and call
//       alloc::handle_alloc_error myself otherwise.
//
//       Obviously, GlobalAlloc layout expectations must also be upheld,
//       including alignment. Until I support overaligned allocations (if ever),
//       this will entail erroring out when requested alignment is higher than
//       global block alignment.
//
//       If I do this, I should mention it in the crate documentation, along
//       with the fact that it's only suitable for specific use cases (due to
//       limited capacity, and possibly no overalignment ability)


/// Small utility to divide two integers, rounding the result up
fn div_round_up(x: usize, y: usize) -> usize {
    // Check interface preconditions in debug builds
    debug_assert!(y != 0, "Attempted to divide by zero");

    // Return rounded division result
    (x / y) + (x % y != 0) as usize
}


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
//
// TODO: Look at assembly and make sure that power-of-two integer manipulation
//       are properly optimized, force inlining of Allocator size queries and
//       div_round_up if necessary.
