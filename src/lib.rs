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
mod transaction;

use crate::{
    bitmap::AtomicSuperblockBitmap,
    transaction::AllocTransaction,
};

use std::{
    alloc::{self, Layout},
    mem::{self, MaybeUninit},
    ptr::NonNull,
    sync::atomic::{self, Ordering},
};


// Re-export allocator builder at the crate root
pub use builder::Builder;

// Re-export allocation transaction for other components
pub(crate) use crate::bitmap::SuperblockBitmap;


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
    pub(crate) unsafe fn new_unchecked(block_align: usize,
                                       block_size: usize,
                                       capacity: usize) -> Self {
        // Allocate the backing store
        //
        // This is safe because we've checked all preconditions of `Layout`
        // and `alloc()` during the `Builder` construction process, including
        // the fact that capacity is not zero.
        let backing_store_layout =
            Layout::from_size_align(capacity, block_align)
                   .expect("All Layout preconditions should have been checked");
        let backing_store_start =
            NonNull::new(alloc::alloc(backing_store_layout))
                    .unwrap_or_else(|| alloc::handle_alloc_error(backing_store_layout))
                    .cast::<MaybeUninit<u8>>();

        // Build the usage-tracking bitmap
        let superblock_size = block_size * Self::blocks_per_superblock();
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
    pub const fn blocks_per_superblock() -> usize {
        mem::size_of::<usize>() * 8
    }

    /// Block size of this allocator (in bytes)
    ///
    /// This is the granularity at which the allocator's internal bitmap tracks
    /// which regions of the backing store are used and unused.
    pub const fn block_size(&self) -> usize {
        1 << (self.block_size_shift as usize)
    }

    /// Superblock size of this allocator (in bytes)
    ///
    /// This is the allocation granularity at which this allocator should
    /// exhibit optimal CPU performance.
    pub const fn superblock_size(&self) -> usize {
        self.block_size() * Self::blocks_per_superblock()
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
        // TODO: Decide if this code is needed by studying how well the general
        //       algorithm would handle this edge case
        //
        // TODO: In general, go through the code once finished and try to
        //       simplify and deduplicate it as done for `dealloc_unbound()`.
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

        // Determine how many complete superblocks must be allocated at minimum.
        // If we denote N = Self::blocks_per_superblock() and use "SB" as a
        // shorthand for superblock, the following schematics of possible
        // allocation patterns should help you see what's going on:
        //
        // From 0 to N-1, 1 partial SB is OK: |0000| -> |1110|
        // From N to 2*N-2, 2 partial SBs are OK: |0111|1000| -> |0111|1110|
        // At 2*N-1 blocks, need 1 full SB and N-1 other blocks: |1111|1110|
        // Every N blocks after that, need one more full SB: |1111|1111|1110|
        //
        let num_superblocks =
            num_blocks.saturating_sub(Self::blocks_per_superblock() - 1)
                      / Self::blocks_per_superblock();
        let remaining_blocks =
            num_blocks - num_superblocks * Self::blocks_per_superblock();

        // TODO: May want to add a special path for the num_blocks == 0 case
        //       since the logic can be quite different (e.g. head/tail blocks
        //       may not be necessary)

        // Look for that number of completely free supeblocks
        //
        // TODO: Use a state variable to avoid always scanning through the
        //       superblock list from 0, instead picking up where the previous
        //       thread left off. That variable does not need to be consistent,
        //       it's just an optimization, so loads and stores are enough.
        //
        //       We could go one step further and store both a superblock index
        //       and a block index withing that variable, via some bit-packing
        //       trick : |superblock_idx|block_idx|.
        //
        //       Make sure not to update this variable early on durign search,
        //       without going O(N), to avoid increasing the odd of an
        //       allocation race between two threads by accidentally hinting
        //       other threads to search where we're currently allocating. An
        //       alternate way to avoid this would be to start search at a
        //       random point in the usage bitmap, but that will have bad
        //       fragmentation behavior.
        //
        //       Bear in mind that this will require some modulo arith instead
        //       of plain superblock index addition and subtraction. Also, the
        //       range of superblocks that we're looking for may straddle the
        //       boundary between the first superblock we'll be looking at and
        //       the last one.
        //
        //       This should wait on validity-checking tests, and on benchmarks
        //       to study the impact and see if the complexity is worth it.
        //
        // This variable tracks our knowledge of the first superblock which
        // _might_ be the beginning of a continuous free superblock sequence,
        // bearing in mind that we haven't checked the current superblock.
        let mut first_superblock_idx = 0;
        'sb_search: for (superblock_idx, superblock) in
            self.usage_bitmap.iter().enumerate()
        {
            // Have we found the right amount of free superblocks already?
            // (can succeed on first iteration if none are needed)
            if superblock_idx - first_superblock_idx == num_superblocks {
                // Try to allocate these superblocks as our allocation's body
                let mut transaction = match AllocTransaction::with_body(
                    self,
                    first_superblock_idx,
                    num_superblocks
                ) {
                    // On success, bubble up transaction object
                    Ok(transaction) => transaction,

                    // On failure, use what we learned about the superblock
                    // landscape to guide the remainder of our search.
                    Err(bad_superblock_idx) => {
                        first_superblock_idx = bad_superblock_idx + 1;
                        continue 'sb_search;
                    }
                };

                // Alright, we have allocated body superblocks, now let's try to
                // allocate head and tail blocks.
                //
                // We'll start by trying to allocate as much as possible on the
                // head block's side (that is necessary to guarantee a compact
                // allocation layout in a classing mostly-FIFO
                // allocation/liberation scenario), then gradually revise that
                // expectation towards zero head blocks as our failures tell us.
                //
                let (mut num_head_blocks, min_head_blocks) =
                    // Do we need to have a tail?
                    if remaining_blocks >= Self::blocks_per_superblock() {
                        // If so, we start to push as much as possible on the
                        // head side, and stop when even a fully allocated tail
                        // superblock wouldn't be enough.
                        (Self::blocks_per_superblock() - 1,
                         remaining_blocks - Self::blocks_per_superblock())
                    } else {
                        // If not, we start by pushing everything on the head
                        // side, and stop when we have moved everything to the
                        // tail side.
                        (remaining_blocks, 0)
                    };
                'head_search: while num_head_blocks >= min_head_blocks {
                    // Try to allocate the desired number of head blocks
                    match transaction.try_alloc_head(num_head_blocks) {
                        // We did it, now let's try to allocate the remaining
                        // blocks as tail blocks
                        Ok(_) => unimplemented!(),

                        // We didn't manage, but now we know how many head
                        // blocks we actually have. Check if that's enough
                        Err(actual_head_blocks) => unimplemented!(),
                    }
                }

                // TODO: Try to allocate neighboring blocks, searching for all
                //       acceptable bit patterns with `remaining_blocks` bits.
                //
                //       Bear in mind that num_superblocks == 0 is a special
                //       case that allows more bit patterns (no need to touch
                //       edge of previously allocated superblocks).
                //
                //       Bear in mind that superblock_idx == 0 and
                //       usage_bitmap.len() will require special handling as
                //       there is no preceding/following neighbor.
                //
                //       Do not forget to rollback superblock allocations and
                //       increment first_free_superblock_idx on failure. I
                //       wonder if it would be a good idea to handle those
                //       rollbacks using some sort of RAII object?
                unimplemented!()
            } else {
                // We need more free superblocks, is the current one free?
                // TODO: Extract into a method
                if !superblock.load(Ordering::Relaxed).is_empty() {
                    // If not, restart superblock search at the next loop index
                    first_superblock_idx = superblock_idx + 1;
                }
            }
        }

        // TODO: Handle case where last free superblock is at end of the loop,
        //       will require extracting above allocation logic into a function
        //       or closure to avoid duplicating it

        // TODO: Handle the case where we failed to allocate, exiting here

        // Make sure that the previous reads from the allocation bitmap are
        // ordered before any subsequent access to the buffer by the current
        // thread, to avoid data races with the thread that deallocated the
        // memory that we are in the process of allocating.
        atomic::fence(Ordering::Acquire);

        // TODO: Construct output pointer
        // NOTE: Keep number of atomic operations to a minimum, even Relaxed ops
        //       because LLVM is downright terrible at optimizing them. Cache
        //       useful results of Relaxed loads instead of reloading.
        // NOTE: Make sure to return a slice of the requested size, even if it
        //       is not a multiple of the block size.
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
        let local_start_idx = block_idx % Self::blocks_per_superblock();
        if local_start_idx != 0 {
            // Compute index of that superblock
            let superblock_idx = block_idx / Self::blocks_per_superblock();

            // Compute how many blocks are allocated within the superblock,
            // bearing in mind that the buffer may end there
            let local_len =
                (Self::blocks_per_superblock() - local_start_idx)
                    .min(end_block_idx - block_idx);

            // Deallocate leading buffer blocks in this first superblock
            self.dealloc_blocks(
                superblock_idx,
                SuperblockBitmap::new_mask(local_start_idx, local_len)
            );

            // Advance block pointer, stop if all blocks were liberated
            block_idx += local_len;
            if block_idx == end_block_idx { return; }
        }

        // If control reached this point, block_idx is now at the start of a
        // superblock, so we can switch to faster superblock-wise deallocation.
        // Deallocate all superblocks until the one where end_block_idx resides.
        let start_superblock_idx = block_idx / Self::blocks_per_superblock();
        let end_superblock_idx = end_block_idx / Self::blocks_per_superblock();
        for superblock_idx in start_superblock_idx..end_superblock_idx {
            self.dealloc_superblock(superblock_idx);
        }

        // Advance block pointer, stop if all blocks were liberated
        block_idx = end_superblock_idx * Self::blocks_per_superblock();
        if block_idx == end_block_idx { return; }

        // Deallocate trailing buffer blocks in the last superblock
        let remaining_len = end_block_idx - block_idx;
        self.dealloc_blocks(end_superblock_idx,
                            SuperblockBitmap::new_tail_mask(remaining_len));
    }

    // TODO: Provide an API which creates an AllocTransaction with a certain
    //       body superblock range, or returns the index of the first superblock
    //       that wasn't fully free (= 0) as an error.
    //       It could be called try_alloc_superblocks() or something similar

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
        // Check interface preconditions in debug builds
        debug_assert!(superblock_idx < self.usage_bitmap.len(),
                      "Superblock index is out of bitmap range");

        // The superblock should initially be fully free
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
        // Check interface preconditions in debug builds
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
