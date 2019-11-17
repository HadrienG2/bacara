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
    /// Backing store from which we'll be allocating memory
    ///
    /// Guaranteed by `AllocatorBuilder` to contain an integer number of
    /// superblocks, each of which maps into one `AtomicUsize` in usage_bitmap.
    backing_store: NonNull<[MaybeUninit<u8>]>,

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
        let backing_store_ptr = alloc::alloc(backing_store_layout);
        if backing_store_ptr.is_null() {
            alloc::handle_alloc_error(backing_store_layout);
        }

        // Turn the backing store into its final form. This is safe because...
        // - We don't care about lifetimes as we ultimately want a NonNull.
        // - We checked that the pointer isn't null above, and a pointer to u8
        //   cannot be misaligned as it has alignment 1.
        // - There can be no &mut aliasing as the allocation was just created.
        // - We use MaybeUninit to handle uninitialized bytes.
        let backing_store = std::slice::from_raw_parts_mut(
                                backing_store_ptr.cast::<MaybeUninit<u8>>(),
                                capacity
                            ).into();

        // Build the usage-tracking bitmap
        let superblock_size = block_size * Self::blocks_per_superblock();
        let usage_bitmap = std::iter::repeat(0)
                                     .map(AtomicUsize::new)
                                     .take(capacity / superblock_size)
                                     .collect::<Vec<_>>()
                                     .into_boxed_slice();

        // Build and return the allocator struct
        Allocator {
            backing_store,
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
    /// make sure that the output buffer pointer is passed back to
    /// `Allocator::dealloc_unbound()` before the allocator is dropped, or else:
    ///
    /// 1. `Allocator::drop()` will panic in debug builds
    /// 2. The pointer will be invalidated, and dereferencing it after that
    ///    _will_ unleash undefined behavior.
    //
    // TODO: Support overaligned allocations by accepting `std::alloc::Layout`
    pub fn alloc_unbound(&self, _size: usize) -> Option<NonNull<[MaybeUninit<u8>]>> {
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
    pub unsafe fn dealloc_unbound(&self, _ptr: NonNull<[MaybeUninit<u8>]>) {
        unimplemented!()
    }
}

impl Drop for Allocator {
    fn drop(&mut self) {
        // Make sure that no storage blocks were allocated, as the corresponding
        // pointers will become dangling when the allocator is dropped...
        //
        // TODO: This should only be a debug check, but is currently necessary
        //       for drop to be safe as an &mut to the backing store must be
        //       currently be constructed in order to query its length. Fix this
        //       once one can query the length of a `NonNull<[T]>` without
        //       materializing a reference to it.
        assert!(
            self.usage_bitmap.iter()
                             .all(|bits| bits.load(Ordering::Relaxed) == 0),
            "Allocator was dropped while there were still live allocations"
        );

        // Access the whole backing store mutably. This should be safe as...
        // - We have checked that there are no allocations in the wild, so &mut
        //   backing store aliasing should not occur per this check's result.
        // - We hold a unique &mut self ref, so we can assume that no one is
        //   concurrently allocating or deallocating memory from the allocator,
        //   and that the check's result will thus remain valid.
        // - The backing store pointer should be valid since it was checked to
        //   be valid at construction time, only Drop is allowed to invalidate
        //   it, and Drop will not be called more than once.
        //
        // However, note that there is currently a little unsoundness problem
        // with taking references to data which is going to be deallocated,
        // because rustc is currently unable to tell LLVM that it should not
        // access the data behind the reference after deallocation.
        //
        // Now, in principle, LLVM has no reason to insert new accesses, and it
        // currently doesn't, but once a fix for that is implemented in `std`,
        // we may want to implement the same fix if it requires some manual
        // intervention in the allocating code (TODO).
        //
        // Here's a discussion of this problem for reference:
        // https://github.com/rust-lang/rust/issues/55005
        let backing_store_slice = unsafe { self.backing_store.as_mut() };

        // Deallocate the backing store. This is safe because...
        // - An allocator is always created with a backing store allocation
        // - Only Drop, which happens at most once, can liberate that allocation
        // - The layout matches that used in `Allocator::new_unchecked()`
        let backing_store_layout =
            Layout::from_size_align(backing_store_slice.len(), self.alignment)
                   .expect("All Layout preconditions were checked by builder");
        unsafe { alloc::dealloc(backing_store_slice.as_mut_ptr().cast::<u8>(),
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
