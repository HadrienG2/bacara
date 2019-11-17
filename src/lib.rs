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
    alloc::{self, Layout},
    mem::{self, MaybeUninit},
    ptr::NonNull,
    sync::atomic::{AtomicUsize, Ordering},
};


/// Builder for a bitmap allocator
//
// TODO: Make everything `const fn` so that the builder can be used to build a
//       global allocator without dirty lazy_static/OnceCell tricks, once
//       Rust's const-eval is powerful enough for that.
#[derive(Debug)]
pub struct AllocatorBuilder {
    /// Block alignment in bytes
    /// - Will be set to 1 if unspecified
    /// - Must be nonzero and a power of 2, per `std::alloc::Layout` demands
    block_align: Option<usize>,

    /// Block size in bytes
    /// - Must be specified, either directly of via superblock size
    /// - Must be a multiple of alignment, so that all blocks are aligned
    /// - Must be a power of 2, so that implementation has fast divide/modulo
    block_size: Option<usize>,

    /// Capacity of the allocator's backing store in bytes
    /// - Must be specified
    /// - Must be nonzero, per system allocator demands
    /// - Will be rounded to the next multiple of superblock size, to satisfy
    ///   allocator alignment needs and simplify the bitmap implementation.
    /// - This rounding must not overflow usize::MAX
    capacity: Option<usize>,
}

impl AllocatorBuilder {
    /// Set the allocator's storage block alignment (in bytes)
    ///
    /// All buffers produced by the allocator will have this alignment.
    ///
    /// Alignment must be a power of 2, and will be set to 1 (byte alignment)
    /// by default if left unspecified.
    pub fn alignment(mut self, align: usize) -> Self {
        // NOTE: Being a power of 2 implies being nonzero
        assert!(align.is_power_of_two(), "Alignment must be a power of 2");
        assert!(self.block_align.replace(align).is_none(),
                "Alignment must only be set once");
        self
    }

    /// Set the allocator's storage block size (in bytes)
    ///
    /// The block size is the granularity at which the allocator manages its
    /// backing store, which has a large impact on both performance and memory
    /// usage. Please read this crate's top-level documentation for details.
    ///
    /// The block size must be a multiple of the alignment and a power of 2.
    ///
    /// You must set either the block size and superblock size, but not both.
    pub fn block_size(mut self, block_size: usize) -> Self {
        assert!(block_size.is_power_of_two(),
                "Block size must be a power of 2");
        assert!(self.block_size.replace(block_size).is_none(),
                "Block size must only be set once");
        self
    }

    /// Set the allocator's superblock size (in bytes)
    ///
    /// The superblock size is an implementation-defined multiple of the block
    /// size, which corresponds to the allocation request size for which the
    /// allocator should exhibit optimal CPU performance.
    ///
    /// The superblock size must be a multiple of the alignment, of
    /// `Allocator::blocks_per_superblock()`, and a power of 2.
    ///
    /// You must set either the block size and superblock size, but not both.
    pub fn superblock_size(self, superblock_size: usize) -> Self {
        assert_eq!(superblock_size % Allocator::blocks_per_superblock(), 0,
                   "Superblock size must be a multiple of \
                    Allocator::blocks_per_superblock()");
        let block_size = superblock_size / Allocator::blocks_per_superblock();
        self.block_size(block_size)
    }

    /// Set the allocator's approximate backing store capacity (in bytes)
    ///
    /// This is the amount of memory that is managed by the allocator, and
    /// therefore an estimate of how much memory can be allocated from it,
    /// bearing in mind that block-based memory management implies that some
    /// bytes will go to waste when allocation requests are not a multiple of
    /// the block size of the allocator.
    ///
    /// Due to implementation constraints, the actual backing store capacity
    /// will be rounded up to the next multiple of the superblock size.
    ///
    /// The backing store capacity must not be zero, and the aforementioned
    /// rounding should not result in the requested capacity going above
    /// the `usize::MAX` limit.
    pub fn capacity(mut self, capacity: usize) -> Self {
        assert!(capacity != 0, "Backing store capacity must not be zero");
        assert!(self.capacity.replace(capacity).is_none(),
                "Backing store capacity must only be set once");
        self
    }

    /// Build the previously configured allocator
    ///
    /// You must have configured at least a block size and a backing store
    /// capacity before calling this function.
    pub fn build(self) -> Allocator {
        // Select block alignment (which will be the backing store alignment)
        let block_align = self.block_align.unwrap_or(1);

        // Check requested block size
        let block_size = self.block_size
                             .expect("You must specify a block size");
        assert_eq!(block_size % block_align, 0,
                   "Block size must be a multiple of alignment");

        // Round requested capacity to next multiple of superblock size
        let mut capacity =
            self.capacity.expect("You must specify a backing store capacity");
        let superblock_size = block_size * Allocator::blocks_per_superblock();
        let extra_bytes = capacity % superblock_size;
        if extra_bytes != 0 {
            capacity =
                capacity.checked_add(superblock_size - extra_bytes)
                        .expect("Excessive backing store capacity requested");
        }

        // Allocate the backing store
        //
        // This is safe because we've checked all the preconditions of the
        // memory allocator throughout the allocator building process, including
        // the fact that the backing store capacity is nonzero.
        let backing_store_layout =
            Layout::from_size_align(capacity, block_align)
                   .expect("All Layout preconditions should be checked above");
        let backing_store_ptr = unsafe { alloc::alloc(backing_store_layout) };
        if backing_store_ptr.is_null() {
            alloc::handle_alloc_error(backing_store_layout);
        }

        // Turn the backing store into its final form. This is safe because...
        // - We don't care about lifetimes as we want a NonNull in the end
        // - We checked that the pointer isn't null above, and a pointer to u8
        //   cannot be misaligned as it has alignment 1.
        // - There can be no aliasing issues as the allocation was just created
        // - We use MaybeUninit to handle uninitialized bytes.
        let backing_store = unsafe {
            std::slice::from_raw_parts_mut(
                backing_store_ptr.cast::<MaybeUninit<u8>>(),
                capacity
            ).into()
        };

        // Build the usage-tracking bitmap
        let usage_bitmap = std::iter::repeat(0)
                                     .map(AtomicUsize::new)
                                     .take(capacity / superblock_size)
                                     .collect::<Vec<_>>()
                                     .into_boxed_slice();

        // Return the allocator
        Allocator {
            backing_store,
            usage_bitmap,
            block_size_shift: block_size.trailing_zeros() as u8,
            alignment: block_align,
        }
    }
}


/// A thread-safe bitmap allocator
#[derive(Debug)]
pub struct Allocator {
    /// Backing store from which we'll be allocating memory
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
    /// is most likely to have native atomic operations on a given CPU arch, but
    /// this may lead to very big superblocks on future CPU archs, so we might
    /// need to reconsider this decision in the future and use e.g. `AtomicU32`.
    usage_bitmap: Box<[AtomicUsize]>,

    /// Bitshift-based representation of the block size
    ///
    /// This odd value storage is not a space optimization, but a way to tell
    /// the compiler's optimizer that the block size is a power of two so that
    /// it optimizes our integer divisions and remainders. Please use methods
    /// like block_size() to query the block size.
    block_size_shift: u8,

    /// Backing store alignment (and thus storage block alignment)
    ///
    /// We must keep track of this because Rust's allocator API will expect us
    /// to give back this information upon deallocating the backing store.
    ///
    /// It will also prove useful when we try to implement GlobalAlloc, which
    /// requires support for overaligned allocations.
    alignment: usize,
}

impl Allocator {
    /// Start building an allocator
    ///
    /// See the AllocatorBuilder documentation for more details on the allocator
    /// configuration process.
    pub const fn new() -> AllocatorBuilder {
        AllocatorBuilder {
            block_align: None,
            block_size: None,
            capacity: None,
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
    fn drop(&mut self) {
        // Make sure that no storage blocks were allocated, as the corresponding
        // pointers will become dangling as the allocator is dropped...
        assert!(
            self.usage_bitmap.iter()
                             .all(|bits| bits.load(Ordering::Relaxed) == 0),
            "Allocator was dropped while there were still live allocations"
        );

        // Access the backing store. This should be safe because Drop is the
        // only entity that can drop it, and it takes &mut self so it should be
        // allowed to get a mut reference to it without aliasing.
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
        // - Only Drop can liberate that allocation
        // - Drop is called at most once
        // - The layout matches that of AllocatorBuilder::build()
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
