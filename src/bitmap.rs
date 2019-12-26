//! Mechanisms for managing block allocations within a superblock
//!
//! Since modern CPUs are not bit-addressable, arrays of unsigned integers must
//! be used as vectors of bits. This module provides an abstraction to ease
//! correct manipulation of the inner layer of this data structure.

use crate::Allocator;

use std::{
    ops::{Add, BitAnd, BitOr, Not, Sub},
    sync::atomic::{AtomicUsize, Ordering},
};


/// Block allocation pattern within a superblock
///
/// This bitmap can be used for two purposes: to specify a sequence of blocks
/// that we _want_ to allocate within a superblock (an allocation mask), and to
/// describe which blocks are _currently_ allocated in a superblock.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
#[repr(transparent)]
pub struct SuperblockBitmap(usize);

impl SuperblockBitmap {
    /// The empty (fully unallocated) superblock bitmap
    pub const EMPTY: Self = Self(0);

    /// The fully allocated superblock bitmap
    pub const FULL: Self = Self(std::usize::MAX);

    /// Compute an allocation mask given the index of the first bit that should
    /// be 1 (allocated) and the number of bits that should be 1.
    pub fn new_mask(start: usize, len: usize) -> Self {
        // Check interface preconditions in debug builds
        debug_assert!(start < Allocator::blocks_per_superblock(),
                      "Allocation start is out of superblock range");
        debug_assert!(len < (Allocator::blocks_per_superblock() - start),
                      "Allocation end is out of superblock range");

        // Handle the "full superblock" edge case without overflowing
        if len == Allocator::blocks_per_superblock() {
            return Self(std::usize::MAX);
        }

        // Otherwise, use a general bit pattern computation
        Self(((1 << len) - 1) << start)
    }

    /// Compute an allocation mask for a "head" block sequence, which must end
    /// at the end of the superblock
    pub fn new_head_mask(len: usize) -> Self {
        let len = len.into();
        Self::new_mask(Allocator::blocks_per_superblock() - len - 1, len)
    }

    /// Compute an allocation mask for a "tail" block sequence, which must start
    /// at the beginning of the superblock
    pub fn new_tail_mask(len: usize) -> Self {
        let len = len.into();
        Self::new_mask(0, len)
    }

    /// Truth that allocation bitmap is empty (fully unallocated)
    pub fn is_empty(&self) -> bool {
        *self == Self::EMPTY
    }

    /// Truth that allocation bitmap is fully allocated
    pub fn is_full(&self) -> bool {
        *self == Self::FULL
    }

    /// Truth that allocation bitmap is a mask (contiguous allocation)
    pub fn is_mask(&self) -> bool {
        self.0.count_zeros() == self.0.leading_zeros() + self.0.trailing_zeros()
    }

    /// Number of free blocks which could be used for a "head" block sequence,
    /// which must end at the end of the superblock
    pub fn free_head_blocks(&self) -> usize {
        self.0.leading_zeros() as usize
    }

    /// Number of free blocks which could be used for a "tail" block sequence,
    /// which must start at the start of the superblock
    pub fn free_tail_blocks(&self) -> usize {
        self.0.trailing_zeros() as usize
    }

    /// Search for a hole of N contiguous blocks, as early as possible in the
    /// superblock to keep the bitmap as densely populated as possible.
    ///
    /// On success, return the block index at which a hole was found. On
    /// failure, return the number of free trailing blocks that could be used
    /// as the head of a multi-superblock allocation.
    pub fn search_free_blocks(
        &self,
        start_idx: usize,
        num_blocks: usize
    ) -> Result<usize, usize> {
        // Prepare a working copy of the bitmap's integer representation
        let mut bits = self.0 >> start_idx;

        // Keep track of the index at which we're currently looking for holes
        let mut block_idx = start_idx;
        loop {
            // How many blocks have we not looked at yet?
            let remaining_blocks =
                Allocator::blocks_per_superblock() - block_idx;

            // Can we still find a suitably large hole in here?
            if num_blocks > remaining_blocks {
                return Err(self.free_head_blocks());
            }

            // Find how many blocks are available at the current index. We must
            // use a minimum here because our bit shift-based technique for
            // iterating over bits of the bitmap (see below) will create zeroes
            // at the end of our "bits" working variable.
            let free_blocks =
                (bits.trailing_zeros() as usize).min(remaining_blocks);

            // Have we found a large enough hole?
            if free_blocks as usize >= num_blocks {
                return Ok(block_idx);
            }

            // If not, skip the hole and the block that ended it
            bits >>= free_blocks + 1;
            block_idx += free_blocks + 1;
        }
    }
}

// Use the addition operator (or bit-or) for set union
impl Add for SuperblockBitmap {
    type Output = Self;

    fn add(self, rhs: Self) -> Self {
        self | rhs
    }
}

// Use the bit-and operaztor for set intersection
impl BitAnd for SuperblockBitmap {
    type Output = Self;

    fn bitand(self, rhs: Self) -> Self {
        Self(self.0 & rhs.0)
    }
}

// Use the bit-or operator (or addition) for set union
impl BitOr for SuperblockBitmap {
    type Output = Self;

    fn bitor(self, rhs: Self) -> Self {
        Self(self.0 | rhs.0)
    }
}

// Use the negation operator for absolute set complement
impl Not for SuperblockBitmap {
    type Output = Self;

    fn not(self) -> Self {
        Self(!self.0)
    }
}

// Use the subtraction operator for relative set complement
impl Sub for SuperblockBitmap {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self {
        self & (!rhs)
    }
}


/// Atomic variant of SuperblockBitmap
///
/// This is a wrapper of the corresponding AtomicUint type that provides
/// 1/better independence from the chosen superblock length and 2/clearer naming
/// with respect to the work that we're doing here.
#[derive(Debug, Default)]
pub struct AtomicSuperblockBitmap(AtomicUsize);

impl AtomicSuperblockBitmap {
    /// Create a new atomic superblock bitmap
    pub const fn new(v: SuperblockBitmap) -> Self {
        Self(AtomicUsize::new(v.0))
    }

    /// Get a mutable reference to the underlying bitmap
    pub fn get_mut(&mut self) -> &mut SuperblockBitmap {
        unsafe { &mut *(self.0.get_mut() as *mut _ as *mut SuperblockBitmap) }
    }

    /// Load a value from the atomic bitmap
    pub fn load(&self, order: Ordering) -> SuperblockBitmap {
        SuperblockBitmap(self.0.load(order))
    }

    /// Store a value into the atomic bitmap
    fn store(&self, val: SuperblockBitmap, order: Ordering) {
        self.0.store(val.0, order)
    }

    /// Store a value into the atomic bitmap, returning the previous value
    fn swap(&self,
                val: SuperblockBitmap,
                order: Ordering) -> SuperblockBitmap {
        SuperblockBitmap(self.0.swap(val.0, order))
    }

    /// Store a value into the atomic bitmap if the current value is as expected
    fn compare_exchange(
        &self,
        current: SuperblockBitmap,
        new: SuperblockBitmap,
        success: Ordering,
        failure: Ordering
    ) -> Result<SuperblockBitmap, SuperblockBitmap> {
        self.0.compare_exchange(current.0, new.0, success, failure)
              .map(SuperblockBitmap)
              .map_err(SuperblockBitmap)
    }

    // TODO: When we know the final usage pattern, decide if some form of
    //       compare_exchange_weak could be useful.

    /// Atomically set bits which are set in "val" in the atomic bitmap (aka set
    /// union) and return the previous value.
    fn fetch_add(&self,
                 val: SuperblockBitmap,
                 order: Ordering) -> SuperblockBitmap {
        SuperblockBitmap(self.0.fetch_or(val.0, order))
    }

    /// Atomically clear bits which are set in "val" in the atomic bitmap (aka
    /// relative set complement) and return the previous value.
    fn fetch_sub(&self,
                 val: SuperblockBitmap,
                 order: Ordering) -> SuperblockBitmap {
        SuperblockBitmap(self.0.fetch_and(!val.0, order))
    }

    /// Try to fully allocate a superblock.
    ///
    /// On failure, return the former bitmap value so that one can check which
    /// blocks were already allocated.
    pub fn try_alloc_all(&self,
                         success: Ordering,
                         failure: Ordering) -> Result<(), SuperblockBitmap> {
        self.compare_exchange(SuperblockBitmap::EMPTY,
                              SuperblockBitmap::FULL,
                              success,
                              failure)
            .map(std::mem::drop)
    }

    /// Try to allocate a subset of a superblock, designated by a mask.
    ///
    /// On failure, return the former bitmap value so that one can check which
    /// blocks were already allocated.
    pub fn try_alloc_mask(&self,
                          mask: SuperblockBitmap,
                          success: Ordering,
                          failure: Ordering) -> Result<(), SuperblockBitmap> {
        // Check for suspicious requests in debug builds
        debug_assert!(mask.is_mask(),
                      "Attempted to allocate non-contiguous blocks");
        debug_assert!(!mask.is_empty(),
                      "Useless call to try_alloc_mask with empty mask");
        debug_assert!(!mask.is_full(),
                      "Inefficient call to try_alloc_mask with full mask, \
                       you should be using try_alloc_all instead");

        // Set the required allocation bits, check which were already set
        let mut former_bitmap = self.fetch_add(mask, success);

        // Make sure that none of the target blocks were previously allocated
        if (former_bitmap & mask).is_empty() {
            // All good, ready to return
            Ok(())
        } else {
            // Revert previous block allocation before exiting
            let allocated_bits = mask - former_bitmap;
            former_bitmap = self.fetch_sub(allocated_bits, failure);
            Err(former_bitmap - allocated_bits)
        }
    }

    // TODO: Consider adding a greedy variant of try_alloc_mask that leaves the
    //       block allocated even if allocation did not succeed, if we find a
    //       way to leverage it in future optimization.

    /// Assuming a superblock is fully allocated, fully deallocate it
    pub fn dealloc_all(&self, order: Ordering) {
        if cfg!(debug_assertions) {
            // In debug builds, we make sure that the superblock was indeed
            // marked as fully allocated in order to detect various forms of
            // incorrect allocator usage including double free.
            assert_eq!(
                self.swap(SuperblockBitmap::EMPTY, order),
                SuperblockBitmap::FULL,
                "Tried to deallocate superblock which wasn't fully allocated"
            );
        } else {
            // In release builds, we just store 0 without checking the former
            // value, which avoids use of expensive atomic read-modify-write ops
            self.store(SuperblockBitmap::EMPTY, order);
        }
    }

    /// Deallocate a subset of a superblock, designated by a mask
    pub fn dealloc_mask(&self, mask: SuperblockBitmap, order: Ordering) {
        // Check for suspicious requests in debug builds
        debug_assert!(mask.is_mask(),
                      "Attempted to deallocate non-contiguous blocks");
        debug_assert!(!mask.is_empty(),
                      "Useless call to dealloc_mask with empty mask");
        debug_assert!(!mask.is_full(),
                      "Inefficient call to dealloc_mask with full mask, \
                       you should be using dealloc_all instead");

        // Clear the requested allocation bits
        let old_bitmap = self.fetch_sub(mask, order);

        // In debug builds, make sure that all requested bits were indeed marked
        // as allocated beforehand.
        debug_assert_eq!(old_bitmap & mask, mask,
                         "Tried to deallocate blocks which weren't allocated");
    }
}

impl From<SuperblockBitmap> for AtomicSuperblockBitmap {
    fn from(x: SuperblockBitmap) -> Self {
        Self::new(x)
    }
}
