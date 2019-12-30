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
    pub const FULL: Self = Self(!0);

    /// Compute an allocation mask given the index of the first bit that should
    /// be 1 (allocated) and the number of bits that should be 1.
    pub fn new_mask(start: usize, len: usize) -> Self {
        // Check interface preconditions in debug builds
        debug_assert!(start < Allocator::blocks_per_superblock(),
                      "Allocation start is out of superblock range");
        debug_assert!(len <= (Allocator::blocks_per_superblock() - start),
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
        // Check interface preconditions in debug builds
        debug_assert!(len <= Allocator::blocks_per_superblock(),
                      "Requested head mask length is unfeasible");

        // Modulo needed to avoid going out-of-bounds for len == 0
        const BBS: usize = Allocator::blocks_per_superblock();
        Self::new_mask((BBS - len) % BBS, len)
    }

    /// Compute an allocation mask for a "tail" block sequence, which must start
    /// at the beginning of the superblock
    pub fn new_tail_mask(len: usize) -> Self {
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
        // NOTE: Usually equal for masks, but 2x smaller in the case of EMPTY.
        self.0.count_zeros() <= self.0.leading_zeros() + self.0.trailing_zeros()
    }

    /// Number of free blocks which could be used for a "head" block sequence,
    /// which must end at the end of the superblock
    pub fn free_blocks_at_end(&self) -> usize {
        self.0.leading_zeros() as usize
    }

    /// Number of free blocks which could be used for a "tail" block sequence,
    /// which must start at the start of the superblock
    pub fn free_blocks_at_start(&self) -> usize {
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
        // Check interface preconditions in debug builds
        debug_assert!(start_idx < Allocator::blocks_per_superblock(),
                      "Search start index is out of superblock range");
        debug_assert_ne!(num_blocks, 0,
                         "Searching for zero blocks makes no sense");

        // Look for holes at increasing indices, from start_idx onwards
        let mut block_idx = start_idx;
        let mut bits = self.0.rotate_right(start_idx as u32);
        loop {
            // How many blocks have we not looked at yet?
            let mut remaining_blocks =
                Allocator::blocks_per_superblock() - block_idx;

            // Can we still find a suitably large hole in here?
            if num_blocks > remaining_blocks {
                return Err(self.free_blocks_at_end().min(remaining_blocks));
            }

            // Find how many blocks are available at the current index.
            let free_blocks =
                (bits.trailing_zeros() as usize).min(remaining_blocks);

            // Have we found a large enough hole?
            if free_blocks as usize >= num_blocks {
                return Ok(block_idx);
            }

            // If not, skip that hole...
            bits = bits.rotate_right(free_blocks as u32);
            block_idx += free_blocks;
            remaining_blocks -= free_blocks;

            // ...and the sequence of allocated blocks that follows it
            let allocated_blocks =
                ((!bits).trailing_zeros() as usize).min(remaining_blocks);
            bits = bits.rotate_right(allocated_blocks as u32);
            block_idx += allocated_blocks;
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


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_masks() {
        // Shorthand
        const EMPTY: SuperblockBitmap = SuperblockBitmap::EMPTY;

        // Boolean properties
        assert!(EMPTY.is_empty());
        assert!(!EMPTY.is_full());
        assert!(EMPTY.is_mask());

        // Free space before/after
        assert_eq!(EMPTY.free_blocks_at_start(),
                   Allocator::blocks_per_superblock());
        assert_eq!(EMPTY.free_blocks_at_end(),
                   Allocator::blocks_per_superblock());

        // Hole search
        for start_idx in 0..Allocator::blocks_per_superblock() {
            let max_len = Allocator::blocks_per_superblock() - start_idx;
            for len in 1..=max_len {
                assert_eq!(EMPTY.search_free_blocks(start_idx, len),
                           Ok(start_idx));
            }
            assert_eq!(EMPTY.search_free_blocks(start_idx, max_len + 1),
                       Err(max_len));
        }

        // Other ways to create empty superblocks
        for idx in 0..Allocator::blocks_per_superblock() {
            assert_eq!(SuperblockBitmap::new_mask(idx, 0), EMPTY);
        }
        assert_eq!(SuperblockBitmap::new_head_mask(0), EMPTY);
        assert_eq!(SuperblockBitmap::new_tail_mask(0), EMPTY);
    }

    #[test]
    fn full_masks() {
        // Shorthand
        const FULL: SuperblockBitmap = SuperblockBitmap::FULL;

        // Boolean properties
        assert!(!FULL.is_empty());
        assert!(FULL.is_full());
        assert!(FULL.is_mask());

        // Free space before/after
        assert_eq!(FULL.free_blocks_at_start(), 0);
        assert_eq!(FULL.free_blocks_at_end(), 0);

        // Hole search
        for start_idx in 1..Allocator::blocks_per_superblock() {
            assert_eq!(FULL.search_free_blocks(start_idx, 1), Err(0));
        }

        // Other ways to create full superblocks
        assert_eq!(
            SuperblockBitmap::new_mask(0, Allocator::blocks_per_superblock()),
            FULL
        );
        assert_eq!(
            SuperblockBitmap::new_head_mask(Allocator::blocks_per_superblock()),
            FULL
        );
        assert_eq!(
            SuperblockBitmap::new_tail_mask(Allocator::blocks_per_superblock()),
            FULL
        );
    }

    #[test]
    fn other_head_masks() {
        // Enumerate all non-empty and non-full head masks: |00000111|
        for head_len in 1..Allocator::blocks_per_superblock() {
            let head = SuperblockBitmap::new_head_mask(head_len);

            // Boolean properties
            assert!(!head.is_empty());
            assert!(!head.is_full());
            assert!(head.is_mask());

            // Free space before/after
            assert_eq!(head.free_blocks_at_start(),
                       Allocator::blocks_per_superblock() - head_len);
            assert_eq!(head.free_blocks_at_end(), 0);

            // Hole search
            for start_idx in 0..Allocator::blocks_per_superblock() {
                let max_len =
                    (Allocator::blocks_per_superblock() - start_idx)
                        .saturating_sub(head_len);
                for len in 1..=max_len {
                    assert_eq!(head.search_free_blocks(start_idx, len),
                               Ok(start_idx));
                }
                assert_eq!(head.search_free_blocks(start_idx, max_len+1),
                           Err(0));
            }

            // Other way to create this head mask
            assert_eq!(head,
                       SuperblockBitmap::new_mask(head.free_blocks_at_start(),
                                                  head_len));
        }
    }

    #[test]
    fn other_tail_masks() {
        // Enumerate all non-empty and non-full tail masks: |11100000|
        for tail_len in 1..Allocator::blocks_per_superblock() {
            let tail = SuperblockBitmap::new_tail_mask(tail_len);

            // Boolean properties
            assert!(!tail.is_empty());
            assert!(!tail.is_full());
            assert!(tail.is_mask());

            // Free space before/after
            assert_eq!(tail.free_blocks_at_start(), 0);
            assert_eq!(tail.free_blocks_at_end(),
                       Allocator::blocks_per_superblock() - tail_len);

            // Hole search
            for start_idx in 0..Allocator::blocks_per_superblock() {
                let max_len =
                    Allocator::blocks_per_superblock()
                        - start_idx.max(tail_len);
                for len in 1..=max_len {
                    assert_eq!(tail.search_free_blocks(start_idx, len),
                               Ok(start_idx.max(tail_len)));
                }
                assert_eq!(tail.search_free_blocks(start_idx, max_len+1),
                           Err(max_len));
            }

            // Other way to create this tail mask
            assert_eq!(tail, SuperblockBitmap::new_mask(0, tail_len));
        }
    }

    #[test]
    fn central_masks() {
        // Enumerate all masks which start after the beginning of the superblock
        // and end befor the end of the superblock
        for mask_start_idx in 0..Allocator::blocks_per_superblock() {
            let blocks_after_mask_start =
                Allocator::blocks_per_superblock() - mask_start_idx;
            for mask_len in 1..blocks_after_mask_start {
                let mask = SuperblockBitmap::new_mask(mask_start_idx, mask_len);

                // Boolean properties
                assert!(!mask.is_empty());
                assert!(!mask.is_full());
                assert!(mask.is_mask());

                // Free space before/after
                assert_eq!(mask.free_blocks_at_start(), mask_start_idx);
                assert_eq!(mask.free_blocks_at_end(),
                           blocks_after_mask_start - mask_len);

                // Hole search
                for start_idx in 0..Allocator::blocks_per_superblock() {
                    let first_hole_len =
                        mask_start_idx.saturating_sub(start_idx);
                    let second_hole_start =
                        (mask_start_idx + mask_len).max(start_idx);
                    let second_hole_len =
                        Allocator::blocks_per_superblock() - second_hole_start;

                    let max_len = first_hole_len.max(second_hole_len);
                    for len in 1..=first_hole_len {
                        assert_eq!(mask.search_free_blocks(start_idx, len),
                                   Ok(start_idx));
                    }
                    for len in (first_hole_len + 1)..max_len {
                        assert_eq!(mask.search_free_blocks(start_idx, len),
                                   Ok(second_hole_start));
                    }
                    assert_eq!(mask.search_free_blocks(start_idx, max_len+1),
                               Err(second_hole_len));
                }
            }
        }
    }

    // TODO: Test things which aren't masks and SuperblockBitmap operators,
    //       possibly in a single go because we can't just test every non-mask
    //       usize SuperblockBitmap. That would take way too long.
    // TODO: Test AtomicSuperblockBitmap
}
