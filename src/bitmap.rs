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

// Use the bit-and operator for set intersection
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
#[repr(transparent)]
pub struct AtomicSuperblockBitmap(AtomicUsize);

impl AtomicSuperblockBitmap {
    /// Create a new atomic superblock bitmap
    pub const fn new(v: SuperblockBitmap) -> Self {
        Self(AtomicUsize::new(v.0))
    }

    /// Get a mutable reference to the underlying bitmap
    pub fn get_mut(&mut self) -> &mut SuperblockBitmap {
        unsafe {
            &mut *(self.0.get_mut() as *mut usize as *mut SuperblockBitmap)
        }
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
    /// and return the previous bit pattern + a success/failure signal
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
        } else if former_bitmap & mask != mask {
            // Incorrect allocations occured, must revert them
            let allocated_bits = mask - former_bitmap;
            former_bitmap = self.fetch_sub(allocated_bits, failure);
            Err(former_bitmap - allocated_bits)
        } else {
            // Failed without incorrect allocations
            Err(former_bitmap)
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
        // and end before the end of the superblock
        for mask_start_idx in 0..Allocator::blocks_per_superblock() {
            let max_mask_len =
                Allocator::blocks_per_superblock() - mask_start_idx.max(1);
            for mask_len in 1..=max_mask_len {
                let mask = SuperblockBitmap::new_mask(mask_start_idx, mask_len);
                let mask_end_idx = mask_start_idx + mask_len;

                // Boolean properties
                assert!(!mask.is_empty());
                assert!(!mask.is_full());
                assert!(mask.is_mask());

                // Free space before/after
                assert_eq!(mask.free_blocks_at_start(), mask_start_idx);
                assert_eq!(mask.free_blocks_at_end(),
                           Allocator::blocks_per_superblock() - mask_end_idx);

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

    #[test]
    fn mask_operations() {
        // Generate a first mask and its negation
        for mask1_start in 0..Allocator::blocks_per_superblock() {
            let max_mask1_len =
                Allocator::blocks_per_superblock() - mask1_start;
            for mask1_len in mask1_start..=max_mask1_len {
                let mask1 = SuperblockBitmap::new_mask(mask1_start, mask1_len);
                let mask1_end = mask1_start + mask1_len;
                let neg_mask1 = !mask1;

                // Boolean properties
                assert_eq!(neg_mask1.is_empty(), mask1.is_full());
                assert_eq!(neg_mask1.is_full(), mask1.is_empty());
                assert_eq!(neg_mask1.is_mask(),
                           mask1.is_empty()
                            || mask1.free_blocks_at_start() == 0
                            || mask1.free_blocks_at_end() == 0);

                // Free space before/after
                assert_eq!(
                    neg_mask1.free_blocks_at_start(),
                    if mask1_start == 0 { mask1_len } else { 0 }
                );
                assert_eq!(
                    neg_mask1.free_blocks_at_end(),
                    if mask1_end == Allocator::blocks_per_superblock() {
                        mask1_len
                    } else {
                        0
                    }
                );

                // Now generate another mask to try some mask operations
                for mask2_start in 0..Allocator::blocks_per_superblock() {
                    let max_mask2_len =
                        Allocator::blocks_per_superblock() - mask2_start;
                    for mask2_len in 0..=max_mask2_len {
                        let mask2 = SuperblockBitmap::new_mask(mask2_start,
                                                               mask2_len);

                        // Subtraction means "bits set in A but not B" and its
                        // properties are therefore checked by testing the
                        // properties of AND and NOT.
                        assert_eq!(mask1 - mask2, mask1 & !mask2);

                        // All other set operations are symmetrical and only
                        // need to be tested for mask2_start >= mask1_start
                        if mask2_start < mask1_start { continue; }

                        // Addition is equivalent to bitmap OR
                        assert_eq!(mask1 + mask2, mask1 | mask2);

                        // Intersection properties
                        let mask1_and_2 = mask1 & mask2;
                        assert_eq!(mask1_and_2, mask2 & mask1);
                        assert_eq!(mask1_and_2.is_empty(),
                                   mask1.is_empty() || mask2.is_empty()
                                   || mask1_end <= mask2_start);
                        assert_eq!(mask1_and_2.is_full(),
                                   mask1.is_full() && mask2.is_full());
                        assert!(mask1_and_2.is_mask());
                        assert_eq!(mask1_and_2.free_blocks_at_start(),
                                   if mask1_and_2.is_empty() {
                                       Allocator::blocks_per_superblock()
                                   } else {
                                       mask1.free_blocks_at_start()
                                            .max(mask2.free_blocks_at_start())
                                   });
                        assert_eq!(mask1_and_2.free_blocks_at_end(),
                                   if mask1_and_2.is_empty() {
                                       Allocator::blocks_per_superblock()
                                   } else {
                                       mask1.free_blocks_at_end()
                                            .max(mask2.free_blocks_at_end())
                                   });

                        // Union properties
                        let mask1_or_2 = mask1 | mask2;
                        assert_eq!(mask1_or_2, mask2 | mask1);
                        assert_eq!(mask1_or_2.is_empty(),
                                   mask1.is_empty() && mask2.is_empty());
                        assert_eq!(mask1_or_2.is_full(),
                                   mask1.is_full() || mask2.is_full()
                                   || (mask1_start == 0
                                          && mask1_end >= mask2_start
                                          && mask2_len == max_mask2_len));
                        assert_eq!(mask1_or_2.is_mask(),
                                   mask1.is_empty() || mask2.is_empty()
                                   || mask1_end >= mask2_start);
                        assert_eq!(mask1_or_2.free_blocks_at_start(),
                                   mask1.free_blocks_at_start()
                                        .min(mask2.free_blocks_at_start()));
                        assert_eq!(mask1_or_2.free_blocks_at_end(),
                                   mask1.free_blocks_at_end()
                                        .min(mask2.free_blocks_at_end()));
                    }
                }
            }
        }
    }

    #[test]
    fn atomic_sequential() {
        // Enumerate all possible initial masks
        for orig_mask_start in 0..Allocator::blocks_per_superblock() {
            let max_orig_mask_len =
                Allocator::blocks_per_superblock() - orig_mask_start;
            for orig_mask_len in orig_mask_start..=max_orig_mask_len {
                let orig_mask = SuperblockBitmap::new_mask(orig_mask_start,
                                                           orig_mask_len);

                // Make an atomic version and check get_mut and load
                let mut atomic_mask = AtomicSuperblockBitmap::new(orig_mask);
                assert_eq!(*atomic_mask.get_mut(), orig_mask);
                assert_eq!(atomic_mask.load(Ordering::Relaxed), orig_mask);

                // Try the From-based route to do the same thing
                let mut atomic_from = AtomicSuperblockBitmap::from(orig_mask);
                assert_eq!(*atomic_from.get_mut(), orig_mask);
                assert_eq!(atomic_from.load(Ordering::Relaxed), orig_mask);

                // Try to allocate and deallocate everything
                let alloc_all_result =
                    atomic_mask.try_alloc_all(Ordering::Relaxed,
                                              Ordering::Relaxed);
                if orig_mask.is_empty() {
                    assert_eq!(alloc_all_result, Ok(()));
                    assert_eq!(*atomic_mask.get_mut(), SuperblockBitmap::FULL);
                    atomic_mask.dealloc_all(Ordering::Relaxed);
                    assert_eq!(*atomic_mask.get_mut(), SuperblockBitmap::EMPTY);
                } else {
                    assert_eq!(alloc_all_result, Err(orig_mask));
                    assert_eq!(*atomic_mask.get_mut(), orig_mask);
                }

                // Enumerate every sensible (non-empty/non-full) allocation mask
                for alloc_mask_start in 0..Allocator::blocks_per_superblock() {
                    let max_alloc_mask_len =
                        Allocator::blocks_per_superblock()
                            - alloc_mask_start.max(1);
                    for alloc_mask_len in 1..=max_alloc_mask_len {
                        let alloc_mask =
                            SuperblockBitmap::new_mask(alloc_mask_start,
                                                       alloc_mask_len);

                        // Try to allocate with that mask
                        let alloc_mask_result =
                            atomic_mask.try_alloc_mask(alloc_mask,
                                                       Ordering::Relaxed,
                                                       Ordering::Relaxed);
                        if (orig_mask & alloc_mask).is_empty() {
                            assert_eq!(alloc_mask_result, Ok(()));
                            assert_eq!(*atomic_mask.get_mut(),
                                       orig_mask + alloc_mask);
                            atomic_mask.dealloc_mask(alloc_mask,
                                                     Ordering::Relaxed);
                            assert_eq!(*atomic_mask.get_mut(), orig_mask);
                        } else {
                            assert_eq!(alloc_mask_result, Err(orig_mask));
                            assert_eq!(*atomic_mask.get_mut(), orig_mask);
                        }
                    }
                }
            }
        }
    }

    #[test]
    #[ignore]
    fn atomic_concurrent_mask_all() {
        use rand::prelude::*;
        use std::sync::Arc;

        // Test configuration
        const NUM_MASKS: usize = 1000;
        const ITERS_PER_MASK: usize = 20_000;

        // Set up what we need
        let atomic_bitmap =
            Arc::new(AtomicSuperblockBitmap::new(SuperblockBitmap::EMPTY));
        let mut rng = thread_rng();

        // For a certain amount of masks
        for _ in 0..NUM_MASKS {
            // Generate a random allocation mask
            let mask_start_idx =
                rng.gen_range(0, Allocator::blocks_per_superblock());
            let max_mask_len =
                Allocator::blocks_per_superblock() - mask_start_idx.max(1);
            let mask_len = rng.gen_range(1, max_mask_len + 1);
            let mask = SuperblockBitmap::new_mask(mask_start_idx, mask_len);

            // Start a race between a thread trying to allocate with that mask
            // and another thread trying to allocate everything
            let atomic_bitmap_1 = atomic_bitmap.clone();
            let atomic_bitmap_2 = atomic_bitmap.clone();
            testbench::concurrent_test_2(
                move || {
                    for _ in 0..ITERS_PER_MASK {
                        match atomic_bitmap_1.try_alloc_mask(
                            mask,
                            Ordering::Relaxed,
                            Ordering::Relaxed
                        ) {
                            Ok(()) =>
                                atomic_bitmap_1.dealloc_mask(mask,
                                                             Ordering::Relaxed),
                            Err(bad_mask) =>
                                assert_eq!(bad_mask, SuperblockBitmap::FULL),
                        }
                    }
                },
                move || {
                    for _ in 0..ITERS_PER_MASK {
                        match atomic_bitmap_2.try_alloc_all(Ordering::Relaxed,
                                                            Ordering::Relaxed) {
                            Ok(()) =>
                                atomic_bitmap_2.dealloc_all(Ordering::Relaxed),
                            Err(bad_mask) =>
                                assert_eq!(bad_mask, mask),
                        }
                    }
                },
            );

            // Everything that has been allocated should now be deallocated
            assert_eq!(atomic_bitmap.load(Ordering::Relaxed),
                       SuperblockBitmap::EMPTY);
        }
    }


    #[test]
    #[ignore]
    fn atomic_concurrent_mask_mask() {
        use rand::prelude::*;
        use std::sync::Arc;

        // Test configuration
        const NUM_MASKS: usize = 1000;
        const ITERS_PER_MASK: usize = 10_000;

        // Set up what we need
        let atomic_bitmap =
            Arc::new(AtomicSuperblockBitmap::new(SuperblockBitmap::EMPTY));
        let mut rng = thread_rng();
        let mut gen_mask = || {
            let mask_start_idx =
                rng.gen_range(0, Allocator::blocks_per_superblock());
            let max_mask_len =
                Allocator::blocks_per_superblock() - mask_start_idx.max(1);
            let mask_len = rng.gen_range(1, max_mask_len + 1);
            SuperblockBitmap::new_mask(mask_start_idx, mask_len)
        };

        // For a certain amount of masks
        for _ in 0..NUM_MASKS {
            // Generate a random allocation mask
            let mask1 = gen_mask();
            let mask2 = gen_mask();

            // Start a race between a thread trying to allocate with that mask
            // and another thread trying to allocate everything
            let atomic_bitmap_1 = atomic_bitmap.clone();
            let atomic_bitmap_2 = atomic_bitmap.clone();
            testbench::concurrent_test_2(
                move || {
                    for _ in 0..ITERS_PER_MASK {
                        match atomic_bitmap_1.try_alloc_mask(
                            mask1,
                            Ordering::Relaxed,
                            Ordering::Relaxed
                        ) {
                            Ok(()) =>
                                atomic_bitmap_1.dealloc_mask(mask1,
                                                             Ordering::Relaxed),
                            Err(bad_mask) =>
                                assert!(bad_mask.is_empty()
                                        || bad_mask == mask2),
                        }
                    }
                },
                move || {
                    for _ in 0..ITERS_PER_MASK {
                        match atomic_bitmap_2.try_alloc_mask(
                            mask2,
                            Ordering::Relaxed,
                            Ordering::Relaxed
                        ) {
                            Ok(()) =>
                                atomic_bitmap_2.dealloc_mask(mask2,
                                                             Ordering::Relaxed),
                            Err(bad_mask) =>
                                assert!(bad_mask.is_empty()
                                        || bad_mask == mask1),
                        }
                    }
                },
            );

            // Everything that has been allocated should now be deallocated
            assert_eq!(atomic_bitmap.load(Ordering::Relaxed),
                       SuperblockBitmap::EMPTY);
        }
    }
}
