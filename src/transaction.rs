//! RAII mechanism to automatically rollback partially committed allocations
//! if subsequent steps of the allocation process fail.

use crate::{Allocator, AllocationMask};

use std::ops::Range;


/// RAII guard to automatically rollback failed allocations
///
/// In this lock-free bitmap allocator, the process of allocating memory which
/// spans more than one superblock isn't atomic, and may therefore fail _after_
/// some memory has already been allocated. In that case, the previous partial
/// memory allocation must be reverted.
///
/// This struct imposes some structure on the memory allocation process which
/// allows this transaction rollback process to occur automatically.
///
/// In order to be memory-efficient and allocation-free, it exploits the fact
/// that all allocations can be decomposed into at most a head, a body and a
/// tail, where the head and tail superblocks are partially allocated and the
/// body superblocks are fully allocated:
///
/// `|0011|1111|1111|1111|1000|`
/// ` head <----body----> tail`
///
struct AllocationTransaction<'allocator> {
    // TODO: Investigate simpler data layouts

    /// Partial bit pattern allocated in the "head" superblock at superblock
    /// index (body_indices.start-1), if any
    head_allocation_mask: Option<AllocationMask>,

    /// Superblock index range covererd by the "body" superblocks
    body_indices: Range<usize>,

    /// Partial bit pattern allocated in the "tail" superblock at superblock
    /// index (body_indices.end), if any
    tail_allocation_mask: Option<AllocationMask>,

    /// Back-reference to the host allocator
    allocator: &'allocator Allocator,
}

impl<'allocator> AllocationTransaction<'allocator> {
    /// Start an allocation transaction by trying to allocate a range of
    /// contiguous "body" superblocks.
    ///
    /// If you don't need body superblocks, but only a head/tail block pair,
    /// consider using `with_head()` instead. If you only need a bunch of blocks
    /// that fit in a single superblock, you don't need AllocationTransaction,
    /// just call `Allocator::try_alloc_blocks()` directly.
    ///
    /// If the body allocation succeeds, this function returns the resulting
    /// transaction object. If it fails, the function returns the index of the
    /// first superblock that was already allocated (and thus that led the
    /// allocation transaction to fail).
    pub fn with_body(allocator: &'allocator Allocator,
                     body_indices: Range<usize>) -> Result<Self, usize> {
        // Check that the request makes sense
        debug_assert!(
            body_indices.start < body_indices.end,
            "Requested an empty body allocation, consider using with_head() or \
             calling Allocator::try_alloc_blocks() directly"
        );
        debug_assert!(
            body_indices.end < allocator.capacity()/allocator.superblock_size(),
            "Requested body goes past the end of the allocator's backing store"
        );

        // Create the transaction object right away, initially with an empty
        // body, so that it can track our progress and cancel the body
        // allocation automatically if a problem occurs.
        let mut transaction = Self {
            head_allocation_mask: None,
            body_indices: body_indices.start..body_indices.start,
            tail_allocation_mask: None,
            allocator,
        };

        // Iterate over the requested body indices
        for superblock_idx in body_indices {
            // Try to allocate the current superblock. If that fails, return the
            // index of the superblock that caused the failure. Any previous
            // allocation will be rolled back because "transaction" is dropped.
            allocator.try_alloc_superblock(superblock_idx)
                     .map_err(|_| superblock_idx)?;

            // Whenever allocation succeeds, update the transaction object to
            // take that into account.
            transaction.body_indices.end = superblock_idx;
        }

        // The body allocation transaction was successful, return the
        // transaction object so that the head and tail (if any) may be
        // allocated as well.
        Ok(transaction)
    }

    // TODO: Constructor for the case where only a head/tail pair is needed

    /// Try to allocate N "head" blocks, falling before the body superblocks.
    ///
    /// On failure, will return how many head blocks are actually available
    /// before the first body superblock.
    pub fn try_alloc_head(&mut self,
                          num_blocks: usize) -> Result<&mut Self, usize> {
        // Check preconditions, and that the user request makes sense
        self.debug_check_invariants();
        debug_assert_ne!(self.body_indices.start, 0,
                         "No superblock available for head allocation");
        debug_assert_ne!(num_blocks, 0,
                         "Requested an empty head allocation");

        // Set up an allocation mask that is suitable for a head block
        let allocation_mask = AllocationMask::new(
            Allocator::blocks_per_superblock() - num_blocks - 1,
            num_blocks
        );

        // Try to allocate, and on failure return how many head blocks are
        // actually available (trailing zeros in the head superblock)
        self.allocator
            .try_alloc_blocks(self.body_indices.start - 1, allocation_mask)
            .map_err(|actual_bitmap| actual_bitmap.trailing_zeros() as usize)?;

        // On success, add the head blocks to the transaction, checking that a
        // head block wasn't already allocated.
        assert!(self.head_allocation_mask.replace(allocation_mask).is_none(),
                "Head blocks may only be allocated once");
        Ok(self)
    }

    /// Try to allocate N "tail" blocks, falling after the body superblocks.
    ///
    /// On failure, will return how many tail blocks are actually available
    /// after the last body superblock.
    pub fn try_alloc_tail(&mut self,
                          num_blocks: usize) -> Result<&mut Self, usize> {
        // Check invariants + that the user request makes sense
        self.debug_check_invariants();
        debug_assert!(num_blocks != 0,
                      "Requested an empty tail allocation");

        // Set up an allocation mask that is suitable for a tail block
        let allocation_mask = AllocationMask::new(0, num_blocks);

        // Try to allocate, and on failure return how many tail blocks are
        // actually available (leading zeros in the tail superblock)
        self.allocator
            .try_alloc_blocks(self.body_indices.end, allocation_mask)
            .map_err(|actual_bitmap| actual_bitmap.leading_zeros() as usize)?;

        // On success, add the tail blocks to the transaction, checking that a
        // tail block wasn't already allocated.
        assert!(self.tail_allocation_mask.replace(allocation_mask).is_none(),
                "Tail blocks may only be allocated once");
        Ok(self)
    }

    // TODO: Query that allows debug assertions to check the total amount of
    //       allocated blocks right before calling commit().

    /// Accept the transaction and return the index of the first allocated block
    pub fn commit(self) -> usize {
        // Check invariants
        self.debug_check_invariants();

        // Find the index of the first "one" in the allocation mask
        let first_block_idx =
            if let Some(head_mask) = self.head_allocation_mask {
                (self.body_indices.start-1) * Allocator::blocks_per_superblock()
                    + head_mask.start()
            } else {
                self.body_indices.start * Allocator::blocks_per_superblock()
            };

        // Forget the transaction object so that transaction is not canceled
        std::mem::forget(self);

        // Return the index of the first allocated block
        first_block_idx
    }

    /// Debug-mode check that the transaction object upholds its invariants
    fn debug_check_invariants(&self) {
        // If the transaction has a head block...
        if let Some(head_mask) = self.head_allocation_mask {
            // All head blocks must verify some properties
            debug_assert_ne!(self.body_indices.start, 0,
                             "Head superblock has an out-of-bounds index");
            debug_assert!(!head_mask.empty(),
                          "Head superblock marked as allocated, but isn't");
            debug_assert!(!head_mask.full(),
                          "Head superblock is fully allocated, should be \
                           marked as a body superblock instead");

            // If there are subsequent allocated blocks, the head block must
            // also be contiguous with them, i.e. end where they begin.
            if self.body_indices.end > self.body_indices.start
               || self.tail_allocation_mask.is_some() {
                debug_assert_eq!(
                    head_mask.end(), Allocator::blocks_per_superblock(),
                    "Head block does not reach the end of its superblock, \
                     but there are subsequent allocated blocks"
                );
            }
        }

        // Check that the body superblocks span a valid index range
        let num_superblocks =
            self.allocator.capacity() / self.allocator.superblock_size();
        debug_assert!(self.body_indices.end <= num_superblocks,
                      "Allocation body ends on an out-of-bounds index");

        // If the transaction has a tail block...
        if let Some(tail_mask) = self.tail_allocation_mask {
            // All tail blocks must verify some properties
            debug_assert!(self.body_indices.end < num_superblocks,
                          "Tail superblock has an out-of-bounds index");
            debug_assert!(!tail_mask.empty(),
                          "Tail superblock marked as allocated, but isn't");
            debug_assert!(!tail_mask.full(),
                          "Tail superblock is fully allocated, should be \
                           marked as a body superblock instead");

            // If there are previous allocated blocks, the tail block must
            // also be contiguous with them, i.e. begin where they end.
            if self.body_indices.end > self.body_indices.start
               || self.head_allocation_mask.is_some() {
                debug_assert_eq!(
                    tail_mask.start(), 0,
                    "Tail block does not begin at start of its superblock, \
                     but there are previous allocated blocks");
            }
        }
    }
}

impl Drop for AllocationTransaction<'_> {
    fn drop(&mut self) {
        // Check invariants in debug build
        self.debug_check_invariants();

        // Deallocate head blocks, if any
        if let Some(head_mask) = self.head_allocation_mask {
            self.allocator.dealloc_blocks(self.body_indices.start-1, head_mask);
        }

        // Deallocate fully allocated body superblocks
        for superblock_idx in self.body_indices.clone() {
            self.allocator.dealloc_superblock(superblock_idx);
        }

        // Deallocate tail blocks, if any
        if let Some(tail_mask) = self.tail_allocation_mask {
            self.allocator.dealloc_blocks(self.body_indices.end, tail_mask);
        }
    }
}