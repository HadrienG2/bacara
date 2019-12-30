//! RAII mechanism to automatically rollback partially committed allocations
//! if subsequent steps of the allocation process fail.

use crate::{Allocator, BLOCKS_PER_SUPERBLOCK, SuperblockBitmap};


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
pub struct AllocTransaction<'allocator> {
    /// Number of blocks allocated at the end of the "head" superblock, located
    /// right before the body superblock sequence.
    num_head_blocks: u32,

    /// Index at which the body superblock sequence begins
    body_start_idx: usize,

    /// Number of superblocks in the body (can be zero)
    num_body_superblocks: usize,

    /// Number of blocks allocated at the beginning of the "tail" superblock,
    /// located right after the body superblock sequence.
    num_tail_blocks: u32,

    /// Back-reference to the host allocator
    allocator: &'allocator Allocator,
}

impl<'allocator> AllocTransaction<'allocator> {
    /// Start an allocation transaction by trying to allocate a range of
    /// contiguous "body" superblocks (which can be empty for head/tail pairs).
    ///
    /// If your block sequence fits in a single superblock, you don't need to
    /// use AllocTransaction, just call `Allocator::try_alloc_blocks()`.
    ///
    /// If the body allocation succeeds, this function returns the resulting
    /// transaction object. If it fails, the function returns the index of the
    /// first superblock that was already allocated (and thus that led the
    /// allocation transaction to fail) and the bit pattern that was observed
    /// on this superblock.
    pub fn with_body(
        allocator: &'allocator Allocator,
        body_start_idx: usize,
        num_body_superblocks: usize
    ) -> Result<Self, (usize, SuperblockBitmap)> {
        // Check that the request makes sense
        let superblock_capacity =
            allocator.capacity() / allocator.superblock_size();
        debug_assert!(
            body_start_idx < superblock_capacity,
            "Requested body starts after end of allocator backing store"
        );
        debug_assert!(
            num_body_superblocks < superblock_capacity - body_start_idx,
            "Requested body ends after end of allocator backing store"
        );

        // Create the transaction object right away, initially with an empty
        // body, so that it can track our progress and cancel the body
        // allocation automatically if a problem occurs.
        let mut transaction = Self {
            num_head_blocks: 0,
            body_start_idx,
            num_body_superblocks: 0,
            num_tail_blocks: 0,
            allocator,
        };

        // Iterate over the requested body indices
        let body_end_idx = body_start_idx + num_body_superblocks;
        for superblock_idx in body_start_idx..body_end_idx {
            // Try to allocate the current superblock. If that fails, return the
            // index of the superblock that caused the failure. Any previous
            // allocation will be rolled back because "transaction" is dropped.
            allocator.try_alloc_superblock(superblock_idx)
                     .map_err(|actual_bitmap| (superblock_idx, actual_bitmap))?;

            // Update the transaction after every allocation
            transaction.num_body_superblocks += 1;
        }

        // The body allocation transaction was successful, return the
        // transaction object to allow head/tail allocation
        Ok(transaction)
    }

    /// Try to allocate N "head" blocks, falling before the body superblocks.
    ///
    /// On failure, will return how many head blocks are actually available
    /// before the first body superblock.
    pub fn try_alloc_head(&mut self, num_blocks: u32) -> Result<(), u32> {
        // Check transaction object consistency
        self.debug_check_invariants();

        // Check that there's room for a head allocation
        debug_assert_ne!(self.body_start_idx, 0,
                         "No superblock available for head allocation");
        debug_assert_eq!(self.num_head_blocks, 0,
                         "Head allocation may only be performed once");

        // Reject nonsensical empty head allocations
        debug_assert_ne!(num_blocks, 0,
                         "Requested an empty head allocation");

        // Try to allocate, and on failure return how many head blocks are
        // actually available (trailing zeros in the head superblock)
        self.allocator
            .try_alloc_blocks(self.body_start_idx - 1,
                              SuperblockBitmap::new_head_mask(num_blocks))
            .map_err(|actual_bitmap| actual_bitmap.free_blocks_at_end())?;

        // On success, add the head blocks to the transaction
        self.num_head_blocks = num_blocks;
        Ok(())
    }

    /// Query the index of the superblock after the end of the body, where extra
    /// body superblocks or tail blocks would be allocated.
    pub fn body_end_idx(&self) -> usize {
        self.body_start_idx + self.num_body_superblocks
    }

    /// Try to extend the body by one superblock
    ///
    /// On failure, will return the bit pattern that was actually observed on
    /// that superblock.
    pub fn try_extend_body(&mut self) -> Result<(), SuperblockBitmap> {
        // Check transaction object consistency
        self.debug_check_invariants();

        // Check that there's room for an extra superblock
        let num_body_superblocks =
            self.allocator.capacity() / self.allocator.superblock_size();
        debug_assert!(self.body_end_idx() < num_body_superblocks,
                      "No superblock available for body extension");

        // Try to allocate, and on failure return actual bit pattern
        self.allocator.try_alloc_superblock(self.body_end_idx())?;

        // On success, add the extra body superblock to the transaction
        self.num_body_superblocks += 1;
        Ok(())
    }

    /// Try to allocate N "tail" blocks, falling after the body superblocks.
    ///
    /// On failure, will return the bit pattern that was actually observed on
    /// the last body superblock.
    pub fn try_alloc_tail(&mut self,
                          num_blocks: u32) -> Result<(), SuperblockBitmap> {
        // Check transaction object consistency
        self.debug_check_invariants();

        // Check that there's room for a tail allocation
        let num_body_superblocks =
            self.allocator.capacity() / self.allocator.superblock_size();
        debug_assert!(self.body_end_idx() < num_body_superblocks,
                      "No superblock available for tail allocation");
        debug_assert_eq!(self.num_tail_blocks, 0,
                         "Tail allocation may only be performed once");

        // Reject nonsensical empty tail allocations
        debug_assert_ne!(num_blocks, 0,
                         "Requested an empty tail allocation");

        // Try to allocate, and on failure return how many tail blocks are
        // actually available (leading zeros in the tail superblock)
        self.allocator
            .try_alloc_blocks(self.body_end_idx(),
                              SuperblockBitmap::new_tail_mask(num_blocks))?;

        // On success, add the tail blocks to the transaction
        self.num_tail_blocks = num_blocks;
        Ok(())
    }

    /// Number of blocks which were allocated as part of this transaction
    pub fn num_blocks(&self) -> usize {
        // Check invariants
        self.debug_check_invariants();

        // Count how many blocks were allocated
        (self.num_head_blocks as usize)
            + self.num_body_superblocks * BLOCKS_PER_SUPERBLOCK
            + (self.num_tail_blocks as usize)
    }

    /// Accept the transaction, return the index of the first allocated _block_
    pub fn commit(self) -> usize {
        // Check invariants
        self.debug_check_invariants();

        // Find the index of the first "one" in the allocation mask
        let first_block_idx =
            self.body_start_idx * BLOCKS_PER_SUPERBLOCK
                - (self.num_head_blocks as usize);

        // Forget the transaction object so that transaction is not canceled
        std::mem::forget(self);

        // Return the index of the first allocated block
        first_block_idx
    }

    /// Debug-mode check that the transaction object upholds its invariants
    fn debug_check_invariants(&self) {
        // If the transaction has head blocks...
        if self.num_head_blocks != 0 {
            debug_assert_ne!(self.body_start_idx, 0,
                             "Head superblock has an out-of-bounds index");
            debug_assert_ne!(self.num_head_blocks,
                             BLOCKS_PER_SUPERBLOCK as u32,
                             "Head superblock is fully allocated, should be \
                              marked as a body superblock instead");
        }

        // Check that the body superblocks span a valid index range
        let superblock_capacity =
            self.allocator.capacity() / self.allocator.superblock_size();
        debug_assert!(
            self.body_start_idx < superblock_capacity,
            "Body starts after end of allocator backing store"
        );
        debug_assert!(
            self.num_body_superblocks < superblock_capacity - self.body_start_idx,
            "Body ends after end of allocator backing store"
        );

        // If the transaction has tail blocks...
        if self.num_tail_blocks != 0 {
            // All tail blocks must verify some properties
            let body_end_idx = self.body_start_idx + self.num_body_superblocks;
            debug_assert_ne!(body_end_idx, superblock_capacity - 1,
                             "Tail superblock has an out-of-bounds index");
            debug_assert_ne!(self.num_tail_blocks,
                             BLOCKS_PER_SUPERBLOCK as u32,
                             "Tail superblock is fully allocated, should be \
                              marked as a body superblock instead");
        }
    }
}

impl Drop for AllocTransaction<'_> {
    fn drop(&mut self) {
        // Check invariants in debug build
        self.debug_check_invariants();

        // Deallocate head blocks, if any
        if self.num_head_blocks != 0 {
            self.allocator.dealloc_blocks(
                self.body_start_idx - 1,
                SuperblockBitmap::new_head_mask(self.num_head_blocks)
            );
        }

        // Deallocate fully allocated body superblocks
        let body_end_idx = self.body_start_idx + self.num_body_superblocks;
        for superblock_idx in self.body_start_idx..body_end_idx {
            self.allocator.dealloc_superblock(superblock_idx);
        }

        // Deallocate tail blocks, if any
        if self.num_tail_blocks != 0 {
            self.allocator.dealloc_blocks(
                body_end_idx,
                SuperblockBitmap::new_tail_mask(self.num_tail_blocks)
            );
        }
    }
}
