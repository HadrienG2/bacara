//! RAII mechanism to automatically rollback partially committed allocations
//! if subsequent steps of the allocation process fail.

use crate::{Allocator, SuperblockBitmap, BLOCKS_PER_SUPERBLOCK};

/// RAII guard for iterative allocation, with automatic rollback on failure
///
/// In this lock-free bitmap allocator, the process of allocating memory which
/// spans more than one superblock isn't atomic, and may therefore fail _after_
/// some memory has already been allocated. In that case, the previous partial
/// memory allocation must be reverted.
///
/// This struct imposes some structure on the memory allocation process which
/// allows this rollback process to occur automatically.
///
/// In order to be memory-efficient and allocation-free, it exploits the fact
/// that all allocations can be decomposed into at most a head, a body and a
/// tail, where the head and tail superblocks are partially allocated and the
/// body superblocks are fully allocated:
///
/// `|0011|1111|1111|1111|1000|`
/// ` head <----body----> tail`
///
pub struct AllocGuard<'allocator> {
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

impl<'allocator> AllocGuard<'allocator> {
    /// Start an allocation by trying to allocate a range of contiguous "body"
    /// superblocks (which can be empty for head/tail pairs).
    ///
    /// If your block sequence fits in a single superblock, you don't need to
    /// use AllocGuard, just call `Allocator::try_alloc_blocks()`.
    ///
    /// If the body allocation succeeds, this function returns the resulting
    /// allocation object. If it fails, the function returns the index of the
    /// first superblock that was already allocated (and thus that led the
    /// allocation to fail) and the bit pattern that was observed on this
    /// superblock.
    pub fn with_body(
        allocator: &'allocator Allocator,
        body_start_idx: usize,
        num_body_superblocks: usize,
    ) -> Result<Self, (usize, SuperblockBitmap)> {
        // Check that the request makes sense
        let superblock_capacity = allocator.capacity() / allocator.superblock_size();
        debug_assert!(
            body_start_idx < superblock_capacity,
            "Requested body starts after end of allocator backing store"
        );
        debug_assert!(
            num_body_superblocks < superblock_capacity - body_start_idx,
            "Requested body ends after end of allocator backing store"
        );

        // Create the allocation object right away, initially with an empty
        // body, so that it can track our progress and cancel the body
        // allocation automatically if a problem occurs.
        let mut allocation = Self {
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
            // allocation will be rolled back because "allocation" is dropped.
            allocator
                .try_alloc_superblock(superblock_idx)
                .map_err(|actual_bitmap| (superblock_idx, actual_bitmap))?;

            // Update the allocation object after every allocation
            debug_assert_eq!(superblock_idx, allocation.body_end_idx());
            allocation.num_body_superblocks += 1;
        }

        // The body allocation  was successful, return the allocation object to
        // allow head/tail allocation
        Ok(allocation)
    }

    /// Try to allocate N "head" blocks, falling before the body superblocks.
    ///
    /// On failure, will return how many head blocks are actually available
    /// before the first body superblock.
    ///
    /// Head allocation is optional, and may only be carried out once.
    pub fn try_alloc_head(&mut self, num_blocks: u32) -> Result<(), u32> {
        // Check allocation object consistency
        self.debug_check_invariants();

        // Check that there's room for a head allocation
        debug_assert_ne!(
            self.body_start_idx, 0,
            "No superblock available for head allocation"
        );
        debug_assert_eq!(
            self.num_head_blocks, 0,
            "Head allocation may only be performed once"
        );

        // Reject nonsensical empty head allocations
        debug_assert_ne!(num_blocks, 0, "Requested an empty head allocation");

        // Try to allocate, and on failure return how many head blocks are
        // actually available (trailing zeros in the head superblock)
        self.allocator
            .try_alloc_mask(
                self.body_start_idx - 1,
                SuperblockBitmap::new_head_mask(num_blocks),
            )
            .map_err(|actual_bitmap| actual_bitmap.free_blocks_at_end())?;

        // On success, add the head blocks to the allocation
        self.num_head_blocks = num_blocks;
        Ok(())
    }

    /// Query the index of the superblock after the end of the body, where extra
    /// body superblocks or tail blocks would be allocated.
    pub fn body_end_idx(&self) -> usize {
        // Check invariants
        self.debug_check_invariants();

        // Compute index of past-the-end superblock
        self.body_start_idx + self.num_body_superblocks
    }

    /// Try to extend the body by one superblock
    ///
    /// On failure, will return the bit pattern that was actually observed on
    /// that superblock.
    ///
    /// Body extension is typically needed after a race with another thread
    /// forced us to allocate a smaller head than initially expected. It is
    /// optional, and may only be carried out before tail allocation.
    pub fn try_extend_body(&mut self) -> Result<(), SuperblockBitmap> {
        // Check allocation object consistency
        self.debug_check_invariants();

        // Check that the tail hasn't already been allocated
        debug_assert_eq!(
            self.num_tail_blocks, 0,
            "Body extension must be performed before tail allocation"
        );

        // Check that there's room for an extra superblock
        let superblock_capacity = self.allocator.capacity() / self.allocator.superblock_size();
        debug_assert!(
            self.body_end_idx() < superblock_capacity,
            "No superblock available for body extension"
        );

        // Try to allocate, and on failure return actual bit pattern
        self.allocator.try_alloc_superblock(self.body_end_idx())?;

        // On success, add the extra body superblock to the allocation
        self.num_body_superblocks += 1;
        Ok(())
    }

    /// Try to allocate N "tail" blocks, falling after the body superblocks.
    ///
    /// On failure, will return the bit pattern that was actually observed on
    /// the last body superblock.
    ///
    /// Tail allocation is optional, and may only be carried out once.
    pub fn try_alloc_tail(&mut self, num_blocks: u32) -> Result<(), SuperblockBitmap> {
        // Check allocation object consistency
        self.debug_check_invariants();

        // Check that there's room for a tail allocation
        let superblock_capacity = self.allocator.capacity() / self.allocator.superblock_size();
        debug_assert!(
            self.body_end_idx() < superblock_capacity,
            "No superblock available for tail allocation"
        );
        debug_assert_eq!(
            self.num_tail_blocks, 0,
            "Tail allocation may only be performed once"
        );

        // Reject nonsensical empty tail allocations
        debug_assert_ne!(num_blocks, 0, "Requested an empty tail allocation");

        // Try to allocate, and on failure return how many tail blocks are
        // actually available (leading zeros in the tail superblock)
        self.allocator.try_alloc_mask(
            self.body_end_idx(),
            SuperblockBitmap::new_tail_mask(num_blocks),
        )?;

        // On success, add the tail blocks to the allocation
        self.num_tail_blocks = num_blocks;
        Ok(())
    }

    /// Number of blocks which were allocated as part of this allocation
    pub fn num_blocks(&self) -> usize {
        // Check invariants
        self.debug_check_invariants();

        // Count how many blocks were allocated
        (self.num_head_blocks as usize)
            + self.num_body_superblocks * BLOCKS_PER_SUPERBLOCK
            + (self.num_tail_blocks as usize)
    }

    /// Accept the allocation, return the index of the first allocated _block_
    pub fn commit(self) -> usize {
        // Check invariants
        self.debug_check_invariants();

        // Find the index of the first "one" in the allocation mask
        let first_block_idx =
            self.body_start_idx * BLOCKS_PER_SUPERBLOCK - (self.num_head_blocks as usize);

        // Forget the allocation object so that allocation is not canceled
        std::mem::forget(self);

        // Return the index of the first allocated block
        first_block_idx
    }

    /// Debug-mode check that the allocation object upholds its invariants
    fn debug_check_invariants(&self) {
        // If the allocation has head blocks...
        if self.num_head_blocks != 0 {
            debug_assert_ne!(
                self.body_start_idx, 0,
                "Head superblock has an out-of-bounds index"
            );
            debug_assert_ne!(
                self.num_head_blocks, BLOCKS_PER_SUPERBLOCK as u32,
                "Head superblock is fully allocated, should be \
                 marked as a body superblock instead"
            );
        }

        // Check that the body superblocks span a valid index range
        let superblock_capacity = self.allocator.capacity() / self.allocator.superblock_size();
        debug_assert!(
            self.body_start_idx < superblock_capacity,
            "Body starts after end of allocator backing store"
        );
        debug_assert!(
            self.num_body_superblocks < superblock_capacity - self.body_start_idx,
            "Body ends after end of allocator backing store"
        );

        // If the allocation has tail blocks...
        if self.num_tail_blocks != 0 {
            // All tail blocks must verify some properties
            debug_assert_ne!(
                self.body_end_idx(),
                superblock_capacity - 1,
                "Tail superblock has an out-of-bounds index"
            );
            debug_assert_ne!(
                self.num_tail_blocks, BLOCKS_PER_SUPERBLOCK as u32,
                "Tail superblock is fully allocated, should be \
                 marked as a body superblock instead"
            );
        }
    }
}

impl Drop for AllocGuard<'_> {
    fn drop(&mut self) {
        // Check invariants in debug build
        self.debug_check_invariants();

        // Deallocate head blocks, if any
        if self.num_head_blocks != 0 {
            // This is safe because...
            // - The only code path that sets num_head_blocks to a nonzero value
            //   does so after a successful allocation at this superblock index
            //   and with this mask.
            // - Head allocation may only occur once and body_start_idx is never
            //   modified after AllocGuard creation.
            unsafe {
                self.allocator.dealloc_mask(
                    self.body_start_idx - 1,
                    SuperblockBitmap::new_head_mask(self.num_head_blocks),
                );
            }
        }

        // Deallocate fully allocated body superblocks
        //
        // This is safe because...
        // - `body_start_idx` is never modified after AllocGuard creation.
        // - `body_end_idx()` is initially at `body_start_idx`, only changes by
        //   being incremented after successfully allocating a superblock there.
        unsafe {
            self.allocator
                .dealloc_superblocks(self.body_start_idx, self.body_end_idx());
        }

        // Deallocate tail blocks, if any
        if self.num_tail_blocks != 0 {
            // This is safe because...
            // - The only code path that sets num_tail_blocks to a nonzero value
            //   does so after a successful allocation at this superblock index
            //   and with this mask.
            // - Tail allocation may only occur once and `body_end_idx()` cannot
            //   change afterwards, because the only way to change it after
            //   `AllocGuard` creation is `try_extend_body()`, and that will
            //   fail after tail allocation.
            unsafe {
                self.allocator.dealloc_mask(
                    self.body_end_idx(),
                    SuperblockBitmap::new_tail_mask(self.num_tail_blocks),
                );
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // TODO: Tester avec 8 superblocs, cf hole.rs pour l'explication
    // TODO: Vérifier body_end_idx et num_blocks à chaque opération
    // TODO: Vérifier effet du Drop sur le bitmap à chaque fois

    // TODO: Constructeur with_body avec body de taille nulle
    // TODO: Constructeur with_body avec body non-nul sur bitmap pas OK
    // TODO: Constructeur with_body avec body non-nul sur bitmap OK
    // TODO: commit directement après with_body

    // TODO: try_alloc_head sur bitmap pas OK
    // TODO: try_alloc_head sur bitmap OK
    // TODO: commit après with_body + try_alloc_head

    // TODO: try_extend_body sur bitmap pas OK
    // TODO: try_extend_body sur bitmap OK
    // TODO: commit après with_body + try_extend_body

    // TODO: try_alloc_tail sur bitmap pas OK
    // TODO: try_alloc_tail sur bitmap OK
    // TODO: commit après with_body + try_alloc_tail

    // TODO: Séquence with_body + try_alloc_head + try_extend_body + try_alloc_tail + commit
}
