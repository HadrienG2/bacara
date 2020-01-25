/// Multi-superblock memory allocation and deallocation functions
mod guard;

use crate::{Allocator, Hole, SuperblockBitmap, BLOCKS_PER_SUPERBLOCK};

use guard::AllocGuard;

#[cfg(test)]
use require_unsafe_in_body::require_unsafe_in_body;

/// Try to allocate from a hole of free memory found by HoleSearch
///
/// This operation is not atomic (other threads may observe the allocation in
/// progress), but it cleans up after itself: by the time the function returns,
/// either a suitably sized memory region has been allocated, or all allocations
/// have been reverted.
///
/// Returns the block index of the start of the allocated region on success,
/// and the index of the superblock where things went wrong and the bit
/// pattern that was observed there on failure.
///
/// This operation has `Relaxed` memory ordering, and must be followed by an
/// `Acquire` memory barrier if successful in order to avoid allocation being
/// reordered after usage of the memory block by the compiler or CPU.
pub fn try_alloc_hole(
    allocator: &Allocator,
    hole: Hole,
    num_blocks: usize,
) -> Result<usize, (usize, SuperblockBitmap)> {
    // Check preconditions
    debug_assert_ne!(
        num_blocks, 0,
        "No need for this primitive in zero_sized allocation"
    );

    // Check what kind of hole we're dealing with
    match hole {
        // All blocks are in a single superblock, no guard needed
        Hole::SingleSuperblock {
            superblock_idx,
            first_block_subidx,
        } => {
            // ...so we just try to allocate
            let alloc_result = if num_blocks == BLOCKS_PER_SUPERBLOCK {
                debug_assert_eq!(first_block_subidx, 0);
                allocator.try_alloc_superblock(superblock_idx)
            } else {
                let mask = SuperblockBitmap::new_mask(first_block_subidx, num_blocks as u32);
                allocator.try_alloc_mask(superblock_idx, mask)
            };

            // Did we succeed?
            match alloc_result {
                // We managed to allocate this hole
                Ok(()) => {
                    let first_block_idx =
                        superblock_idx * BLOCKS_PER_SUPERBLOCK + (first_block_subidx as usize);
                    Ok(first_block_idx)
                }

                // We failed to allocate this hole, but we got an
                // updated view of the bit pattern for this superblock
                Err(observed_bitmap) => Err((superblock_idx, observed_bitmap)),
            }
        }

        // Blocks span more than one superblock, need an AllocGuard
        Hole::MultipleSuperblocks {
            body_start_idx,
            mut num_head_blocks,
        } => {
            // Given the number of head blocks, we can find all other
            // parameters of the active allocation.
            let other_blocks = num_blocks - num_head_blocks as usize;
            let num_body_superblocks = other_blocks / BLOCKS_PER_SUPERBLOCK;
            let mut num_tail_blocks = (other_blocks % BLOCKS_PER_SUPERBLOCK) as u32;

            // Try to allocate the body of the allocation
            let mut allocation =
                AllocGuard::with_body(allocator, body_start_idx, num_body_superblocks)?;

            // Try to allocate the head of the hole (if any)
            while num_head_blocks > 0 {
                if let Err(observed_head_blocks) = allocation.try_alloc_head(num_head_blocks) {
                    // On head allocation failure, try to "move the hole
                    // forward", pushing more blocks to the tail.
                    num_tail_blocks += num_head_blocks - observed_head_blocks;
                    num_head_blocks = observed_head_blocks;
                }
            }

            // If needed, allocate one more body superblock.
            // This can happen as a result of moving the hole forward:
            //     |0011|1111|1110|0000| -> |0000|1111|1111|1000|
            if num_tail_blocks >= BLOCKS_PER_SUPERBLOCK as u32 {
                if let Err(observed_bitmap) = allocation.try_extend_body() {
                    return Err((allocation.body_end_idx(), observed_bitmap));
                } else {
                    num_tail_blocks -= BLOCKS_PER_SUPERBLOCK as u32;
                }
            }

            // Try to allocate the tail of the hole (if any)
            if num_tail_blocks > 0 {
                if let Err(observed_bitmap) = allocation.try_alloc_tail(num_tail_blocks) {
                    return Err((allocation.body_end_idx(), observed_bitmap));
                }
            }

            // We managed to allocate everything! Isn't that right?
            debug_assert_eq!(
                allocation.num_blocks(),
                num_blocks,
                "Allocated an incorrect number of blocks"
            );

            // Commit the allocation and get the first block index
            Ok(allocation.commit())
        }
    }
}

/// Deallocate a contiguous range of blocks from the allocator
///
/// This operation has `Relaxed` memory ordering and must be preceded by a
/// `Release` memory barrier in order to avoid deallocation being reordered
/// before usage of the memory block by the compiler or CPU.
///
/// # Safety
///
/// This function must not be targeted at a blocks which are still in use,
/// otherwise many forms of undefined behavior will occur (&mut aliasing, race
/// conditions, double-free...).
#[cfg_attr(test, require_unsafe_in_body)]
#[cfg_attr(not(test), allow(unused_unsafe))]
pub unsafe fn dealloc_blocks(allocator: &Allocator, start_idx: usize, num_blocks: usize) {
    // Check some preconditions
    let block_capacity = allocator.capacity() / allocator.block_size();
    debug_assert!(
        start_idx < block_capacity,
        "Start of target block range is out of bounds"
    );
    debug_assert!(
        num_blocks < block_capacity - start_idx,
        "End of target block range is out of bounds"
    );

    // Set up some progress accounting
    let mut block_idx = start_idx;
    let end_block_idx = block_idx + num_blocks;

    // Does our first block fall in the middle of a superblock?
    let local_start_idx = (block_idx % BLOCKS_PER_SUPERBLOCK) as u32;
    if local_start_idx != 0 {
        // Compute index of that superblock
        let superblock_idx = block_idx / BLOCKS_PER_SUPERBLOCK;

        // Compute how many blocks are allocated within the superblock,
        // bearing in mind that the buffer may end there
        let local_len = (BLOCKS_PER_SUPERBLOCK - local_start_idx as usize)
            .min(end_block_idx - block_idx) as u32;

        // Deallocate leading buffer blocks in this first superblock
        //
        // This is safe because if the above computations are correct, those
        // blocks belong to the target block range, which our safety contract
        // assumes to be safe to deallocate
        unsafe {
            allocator.dealloc_mask(
                superblock_idx,
                SuperblockBitmap::new_mask(local_start_idx, local_len),
            )
        };

        // Advance block pointer, stop if all blocks were liberated
        block_idx += local_len as usize;
        if block_idx == end_block_idx {
            return;
        }
    }

    // If control reached this point, block_idx is now at the start of a
    // superblock, so we can switch to faster superblock-wise deallocation.
    // Deallocate all superblocks until the one where end_block_idx resides.
    //
    // This is safe because if the above computations are correct, those blocks
    // belong to the target block range, which our safety contract assumes to be
    // safe to deallocate
    debug_assert_eq!(
        block_idx % BLOCKS_PER_SUPERBLOCK,
        0,
        "Head deallocation did not work as expected"
    );
    let start_superblock_idx = block_idx / BLOCKS_PER_SUPERBLOCK;
    let end_superblock_idx = end_block_idx / BLOCKS_PER_SUPERBLOCK;
    unsafe {
        allocator.dealloc_superblocks(start_superblock_idx, end_superblock_idx);
    }

    // Advance block pointer, stop if all blocks were liberated
    block_idx = end_superblock_idx * BLOCKS_PER_SUPERBLOCK;
    if block_idx == end_block_idx {
        return;
    }

    // Deallocate trailing buffer blocks in the last superblock
    //
    // This is safe because if the above computations are correct, those blocks
    // belong to the target block range, which our safety contract assumes to be
    // safe to deallocate
    let remaining_len = (end_block_idx - block_idx) as u32;
    unsafe {
        allocator.dealloc_mask(
            end_superblock_idx,
            SuperblockBitmap::new_tail_mask(remaining_len),
        );
    }
}

// TODO: Test this
