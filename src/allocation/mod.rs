/// Multi-superblock memory allocation and deallocation functions

mod guard;

use crate::{Allocator, Hole, SuperblockBitmap, BLOCKS_PER_SUPERBLOCK};

use guard::AllocGuard;


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
    // TODO: Add some debug_asserts in there
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
                allocator.try_alloc_blocks(superblock_idx, mask)
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

// TODO: Test this