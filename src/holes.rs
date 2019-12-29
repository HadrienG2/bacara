//! Mechanism for searching holes in the allocation bitmap

use crate::{Allocator, SuperblockBitmap};


/// Location of a free memory "hole" within the allocation bitmap
///
/// The hole can be bigger than requested, so the memory allocation code is
/// encouraged to try translating its hole forward when allocation fails instead
/// of rolling back the full memory allocation transaction right away.
pub enum Hole {
    /// Hole that is concentrated within a single superblock
    SingleSuperblock {
        /// Superblock of interest
        superblock_idx: usize,

        /// Block within that superblock in which the hole starts
        first_block_subidx: usize,
    },

    /// Hole that spans multiple superblocks
    MultipleSuperblocks {
        /// Index of first "body" (fully empty) superblock in the hole, or of
        /// the tail block if there is no body block
        body_start_idx: usize,

        /// Number of head blocks (if any) before the body superblocks
        num_head_blocks: usize,
    },
}


/// Ongoing search for holes in the allocator's allocation-tracking bitmap
///
/// The search behaves like an iterator of holes, but can additionally be
/// informed with updated information about the contents of the bitmap, which is
/// obtained through unsuccessful allocation attempts.
///
/// The search must move forward through the allocation bitmap at every step of
/// the search. It is not allowed to go back to a prior (super)block, nor is it
/// allowed to stand still at a given place. This guarantees bounded allocation
/// timings, which are important for real-time applications.
pub struct HoleSearch<SuperblockIter: Iterator<Item=SuperblockBitmap>> {
    // Requested hole size
    requested_blocks: usize,

    // Number of blocks that remain to be found before emitting the next hole,
    // accumulates knowledge from previous superblocks.
    remaining_blocks: usize,

    // Iterator over superblock bit patterns
    superblock_iter: SuperblockIter,

    // Index of the superblock that we're currently looking at
    current_superblock_idx: usize,

    // Bitmap of the superblock that we're currently looking at
    current_bitmap: SuperblockBitmap,

    // Index of the block that we're looking at within the current superblock
    current_search_subidx: usize,
}

impl<SuperblockIter> HoleSearch<SuperblockIter>
    where SuperblockIter: Iterator<Item=SuperblockBitmap>,
{
    /// Start searching for holes and find the first suitable hole (if any)
    pub fn new(requested_blocks: usize,
               mut superblock_iter: SuperblockIter) -> (Self, Option<Hole>) {
        // Look at the first superblock. There must be one, since allocator
        // capacity cannot be zero per std::alloc rules.
        let first_bitmap =
            superblock_iter.next()
                           .expect("Allocator capacity can't be zero");

        // Build the hole search state
        let mut hole_search = Self {
            requested_blocks,
            remaining_blocks: requested_blocks,
            superblock_iter,
            current_superblock_idx: 0,
            current_bitmap: first_bitmap,
            current_search_subidx: 0,
        };

        // Find the first suitable hole (if any) and return it
        let first_hole = hole_search.search_next();
        (hole_search, first_hole)
    }

    /// Search the next hole, explaining why the previous hole couldn't be used
    ///
    /// This interface is similar to that of `Iterator`: the caller can keep
    /// calling this method as long as it needs more holes, but must stop
    /// calling it as soon as a `None` has been emitted, which signals that the
    /// allocation bitmap has been fully scanned without finding a proper hole.
    ///
    /// However, `HoleSearch` cannot be an `Iterator`, because it asks for some
    /// information to be passed back in on every iteration, namely the reason
    /// why the previous hole couldn't be allocated. This reason is specified by
    /// indicating the superblock on which allocation failed and the allocation
    /// pattern that was observed on that superblock.
    pub fn retry(&mut self,
                 bad_superblock_idx: usize,
                 observed_bitmap: SuperblockBitmap) -> Option<Hole> {
        // Nothing can go wrong in a fully free superblock
        debug_assert!(!observed_bitmap.is_empty(),
                      "Nothing can go wrong with a fully free bitmap");

        // During single-superblock allocation, things can only go wrong at the
        // current superblock index. This means that the code below won't move
        // self.current_superblock_idx and therefore we don't need to worry
        // about resetting self.current_search_subidx.
        if self.current_search_subidx > 0 {
            debug_assert_eq!(
                bad_superblock_idx, self.current_superblock_idx,
                "Single-superblock allocation should fail at the current index"
            );
        }

        // If something went wrong, we need to reset `self.remaining_blocks`
        self.remaining_blocks = self.requested_blocks;

        // Where did things go wrong?
        if bad_superblock_idx < self.current_superblock_idx {
            // Before the current superblock: this reduces the number of
            // previous blocks in the hole that is being investigated.
            let num_prev_superblocks =
                self.current_superblock_idx - bad_superblock_idx - 1;
            self.remaining_blocks -=
                observed_bitmap.free_blocks_at_end()
                + num_prev_superblocks * Allocator::blocks_per_superblock();
        } else {
            // At the current superblock or after (the latter can happen if
            // allocation tried to shift the hole forward). Move to that
            // location if need be and update the current bitmap info.
            while self.current_superblock_idx < bad_superblock_idx {
                let bitmap_opt = self.superblock_iter.next();
                debug_assert!(bitmap_opt.is_some(),
                              "Allocation claims to have observed a block
                               after the end of the backing store");
                self.current_superblock_idx += 1;
            }
            self.current_bitmap = observed_bitmap;
        }

        // Find the next hole
        self.search_next()
    }

    /// Search the next hole in the allocation bitmap
    ///
    /// This method is private because for every hole after the first one (which
    /// is provided by the constructor), the user must tell why the previous
    /// hole wasn't suitable so that the hole search state can be updated.
    fn search_next(&mut self) -> Option<Hole> {
        // Loop over superblocks, starting from the current one
        loop {
            // Holes should be emitted as soon as possible, and remaining_blocks
            // should be reset every time a new hole is found.
            debug_assert_ne!(self.remaining_blocks, 0,
                             "A Hole has not been yielded at the right time, or
                              remaining_blocks has not been properly reset");

            // Are we currently investigating a multi-superblock hole?
            if self.remaining_blocks < self.requested_blocks {
                // How many blocks can we append at the end of the current hole?
                //
                // TODO: Cross-check if this optimization is effective, maybe
                //       self.current_bitmap.free_blocks_at_start() is enough.
                let found_blocks = if self.current_bitmap.is_empty() {
                    Allocator::blocks_per_superblock()
                } else {
                    self.current_bitmap.free_blocks_at_start()
                };

                // Is this enough to complete the current hole?
                if found_blocks >= self.remaining_blocks {
                    // Recover hole shape. The current superblock is preceded by
                    // an integer number of free superblocks (as one that's not
                    // fully free would "break the chain"), possibly preceded by
                    // some head blocks at the end of the previous superblock.
                    let previous_blocks =
                        self.requested_blocks - self.remaining_blocks;
                    let num_head_blocks =
                        previous_blocks % Allocator::blocks_per_superblock();
                    let num_prev_superblocks =
                        previous_blocks / Allocator::blocks_per_superblock();

                    // Emit the current hole
                    return Some(Hole::MultipleSuperblocks {
                        body_start_idx:
                            self.current_superblock_idx - num_prev_superblocks,
                        num_head_blocks,
                    });
                } else if self.current_bitmap.is_empty() {
                    // The hole is not big enough yet, but can be extended.
                    self.remaining_blocks -= Allocator::blocks_per_superblock();
                } else {
                    // The hole is too small and can't be extended.
                    // Reset hole search to start looking for another hole.
                    self.remaining_blocks = self.requested_blocks;
                }
            }

            // Are we starting a new hole?
            if self.remaining_blocks == self.requested_blocks {
                // Look for a suitable hole in the current superblock, at the
                // block index where we left off last time.
                //
                // FIXME: There's probably a more efficient way to do the search
                //        which does not require querying head blocks until
                //        we're sure that we have enough body superblocks. But
                //        do not experiment with this until we have benchmarks.
                match self.current_bitmap.search_free_blocks(
                    self.current_search_subidx,
                    self.requested_blocks,
                ) {
                    // We found all we need within a single superblock
                    Ok(first_block_subidx) => {
                        // If we need to resume the search, we'll do so at the
                        // next block index. This guarantees that hole search
                        // completes in a finite number of steps.
                        self.current_search_subidx = first_block_subidx + 1;

                        // Return this hole to the client
                        return Some(Hole::SingleSuperblock {
                            superblock_idx: self.current_superblock_idx,
                            first_block_subidx,
                        });
                    }

                    // We only found some head blocks (maybe none). Search
                    // for more free blocks in the next superblocks.
                    Err(num_head_blocks) => {
                        self.remaining_blocks =
                            self.requested_blocks - num_head_blocks;
                        self.current_search_subidx = 0;
                    }
                }
            }

            // Go to the next superblock, propagate end of iteration
            self.current_bitmap = self.superblock_iter.next()?;
            self.current_superblock_idx += 1;
        }
    }
}
