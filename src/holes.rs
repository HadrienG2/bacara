//! Mechanism for searching holes in the allocation bitmap

use crate::{BLOCKS_PER_SUPERBLOCK, SuperblockBitmap};


/// Location of a free memory "hole" within the allocation bitmap
///
/// The hole can be bigger than requested, so the memory allocation code is
/// encouraged to try translating its hole forward when allocation fails instead
/// of rolling back the full memory allocation transaction right away.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum Hole {
    /// Hole that fits within a single superblock
    SingleSuperblock {
        /// Superblock of interest
        superblock_idx: usize,

        /// Block within that superblock in which the hole starts
        first_block_subidx: u32,
    },

    /// Hole that spans multiple superblocks
    MultipleSuperblocks {
        /// Index of first "body" (fully empty) superblock in the hole, or of
        /// the tail block if there is no body block
        body_start_idx: usize,

        /// Number of head blocks (if any) before the body superblocks
        num_head_blocks: u32,
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
    current_search_subidx: u32,
}

impl<SuperblockIter> HoleSearch<SuperblockIter>
    where SuperblockIter: Iterator<Item=SuperblockBitmap>,
{
    /// Start searching for holes and find the first suitable hole (if any)
    pub fn new(requested_blocks: usize,
               mut superblock_iter: SuperblockIter) -> (Self, Option<Hole>) {
        // Zero-sized holes are not worth the trouble of being supported here
        debug_assert_ne!(requested_blocks, 0,
                         "No need for HoleSearch to invent a zero-sized hole");

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

        // Where did things go wrong?
        if bad_superblock_idx < self.current_superblock_idx {
            // Before the current superblock: this reduces the number of
            // previous blocks in the hole that is being investigated.
            let num_prev_superblocks =
                self.current_superblock_idx - bad_superblock_idx - 1;
            self.remaining_blocks =
                self.requested_blocks
                    - (observed_bitmap.free_blocks_at_end() as usize)
                    + num_prev_superblocks * BLOCKS_PER_SUPERBLOCK;
        } else {
            // At the current superblock or after (the latter can happen if
            // allocation tried to shift the hole forward). Move to that
            // location if need be and update the current bitmap info.
            while bad_superblock_idx > self.current_superblock_idx {
                debug_assert!(self.next_superblock().is_some(),
                              "Allocation claims to have observed a block after
                               the end of the allocator's backing store");
            }
            self.current_bitmap = observed_bitmap;
            self.remaining_blocks = self.requested_blocks;
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
            // should be reset after an allocation failure.
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
                    BLOCKS_PER_SUPERBLOCK
                } else {
                    self.current_bitmap.free_blocks_at_start() as usize
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
                        (previous_blocks % BLOCKS_PER_SUPERBLOCK) as u32;
                    let num_prev_superblocks =
                        previous_blocks / BLOCKS_PER_SUPERBLOCK;

                    // Emit the current hole
                    return Some(Hole::MultipleSuperblocks {
                        body_start_idx:
                            self.current_superblock_idx - num_prev_superblocks,
                        num_head_blocks,
                    });
                } else if self.current_bitmap.is_empty() {
                    // The hole is not big enough yet, but can be extended.
                    self.remaining_blocks -= BLOCKS_PER_SUPERBLOCK;
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
                        // current hole's start.
                        self.current_search_subidx = first_block_subidx;

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
                            self.requested_blocks - (num_head_blocks as usize);
                        self.current_search_subidx = 0;
                    }
                }
            }

            // Go to the next superblock, propagate end of iteration
            self.current_bitmap = self.next_superblock()?;
        }
    }

    /// Go to the next superblock (if any) and return its bitmap
    fn next_superblock(&mut self) -> Option<SuperblockBitmap> {
        debug_assert_eq!(self.current_search_subidx, 0,
                         "Moved to next superblock before end of block iteration
                          or failed to reset block iteration state.");
        self.current_superblock_idx += 1;
        self.superblock_iter.next()
    }
}


#[cfg(test)]
mod tests {
    use super::*;


    /// Suggestions of (number of superblocks, request size) tuples to try out
    ///
    /// Need to try with 8 superblocks in order to allow a maximally complex
    /// pattern (skip two superblocks, allocate a head, three body superblocks
    /// and a tail, don't touch the last superblock). Trying with minimal
    /// resources, like one or two superblocks, is also a good idea.
    ///
    /// Request sizes were defined for 8 superblocks in order to cover a broad
    /// range of situations, then adapted for 1 or 2 superblocks.
    const TEST_CONFIGURATIONS: &[(usize, usize)] = &[
        (1, 1),
        (1, BLOCKS_PER_SUPERBLOCK / 2),
        (1, BLOCKS_PER_SUPERBLOCK - 1),
        (1, BLOCKS_PER_SUPERBLOCK),
        (1, BLOCKS_PER_SUPERBLOCK + 1),
        (1, 2 * BLOCKS_PER_SUPERBLOCK),

        (2, 1),
        (2, BLOCKS_PER_SUPERBLOCK / 2),
        (2, BLOCKS_PER_SUPERBLOCK - 1),
        (2, BLOCKS_PER_SUPERBLOCK),
        (2, BLOCKS_PER_SUPERBLOCK + 1),
        (2, 2 * BLOCKS_PER_SUPERBLOCK - 1),
        (2, 2 * BLOCKS_PER_SUPERBLOCK),
        (2, 2 * BLOCKS_PER_SUPERBLOCK + 1),
        (2, 3 * BLOCKS_PER_SUPERBLOCK),

        (8, 1),
        (8, BLOCKS_PER_SUPERBLOCK / 2),
        (8, BLOCKS_PER_SUPERBLOCK - 1),
        (8, BLOCKS_PER_SUPERBLOCK),
        (8, BLOCKS_PER_SUPERBLOCK + 1),
        (8, 2 * BLOCKS_PER_SUPERBLOCK - 1),
        (8, 2 * BLOCKS_PER_SUPERBLOCK),
        (8, 2 * BLOCKS_PER_SUPERBLOCK + 1),
        (8, 3 * BLOCKS_PER_SUPERBLOCK - 1),
        (8, 3 * BLOCKS_PER_SUPERBLOCK),
        (8, 3 * BLOCKS_PER_SUPERBLOCK + 2),
        (8, 8 * BLOCKS_PER_SUPERBLOCK),
        (8, 8 * BLOCKS_PER_SUPERBLOCK + 1),
        (8, 9 * BLOCKS_PER_SUPERBLOCK),
    ];

    #[test]
    fn build_full() {
        for &(num_superblocks, requested_blocks) in TEST_CONFIGURATIONS {
            let (hole_search, hole) =
                HoleSearch::new(requested_blocks,
                                std::iter::repeat(SuperblockBitmap::FULL)
                                          .take(num_superblocks));
            assert_eq!(hole, predict_search_result(requested_blocks, 0, 0));
            assert_eq!(hole_search.requested_blocks, requested_blocks);
            assert_eq!(hole_search.remaining_blocks,
                       predict_remaining_blocks(requested_blocks,
                                                0,
                                                0,
                                                num_superblocks));
            assert_eq!(hole_search.current_superblock_idx, num_superblocks);
            assert_eq!(hole_search.current_bitmap, SuperblockBitmap::FULL);
            assert_eq!(hole_search.current_search_subidx, 0);
            check_superblock_iter(hole_search, num_superblocks);
        }
    }

    #[test]
    fn build_empty() {
        for &(num_superblocks, requested_blocks) in TEST_CONFIGURATIONS {
            let (hole_search, hole) =
                HoleSearch::new(requested_blocks,
                                std::iter::repeat(SuperblockBitmap::EMPTY)
                                          .take(num_superblocks));

            let hole_blocks = num_superblocks * BLOCKS_PER_SUPERBLOCK;
            assert_eq!(hole, predict_search_result(requested_blocks,
                                                   0,
                                                   hole_blocks));
            assert_eq!(hole_search.requested_blocks, requested_blocks);
            assert_eq!(hole_search.remaining_blocks,
                       predict_remaining_blocks(requested_blocks,
                                                0,
                                                hole_blocks,
                                                num_superblocks));

            let trailing_blocks =
                if requested_blocks % BLOCKS_PER_SUPERBLOCK == 0 {
                    BLOCKS_PER_SUPERBLOCK
                } else {
                    requested_blocks % BLOCKS_PER_SUPERBLOCK
                };
            let previous_superblocks =
                (requested_blocks - trailing_blocks) / BLOCKS_PER_SUPERBLOCK;

            assert_eq!(hole_search.current_superblock_idx,
                       previous_superblocks.min(num_superblocks));

            assert_eq!(hole_search.current_bitmap, SuperblockBitmap::EMPTY);

            assert_eq!(hole_search.current_search_subidx, 0);

            check_superblock_iter(hole_search, num_superblocks);
        }
    }

    #[test]
    fn build_exact() {
        'conf: for &(num_superblocks, requested_blocks) in TEST_CONFIGURATIONS {
            // We'll try to generate all possible holes of the right size. If
            // none exist, we skip this configuration.
            if requested_blocks > num_superblocks { continue 'conf; }
            let num_blocks = num_superblocks * BLOCKS_PER_SUPERBLOCK;
            for hole_shift in 0..=(num_blocks - requested_blocks) {
                // Start the hole search and validate its results
                let (hole_search, hole) =
                    HoleSearch::new(requested_blocks,
                                    hole_iter(hole_shift, requested_blocks)
                                        .take(num_superblocks));

                assert_eq!(hole, predict_search_result(requested_blocks,
                                                       hole_shift,
                                                       requested_blocks));
                assert_eq!(hole_search.requested_blocks, requested_blocks);
                assert_eq!(hole_search.remaining_blocks,
                           predict_remaining_blocks(requested_blocks,
                                                    hole_shift,
                                                    requested_blocks,
                                                    num_superblocks));

                // Convert the hole block-wise shift in superblocks+tail
                let start_superblock_idx = hole_shift / BLOCKS_PER_SUPERBLOCK;
                let start_subidx = (hole_shift % BLOCKS_PER_SUPERBLOCK) as u32;

                // Compute number of free blocks in the first hole superblock
                let first_blocks =
                    (BLOCKS_PER_SUPERBLOCK - start_subidx as usize)
                        .min(requested_blocks);

                if first_blocks == requested_blocks {
                    // Hole fits in a single superblock
                    assert_eq!(hole_search.current_superblock_idx,
                               start_superblock_idx);
                    assert_eq!(hole_search.current_bitmap,
                               !SuperblockBitmap::new_mask(
                                   start_subidx,
                                   requested_blocks as u32
                               ));
                    assert_eq!(hole_search.current_search_subidx, start_subidx);
                } else {
                    // Hole has a head/body/tail layout
                    let body_start_idx =
                        start_superblock_idx +
                            (first_blocks != BLOCKS_PER_SUPERBLOCK) as usize;
                    let other_blocks = requested_blocks - first_blocks;
                    let trailing_blocks =
                        if other_blocks % BLOCKS_PER_SUPERBLOCK == 0 {
                            BLOCKS_PER_SUPERBLOCK
                        } else {
                            other_blocks % BLOCKS_PER_SUPERBLOCK
                        };
                    let previous_superblocks =
                        (other_blocks-trailing_blocks) / BLOCKS_PER_SUPERBLOCK;
                    assert_eq!(hole_search.current_superblock_idx,
                               body_start_idx + previous_superblocks);
                    assert_eq!(hole_search.current_bitmap,
                               SuperblockBitmap::new_tail_mask(
                                   trailing_blocks as u32
                               ));
                    assert_eq!(hole_search.current_search_subidx, 0);
                }

                check_superblock_iter(hole_search, num_superblocks);
            }
        }
    }

    #[test]
    #[should_panic]
    fn bad_build_empty_iter() {
        HoleSearch::new(1, std::iter::empty());
    }

    // TODO: Retrying and ending iteration
    // TODO: Tester avec des trous un peu trop petits et un peu trop grands
    // TODO: Tester avec une suite de deux trous (trop court -> OK)

    // Generate an infinite bitmap which is entirely allocated except for a hole
    // of a certain size at a certain (block-wise) offset.
    fn hole_iter(offset: usize, size: usize) -> impl Iterator<Item=SuperblockBitmap> {
        // Generate the desired bitmap using genawaiter trickery
        //
        // Must use a reference counted generator here because the state exits
        // the current function's scope, and stack-pinned generators don't
        // support that at this point in time.
        genawaiter::rc::Gen::new(|co| async move {
            // Convert the hole block-wise shift in superblocks+tail
            let start_superblock_idx = offset / BLOCKS_PER_SUPERBLOCK;
            let start_subidx = (offset % BLOCKS_PER_SUPERBLOCK) as u32;

            // Compute number of free blocks in the first hole superblock
            let first_blocks =
                (BLOCKS_PER_SUPERBLOCK - start_subidx as usize)
                    .min(size);

            // Full superblocks before the hole
            for _header in 0..start_superblock_idx {
                co.yield_(SuperblockBitmap::FULL).await;
            }

            // First superblock in the hole, need to handle the case of
            // a hole that fits in a single superblock: |11100011|
            co.yield_(
                !SuperblockBitmap::new_mask(start_subidx,
                                            first_blocks as u32)
            ).await;

            // Start keeping track of remaining blocks, emit superblocks
            let mut remaining_blocks = size - first_blocks;
            while remaining_blocks > BLOCKS_PER_SUPERBLOCK {
                co.yield_(SuperblockBitmap::EMPTY).await;
                remaining_blocks -= BLOCKS_PER_SUPERBLOCK;
            }

            // Now we can emit the tail of the hole (if any, otherwise
            // this code nicely degrades into SuperblockBitmap::FULL.
            co.yield_(
                !SuperblockBitmap::new_tail_mask(
                    remaining_blocks as u32
                )
            ).await;

            // And after that we emit full superblocks again
            loop {
                co.yield_(SuperblockBitmap::FULL).await;
            }
        }).into_iter()
    }

    // Predict the result of a hole search on a bitmap with a single hole
    fn predict_search_result(requested_size: usize,
                             hole_offset: usize,
                             hole_size: usize) -> Option<Hole> {
        // Search will fail if request is too big
        if hole_size < requested_size {
            return None;
        }

        // Convert the hole block-wise shift in superblocks+tail
        let superblock_idx = hole_offset / BLOCKS_PER_SUPERBLOCK;
        let start_subidx = hole_offset % BLOCKS_PER_SUPERBLOCK;

        // Dertermine number of blocks after hole start in initial superblock
        let blocks_after_start = BLOCKS_PER_SUPERBLOCK - start_subidx;

        // Figure out hole topology accordingly
        if blocks_after_start >= requested_size {
            Some(Hole::SingleSuperblock {
                superblock_idx,
                first_block_subidx: start_subidx as u32,
            })
        } else if blocks_after_start == BLOCKS_PER_SUPERBLOCK {
            Some(Hole::MultipleSuperblocks {
                body_start_idx: superblock_idx,
                num_head_blocks: 0,
            })
        } else {
            Some(Hole::MultipleSuperblocks {
                body_start_idx: superblock_idx + 1,
                num_head_blocks: blocks_after_start as u32,
            })
        }
    }

    // Predict "remaining blocks" state on a bitmap with a single hole
    fn predict_remaining_blocks(requested_size: usize,
                                hole_offset: usize,
                                hole_size: usize,
                                num_superblocks: usize) -> usize {
        // The result is easiest to predict starting from search results
        match predict_search_result(requested_size,
                                    hole_offset,
                                    hole_size) {
            // If search failed, the end of the bitmap was reached
            None => {
                let hole_end = hole_offset + hole_size;
                let bitmap_end = num_superblocks * BLOCKS_PER_SUPERBLOCK;
                if hole_end == bitmap_end {
                    // If the hole reached there, remaining_blocks wasn't reset
                    requested_size - hole_size
                } else {
                    // If the hole stopped before, remaining_blocks was reset
                    requested_size
                }
            }

            // On single-superblock allocs, remaining_blocks isn't updated
            Some(Hole::SingleSuperblock { .. }) => requested_size,

            // On multi-superblock allocs, remaining_blocks reflects the number
            // of remaining blocks when the last superblock was investigated.
            Some(Hole::MultipleSuperblocks { num_head_blocks, .. }) => {
                let non_head_blocks = requested_size - num_head_blocks as usize;
                let tail_blocks = non_head_blocks % BLOCKS_PER_SUPERBLOCK;
                if tail_blocks != 0 {
                    // That would be the tail blocks, if any...
                    tail_blocks
                } else {
                    // ...or else the last fully allocated body superblock
                    BLOCKS_PER_SUPERBLOCK
                }
            }
        }
    }

    // Check that the hole search iterator is in sync with its superblock idx,
    // must be called at the end of a test since it consumes the iterator.
    fn check_superblock_iter<It>(hole_search: HoleSearch<It>,
                                 num_superblocks: usize)
        where It: Iterator<Item=SuperblockBitmap>,
    {
        assert_eq!(hole_search.superblock_iter.count(),
                   (num_superblocks - hole_search.current_superblock_idx)
                       .saturating_sub(1));
    }
}
