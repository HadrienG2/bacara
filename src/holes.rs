//! Mechanism for searching holes in the allocation bitmap

use crate::SuperblockBitmap;


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

    // Number of blocks that remain to be found before emitting the next hole
    remaining_blocks: usize,

    // Iterator over superblock bit patterns
    superblock_iter: SuperblockIter,

    // Index of the superblock that we're currently looking at
    current_superblock_idx: usize,

    // Bitmap of the superblock that we're currently looking at
    current_bitmap: SuperblockBitmap,

    // Index of the block that we're looking at within the current superblock
    current_block_subidx: usize,
}

impl<SuperblockIter> HoleSearch<SuperblockIter>
    where SuperblockIter: Iterator<Item=SuperblockBitmap>
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
            current_block_subidx: 0,
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
    pub fn next(&mut self,
                bad_superblock_idx: usize,
                observed_bitmap: SuperblockBitmap) -> Option<Hole> {
        unimplemented!()
    }

    /// Search the next hole in the allocation bitmap
    ///
    /// This method is private because for every hole after the first one (which
    /// is provided by the constructor), the user must tell why the previous
    /// hole wasn't suitable so that the hole search state can be updated.
    fn search_next(&mut self) -> Option<Hole> {
        unimplemented!()
    }
}
