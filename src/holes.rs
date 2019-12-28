//! Mechanism for searching holes in the allocation bitmap

use crate::SuperblockBitmap;


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
pub struct HoleSearch<SuperblockIter>
    where SuperblockIter: Iterator<Item=SuperblockBitmap>
{
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

// TODO: Add constructor. Constructor reads the first superblock from the iter,
//       which is safe to unwrap because allocator capacity can't be zero, and
//       returns a Self _and_ the first hole.
// TODO: Add method that feeds back information from a bad alloc (bad superblock
//       idx + bitmap observed there) and tries to yield another hole.
