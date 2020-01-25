//! Mechanism for building an `Allocator`, with proper invariant checking

use crate::{Allocator, BLOCKS_PER_SUPERBLOCK};

/// Builder for a bitmap allocator
//
// NOTE: The main purpose of this builder is to ensure that a certain number of
//       preconditions are upheld upon constructing the allocator. These are
//       listed as "must" bullet points in the struct members' doc comments.
#[derive(Debug, Default, PartialEq)]
pub struct Builder {
    /// Block alignment in bytes
    /// - Will be set to 1 if unspecified
    /// - Must be a power of 2 (and thus nonzero), per `alloc::Layout` demands
    block_align: Option<usize>,

    /// Block size in bytes
    /// - Must be specified, either directly of via superblock size
    /// - Must be a multiple of block alignment, so that all blocks are aligned
    /// - Must be a power of 2, so that implementation has fast divide/modulo
    block_size: Option<usize>,

    /// Capacity of the allocator's backing store in bytes
    /// - Must be specified
    /// - Must be nonzero, per system allocator demands
    /// - Will be rounded to the next multiple of superblock size, to satisfy
    ///   allocator alignment needs and simplify the bitmap implementation.
    /// - This rounding must not overflow isize::MAX
    capacity: Option<usize>,
}

// TODO: Make everything `const fn` so that the builder can be used to build a
//       global allocator without dirty lazy_static/OnceCell tricks, once
//       Rust's const-eval is powerful enough for that.
impl Builder {
    /// Start building an allocator
    pub const fn new() -> Self {
        Self {
            block_align: None,
            block_size: None,
            capacity: None,
        }
    }

    /// Set the allocator's storage block alignment (in bytes)
    ///
    /// All buffers produced by the allocator will have this alignment.
    ///
    /// Alignment must be a power of 2, and will be set to 1 (byte alignment)
    /// by default if left unspecified.
    pub fn alignment(mut self, align: usize) -> Self {
        assert!(align.is_power_of_two(), "Alignment must be a power of 2");
        assert!(
            self.block_align.replace(align).is_none(),
            "Alignment must only be set once"
        );
        self
    }

    /// Set the allocator's storage block size (in bytes)
    ///
    /// The block size is the granularity at which the allocator manages its
    /// backing store, which has a large impact on both performance and memory
    /// usage. Please read this crate's top-level documentation for details.
    ///
    /// The block size must be a multiple of the alignment and a power of 2.
    ///
    /// You must set either the block size or the superblock size, but not both.
    pub fn block_size(mut self, block_size: usize) -> Self {
        assert!(
            block_size.is_power_of_two(),
            "Block size must be a power of 2"
        );
        assert!(
            self.block_size.replace(block_size).is_none(),
            "Block size must only be set once"
        );
        self
    }

    /// Set the allocator's superblock size (in bytes)
    ///
    /// The superblock size is an implementation-defined multiple of the block
    /// size, which corresponds to the allocation request size for which the
    /// allocator should exhibit optimal CPU performance.
    ///
    /// The superblock size must be a multiple of the alignment, and the product
    /// of `Allocator::BLOCKS_PER_SUPERBLOCK` by a power of 2.
    ///
    /// You must set either the block size or the superblock size, but not both.
    pub fn superblock_size(self, superblock_size: usize) -> Self {
        assert_eq!(
            superblock_size % BLOCKS_PER_SUPERBLOCK,
            0,
            "Superblock size must be a multiple of \
             Allocator::BLOCKS_PER_SUPERBLOCK"
        );
        let block_size = superblock_size / BLOCKS_PER_SUPERBLOCK;
        self.block_size(block_size)
    }

    /// Set the allocator's approximate backing store capacity (in bytes)
    ///
    /// This is the amount of memory that is managed by the allocator, and
    /// therefore an estimate of how much memory can be allocated from it,
    /// bearing in mind that block-based memory management implies that some
    /// bytes will go to waste when allocation requests are not a multiple of
    /// the block size of the allocator.
    ///
    /// Due to implementation constraints, the actual backing store capacity
    /// will be rounded up to the next multiple of the superblock size.
    ///
    /// The backing store capacity must not be zero, and the aforementioned
    /// rounding should not result in the requested capacity going above
    /// the `isize::MAX` limit. Yes, `isize`, you read that right. Rust pointers
    /// have some mysterious limitations, among which the one that it is
    /// forbidden to have pointer offsets that overflow `isize`.
    pub fn capacity(mut self, capacity: usize) -> Self {
        assert!(capacity != 0, "Backing store capacity must not be zero");
        assert!(
            capacity <= (std::isize::MAX as usize),
            "Backing store capacity cannot overflow isize::MAX"
        );
        assert!(
            self.capacity.replace(capacity).is_none(),
            "Backing store capacity must only be set once"
        );
        self
    }

    /// Build the previously configured allocator
    ///
    /// You must have configured at least a block size and a backing store
    /// capacity before calling this function.
    pub fn build(&self) -> Allocator {
        // Select block alignment (which will be the backing store alignment)
        let block_align = self.block_align.unwrap_or(1);

        // Check requested block size
        let block_size = self.block_size.expect("You must specify a block size");
        assert_eq!(
            block_size % block_align,
            0,
            "Block size must be a multiple of alignment"
        );

        // Round requested capacity to next multiple of superblock size
        let mut capacity = self
            .capacity
            .expect("You must specify a backing store capacity");
        let superblock_size = block_size * BLOCKS_PER_SUPERBLOCK;
        let extra_bytes = capacity % superblock_size;
        if extra_bytes != 0 {
            capacity += superblock_size - extra_bytes;
            assert!(
                capacity <= (std::isize::MAX as usize),
                "Excessive backing store capacity requested"
            );
        }

        // Build the allocator, this is safe because we have checked all the
        // preconditions listed in the Builder struct documentation.
        unsafe { Allocator::new_unchecked(block_align, block_size, capacity) }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn initial_state() {
        // Constructor route
        let builder = Builder::new();
        assert_eq!(builder.block_align, None);
        assert_eq!(builder.block_size, None);
        assert_eq!(builder.capacity, None);

        // Default-constructor route
        let builder = Builder::default();
        assert_eq!(builder.block_align, None);
        assert_eq!(builder.block_size, None);
        assert_eq!(builder.capacity, None);
    }

    #[test]
    #[should_panic]
    fn build_empty() {
        Builder::new().build();
    }

    #[test]
    fn good_alignment() {
        let builder = Builder::new().alignment(1);
        assert_eq!(builder.block_align, Some(1));
        assert_eq!(builder.block_size, None);
        assert_eq!(builder.capacity, None);

        let builder = Builder::new().alignment(2);
        assert_eq!(builder.block_align, Some(2));
        assert_eq!(builder.block_size, None);
        assert_eq!(builder.capacity, None);

        let builder = Builder::new().alignment(4);
        assert_eq!(builder.block_align, Some(4));
        assert_eq!(builder.block_size, None);
        assert_eq!(builder.capacity, None);

        let builder = Builder::new().alignment(8);
        assert_eq!(builder.block_align, Some(8));
        assert_eq!(builder.block_size, None);
        assert_eq!(builder.capacity, None);
    }

    #[test]
    #[should_panic]
    fn bad_alignment_0() {
        Builder::new().alignment(0);
    }

    #[test]
    #[should_panic]
    fn bad_alignment_3() {
        Builder::new().alignment(3);
    }

    #[test]
    #[should_panic]
    fn bad_alignment_5() {
        Builder::new().alignment(5);
    }

    #[test]
    #[should_panic]
    fn bad_alignment_6() {
        Builder::new().alignment(6);
    }

    #[test]
    #[should_panic]
    fn bad_alignment_7() {
        Builder::new().alignment(7);
    }

    #[test]
    #[should_panic]
    fn multiple_alignment() {
        Builder::new().alignment(1).alignment(2);
    }

    #[test]
    #[should_panic]
    fn build_alignment_only() {
        Builder::new().alignment(1).build();
    }

    #[test]
    fn good_block_size() {
        let builder = Builder::new().block_size(1);
        assert_eq!(builder.block_align, None);
        assert_eq!(builder.block_size, Some(1));
        assert_eq!(builder.capacity, None);

        let builder = Builder::new().block_size(2);
        assert_eq!(builder.block_align, None);
        assert_eq!(builder.block_size, Some(2));
        assert_eq!(builder.capacity, None);

        let builder = Builder::new().block_size(4);
        assert_eq!(builder.block_align, None);
        assert_eq!(builder.block_size, Some(4));
        assert_eq!(builder.capacity, None);

        let builder = Builder::new().block_size(8);
        assert_eq!(builder.block_align, None);
        assert_eq!(builder.block_size, Some(8));
        assert_eq!(builder.capacity, None);
    }

    #[test]
    #[should_panic]
    fn bad_block_size_0() {
        Builder::new().block_size(0);
    }

    #[test]
    #[should_panic]
    fn bad_block_size_3() {
        Builder::new().block_size(3);
    }

    #[test]
    #[should_panic]
    fn bad_block_size_5() {
        Builder::new().block_size(5);
    }

    #[test]
    #[should_panic]
    fn bad_block_size_6() {
        Builder::new().block_size(6);
    }

    #[test]
    #[should_panic]
    fn bad_block_size_7() {
        Builder::new().block_size(7);
    }

    #[test]
    #[should_panic]
    fn multiple_block_sizes() {
        Builder::new().block_size(1).block_size(2);
    }

    #[test]
    #[should_panic]
    fn build_block_size_only() {
        Builder::new().block_size(1).build();
    }

    #[test]
    fn good_superblock_size() {
        let builder = Builder::new().superblock_size(BLOCKS_PER_SUPERBLOCK);
        assert_eq!(builder.block_align, None);
        assert_eq!(builder.block_size, Some(1));
        assert_eq!(builder.capacity, None);

        let builder = Builder::new().superblock_size(2 * BLOCKS_PER_SUPERBLOCK);
        assert_eq!(builder.block_align, None);
        assert_eq!(builder.block_size, Some(2));
        assert_eq!(builder.capacity, None);

        let builder = Builder::new().superblock_size(4 * BLOCKS_PER_SUPERBLOCK);
        assert_eq!(builder.block_align, None);
        assert_eq!(builder.block_size, Some(4));
        assert_eq!(builder.capacity, None);

        let builder = Builder::new().superblock_size(8 * BLOCKS_PER_SUPERBLOCK);
        assert_eq!(builder.block_align, None);
        assert_eq!(builder.block_size, Some(8));
        assert_eq!(builder.capacity, None);
    }

    #[test]
    #[should_panic]
    fn bad_superblock_size_0() {
        Builder::new().superblock_size(0);
    }

    #[test]
    #[should_panic]
    fn bad_superblock_size_half() {
        Builder::new().superblock_size(BLOCKS_PER_SUPERBLOCK / 2);
    }

    #[test]
    #[should_panic]
    fn bad_superblock_size_m1() {
        Builder::new().superblock_size(BLOCKS_PER_SUPERBLOCK.saturating_sub(1));
    }

    #[test]
    #[should_panic]
    fn bad_superblock_size_p1() {
        Builder::new().superblock_size(BLOCKS_PER_SUPERBLOCK + 1);
    }

    #[test]
    #[should_panic]
    fn bad_superblock_size_3x() {
        Builder::new().superblock_size(3 * BLOCKS_PER_SUPERBLOCK);
    }

    #[test]
    #[should_panic]
    fn block_and_superblock_sizes() {
        Builder::new()
            .block_size(1)
            .superblock_size(2 * BLOCKS_PER_SUPERBLOCK);
    }

    #[test]
    #[should_panic]
    fn multiple_superblock_sizes() {
        Builder::new()
            .superblock_size(BLOCKS_PER_SUPERBLOCK)
            .superblock_size(2 * BLOCKS_PER_SUPERBLOCK);
    }

    #[test]
    #[should_panic]
    fn build_superblock_size_only() {
        Builder::new()
            .superblock_size(BLOCKS_PER_SUPERBLOCK)
            .build();
    }

    #[test]
    fn good_capacity() {
        let builder = Builder::new().capacity(1);
        assert_eq!(builder.block_align, None);
        assert_eq!(builder.block_size, None);
        assert_eq!(builder.capacity, Some(1));

        let builder = Builder::new().capacity(2);
        assert_eq!(builder.block_align, None);
        assert_eq!(builder.block_size, None);
        assert_eq!(builder.capacity, Some(2));

        let builder = Builder::new().capacity(3);
        assert_eq!(builder.block_align, None);
        assert_eq!(builder.block_size, None);
        assert_eq!(builder.capacity, Some(3));

        let builder = Builder::new().capacity(4);
        assert_eq!(builder.block_align, None);
        assert_eq!(builder.block_size, None);
        assert_eq!(builder.capacity, Some(4));

        let builder = Builder::new().capacity(5);
        assert_eq!(builder.block_align, None);
        assert_eq!(builder.block_size, None);
        assert_eq!(builder.capacity, Some(5));

        let builder = Builder::new().capacity(6);
        assert_eq!(builder.block_align, None);
        assert_eq!(builder.block_size, None);
        assert_eq!(builder.capacity, Some(6));

        let builder = Builder::new().capacity(std::isize::MAX as usize - 1);
        assert_eq!(builder.block_align, None);
        assert_eq!(builder.block_size, None);
        assert_eq!(builder.capacity, Some(std::isize::MAX as usize - 1));

        let builder = Builder::new().capacity(std::isize::MAX as usize);
        assert_eq!(builder.block_align, None);
        assert_eq!(builder.block_size, None);
        assert_eq!(builder.capacity, Some(std::isize::MAX as usize));
    }

    #[test]
    #[should_panic]
    fn bad_capacity_0() {
        Builder::new().capacity(0);
    }

    #[test]
    #[should_panic]
    fn bad_capacity_p1() {
        Builder::new().capacity(std::isize::MAX as usize + 1);
    }

    #[test]
    #[should_panic]
    fn multiple_capacities() {
        Builder::new().capacity(1).capacity(2);
    }

    #[test]
    #[should_panic]
    fn build_capacity_only() {
        Builder::new().capacity(1).build();
    }

    #[test]
    fn minimal_builds() {
        let builder = Builder::new().block_size(1).capacity(1);
        assert_eq!(builder.block_align, None);
        assert_eq!(builder.block_size, Some(1));
        assert_eq!(builder.capacity, Some(1));
        let allocator = builder.build();
        assert_eq!(allocator.block_size(), 1);
        assert_eq!(allocator.capacity(), BLOCKS_PER_SUPERBLOCK);
        assert_eq!(allocator.block_alignment(), 1);

        let builder = Builder::new()
            .superblock_size(BLOCKS_PER_SUPERBLOCK)
            .capacity(BLOCKS_PER_SUPERBLOCK);
        assert_eq!(builder.block_align, None);
        assert_eq!(builder.block_size, Some(1));
        assert_eq!(builder.capacity, Some(BLOCKS_PER_SUPERBLOCK));
        let allocator = builder.build();
        assert_eq!(allocator.block_size(), 1);
        assert_eq!(allocator.capacity(), BLOCKS_PER_SUPERBLOCK);
        assert_eq!(allocator.block_alignment(), 1);

        let builder = Builder::new().block_size(2).capacity(1);
        assert_eq!(builder.block_align, None);
        assert_eq!(builder.block_size, Some(2));
        assert_eq!(builder.capacity, Some(1));
        let allocator = builder.build();
        assert_eq!(allocator.block_size(), 2);
        assert_eq!(allocator.capacity(), 2 * BLOCKS_PER_SUPERBLOCK);
        assert_eq!(allocator.block_alignment(), 1);

        let builder = Builder::new()
            .block_size(1)
            .capacity(BLOCKS_PER_SUPERBLOCK + 1);
        assert_eq!(builder.block_align, None);
        assert_eq!(builder.block_size, Some(1));
        assert_eq!(builder.capacity, Some(BLOCKS_PER_SUPERBLOCK + 1));
        let allocator = builder.build();
        assert_eq!(allocator.block_size(), 1);
        assert_eq!(allocator.capacity(), 2 * BLOCKS_PER_SUPERBLOCK);
        assert_eq!(allocator.block_alignment(), 1);

        let builder = Builder::new()
            .block_size(4)
            .capacity(8 * BLOCKS_PER_SUPERBLOCK + 1);
        assert_eq!(builder.block_align, None);
        assert_eq!(builder.block_size, Some(4));
        assert_eq!(builder.capacity, Some(8 * BLOCKS_PER_SUPERBLOCK + 1));
        let allocator = builder.build();
        assert_eq!(allocator.block_size(), 4);
        assert_eq!(allocator.capacity(), 12 * BLOCKS_PER_SUPERBLOCK);
        assert_eq!(allocator.block_alignment(), 1);
    }

    #[test]
    #[should_panic]
    fn capacity_overflow() {
        // Capacity will go above std::isize::MAX due to block rounding
        Builder::new()
            .block_size(2)
            .capacity(std::isize::MAX as usize)
            .build();
    }

    #[test]
    fn fully_specified_builds() {
        let builder = Builder::new().alignment(1).block_size(1).capacity(1);
        assert_eq!(builder.block_align, Some(1));
        assert_eq!(builder.block_size, Some(1));
        assert_eq!(builder.capacity, Some(1));
        let allocator = builder.build();
        assert_eq!(allocator.block_size(), 1);
        assert_eq!(allocator.capacity(), BLOCKS_PER_SUPERBLOCK);
        assert_eq!(allocator.block_alignment(), 1);

        let builder = Builder::new().alignment(2).block_size(2).capacity(1);
        assert_eq!(builder.block_align, Some(2));
        assert_eq!(builder.block_size, Some(2));
        assert_eq!(builder.capacity, Some(1));
        let allocator = builder.build();
        assert_eq!(allocator.block_size(), 2);
        assert_eq!(allocator.capacity(), 2 * BLOCKS_PER_SUPERBLOCK);
        assert_eq!(allocator.block_alignment(), 2);

        let builder = Builder::new().alignment(2).block_size(4).capacity(1);
        assert_eq!(builder.block_align, Some(2));
        assert_eq!(builder.block_size, Some(4));
        assert_eq!(builder.capacity, Some(1));
        let allocator = builder.build();
        assert_eq!(allocator.block_size(), 4);
        assert_eq!(allocator.capacity(), 4 * BLOCKS_PER_SUPERBLOCK);
        assert_eq!(allocator.block_alignment(), 2);

        let builder = Builder::new().alignment(2).block_size(8).capacity(1);
        assert_eq!(builder.block_align, Some(2));
        assert_eq!(builder.block_size, Some(8));
        assert_eq!(builder.capacity, Some(1));
        let allocator = builder.build();
        assert_eq!(allocator.block_size(), 8);
        assert_eq!(allocator.capacity(), 8 * BLOCKS_PER_SUPERBLOCK);
        assert_eq!(allocator.block_alignment(), 2);

        let builder = Builder::new().alignment(4).block_size(4).capacity(1);
        assert_eq!(builder.block_align, Some(4));
        assert_eq!(builder.block_size, Some(4));
        assert_eq!(builder.capacity, Some(1));
        let allocator = builder.build();
        assert_eq!(allocator.block_size(), 4);
        assert_eq!(allocator.capacity(), 4 * BLOCKS_PER_SUPERBLOCK);
        assert_eq!(allocator.block_alignment(), 4);

        let builder = Builder::new().alignment(4).block_size(8).capacity(1);
        assert_eq!(builder.block_align, Some(4));
        assert_eq!(builder.block_size, Some(8));
        assert_eq!(builder.capacity, Some(1));
        let allocator = builder.build();
        assert_eq!(allocator.block_size(), 8);
        assert_eq!(allocator.capacity(), 8 * BLOCKS_PER_SUPERBLOCK);
        assert_eq!(allocator.block_alignment(), 4);

        let builder = Builder::new().alignment(8).block_size(8).capacity(1);
        assert_eq!(builder.block_align, Some(8));
        assert_eq!(builder.block_size, Some(8));
        assert_eq!(builder.capacity, Some(1));
        let allocator = builder.build();
        assert_eq!(allocator.block_size(), 8);
        assert_eq!(allocator.capacity(), 8 * BLOCKS_PER_SUPERBLOCK);
        assert_eq!(allocator.block_alignment(), 8);

        let builder = Builder::new()
            .alignment(2)
            .block_size(8)
            .capacity(24 * BLOCKS_PER_SUPERBLOCK);
        assert_eq!(builder.block_align, Some(2));
        assert_eq!(builder.block_size, Some(8));
        assert_eq!(builder.capacity, Some(24 * BLOCKS_PER_SUPERBLOCK));
        let allocator = builder.build();
        assert_eq!(allocator.block_size(), 8);
        assert_eq!(allocator.capacity(), 24 * BLOCKS_PER_SUPERBLOCK);
        assert_eq!(allocator.block_alignment(), 2);
    }

    #[test]
    #[should_panic]
    fn incompatible_block_size_alignment() {
        Builder::new()
            .alignment(2)
            .block_size(1)
            .capacity(1)
            .build();
    }
}
