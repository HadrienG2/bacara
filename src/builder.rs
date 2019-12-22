//! Mechanism for building an `Allocator`, with proper invariant checking

use crate::Allocator;


/// Builder for a bitmap allocator
//
// NOTE: The main purpose of this builder is to ensure that a certain number of
//       preconditions are upheld upon constructing the allocator. These are
//       listed as "must" bullet points in the struct members' doc comments.
#[derive(Debug)]
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
    /// - This rounding must not overflow usize::MAX
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
    pub fn alignment(&mut self, align: usize) -> &mut Self {
        assert!(align.is_power_of_two(), "Alignment must be a power of 2");
        assert!(self.block_align.replace(align).is_none(),
                "Alignment must only be set once");
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
    /// You must set either the block size and superblock size, but not both.
    pub fn block_size(&mut self, block_size: usize) -> &mut Self {
        assert!(block_size.is_power_of_two(),
                "Block size must be a power of 2");
        assert!(self.block_size.replace(block_size).is_none(),
                "Block size must only be set once");
        self
    }

    /// Set the allocator's superblock size (in bytes)
    ///
    /// The superblock size is an implementation-defined multiple of the block
    /// size, which corresponds to the allocation request size for which the
    /// allocator should exhibit optimal CPU performance.
    ///
    /// The superblock size must be a multiple of the alignment, of
    /// `Allocator::blocks_per_superblock()`, and a power of 2.
    ///
    /// You must set either the block size and superblock size, but not both.
    pub fn superblock_size(&mut self, superblock_size: usize) -> &mut Self {
        assert_eq!(superblock_size % Allocator::blocks_per_superblock(), 0,
                   "Superblock size must be a multiple of \
                    Allocator::blocks_per_superblock()");
        let block_size = superblock_size / Allocator::blocks_per_superblock();
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
    /// the `usize::MAX` limit.
    pub fn capacity(&mut self, capacity: usize) -> &mut Self {
        assert!(capacity != 0, "Backing store capacity must not be zero");
        assert!(self.capacity.replace(capacity).is_none(),
                "Backing store capacity must only be set once");
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
        let block_size = self.block_size
                             .expect("You must specify a block size");
        assert_eq!(block_size % block_align, 0,
                   "Block size must be a multiple of alignment");

        // Round requested capacity to next multiple of superblock size
        let mut capacity =
            self.capacity.expect("You must specify a backing store capacity");
        let superblock_size = block_size * Allocator::blocks_per_superblock();
        let extra_bytes = capacity % superblock_size;
        if extra_bytes != 0 {
            capacity =
                capacity.checked_add(superblock_size - extra_bytes)
                        .expect("Excessive backing store capacity requested");
        }

        // Build the allocator, this is safe because we have checked all the
        // preconditions listed in the Builder struct documentation.
        unsafe { Allocator::new_unchecked(block_align, block_size, capacity) }
    }
}


#[cfg(test)]
mod tests {
    #[test]
    fn it_works() {
        assert_eq!(2 + 2, 4);
    }

    // TODO: Tests tests test absolutely everything as this code is going to be
    //       super-extremely tricky
}
