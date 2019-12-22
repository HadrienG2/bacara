//! Mechanisms for managing block allocation masks within a superblock
//!
//! Since modern CPUs are not bit-addressable, unsigned integers must be used
//! as a substitute for vector of bits. This module provides an abstraction to
//! ease correct manipulation of such homegrown bitfields in the standard use
//! case of allocating a contiguous chain of blocks.

use crate::Allocator;


/// Mask for allocating a sequence of blocks within a superblock
///
/// Will contain a superblock bitmask of the form 0b001111110000..., which can
/// be used for targeting a subset of blocks within a superblock for the purpose
/// of allocating and deallocating them without touching the rest.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct AllocationMask(usize);

impl AllocationMask {
    /// Compute an allocation mask given the index of the first bit that should
    /// be 1 (allocated) and the number of bits that should be 1.
    pub fn new(start: usize, len: usize) -> Self {
        // Check interface preconditions in debug builds
        debug_assert!(start < Allocator::blocks_per_superblock(),
                      "Allocation start is out of superblock range");
        debug_assert!(len < (Allocator::blocks_per_superblock() - start),
                      "Allocation end is out of superblock range");

        // Handle the "full superblock" edge case without overflowing
        if len == Allocator::blocks_per_superblock() {
            return Self(std::usize::MAX);
        }

        // Otherwise, use a general bit pattern computation
        Self(((1 << len) - 1) << start)
    }

    /// Index of the first allocated block
    pub fn start(&self) -> usize {
        self.0.trailing_zeros() as usize
    }

    /// Number of allocated blocks
    pub fn len(&self) -> usize {
        self.0.count_ones() as usize
    }

    /// Index after the last allocated block (start + len == end)
    pub fn end(&self) -> usize {
        Allocator::blocks_per_superblock() - self.0.leading_zeros() as usize
    }

    /// Truth that allocation mask is empty (has no allocated block)
    pub fn empty(&self) -> bool {
        self.0 == 0
    }

    /// Truth that allocation mask is full (all blocks are allocated)
    pub fn full(&self) -> bool {
        self.0 == std::usize::MAX
    }
}

impl From<AllocationMask> for usize {
    fn from(x: AllocationMask) -> usize {
        x.0
    }
}