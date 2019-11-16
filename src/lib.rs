//! This crate provides a Concurrent Bitmap Allocator, which you can use for
//! your dynamic memory allocation needs in those real-time threads where the
//! system memory allocator should not be used.
//!
//! The implementation is both thread-safe and real-time-friendly: after an
//! RT-unsafe initialization stage, you can use the allocator to allocate and
//! liberate memory from multiple RT threads, and none of them will acquire a
//! lock or call into any RT-unsafe operating system facility in the process.
//!
//! # Bitmap allocation primer
//!
//! A bitmap allocator is a general-purpose memory allocator, in the sense that
//! it allows allocating variable-sized objects and deallocating them one by one
//! in a different order. It works by dividing the buffer of memory that it is
//! managing (its **backing store**) into evenly sized **blocks**, and tracking
//! which blocks are available and used using an array of bits, a **bitmap**.
//!
//! Allocation is done by scanning the bitmap for a suitably large hole
//! (continuous sequence of zeroes), filling that hole with ones, and mapping 
//! its index in the bitmap into a pointer within the backing store.
//! Deallocation is done by mapping back from the user-provided pointer to a
//! range of indices within the bitmap and resetting those bits to zero.
//!
//! The **block size** is the most important tuning parameter of a bitmap
//! allocator, and should be selected wisely:
//!
//! - Because the allocation overhead and bitmap size are proportional to the
//!   number of blocks managed by the allocator, the CPU and memory overhead of
//!   a bitmap allocator will grow as its block size shrinks. From this
//!   perspective, using the highest block size you can get away with is best.
//! - But since allocations are tracked with block granularity, higher block
//!   sizes mean less efficient use of the backing store, as the allocator is
//!   more likely to allocate more memory than the client needs on each request.
//!
//! You should tune your block size based on your full range of envisioned
//! allocation workloads, and even consider instantiating multiple allocators
//! with different block sizes if your allocation workloads vary widely, because
//! a block size that is a good compromise for a given allocation pattern may be
//! a poor choice for another allocation pattern.
//!
//! # Example
//!
//! FIXME: Oh yes I do need those


#[cfg(test)]
mod tests {
    #[test]
    fn it_works() {
        assert_eq!(2 + 2, 4);
    }
}
