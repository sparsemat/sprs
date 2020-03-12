/// A set for unsigned integers backed by a flat array inspired by BTrees
/// and bit sets.
///
/// The structure is designed to be useful for sparse matrix multiplication,
/// the goal is to obtain the sorted set of nonzero indices with a low
/// complexity, taking advantage of a known upper bound. For memory efficiency,
/// a simple bitset would be nice, but collecting the set entries would be
/// linear in the upper bound, and not in the number of entries. To deal with
/// this issue, we adopt a tree-like structure as in a BTree, packed at the
/// bit level.
///
/// Concretely, the set entries are stored in a `Vec<u32>` as follows:
/// - the first entry of the vector divides the range `0..max_entry` in 32
/// equal parts, and each bit of the entry indicates if the corresponding part
/// contains an entry
/// - the next 32 entries further divides each first level part in 32 parts,
/// using the same bit-flagging to indicate wich parts contain entries.
/// - this schema is repeated recursively up to a maximum depth. At the
/// end of the `Vec<u32>` comes a bitset representing the actual entries.
pub struct BTreeBitSet {
    depth: u32,
    max_entry: usize,
    entries: Vec<u32>,
}

impl BTreeBitSet {
    pub fn new(max_entry: usize) -> Self {
        // Each level addresses 2^5 entries more than the previous
        // one.
        let mut depth = 0;
        let mut rem_entries = max_entry;
        let mut vec_size = 1;
        while rem_entries > 32 * 2 {
            rem_entries = rem_entries / 32;
            vec_size += 32 * (depth as usize);
            depth += 1;
            dbg!(depth);
            dbg!(rem_entries);
        }
        vec_size += max_entry / 32;
        if max_entry % 32 > 0 {
            vec_size += 1;
        }
        BTreeBitSet {
            depth,
            max_entry,
            entries: vec![0; vec_size],
        }
    }

    pub fn entries(&self) -> &[u32] {
        &self.entries
    }

    pub fn insert(&mut self, elem: usize) -> bool {
        // TODO
        false
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn construction() {
        let set = super::BTreeBitSet::new(1024);
        assert_eq!(set.entries.len(), 33);

        let set = super::BTreeBitSet::new(1025);
        assert_eq!(set.entries.len(), 34);
        let set = super::BTreeBitSet::new(1057);
        assert_eq!(set.entries.len(), 35);
        let set = super::BTreeBitSet::new(1089);
        assert_eq!(set.entries.len(), 36);
        let set = super::BTreeBitSet::new(8192);
        assert_eq!(set.entries.len(), 33 + 256);
    }
}
