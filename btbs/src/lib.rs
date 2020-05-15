use bit_vec::BitVec;

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
/// Possible optim: take the useful parts from the bitvec library, make them
/// work on slices, and have all our bitvecs inlined in a single vector
pub struct BTreeBitSet {
    max_entry: usize,
    bitvecs: Vec<BitVec>,
    dividers: Vec<usize>,
    nb_inserted: usize,
    stack: Vec<Location>,
}

#[derive(Debug)]
struct Location {
    lvl: usize,
    start: usize,
    stop: usize,
}

impl BTreeBitSet {
    pub fn new(max_entry: usize) -> Self {
        let mut bitvecs = Vec::with_capacity(5);
        let mut dividers = Vec::with_capacity(5);
        let mut nbits = 0;
        let mut lvl_bits = 1;
        while nbits < max_entry {
            dividers.push(lvl_bits);
            lvl_bits *= 32;
            let bv = BitVec::from_elem(lvl_bits, false);
            nbits = bv.len();
            bitvecs.push(bv);
        }
        dividers.reverse();
        BTreeBitSet {
            max_entry,
            bitvecs,
            dividers,
            nb_inserted: 0,
            stack: Vec::with_capacity(32),
        }
    }

    /// The number of elements in the set
    pub fn len(&self) -> usize {
        self.nb_inserted
    }

    pub fn depth(&self) -> usize {
        self.bitvecs.len()
    }

    /// Insert a value in the set, returning true is the value was not
    /// already present, false otherwise
    pub fn insert(&mut self, elem: usize) -> bool {
        assert!(elem < self.max_entry);
        let already_present =
            self.bitvecs.last().map(|bv| bv[elem]).unwrap_or(false);
        for (bv, div) in self.bitvecs.iter_mut().zip(&self.dividers) {
            bv.set(elem / div, true);
        }
        if !already_present {
            self.nb_inserted += 1;
        }
        !already_present
    }

    /// Extracts the elements in the set in sorted order,
    /// clearing the set in the process
    pub fn drain(&mut self) -> Vec<usize> {
        assert!(self.depth() > 0);
        let mut res = Vec::with_capacity(self.len());
        self.stack.clear();
        self.stack.push(Location { lvl: 0, start: 0, stop: 32 });
        while let Some(loc) = self.stack.pop() {
            let start = self.stack.len();
            for ind in loc.start..loc.stop {
                let bit = self.bitvecs[loc.lvl].get(ind).unwrap();
                if !bit {
                    continue;
                }
                if loc.lvl + 1 == self.depth() {
                    res.push(ind);
                    self.nb_inserted -= 1;
                } else {
                    self.stack.push(Location {
                        lvl: loc.lvl + 1,
                        start: ind,
                        stop: 32 * ind + 32,
                    });
                }
                self.bitvecs[loc.lvl].set(ind, false);
            }
            // need to reverse the inserted locations to produce the correct
            // ordering
            self.stack[start..].reverse();
        }
        res
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn construction() {
        let set = super::BTreeBitSet::new(1024);
        assert_eq!(set.depth(), 2);
        let set = super::BTreeBitSet::new(1025);
        assert_eq!(set.depth(), 3);
    }

    #[test]
    fn insertion() {
        let mut set = super::BTreeBitSet::new(1024);
        assert_eq!(set.insert(3), true);
        assert_eq!(set.insert(5), true);
        assert_eq!(set.len(), 2);
        assert_eq!(set.insert(5), false);
        assert_eq!(set.len(), 2);
    }

    #[test]
    fn drain() {
        let mut set = super::BTreeBitSet::new(3217);
        set.insert(2323);
        set.insert(3);
        set.insert(1);
        set.insert(80);
        set.insert(512);
        set.insert(999);
        set.insert(124);
        set.insert(1001);
        set.insert(1000);
        let elems = set.drain();
        assert_eq!(&elems[..], &[1, 3, 80, 124, 512, 999, 1000, 1001, 2323]);
        assert_eq!(set.len(), 0);
    }
}
