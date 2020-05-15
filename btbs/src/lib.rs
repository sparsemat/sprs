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
    /// Create a set that can hold values in the range `0..max_entry`.
    pub fn new(max_entry: usize) -> Self {
        let mut bitvecs = Vec::with_capacity(5);
        let mut nbits = 0;
        let mut lvl_bits = 1;
        while nbits < max_entry {
            lvl_bits *= 32;
            let bv = BitVec::from_elem(lvl_bits, false);
            nbits = bv.len();
            bitvecs.push(bv);
        }
        BTreeBitSet {
            max_entry,
            bitvecs,
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

    pub fn max_entry(&self) -> usize {
        self.max_entry
    }

    /// Removes all elements in the set
    pub fn clear(&mut self) {
        for bv in self.bitvecs.iter_mut() {
            bv.clear();
        }
        self.nb_inserted = 0;
    }


    /// Insert a value in the set, returning true is the value was not
    /// already present, false otherwise
    pub fn insert(&mut self, elem: usize) -> bool {
        assert!(elem < self.max_entry);
        let already_present =
            self.bitvecs.last().map(|bv| bv[elem]).unwrap_or(false);
        let mut elem = elem;
        for bv in self.bitvecs.iter_mut().rev() {
            bv.set(elem, true);
            elem /= 32;
        }
        if !already_present {
            self.nb_inserted += 1;
        }
        !already_present
    }

    /// Extracts the elements in the set in sorted order,
    /// clearing the set in the process
    pub fn drain_to_extend_vec(&mut self, vec: &mut Vec<usize>) {
        assert!(self.depth() > 0);
        if vec.capacity() < vec.len() + self.len() {
            vec.reserve(vec.len() + self.len() - vec.capacity());
        }
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
                    vec.push(ind);
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
    }

    /// Return a draining iterator that will remove the elements from the
    /// sets, yielding them on sorted order.
    pub fn drain(&mut self) -> impl Iterator<Item=usize> + '_ {
        assert!(self.depth() > 0);
        self.stack.clear();
        self.stack.push(Location { lvl: 0, start: 0, stop: 32 });
        let depth = self.depth();
        Drain {
            stack: &mut self.stack,
            nb_inserted: &mut self.nb_inserted,
            bitvecs: &mut self.bitvecs,
            depth,
        }
    }
}

struct Drain<'a> {
    stack: &'a mut Vec<Location>,
    nb_inserted: &'a mut usize,
    bitvecs: &'a mut [BitVec],
    depth: usize,
}

impl<'a> Iterator for Drain<'a> {
    type Item = usize;

    fn next(&mut self) -> Option<usize> {
        while let Some(loc) = self.stack.pop() {
            let start = self.stack.len();
            for ind in loc.start..loc.stop {
                let bit = self.bitvecs[loc.lvl].get(ind).unwrap();
                if !bit {
                    continue;
                }
                self.bitvecs[loc.lvl].set(ind, false);
                if loc.lvl + 1 == self.depth {
                    *self.nb_inserted -= 1;
                    self.stack.push(Location {
                        lvl: loc.lvl,
                        start: ind + 1,
                        stop: loc.stop,
                    });
                    return Some(ind);
                } else {
                    self.stack.push(Location {
                        lvl: loc.lvl + 1,
                        start: 32 * ind,
                        stop: 32 * ind + 32,
                    });
                }
            }
            // need to reverse the inserted locations to produce the correct
            // ordering
            self.stack[start..].reverse();
        }
        None
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (*self.nb_inserted, Some(*self.nb_inserted))
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
        let mut elems = Vec::new();
        set.drain_to_extend_vec(&mut elems);
        assert_eq!(&elems[..], &[1, 3, 80, 124, 512, 999, 1000, 1001, 2323]);
        assert_eq!(set.len(), 0);

        set.insert(2324);
        set.insert(3);
        set.insert(0);
        set.insert(81);
        set.insert(524);
        set.insert(999);
        set.insert(124);
        set.insert(1001);
        set.insert(1000);
        let elems: Vec<_> = set.drain().collect();
        assert_eq!(&elems[..], &[0, 3, 81, 124, 524, 999, 1000, 1001, 2324]);
        assert_eq!(set.len(), 0);
    }
}
