
use std::sync::atomic::{AtomicI32, AtomicPtr, Ordering};
use std::{ops::Deref, ptr, fmt::Display, marker::PhantomData};
use rand::{distributions::{Uniform, Distribution}};
use utils::RawPtr;

/// A concurrent, lock-free, key-value, value-versioning skiplist.
///
/// This is a key-value-based skiplist, where nodes are sorted by 
/// key. Each value is actually a set of value versions (history),
/// where the latest is the current value, and all others are past
/// values, ordered by write epoch. This value versioning allows the skiplist
/// to perform concurrent writes and reads, where as a read can effectively
/// look at a snapshot of the skiplist at the start of the read. Writes and updates,
/// however, require the latest values to work correctly.
///
/// Concurrent reads and writes are lock-free, meaning no mutex is used. All
/// coordination is done via atomic primitives. This skiplist is not wait free
/// at the moment. If it really needs to be for performance, the adjustments can
/// be made as necessary.
///
/// The skiplist does not really support deletes. It can only create a copy
/// of itself with some chunk cut out. This clone & delete is only allowed on
/// an immutable skiplist, meaning one that no longer allows writes. 
///
/// Reads support different epoch-based read options including: 
/// `Exact`, `Current`, `Latest`. `Exact` allows you to read only up
/// to a manually-specified epoch. `Current` allows you to read only up 
/// to the epoch present at the start of the read. `Latest` let's you read
/// whatever latest value is present on every individual key when that key is read.
/// This option is best for iterating over the list in attempt to update some set of 
/// values. 
pub struct VersionedSkiplist<'a, K: Ord + Clone, V: Ord + Clone> {
    // pointers to the head node of each level, starting with the largest, bottom level
    levels: Vec<AtomicPtr<Node<'a, K, V>>>,
    height: i32,
    // uniform distribution, stored so we don't have the overhead of remaking each time we insert a node
    uniform: Uniform<i32>,
    current_epoch: AtomicI32,
    mutable: bool,
    _phantom: PhantomData<Node<'a, K, V>> // simulate the skiplist owning Nodes
}

/// An iterator over the skiplist. It can only reference the nodes.
/// The iterator supports different epoch-based reads as described in `VersionedSkipList`
pub struct VersionedListIterator<'a, K: Ord + Clone, V: Ord + Clone> {
    epoch_option: EpochOption,
    pinned_epoch: i32, // epoch at the creation of the list
    latest_epoch: RawPtr<AtomicI32>, // pointer to skiplist's epoch atomic counter
    current: RawPtr<Node<'a, K, V>>,
    started: bool,
    finished: bool,
    _phantom: PhantomData<&'a Node<'a, K, V>> // simulate references to Nodes
}

/// A node of the skiplist
///
/// A node includes a key, a value version history, and the next values for each 
/// skiplist level. The skiplist height is set at initialization and does not change,
/// so `nexts` only experiences concurrent reads, not updates.
#[derive(Debug)]
pub struct Node<'a, K: Ord + Clone, V: Ord + Clone> {
    key: K,
    versions: AtomicPtr<Vec<VersionedValue<V>>>,
    nexts: Vec<AtomicPtr<Node<'a, K, V>>>,
    _phantom_versions: PhantomData<VersionedValue<V>>, // simulate owning value versions
    _phantom_nodes: PhantomData<&'a Node<'a, K, V>> // simulate references to other Nodes
}

/// A value and the epoch it was added
#[derive(Debug)]
struct VersionedValue<V: Ord + Clone> {
    epoch: AtomicI32,
    value: V
}

/// Different epoch-based read options. 
///
/// `Exact` allows you to read only up to a manually-specified epoch.
///
/// `Current` allows you to read only up to the epoch present at the start of the 
///read. 
///
/// `Latest` let's you read whatever latest value is present on every individual key when that key is read.
pub enum EpochOption {
    Exact(i32),
    Current,
    Latest
}

/// A search result for a node. 
///
/// Returns either the `Exact` node, if found, or a set of `LeftBounds`
/// for each skiplist level. A leftbound node is a node with the greatest key
/// less than the search key. 
///
/// `LeftBounds` is useful for searching for the location to insert a node on each
/// level when performing a write.
///
/// `LeftBounds` is in reverse order of levels, starting with the highest level.
enum FoundNode<'a, K: Ord + Clone, V: Ord + Clone> {
    LeftBounds(Vec<RawPtr<Node<'a, K, V>>>),
    Exact(RawPtr<Node<'a, K, V>>)
}

/// The same as `FoundNode`, but for a single level. 
/// 
/// All the values correspond to `FoundNode` except the extra `MissingLeftBound`,
/// which denotes that the search key is smaller than the smallest node in the level,
/// and should therefore become the new start of that level.
enum FoundLevelNode<'a, K: Ord + Clone, V: Ord + Clone> {
    MissingLeftBound,
    LeftBound(RawPtr<Node<'a, K, V>>),
    Exact(RawPtr<Node<'a, K, V>>)
}

#[derive(Debug)]
pub struct UpdateError<'a, K: Ord + Clone, V: Ord + Clone> {
    kind: UpdateErrorKind<'a, K, V>
}

#[derive(Debug)]
pub enum UpdateErrorKind<'a, K: Ord + Clone, V: Ord + Clone> {
    DuplicateKey(&'a Node<'a, K, V>),
    MissingKey,
    StaleValue(&'a V)
}

impl<'a, K: Ord + Clone, V: Ord + Clone> From<UpdateErrorKind<'a, K, V>> for UpdateError<'a, K, V> {
    fn from(kind: UpdateErrorKind<'a, K, V>) -> Self {
        UpdateError { kind }
    }
}

impl<'a, K: Ord + Clone, V: Ord + Clone> Display for UpdateError<'a, K, V> {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        let msg = match self.kind {
            UpdateErrorKind::DuplicateKey(node) => "Attempted to insert a key that already exists",
            UpdateErrorKind::MissingKey => "Attempted to update a key that does not exist",
            UpdateErrorKind::StaleValue(_) => "Attempted to update a key based on a stale value"
        };

        write!(f, "{}", msg)
    }
}

impl<'a, K: Ord + Clone, V: Ord + Clone> Deref for UpdateError<'a, K, V> {
    type Target = UpdateErrorKind<'a, K, V>;
    fn deref(&self) -> &Self::Target {
        &self.kind
    }
}

impl<V: Ord + Clone> Clone for VersionedValue<V> {
    fn clone(&self) -> Self {
        // Clone the value itself, and create a new epoch by loading and setting
        VersionedValue { 
            epoch: AtomicI32::new(self.epoch.load(Ordering::SeqCst)), 
            value: self.value.clone() 
        }
    }
}

impl<'a, K: Ord + Clone, V: Ord + Clone> Node<'a, K, V> {
    /// Creates a new node, specifying the max skiplist height
    pub fn new(key: K, value: V, height: i32) -> Node<'a, K, V> {
        let mut nexts: Vec<AtomicPtr<Node<K, V>>> = Vec::with_capacity(height as usize);

        // Push a null atomic pointer at each level
        // This is basically an alternative to Option that allows us to stay atomic
        for _ in 0..height {
            nexts.push(AtomicPtr::new(ptr::null_mut()));
        }

        // Allocate value versions on the heap. Start the epoch at -1, indicating 
        // that a value corresponding to the latest epoch could exist, but needs to be set first.
        // We have to add a node, and only then set and increment the epoch, since otherwise 
        // we could attempt to read the latest epoch, but miss this node as it hasn't been added yet. 
        // So instead we add it with an epoch of -1, add the node, then increment the global epoch and set
        // the true epoch on the node. This way, the node will always be present in the list, and therefore
        // always readable as soon as it's epoch is globally exposed. 
        let versions = Box::new(
            vec![VersionedValue { epoch: AtomicI32::from(-1), value }]
        );

        // move ownership of the versions pointer out of the box
        let versions_ptr = Box::into_raw(versions);

        // Node "owns" the versions pointer now, and effectively the versions themselves
        Node {
            key,
            versions: AtomicPtr::new(versions_ptr),
            nexts,
            _phantom_versions: PhantomData,
            _phantom_nodes: PhantomData
        }
    }
}

impl<'a, K: Ord + Clone, V: Ord + Clone> Clone for Node<'a, K, V> {
    fn clone(&self) -> Self {
        // load and clone the versions
        let cur_versions = unsafe { (*(self.versions.load(Ordering::SeqCst))).clone() };
        
        // allocate on the heap and take ownership from the Box
        let cur_versions_ptr = Box::into_raw(Box::new(cur_versions));

        // nexts just have to clone pointers, so we'll just load each pointer and create a new atomic 
        // pointer from it
        Node { 
            key: self.key.clone(),
            versions: AtomicPtr::new(cur_versions_ptr),
            nexts: self.nexts.iter().map(|n| AtomicPtr::new(n.load(Ordering::SeqCst))).collect(),
            _phantom_versions: PhantomData,
            _phantom_nodes: PhantomData
        }
    }
}

impl<'a, K: Ord + Clone, V: Ord + Clone> Drop for Node<'a, K, V> {
    fn drop(&mut self) {
        // deallocate versions and call drop on the versions Vec and it's contents
        let versions: RawPtr<Vec<VersionedValue<V>>> = self.versions.load(Ordering::SeqCst).into();
        unsafe {
            versions.dealloc();
        }
    }
}

impl<'a, K: Ord + Clone, V: Ord + Clone> VersionedSkiplist<'a, K, V> {
    /// Creates a new skiplist with a set height
    pub fn new(max_height: i32) -> VersionedSkiplist<'a, K, V> {
        let mut levels: Vec<AtomicPtr<Node<K, V>>> = Vec::with_capacity(max_height as usize);
        
        // with_capacity is more efficient, since it allocates an array of the correct size,
        // but len() of the Vec will still be set to 0. So we push null atomic pointers
        // as a Option::None replacement
        for _ in 0..max_height {
            levels.push(AtomicPtr::new(ptr::null_mut()));
        }

        VersionedSkiplist {
            levels,
            height: max_height,
            uniform: Uniform::new(0, 2), // uniform distribution selecting 0 or 1
            current_epoch: AtomicI32::new(0),
            mutable: true,
            _phantom: PhantomData
        }
    }

    /// Clones the entire skiplist with a specified range excluded
    ///
    /// There are some conditions to this that are critical for the user to understand.
    /// This can only run on an immutable list. Trying to clone a mutable skiplist will fail.
    /// The result is always immutable.
    pub fn clone_without_range(&self, range_start: K, range_end: K) -> Self {
        // Note that nothing will be touching the new skiplist until it's returned.
        // Since this method is single-threaded and nothing will interact with the new skiplist
        // while it's mid-creation, we don't have to worry about thread-safety. This means
        // our gets/sets can be more efficient than normal skiplist gets/sets because they don't
        // have to load/store atomically, they can just assign atomic values instead.

        let mut new_list = VersionedSkiplist::new(self.height);
        let mut node: RawPtr<Node<K, V>> = self.levels[0].load(Ordering::SeqCst).into(); // head node

        // find the correct start node based on range_start
        // keep loading the next node until either the node is null or
        // the node >= range_start
        while node.key < range_start && !node.is_null() {
            node = node.nexts[0].load(Ordering::SeqCst).into();
        }
        
        // the node is null if the list is empty or range_start > max(skiplist) 
        if node.is_null() {
            new_list // return an empty list
        } else {
            // we actually have some work to do...
            // clone the start node and save a pointer to it
            let mut new_node: RawPtr<Node<K, V>> = Box::into_raw(Box::new((*node).clone())).into();
            
            // grab the next node from the original list
            // next_node will always be a pointer to the node in the original list, not the new list
            let mut next_node: RawPtr<Node<K, V>> = node.nexts[0].load(Ordering::SeqCst).into();
            new_list.levels[0] = AtomicPtr::new(new_node.into_raw()); // init the head of the new skiplist

            // keep cloning and adding nodes to the base level while
            // the next_node is within the range and non-null
            while next_node.key < range_end && !next_node.is_null() {
                // clone the next node and save a pointer to it
                let next_new_node: RawPtr<Node<K, V>> = Box::into_raw(Box::new((*next_node).clone())).into();
                new_node.nexts[0] = AtomicPtr::new(next_new_node.into_raw()); // set it as next for the current new_node
                new_node = next_new_node; // make the current new_node the next node that we just cloned
                next_node = new_node.nexts[0].load(Ordering::SeqCst).into(); // load the next node from the original skiplist
            }
            
            // By this point we've finished constructing the base layer, now
            // we need to actually connect pointers in all the other layers

            // The last new_node is the tail, but since it's a cut from the original list, it
            // could have pointers to the original list's nodes. Look at how a Node is cloned to 
            // see why. We want to break this connection by assigning null atomic pointers to all the levels
            // of this tail node.
            new_node.nexts.iter_mut().for_each(|n| *n = AtomicPtr::new(ptr::null_mut()));
            
            // We could assign nodes by using the insert_into_level method that we use for a regular insert.
            // However, this isn't as efficient as possible since we know we're always going to be appending here,
            // whereas insert_into_level cannot make that assumption. As a result, using insert_into_level 
            // requires more traversal. Here, we store the left bound for each level. Since we know the next
            // node we traverse will be greater than this left bound, if we decide to add the node to a level,
            // we can just do so by setting the next of the left bound, and then overwriting this left bound
            // to be this new node.

            // Init the left bounds 
            let mut left_bounds: Vec<RawPtr<Node<K, V>>> = Vec::with_capacity(self.height as usize);
            for _ in 0..left_bounds.capacity() {
                left_bounds.push(RawPtr::null());
            }

            // Grab the head. We don't really have to do a load here, sort of a byproduct of the lock-free-ness
            // we depend on in the rest of the algorithms
            let mut cur_node: RawPtr<Node<K, V>> = new_list.levels[0].load(Ordering::SeqCst).into();
            
            // go through all the nodes in order
            while !cur_node.is_null() {
                // for each node, go through all the levels, starting from the base
                for (level_idx, _) in cur_node.nexts.iter().enumerate() {

                    // base level is already set up, so go to the next level
                    if level_idx == 0 {
                        continue;
                    }

                    let sample = self.uniform.sample(&mut rand::thread_rng());

                    // if the node won't get added to more levels, set the remaining levels
                    // of this node to be null pointers and exit the loop for this node
                    if sample == 0 {
                        for idx in level_idx..new_list.levels.len() {
                            cur_node.nexts[idx] = AtomicPtr::new(ptr::null_mut());
                        }

                        break;
                    }

                    // otherwise the node is getting added to this level...
                    // if the level already has a left bound...
                    if !left_bounds[level_idx].is_null() {
                        // set that left bound's next to be this node on this level
                        left_bounds[level_idx].nexts[level_idx] = AtomicPtr::new(cur_node.into_raw());
                    } else {
                        // otherwise add it as the head of the level
                        new_list.levels[level_idx] = AtomicPtr::new(cur_node.into_raw());
                    }

                    // update the left bound at this level to be the current node
                    left_bounds[level_idx] = cur_node;
                }

                // get the next node
                cur_node = cur_node.nexts[0].load(Ordering::SeqCst).into();            
            }

            new_list
        }
    }

    /// Finds a value in the versions Vec that has the highest epoch that's less than 
    /// the request max epoch. Essentially, this is the last version we're allowed to read.
    /// This is effectively a binary search over version epochs
    fn find_versioned_value(node: &Node<K, V>, max_epoch: i32) -> Option<RawPtr<V>> {
        let versions = RawPtr::from(node.versions.load(Ordering::SeqCst));
        let total = versions.len();
        let mut idx: usize = total / 2; // start in the middle
        let mut best_idx: i32 = -1;
        let mut best_epoch: i32 = -1;
        let mut lower_bound = 0; // start at the beginning of the Vec
        let mut upper_bound = total - 1; // end at the end of the Vec

        loop {
           let epoch = versions[idx].epoch.load(Ordering::SeqCst);

           // found an exact epoch match, so just return the value here
            if epoch == max_epoch {
                break Some((&versions[idx].value).into());
            }

            // This happens when a write/update just happened and there's a slight
            // delay between the Node being written and setting it's update/write epoch.
            // We don't know what this is going to be, so we loop again until it's a valid epoch.
            // Don't update the idx to try this epoch again.
            if epoch == -1 {
                continue;
            }

            // this is a valid epoch and better than the best epoch we've found so far
            if epoch < max_epoch && epoch > best_epoch {
                best_epoch = epoch;
                best_idx = idx as i32;
                lower_bound = idx; // the epoch could be in the upper interval...
            }

            // if it's not a valid epoch, just this idx as the upper bound
            if epoch > max_epoch {
                upper_bound = idx;
            }
            
            // find the middle of the bounds to continue the binary search
            idx = (lower_bound + upper_bound) / 2;

            // if the idx is one of the bounds, we've finished our search
            if idx == upper_bound || idx == lower_bound {
                break if best_epoch == -1 {
                    None
                } else {
                    Some((&versions[best_idx as usize].value).into())
                };
            }
        }
    }

    /// Consumes the current skiplist, returning an immutable one. It will have the
    /// immutable flag set to true and will no longer allow writes/updates. This is
    /// irreversible.
    pub fn to_immutable(mut self) -> Self {
        self.mutable = false;
        self
    }

    /// Creates an iterator over the skiplist
    pub fn iter(&self, epoch_option: EpochOption) -> VersionedListIterator<'a, K, V> {
        VersionedListIterator {
            epoch_option,
            pinned_epoch: self.current_epoch.load(Ordering::SeqCst),
            latest_epoch: (&self.current_epoch).into(),
            current: self.levels[0].load(Ordering::SeqCst).into(),
            started: false,
            finished: false,
            _phantom: PhantomData
        }
    }

    /// Attempts to insert a key into the skiplist if the key doesn't exist yet.
    pub fn insert(&self, key: K, value: V) -> Result<&'a Node<'a, K, V>, UpdateError<'a, K, V>> {
        let found_result = self.find(&key); 
        let new_node = Box::new(Node::new(key, value, self.height));
        let new_node_ptr = Box::into_raw(new_node);
        let mut status = unsafe { Ok(&*new_node_ptr) };
        match found_result {
            // Node doesn't exist yet, add it to each level using the left bound as a start  
            FoundNode::LeftBounds(bounds) => {
                // Reverse the bounds so they're in the same order as levels (bottom -> top)
                // The crucial part here is atomically inserting into the bottom level, that is,
                // the source-of-truth level. Once that succeeds, the list doesn't have to stay locked
                // until all the levels are linked up. The node already exists; the remaining levels
                // are just for probabilistically improving search time. If they're not linked yet,
                // search time will suffer, but we don't lose any correctness guarantees. So we link
                // the bottom level first, then keep going.
                for (level_idx, &bound_node_ptr) in bounds.iter().rev().enumerate() {
                    let sample = self.uniform.sample(&mut rand::thread_rng());

                    // always insert into the first level, then insert based on probability
                    if level_idx > 0 && sample == 0 {
                        break;
                    }
                    
                    // Insert into the level using our best guess of a left bound. This can
                    // error out if a duplicate key was inserted after we ran find(), but before
                    // we started the insert. Since keys are always inserted at the bottom level,
                    // an error is only possible for the bottom level, so there won't be error traces
                    // laying around in other levels.
                    if let Err(err) = self.insert_into_level(new_node_ptr.into(), bound_node_ptr, level_idx) {
                        status = Err(err);
                        break;
                    }
    
                    // If this is the first level, we just performed the "true" insert
                    // The epoch defaults to -1, so anything reading it has to wait for this
                    // piece to complete. Here we update the global epoch, and set the node's
                    // epoch to be that. 
                    if level_idx == 0 {
                        let last_epoch = self.current_epoch.fetch_add(1, Ordering::SeqCst);
                        unsafe {
                            let versions = (*new_node_ptr).versions.load(Ordering::SeqCst);
                            (*versions)[0].epoch.store(last_epoch + 1, Ordering::SeqCst);
                        }
                    }
                }
            },
            // Node already exists...
            FoundNode::Exact(node) => status = unsafe {
                Err(UpdateErrorKind::DuplicateKey(&*(node.into_raw())).into())
            }
        };

       status
    }

    /// Checks if the key already exists.
    pub fn contains(&self, key: K) -> bool {
        // just a convenience wrapper over find()
        match self.find(&key) {
            FoundNode::Exact(_) => true,
            _ => false
        }
    }

    /// Returns a reference to a node with a specific key.
    /// This does not do any epoch checking. It will return 
    /// if a key is present by the time it finds it in the skiplist.
    /// The user can record the epoch and check the value versions manually
    /// if needed.
    pub fn get_node(&self, key: K) -> Option<&Node<'a, K, V>> {
        // also a convenience wrapper over find()
        match self.find(&key) {
            FoundNode::Exact(node) => unsafe { Some(&*(node.into_raw())) },
            _ => None
        }
    }

    /// Returns a value for a key given an epoch option
    pub fn get_value(&self, key: K, epoch_option: EpochOption) -> Option<&V> {
        // essentially a find() followed by a find_versioned_value()
        let mut current_epoch = -1;
        if let EpochOption::Current = epoch_option {
            current_epoch = self.current_epoch.load(Ordering::SeqCst);
        };

        match self.find(&key) {
            FoundNode::Exact(node_ptr) => {
                unsafe {
                    let node = &*node_ptr.into_raw();
                    match epoch_option {
                        EpochOption::Exact(epoch) => {
                            // use requested epoch
                            let value = &*(VersionedSkiplist::find_versioned_value(&node, epoch)?.into_raw());
                            Some(value)
                        },
                        EpochOption::Latest => {
                            // grab the last value from the versions Vec
                            let latest_value = (*(node.versions.load(Ordering::SeqCst))).last()?;
                            Some(&latest_value.value)
                        },
                        EpochOption::Current => {
                            // use the global epoch at the start of the method
                            let value = &*(VersionedSkiplist::find_versioned_value(&node, current_epoch)?.into_raw());
                            Some(value)
                        }
                    }
                }
            },
            _ => None
        }
    }

    /// Attempts to update a node with a given key. We cannot just force `update()`
    /// to loop until it updates the versions Vec. We can do that with `insert()` because
    /// the only uncertainty there is whether the key already exists or not.
    /// With update, the user has to be actively aware of the update they're proposing. 
    /// If the latest value of the versions is the old value that the user expects, `update()`
    /// will commit the proposed new value. If, however, the old value has since changed, `update()`
    /// will return an error with the new latest value so the user can explicitly decide how to proceed.
    ///
    /// An update cannot be forced like an insert because invalid updates may occur when the user 
    /// doesn't expect, something not possible when inserting.
    pub fn update(&self, key: K, new_value: V, old_value: V) -> Result<&'a Node<K, V>, UpdateError<'a, K, V>> {
        // Since a Vec isn't thread-safe, we can't just concurrently push a value to it.
        // Instead, we load the current versions Vec, push to it, and then try to swap an atomic pointer
        // to the new versions Vec in
        if let Some(node) = self.get_node(key) {
            let initial_versions = RawPtr::from(node.versions.load(Ordering::SeqCst));
            let initial_last_value = initial_versions.last().unwrap();
            loop {
                // same as an insert, wait for the epoch to be valid before checking the value
                if initial_last_value.epoch.load(Ordering::SeqCst) == -1 {
                    continue;
                } else if initial_last_value.value == old_value {
                    // update is valid since the old_value matches the actual last value
                    let mut new_versions = (*initial_versions).clone();
                    let mut new_versioned_value = VersionedValue { 
                        epoch: AtomicI32::new(-1), 
                        value: new_value 
                    };

                    let new_versioned_value_ptr = &mut new_versioned_value as *mut VersionedValue<V>;
                    new_versions.push(new_versioned_value);
        
                    let new_versions_ptr = Box::into_raw(Box::new(new_versions));
                    let pre_swap_versions = node.versions.compare_and_swap(initial_versions.into_raw(), new_versions_ptr, Ordering::SeqCst);
                    
                    unsafe {
                        // the compare and swap for the new versions pointer succeeded...
                        if pre_swap_versions == new_versions_ptr {
                            initial_versions.dealloc(); // dealloc the previous versions Vec
                            // same as an insert, increment the global epoch and store a valid epoch in the updated version
                            let last_epoch = self.current_epoch.fetch_add(1, Ordering:: SeqCst);
                            (*new_versioned_value_ptr).epoch.store(last_epoch + 1, Ordering::SeqCst);
                            break Ok(node);
                        } else {
                            // compare and swap failed, return an error with the current last version value
                            let last_value = &((*pre_swap_versions).last().unwrap().value);
                            RawPtr::from(new_versions_ptr).dealloc(); // dealloc the versions Vec we tried to swap in
                            break Err(UpdateErrorKind::StaleValue(last_value.into()).into());
                        }
                    }
                } else {
                    // the last version value does not match the expected old_value
                    unsafe {
                        let val = &initial_last_value.value as *const V as *mut V;
                        break Err(UpdateErrorKind::StaleValue(&*val).into());
                    }
                }
            }
        } else {
            Err(UpdateErrorKind::MissingKey.into())
        }
    }

    /// Inserts a node into a specified level using a suggested starting node to find the
    /// correct left bound.
    fn insert_into_level(&self, mut new_node_ptr: RawPtr<Node<'a, K, V>>, start_node: RawPtr<Node<'a, K, V>>, level_idx: usize) -> Result<(), UpdateError<'a, K, V>> {
        let mut left_bound = start_node;
        let mut cur_node: RawPtr<Node<K, V>> = start_node;
        loop {
            let cur_ptr = if left_bound.is_null() {
                &self.levels[level_idx]
            } else {
                &left_bound.nexts[level_idx]
            };

            cur_node = cur_ptr.load(Ordering::SeqCst).into();
            if cur_node.is_null() || cur_node.key > new_node_ptr.key {
                new_node_ptr.nexts[level_idx] = AtomicPtr::new(cur_node.into_raw());
                let last_updated_val = cur_ptr.compare_and_swap(cur_node.into_raw(), new_node_ptr.into_raw(), Ordering::SeqCst);
                if cur_node.into_raw() == last_updated_val {
                    break Ok(());
                }

                if start_node.is_null() {
                    left_bound = last_updated_val.into();
                }
            } else if cur_node.key < new_node_ptr.key {
                left_bound = cur_node;
            } else {
                unsafe {
                    break Err(UpdateErrorKind::DuplicateKey(&*(cur_node.into_raw())).into());
                }
            }
        }
    }

    fn find(&self, key: &K) -> FoundNode<'a, K, V> {
        let mut cur_left_bound_node: Option<RawPtr<Node<K, V>>> = None;
        let mut left_bound_nodes: Vec<RawPtr<Node<K, V>>> = Vec::with_capacity(self.height as usize);
        let mut exact_node: Option<FoundNode<'a, K, V>> = None;
        
        for (level_idx, _) in self.levels.iter().enumerate().rev(){
            let best_start_node = if let Some(node) = cur_left_bound_node {
                node
            } else {
                self.levels[level_idx].load(Ordering::SeqCst).into()
            };

            let found_node = self.find_on_level(&key, best_start_node, level_idx);
            if let FoundLevelNode::Exact(node) = found_node {
                exact_node = Some(FoundNode::Exact(node));
                break;
            }

            if let FoundLevelNode::LeftBound(node) = found_node {
                cur_left_bound_node = Some(node);
                left_bound_nodes.push(node);
            } else {
                cur_left_bound_node = None;
                left_bound_nodes.push(RawPtr::null());
            }
        }

        if let Some(node) = exact_node {
            node
        } else {
            FoundNode::LeftBounds(left_bound_nodes)
        }
    }

    fn find_on_level(&self, key: &K, start_node: RawPtr<Node<'a, K, V>>, level_idx: usize) -> FoundLevelNode<'a, K, V> {
        let mut cur_node = start_node;
        let mut left_bound = FoundLevelNode::MissingLeftBound;

        loop {
            if cur_node.is_null() {
                break left_bound
            } else if cur_node.key == *key {
                break FoundLevelNode::Exact(cur_node);
            } else if cur_node.key < *key {
                left_bound = FoundLevelNode::LeftBound(cur_node);
                cur_node = cur_node.nexts[level_idx].load(Ordering::SeqCst).into();
            } else {
                break left_bound;
            }
        }
    }
}

impl<'a, K: Ord + Clone, V: Ord + Clone> Drop for VersionedSkiplist<'a, K, V> {
    fn drop(&mut self) {
        let mut node: RawPtr<Node<K, V>> = self.levels[0].load(Ordering::SeqCst).into();
        while !node.is_null() {
            let next_node = node.nexts[0].load(Ordering::SeqCst);
            unsafe {
                node.dealloc();
            }

            node = next_node.into();
        }
    }
}

impl<'a, K: Ord + Clone, V: Ord + Clone> Iterator for VersionedListIterator<'a, K, V> {
    type Item = RawPtr<V>;
    fn next(&mut self) -> Option<Self::Item> {
        if self.finished {
            None
        } else {
            if self.started {
                self.current = self.current.nexts[0].load(Ordering::SeqCst).into();
            }
    
            if !self.started {
                self.started = true;
            }

            loop {
                if self.current.is_null() {
                    self.finished = true;
                    break None;
                }

                let found_value = match self.epoch_option {
                    EpochOption::Exact(epoch) => VersionedSkiplist::find_versioned_value(&*self.current, epoch),
                    EpochOption::Latest => VersionedSkiplist::find_versioned_value(&*self.current, self.latest_epoch.load(Ordering::SeqCst)),
                    EpochOption::Current => VersionedSkiplist::find_versioned_value(&*self.current, self.pinned_epoch)
                };

                if let Some(value) = found_value {
                    break Some(value);
                }

                self.current = self.current.nexts[0].load(Ordering::SeqCst).into();
            }
        }
    }
}