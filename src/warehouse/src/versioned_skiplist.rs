
use std::sync::atomic::{AtomicI32, AtomicPtr, Ordering};
use std::{ops::Deref, ptr, fmt::Display, marker::PhantomData};
use rand::{distributions::{Uniform, Distribution}};
use utils::RawPtr;

pub struct VersionedSkiplist<'a, K: Ord + Clone, V: Ord + Clone> {
    levels: Vec<AtomicPtr<Node<'a, K, V>>>,
    max_height: i32,
    uniform: Uniform<i32>,
    current_epoch: AtomicI32,
    mutable: bool,
    _phantom: PhantomData<&'a Node<'a, K, V>>
}

pub struct VersionedListIterator<'a, K: Ord + Clone, V: Ord + Clone> {
    epoch_option: EpochOption,
    pinned_epoch: i32,
    latest_epoch: RawPtr<AtomicI32>,
    current: RawPtr<Node<'a, K, V>>,
    started: bool,
    finished: bool,
    _phantom: PhantomData<&'a Node<'a, K, V>>
}

#[derive(Debug)]
pub struct Node<'a, K: Ord + Clone, V: Ord + Clone> {
    key: K,
    versions: AtomicPtr<Vec<VersionedValue<V>>>,
    nexts: Vec<AtomicPtr<Node<'a, K, V>>>,
    _phantom_versions: PhantomData<&'a VersionedValue<V>>,
    _phantom_nodes: PhantomData<&'a Node<'a, K, V>>
}

#[derive(Debug)]
struct VersionedValue<V: Ord + Clone> {
    epoch: AtomicI32,
    value: V
}

pub enum EpochOption {
    Exact(i32),
    Current,
    Latest
}

enum FoundNode<'a, K: Ord + Clone, V: Ord + Clone> {
    LeftBounds(Vec<RawPtr<Node<'a, K, V>>>),
    Exact(RawPtr<Node<'a, K, V>>)
}

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
        VersionedValue { 
            epoch: AtomicI32::new(self.epoch.load(Ordering::SeqCst)), 
            value: self.value.clone() 
        }
    }
}

impl<'a, K: Ord + Clone, V: Ord + Clone> Node<'a, K, V> {
    pub fn new(key: K, value: V, max_height: i32) -> Node<'a, K, V> {
        let mut nexts: Vec<AtomicPtr<Node<K, V>>> = Vec::with_capacity(max_height as usize);
        for _ in 0..max_height {
            nexts.push(AtomicPtr::new(ptr::null_mut()));
        }

        let versions = Box::new(vec![VersionedValue { epoch: AtomicI32::from(-1), value }]);
        let versions_ptr = Box::into_raw(versions);

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
        let cur_versions = unsafe { (*(self.versions.load(Ordering::SeqCst))).clone() };
        let cur_versions_ptr = Box::into_raw(Box::new(cur_versions));

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
        let versions: RawPtr<Vec<VersionedValue<V>>> = self.versions.load(Ordering::SeqCst).into();
        unsafe {
            versions.dealloc();
        }
    }
}

impl<'a, K: Ord + Clone, V: Ord + Clone> Clone for VersionedSkiplist<'a, K, V> {
    fn clone(&self) -> Self {
        VersionedSkiplist {
            levels: self.levels.iter().map(|n| {
                let cloned_node = unsafe { (*(n.load(Ordering::SeqCst))).clone() };
                let cloned_node_ptr = Box::into_raw(Box::new(cloned_node));
                AtomicPtr::new(cloned_node_ptr)
            })
            .collect(),
            max_height: self.max_height,
            uniform: self.uniform,
            current_epoch: AtomicI32::new(self.current_epoch.load(Ordering::SeqCst)),
            mutable: self.mutable,
            _phantom: PhantomData
        }
    }
}

impl<'a, K: Ord + Clone, V: Ord + Clone> VersionedSkiplist<'a, K, V> {
    pub fn new(max_height: i32) -> VersionedSkiplist<'a, K, V> {
        let mut levels: Vec<AtomicPtr<Node<K, V>>> = Vec::with_capacity(max_height as usize);
        for i in 0..max_height {
            levels.push(AtomicPtr::new(ptr::null_mut()));
        }

        VersionedSkiplist {
            levels,
            max_height,
            uniform: Uniform::new(0, 2),
            current_epoch: AtomicI32::new(0),
            mutable: true,
            _phantom: PhantomData
        }
    }

    pub fn copy_without_range(&self, range_start: K, range_end: K) -> Self {
        let new_list = self.clone();
        for (level_idx, node) in new_list.levels.iter().enumerate().rev() {
            self.reassign_viable_next(
                node.load(Ordering::SeqCst).into(), 
                &range_start,
                &range_end, 
                level_idx, 
                level_idx == 0
            );
        }

        new_list
    }

    fn reassign_viable_next(
        &self, node: RawPtr<Node<K, V>>, 
        range_start: &K, 
        range_end: &K, 
        level_idx: usize, 
        remove_invalid_node: bool
    ) {
        let mut next_node: RawPtr<Node<K, V>> = node.nexts[level_idx].load(Ordering::SeqCst).into();
        while !next_node.is_null() && next_node.key > *range_start && next_node.key < *range_end {
            let invalid_node = next_node;
            next_node = next_node.nexts[level_idx].load(Ordering::SeqCst).into();
            if remove_invalid_node {
                unsafe {
                    invalid_node.dealloc();
                }
            }
        }

        node.nexts[level_idx].store(next_node.into_raw(), Ordering::SeqCst);
    }

    fn find_versioned_value(node: &Node<K, V>, max_epoch: i32) -> Option<RawPtr<V>> {
        let versions = RawPtr::from(node.versions.load(Ordering::SeqCst));
        let total = versions.len();
        let mut idx: usize = total / 2;
        let mut best_idx: i32 = -1;
        let mut best_epoch: i32 = -1;
        let mut lower_bound = 0;
        let mut upper_bound = total - 1;

        loop {
           let epoch = versions[idx].epoch.load(Ordering::SeqCst);
            if epoch == max_epoch {
                break Some((&versions[idx].value).into());
            }

            if epoch == -1 {
                continue;
            }

            if epoch < max_epoch && epoch > best_epoch {
                best_epoch = epoch;
                best_idx = idx as i32;
                lower_bound = idx;
            }

            if epoch > max_epoch {
                upper_bound = idx;
            }
            
            idx = (lower_bound + upper_bound) / 2;
            if idx == upper_bound || idx == lower_bound {
                break if best_epoch == -1 {
                    None
                } else {
                    Some((&versions[best_idx as usize].value).into())
                };
            }
        }
    }

    pub fn to_immutable(mut self) -> Self {
        self.mutable = false;
        self
    }

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

    pub fn insert(&self, key: K, value: V) -> Result<&'a Node<'a, K, V>, UpdateError<'a, K, V>> {
        let found_result = self.find(&key);
        let new_node = Box::new(Node::new(key, value, self.max_height));
        let new_node_ptr = Box::into_raw(new_node);
        let mut status = unsafe { Ok(&*new_node_ptr) };
        match found_result {
            FoundNode::LeftBounds(bounds) => {
                for (level_idx, &bound_node_ptr) in bounds.iter().rev().enumerate() {
                    let sample = self.uniform.sample(&mut rand::thread_rng());
                    if level_idx > 0 && sample == 0 {
                        break;
                    }
    
                    if let Err(err) = self.insert_into_level(new_node_ptr.into(), bound_node_ptr, level_idx) {
                        status = Err(err);
                        break;
                    }
    
                    if level_idx == 0 {
                        let last_epoch = self.current_epoch.fetch_add(1, Ordering::SeqCst);
                        unsafe {
                            let versions = (*new_node_ptr).versions.load(Ordering::SeqCst);
                            (*versions)[0].epoch.store(last_epoch + 1, Ordering::SeqCst);
                        }
                    }
                }
            },
            FoundNode::Exact(node) => status = unsafe {
                Err(UpdateErrorKind::DuplicateKey(&*(node.into_raw())).into())
            }
        };

       status
    }

    pub fn contains(&self, key: K) -> bool {
        match self.find(&key) {
            FoundNode::Exact(_) => true,
            _ => false
        }
    }

    pub fn get_node(&self, key: K) -> Option<&Node<'a, K, V>> {
        match self.find(&key) {
            FoundNode::Exact(node) => unsafe { Some(&*(node.into_raw())) },
            _ => None
        }
    }

    pub fn get_value(&self, key: K, epoch_option: EpochOption) -> Option<&V> {
        match self.find(&key) {
            FoundNode::Exact(node_ptr) => {
                unsafe {
                    let node = &*node_ptr.into_raw();
                    match epoch_option {
                        EpochOption::Exact(epoch) => {
                            let value = &*(VersionedSkiplist::find_versioned_value(&node, epoch)?.into_raw());
                            Some(value)
                        },
                        EpochOption::Latest => {
                            let latest_value = (*(node.versions.load(Ordering::SeqCst))).last()?;
                            Some(&latest_value.value)
                        },
                        EpochOption::Current => {
                            let epoch = self.current_epoch.load(Ordering::SeqCst);
                            let value = &*(VersionedSkiplist::find_versioned_value(&node, epoch)?.into_raw());
                            Some(value)
                        }
                    }
                }
            },
            _ => None
        }
    }

    pub fn update(&self, key: K, new_value: V, old_value: V) -> Result<&'a Node<K, V>, UpdateError<'a, K, V>> {
        if let Some(node) = self.get_node(key) {
            let initial_versions = RawPtr::from(node.versions.load(Ordering::SeqCst));
            let initial_last_value = initial_versions.last().unwrap();
            loop {
                if initial_last_value.epoch.load(Ordering::SeqCst) == -1 {
                    continue;
                } else if initial_last_value.value == old_value {
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
                        if pre_swap_versions == new_versions_ptr {
                            initial_versions.dealloc();
                            let last_epoch = self.current_epoch.fetch_add(1, Ordering:: SeqCst);
                            (*new_versioned_value_ptr).epoch.store(last_epoch + 1, Ordering::SeqCst);
                            break Ok(node);
                        } else {
                            let last_value = &((*pre_swap_versions).last().unwrap().value);
                            break Err(UpdateErrorKind::StaleValue(last_value.into()).into());
                        }
                    }
                } else {
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
        let mut left_bound_nodes: Vec<RawPtr<Node<K, V>>> = Vec::with_capacity(self.max_height as usize);
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

#[test]
fn insert() {
    /* let list: ConcurrentSkiplist<i32> = ConcurrentSkiplist::new(10);
    list.insert(10);
    list.insert(-20);
    list.insert(20);
    println!("done"); */
}

#[test]
fn uniform() {
    let uniform = Uniform::new(0, 2);
    println!("{}", uniform.sample(&mut rand::thread_rng()));
}