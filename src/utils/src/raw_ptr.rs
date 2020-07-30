use std::{ptr, ops::{DerefMut, Deref}, alloc::{dealloc, Layout}};


/// A zero-cost convenience wrapper over a raw pointer.
/// 
/// This provides some conveniences over a regular pointer, mostly for
/// convenient dereferencing. With regular pointers, you have to dereference 
/// explicitly, which can make accessing a struct's method's a bit messy in 
/// the code. 
/// Instead of doing: `(*ptr).field` each time, RawPtr let's you dereference
/// without parenthesis. 
///
///Coupled with this, RawPtr has conversion methods for all
/// reference types and raw pointers into `*mut T` to make it trivial to obtain a 
/// RawPtr.
///
/// Finally, RawPtr provides a `dealloc` method for easily deallocating heap-residing
/// values. A great use case for RawPtr is extracting the raw pointer from a 
/// Box-created value via methods like `Box::into_raw`. This transfers all control
/// of the value's lifetime to the user, preventing `Box` from dropping the
/// value as normal. Sometimes we want that, especially when managing lock-free
/// data structures. `RawPtr::dealloc` performs a function similar dropping a `Box`.
///
/// Note that since this is just used for internal convenience, some safety is perhaps
/// incorrectly abstracted away. Specifically, `Deref` forces us to deref safetly, 
/// so a normally unsafe raw pointer deref is treated as safe in RawPtr. This helps
/// avoid throwing `unsafe` all over the place, but creates the potential for safetly
/// derefing a null pointer and panicking. Any user of RawPtr should be aware of this,
/// and should make sure to check the pointer before dereferencing. 
#[derive(Debug)]
pub struct RawPtr<T> {
    ptr: *mut T
}

impl<T> RawPtr<T> {
    /// Creates a RawPtr with an underlying null `*mut T` pointer
    pub fn null() -> Self {
        RawPtr::from(ptr::null_mut())
    }

    /// Checks if the underlying raw pointer is null
    pub fn is_null(&self) -> bool {
        self.ptr.is_null()
    }

    /// Returns the underlying raw pointer, consuming the RawPtr
    pub fn into_raw(self) -> *mut T {
        self.ptr
    }

    /// Drops and deallocates the value pointed to by the RawPtr 
    pub unsafe fn dealloc(&self) {
        ptr::drop_in_place(self.ptr);
        dealloc(self.ptr.cast(), Layout::for_value(&*self.ptr));
    }
}

impl<T> Copy for RawPtr<T> { }
impl<T> Clone for RawPtr<T> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<T> From<&T> for RawPtr<T> {
    fn from(ptr: &T) -> Self {
        RawPtr { ptr: ptr as *const T as *mut T }
    }
}

impl<T> From<&mut T> for RawPtr<T> {
    fn from(ptr: &mut T) -> Self {
        RawPtr { ptr: ptr as *const T as *mut T }
    }
}

impl<T> From<*mut T> for RawPtr<T> {
    fn from(ptr: *mut T) -> Self {
        RawPtr { ptr }
    }
}

impl<T> From<*const T> for RawPtr<T> {
    fn from(ptr: *const T) -> Self {
        RawPtr { ptr: ptr as *mut T }
    }
}

impl<T> Deref for RawPtr<T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        unsafe {
            &*self.ptr
        }
    }
}

impl<T> DerefMut for RawPtr<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        unsafe {
            &mut *self.ptr
        }
    }
}