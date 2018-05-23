
use std::ops::{Deref, DerefMut};
use cumath::CuDataType;
use ffi::_TensorDescriptorStruct;


pub struct CuTensorArrayDeref<T: CuDataType> {
    pub(crate) descriptors: Box<[*const _TensorDescriptorStruct]>,
    pub(crate) data: *mut T,
}


pub struct CuTensorArray<T: CuDataType> {
    pub(crate) deref: CuTensorArrayDeref<T>,
}
impl<T: CuDataType> Deref for CuTensorArray<T> {
    type Target = CuTensorArrayDeref<T>;
    fn deref(&self) -> &CuTensorArrayDeref<T> { &self.deref }
}

pub struct CuTensorArrayMut<T: CuDataType> {
    pub(crate) deref: CuTensorArrayDeref<T>,
}
impl<T: CuDataType> Deref for CuTensorArrayMut<T> {
    type Target = CuTensorArrayDeref<T>;
    fn deref(&self) -> &CuTensorArrayDeref<T> { &self.deref }
}
impl<T: CuDataType> DerefMut for CuTensorArrayMut<T> {
    fn deref_mut(&mut self) -> &mut CuTensorArrayDeref<T> { &mut self.deref }
}
