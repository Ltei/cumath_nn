
use super::*;
use std::ops::{Deref, DerefMut};
use cumath::CuDataType;




pub struct CuFilterDeref<'a, T: CuDataType + 'a> {
    pub(crate) descriptor: &'a CuFilterDescriptor<T>,
    pub(crate) data: *mut T,
}

impl<'a, T: CuDataType + 'a> CuFilterDeref<'a, T> {

    pub fn descriptor(&self) -> &CuFilterDescriptor<T> {
        self.descriptor
    }

}


pub struct CuFilter<'a, T: CuDataType + 'a> {
    pub(crate) deref: CuFilterDeref<'a, T>,
}
impl<'a, T: CuDataType + 'a> Deref for CuFilter<'a, T> {
    type Target = CuFilterDeref<'a, T>;
    fn deref(&self) -> &CuFilterDeref<'a, T> { &self.deref }
}

pub struct CuFilterMut<'a, T: CuDataType + 'a> {
    pub(crate) deref: CuFilterDeref<'a, T>,
}
impl<'a, T: CuDataType + 'a> Deref for CuFilterMut<'a, T> {
    type Target = CuFilterDeref<'a, T>;
    fn deref(&self) -> &CuFilterDeref<'a, T> { &self.deref }
}
impl<'a, T: CuDataType + 'a> DerefMut for CuFilterMut<'a, T> {
    fn deref_mut(&mut self) -> &mut CuFilterDeref<'a, T> { &mut self.deref }
}
