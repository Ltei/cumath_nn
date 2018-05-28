
use super::ffi::*;
use std::ptr;


pub struct Cudnn {
    pub(crate) handle: *mut _CudnnStruct,
}

impl Drop for Cudnn {
    fn drop(&mut self) {
        cudnn_destroy(self.handle)
    }
}

impl Cudnn {

    pub fn new() -> Cudnn {
        let mut data = ptr::null_mut();
        cudnn_create(&mut data);
        Cudnn { handle: data }
    }

}