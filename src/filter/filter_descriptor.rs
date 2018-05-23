
use std::ptr;
use std::marker::PhantomData;
use std::fmt::{self, Debug};
use cumath::CuDataType;
use ffi::*;


pub struct CuFilterDescriptor<T: CuDataType> {
    _phantom: PhantomData<T>,
    pub(crate) data: *mut _FilterDescriptorStruct,
}

impl<T: CuDataType> Drop for CuFilterDescriptor<T> {
    fn drop(&mut self) {
        cudnn_destroy_filter_descriptor(self.data)
    }
}

impl<T: CuDataType> CuFilterDescriptor<T> {

    /*pub fn link<'a>(&'a self, data: &'a CuVectorDeref<T>) -> CuTensor<'a, T> {
        #[cfg(not(feature = "disable_checks"))] {
            assert_eq!(data.len(), self.data_len(), "data.len() != self.data_len()");
        }
        CuTensor { deref: CuTensorDeref { descriptor: self, data: data.as_ptr() as *mut T } }
    }

    pub fn link_mut<'a>(&'a self, data: &'a mut CuVectorDeref<T>) -> CuTensorMut<'a, T> {
        #[cfg(not(feature = "disable_checks"))] {
            assert_eq!(data.len(), self.data_len(), "data.len() != self.data_len()");
        }
        CuTensorMut { deref: CuTensorDeref { descriptor: self, data: data.as_mut_ptr() } }
    }*/

    pub fn get_info(&self, nb_dims_requested: i32) -> CuFilterDescriptorInfo {
        let mut data_type = CudnnDataType::Int8x4;
        let mut format = CudnnTensorFormat::Nchw;
        let mut nb_dims = -1;
        let mut filter_dims = vec![-1; nb_dims_requested as usize];
        cudnn_get_filter_nd_descriptor(self.data, nb_dims_requested, &mut data_type, &mut format, &mut nb_dims, filter_dims.as_mut_ptr());
        CuFilterDescriptorInfo { data_type, format, nb_dims, filter_dims }
    }

}

impl CuFilterDescriptor<f32> {

    pub fn new(format: CudnnTensorFormat, filter_dims: &[i32]) -> CuFilterDescriptor<f32> {
        let mut data = ptr::null_mut();
        cudnn_create_filter_descriptor(&mut data);
        cudnn_set_filter_nd_descriptor(data, CudnnDataType::Float, format, filter_dims.len() as i32, filter_dims.as_ptr());
        CuFilterDescriptor { _phantom: PhantomData, data }
    }

    pub fn new_4d(format: CudnnTensorFormat, k: i32, c: i32, h: i32, w: i32) -> CuFilterDescriptor<f32> {
        let mut data = ptr::null_mut();
        cudnn_create_filter_descriptor(&mut data);
        cudnn_set_filter4d_descriptor(data, CudnnDataType::Float, format, k, c, h, w);
        CuFilterDescriptor { _phantom: PhantomData, data }
    }

}


pub struct CuFilterDescriptorInfo {
    pub data_type: CudnnDataType,
    pub format: CudnnTensorFormat,
    pub nb_dims: i32,
    pub filter_dims: Vec<i32>,
}

impl Debug for CuFilterDescriptorInfo {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "Data type:{:?}, Format = {:?}, Nb dims:{}, Filter dims:{:?}", self.data_type, self.format, self.nb_dims, self.filter_dims)
    }
}


#[cfg(test)]
mod tests {

    use super::*;

    #[test]
    fn init_nchw() {
        let descriptor = CuFilterDescriptor::<f32>::new(CudnnTensorFormat::Nchw, &[2, 4, 1, 7]);
        let info = descriptor.get_info(4);
        assert_eq!(info.data_type, CudnnDataType::Float);
        assert_eq!(info.format, CudnnTensorFormat::Nchw);
        assert_eq!(info.nb_dims, 4);
        assert_eq!(info.filter_dims.len(), 4);
        assert_eq!(info.filter_dims[0], 2);
        assert_eq!(info.filter_dims[1], 4);
        assert_eq!(info.filter_dims[2], 1);
        assert_eq!(info.filter_dims[3], 7);
    }

}