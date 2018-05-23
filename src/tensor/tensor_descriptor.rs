
use std::{ptr, marker::PhantomData, fmt::{self, Debug}, mem::size_of};
use cumath::{CuDataType, CuVectorDeref};
use ffi::*;
use super::*;



// Descriptor

pub struct CuTensorDescriptor<T: CuDataType> {
    _phantom: PhantomData<T>,
    pub(crate) data: *mut _TensorDescriptorStruct,
    nb_dimensions: i32,
    dimensions: Vec<i32>,
    strides: Vec<i32>,
    data_len: usize,
}

impl<T: CuDataType> Drop for CuTensorDescriptor<T> {
    fn drop(&mut self) {
        cudnn_destroy_tensor_descriptor(self.data);
    }
}

impl<T: CuDataType> Debug for CuTensorDescriptor<T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "Nb dims:{}, Dimensions:{:?}, Strides:{:?}, Data len:{}", self.nb_dimensions, self.dimensions, self.strides, self.data_len)
    }
}

impl<T: CuDataType> Clone for CuTensorDescriptor<T> {
    fn clone(&self) -> CuTensorDescriptor<T> {
        let info = self.get_info();
        let mut data = ptr::null_mut();
        cudnn_create_tensor_descriptor(&mut data);
        cudnn_set_tensor_nd_descriptor(data, info.data_type, info.nb_dims, info.dimensions.as_ptr(), info.strides.as_ptr());
        CuTensorDescriptor {
            _phantom: PhantomData,
            data,
            nb_dimensions: self.nb_dimensions,
            dimensions: self.dimensions.clone(),
            strides: self.strides.clone(),
            data_len: self.data_len,
        }
    }
    fn clone_from(&mut self, other: &CuTensorDescriptor<T>) {
        let info = other.get_info();
        cudnn_set_tensor_nd_descriptor(self.data, info.data_type, info.nb_dims, info.dimensions.as_ptr(), info.strides.as_ptr());
        self.dimensions.clone_from(&other.dimensions);
        self.strides.clone_from(&other.strides);
        self.data_len = other.data_len;
    }
}

impl<T: CuDataType> CuTensorDescriptor<T> {

    pub fn link<'a>(&'a self, data: &'a CuVectorDeref<T>) -> CuTensor<'a, T> {
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
    }

    pub fn data_len(&self) -> usize {
        let mut output = 0;
        cudnn_get_tensor_size_in_bytes(self.data, &mut output);
        output / size_of::<T>()
    }

    pub fn get_info(&self) -> CuTensorDescriptorInfo {
        let nb_dimensions = self.dimensions.len();
        let mut data_type = CudnnDataType::Int8x4;
        let mut nb_dims = -1;
        let mut dimensions = vec![-1; nb_dimensions];
        let mut strides = vec![-1; nb_dimensions];
        cudnn_get_tensor_nd_descriptor(self.data, self.nb_dimensions,
                                       &mut data_type, &mut nb_dims, dimensions.as_mut_ptr(), strides.as_mut_ptr());
        CuTensorDescriptorInfo { data_type, nb_dims, dimensions, strides }
    }

}

impl CuTensorDescriptor<f32> {

    pub fn new(dimensions: Vec<i32>) -> CuTensorDescriptor<f32> {
        #[cfg(not(feature = "disable_checks"))] {
            if dimensions.len() < 3 { panic!("dimensions.len() must be >= 3") }
        }
        let strides = get_fully_packed_strides(&dimensions);
        let mut data = ptr::null_mut();
        cudnn_create_tensor_descriptor(&mut data);
        cudnn_set_tensor_nd_descriptor(data, CudnnDataType::Float, dimensions.len() as i32, dimensions.as_ptr(), strides.as_ptr());
        let mut data_len = 0;
        cudnn_get_tensor_size_in_bytes(data, &mut data_len);
        data_len /=  size_of::<f32>();

        CuTensorDescriptor {
            _phantom: PhantomData,
            data,
            nb_dimensions: dimensions.len() as i32,
            dimensions,
            strides,
            data_len,
        }
    }

    pub fn new_4d(format: CudnnTensorFormat, n: i32, c: i32, h: i32, w: i32) -> CuTensorDescriptor<f32> {
        let mut data = ptr::null_mut();
        cudnn_create_tensor_descriptor(&mut data);
        cudnn_set_tensor4d_descriptor(data, format, CudnnDataType::Float, n, c, h, w);
        let mut data_len = 0;
        cudnn_get_tensor_size_in_bytes(data, &mut data_len);
        data_len /=  size_of::<f32>();

        let dimensions = match format {
            CudnnTensorFormat::Nchw => vec![n, c, h, w],
            CudnnTensorFormat::Nhwc => vec![n, c, h, w],
        };

        CuTensorDescriptor {
            _phantom: PhantomData,
            nb_dimensions: 4,
            data_len: {
                let mut data_len = 0;
                cudnn_get_tensor_size_in_bytes(data, &mut data_len);
                data_len /  size_of::<f32>()
            },
            strides: get_fully_packed_strides(&dimensions),
            data,
            dimensions,
        }
    }

}


fn get_fully_packed_strides(dims: &[i32]) -> Vec<i32> {
    use std::collections::VecDeque;
    let mut output = VecDeque::with_capacity(dims.len());
    output.push_front(1);
    let mut last = 1;
    for i in (1..dims.len()).rev() {
        last *= dims[i];
        output.push_front(last);
    }
    Vec::from(output)
}


// Descriptor Info

pub struct CuTensorDescriptorInfo {
    pub data_type: CudnnDataType,
    pub nb_dims: i32,
    pub dimensions: Vec<i32>,
    pub strides: Vec<i32>,
}

impl Debug for CuTensorDescriptorInfo {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "Data type:{:?}, Nb dims:{}, Dimensions:{:?}, Strides:{:?}", self.data_type, self.nb_dims, self.dimensions, self.strides)
    }
}




#[cfg(test)]
mod tests {

    use super::*;
    use cumath::CuVector;

    fn assert_validity(descriptor: &CuTensorDescriptor<f32>) {
        println!("Asserting Tensor descriptor {:?}", descriptor);
        assert_eq!(descriptor.dimensions.len(), descriptor.strides.len());
        assert_eq!(descriptor.strides, get_fully_packed_strides(&descriptor.dimensions));

        let info = descriptor.get_info();
        println!("    Info = {:?}", info);
        assert_eq!(info.data_type, CudnnDataType::Float);
        assert_eq!(info.nb_dims, descriptor.dimensions.len() as i32);
        assert_eq!(info.nb_dims, info.dimensions.len() as i32);
        assert_eq!(info.nb_dims, info.strides.len() as i32);

        (0..descriptor.dimensions.len()).for_each(|x| {
            assert_eq!(info.dimensions[x], descriptor.dimensions[x]);
            assert_eq!(info.strides[x], descriptor.strides[x]);
        })
    }

    #[test]
    fn init() {
        assert_validity(&CuTensorDescriptor::<f32>::new(vec![7, 1, 5, 3]));
        assert_validity(&CuTensorDescriptor::<f32>::new_4d(CudnnTensorFormat::Nhwc, 7, 1, 5, 3));
        assert_validity(&CuTensorDescriptor::<f32>::new_4d(CudnnTensorFormat::Nchw, 7, 1, 5, 3));
    }

    #[test]
    fn link() {

        let descriptor = CuTensorDescriptor::<f32>::new(vec![1, 1, 4, 10]);
        let mut data = CuVector::<f32>::new(1.0, descriptor.data_len());

        {
            let _tensor = descriptor.link(&data);
        }
        let _tensor = descriptor.link_mut(&mut data);

    }

}