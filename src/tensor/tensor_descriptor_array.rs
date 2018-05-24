
use cumath::{CuDataType, CuVectorDeref};
use ffi::*;
use super::*;





// Descriptor array

pub trait CuTensorDescriptorArray<T: CuDataType> {
    fn data_len(&self) -> usize;
    fn link(&self, data: &CuVectorDeref<T>) -> CuTensorArray<T>;
    fn link_mut(&self, data: &mut CuVectorDeref<T>) -> CuTensorArrayMut<T>;
}

// Impl for [CuTensorDescriptor]

impl<T: CuDataType> CuTensorDescriptorArray<T> for [CuTensorDescriptor<T>] {
    fn data_len(&self) -> usize {
        self.iter().fold(0, |acc, x| acc + x.data_len())
    }

    fn link(&self, data: &CuVectorDeref<T>) -> CuTensorArray<T> {
        CuTensorArray {
            deref: CuTensorArrayDeref {
                descriptors: self.iter().map(|x| x.data as *const _TensorDescriptorStruct).collect::<Vec<_>>().into_boxed_slice(),
                data: data.as_ptr() as *mut T,
            }
        }
    }

    fn link_mut(&self, data: &mut CuVectorDeref<T>) -> CuTensorArrayMut<T> {
        CuTensorArrayMut {
            deref: CuTensorArrayDeref {
                descriptors: self.iter().map(|x| x.data as *const _TensorDescriptorStruct).collect::<Vec<_>>().into_boxed_slice(),
                data: data.as_ptr() as *mut T,
            }
        }
    }
}



#[cfg(test)]
mod tests {

    use super::*;
    use cumath::CuVector;

    #[test]
    fn link_array() {

        let descriptors = [CuTensorDescriptor::<f32>::fully_packed(&[1, 1, 4, 10]), CuTensorDescriptor::<f32>::fully_packed(&[1, 1, 4, 20])];
        let mut data = CuVector::<f32>::new(1.0, descriptors.data_len());

        {
            let _tensor = &descriptors.link(&data);
        }
        let _tensor = &descriptors.link_mut(&mut data);

    }

}