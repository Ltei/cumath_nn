
use super::ffi::*;
use super::Cudnn;
use std::ptr;
use std::marker::PhantomData;
use std::os::raw::c_void;
use cumath::CuDataType;
use cumath::CuVector;

pub struct CuDropoutDescriptor<T: CuDataType> {
    _phantom: PhantomData<T>,
    _states: CuVector<T>,
    pub(crate) data: *mut _DropoutDescriptorStruct,
}

impl<T: CuDataType> Drop for CuDropoutDescriptor<T> {
    fn drop(&mut self) {
        cudnn_destroy_dropout_descriptor(self.data)
    }
}

impl CuDropoutDescriptor<f32> {

    pub fn new(cudnn: &Cudnn, dropout: f32, seed: u64) -> CuDropoutDescriptor<f32> {
        let mut states = CuVector::<f32>::zero(Self::get_states_size(cudnn));
        let mut data = ptr::null_mut();
        cudnn_create_dropout_descriptor(&mut data);
        cudnn_set_dropout_descriptor(data, cudnn.handle, dropout, states.as_mut_ptr() as *mut c_void,
                                     states.len(), seed);

        CuDropoutDescriptor {
            _phantom: PhantomData,
            _states: states,
            data
        }

    }

    pub fn get_states_size(cudnn: &Cudnn) -> usize {
        let mut states_size = 0;
        cudnn_dropout_get_states_size(cudnn.handle, &mut states_size);
        states_size
    }

}



#[cfg(test)]
mod tests {

    use super::*;

    #[test]
    fn test() {
        let cudnn = Cudnn::new();
        let _dropout = CuDropoutDescriptor::new(&cudnn, 0.5, 545016);
        let _states_size = CuDropoutDescriptor::get_states_size(&cudnn);
    }

}