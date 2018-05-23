
use super::ffi::*;
use super::{Cudnn, CuDropoutDescriptor, CuTensorDescriptor, CuTensorDeref, CuTensorArrayDeref};
use std::ptr;
use std::marker::PhantomData;
use cumath::CuDataType;


pub struct CuRNNDescriptor<T: CuDataType> {
    _phantom: PhantomData<T>,
    _dropout: CuDropoutDescriptor<T>,
    pub(crate) data: *mut _RNNDescriptorStruct,
}

impl<T: CuDataType> Drop for CuRNNDescriptor<T> {
    fn drop(&mut self) {
        cudnn_destroy_rnn_descriptor(self.data)
    }
}

impl CuRNNDescriptor<f32> {

    pub fn new(cudnn: &Cudnn, hidden_size: usize, nb_layers: usize,
               input_mode: CudnnRNNInputMode, direction: CudnnDirectionMode,
               mode: CudnnRNNMode, algo: CudnnRNNAlgo, dropout: f32, seed: u64) -> CuRNNDescriptor<f32> {

        let dropout = CuDropoutDescriptor::<f32>::new(cudnn, dropout, seed);
        let mut data = ptr::null_mut();
        cudnn_create_rnn_descriptor(&mut data);
        cudnn_set_rnn_descriptor(cudnn.handle, data,
                                 hidden_size as i32,
                                 nb_layers as i32, dropout.data, input_mode, direction, mode, algo, CudnnDataType::Float);
        CuRNNDescriptor {
            _phantom: PhantomData,
            _dropout: dropout,
            data
        }
    }

    pub fn get_workspace_size(&self, cudnn: &Cudnn, sequence_len: usize, input_descriptor: &[CuTensorDescriptor<f32>]) -> usize {
        assert_eq!(input_descriptor.len(), sequence_len);
        let mut result = 0;
        cudnn_get_rnn_workspace_size(
            cudnn.handle,
            self.data,
            sequence_len as i32,
            input_descriptor.iter().map(|x| x.data).collect::<Vec<_>>().as_ptr() as *const*const _TensorDescriptorStruct,
            &mut result
        );
        result
    }

    pub fn get_training_reserve_size(&self, cudnn: &Cudnn, sequence_len: usize, input_descriptor: &[CuTensorDescriptor<f32>]) -> usize {
        assert_eq!(input_descriptor.len(), sequence_len);
        let mut result = 0;
        cudnn_get_rnn_training_reserve_size(
            cudnn.handle,
            self.data,
            sequence_len as i32,
            input_descriptor.iter().map(|x| x.data).collect::<Vec<_>>().as_ptr() as *const*const _TensorDescriptorStruct,
            &mut result
        );
        result
    }

    /*pub fn forward_inference(&self, cudnn: &Cudnn, sequence_len: usize,
                             inputs_descriptor: &CuTensorArrayDeref<f32>,
                             hidden_state_input: &CuTensorDeref<f32>,
                             cell_state_input: &CuTensorDeref<f32>, weights: &CuFi)*/

}


#[cfg(test)]
mod tests {

    use super::*;

    #[test]
    fn test() {

        let cudnn = Cudnn::new();
        let rnn = CuRNNDescriptor::new(&cudnn, 512, 64,
                                       CudnnRNNInputMode::LinearInput,
                                       CudnnDirectionMode::Unidirectional,
                                       CudnnRNNMode::Gru,
                                       CudnnRNNAlgo::Standard,
                                       0.5, 100);
        let input_desc = [CuTensorDescriptor::<f32>::new(&[1, 1, 12, 6])];

        rnn.get_workspace_size(&cudnn, 1, &input_desc);
        rnn.get_training_reserve_size(&cudnn, 1, &input_desc);

    }

}