
use super::*;
use super::ffi::*;
use std::ptr;
use std::os::raw::c_void;
use std::fmt::{self, Debug};
use cumath::*;






pub struct CuActivationDescriptor {
    data: *mut _ActivationDescriptorStruct,
}

impl Drop for CuActivationDescriptor {
    fn drop(&mut self) {
        cudnn_destroy_activation_descriptor(self.data)
    }
}

impl CuActivationDescriptor {

    pub fn new(mode: CudnnActivationMode, coef: f64) -> CuActivationDescriptor {
        let mut data = ptr::null_mut();
        cudnn_create_activation_descriptor(&mut data);
        cudnn_set_activation_descriptor(data, mode, CudnnNanPropagation::Propagate, coef);
        CuActivationDescriptor { data }
    }
    pub fn sigmoid() -> CuActivationDescriptor {
        let mut data = ptr::null_mut();
        cudnn_create_activation_descriptor(&mut data);
        cudnn_set_activation_descriptor(data, CudnnActivationMode::Sigmoid, CudnnNanPropagation::Propagate, 1.0);
        CuActivationDescriptor { data }
    }
    pub fn relu() -> CuActivationDescriptor {
        let mut data = ptr::null_mut();
        cudnn_create_activation_descriptor(&mut data);
        cudnn_set_activation_descriptor(data, CudnnActivationMode::Relu, CudnnNanPropagation::Propagate, 1.0);
        CuActivationDescriptor { data }
    }
    pub fn tanh() -> CuActivationDescriptor {
        let mut data = ptr::null_mut();
        cudnn_create_activation_descriptor(&mut data);
        cudnn_set_activation_descriptor(data, CudnnActivationMode::Tanh, CudnnNanPropagation::Propagate, 1.0);
        CuActivationDescriptor { data }
    }
    pub fn clipped_relu(threshold: f64) -> CuActivationDescriptor {
        let mut data = ptr::null_mut();
        cudnn_create_activation_descriptor(&mut data);
        cudnn_set_activation_descriptor(data, CudnnActivationMode::ClippedRelu, CudnnNanPropagation::Propagate, threshold);
        CuActivationDescriptor { data }
    }
    pub fn elu(alpha: f64) -> CuActivationDescriptor {
        let mut data = ptr::null_mut();
        cudnn_create_activation_descriptor(&mut data);
        cudnn_set_activation_descriptor(data, CudnnActivationMode::Elu, CudnnNanPropagation::Propagate, alpha);
        CuActivationDescriptor { data }
    }
    /*pub fn identity() -> CuActivationDescriptor {
        let mut data = ptr::null_mut();
        cudnn_create_activation_descriptor(&mut data);
        cudnn_set_activation_descriptor(data, CudnnActivationMode::Identity, CudnnNanPropagation::Propagate, 1.0);
        CuActivationDescriptor { data }
    }*/

    pub fn get_info(&self) -> CuActivationDescriptorInfo {
        let mut mode = CudnnActivationMode::Elu;
        let mut relu_nan_opt = CudnnNanPropagation::NotPropagate;
        let mut coef = -9999.0;
        cudnn_get_activation_descriptor(
            self.data,
            &mut mode,
            &mut relu_nan_opt,
            &mut coef,
        );
        CuActivationDescriptorInfo { mode, relu_nan_opt, coef }
    }

    pub fn forward<T: CuDataType>(&self, cudnn: &Cudnn, input: &CuTensorDeref<T>, input_scale: T, output: &mut CuTensorDeref<T>, output_scale: T) {
        cudnn_activation_forward(cudnn.handle, self.data,
                                 &input_scale as *const T as *const c_void, input.descriptor.data, input.data as *const c_void,
                                 &output_scale as *const T as *const c_void, output.descriptor.data, output.data as *mut c_void)
    }
    pub fn forward_inplace<T: CuDataType>(&self, cudnn: &Cudnn, vector: &mut CuTensorDeref<T>, input_scale: T, output_scale: T) {
        cudnn_activation_forward(cudnn.handle, self.data,
                                 &input_scale as *const T as *const c_void, vector.descriptor.data, vector.data as *const c_void,
                                 &output_scale as *const T as *const c_void, vector.descriptor.data, vector.data as *mut c_void)
    }
    pub fn backward<T: CuDataType>(&self, cudnn: &Cudnn, alpha: T, beta: T,
                                   input: &CuTensorDeref<T>,
                                   output: &CuTensorDeref<T>,
                                   output_signal: &CuTensorDeref<T>,
                                   input_signal: &mut CuTensorDeref<T>) {
        cudnn_activation_backward(cudnn.handle, self.data,
                                  (&alpha) as *const T as *const c_void,
                                  output.descriptor.data, output.data as *const c_void,
                                  output_signal.descriptor.data, output_signal.data as *const c_void,
                                  input.descriptor.data, input.data as *mut c_void,
                                  (&beta) as *const T as *const c_void,
                                  input_signal.descriptor.data, input_signal.data as *mut c_void)
    }
    pub fn backward_inplace<T: CuDataType>(&self, cudnn: &Cudnn, alpha: T, beta: T,
                                           input: &CuTensorDeref<T>,
                                           output: &CuTensorDeref<T>,
                                           signal: &mut CuTensorDeref<T>) {
        cudnn_activation_backward(cudnn.handle, self.data,
                                  (&alpha) as *const T as *const c_void,
                                  output.descriptor.data, output.data as *const c_void,
                                  signal.descriptor.data, signal.data as *const c_void,
                                  input.descriptor.data, input.data as *mut c_void,
                                  (&beta) as *const T as *const c_void,
                                  signal.descriptor.data, signal.data as *mut c_void)
    }

}


pub struct CuActivationDescriptorInfo {
    pub mode: CudnnActivationMode,
    pub relu_nan_opt: CudnnNanPropagation,
    pub coef: f64,
}

impl Debug for CuActivationDescriptorInfo {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "Mode:{:?}, ReluNanOpt:{:?}, coef:{}", self.mode, self.relu_nan_opt, self.coef)
    }
}



#[cfg(test)]
mod tests {

    use super::*;

    fn test_activation(name: &str, activation: CuActivationDescriptor) {
        let cudnn = Cudnn::new();

        let input_data = [-0.75, -0.5, 0.0, 1.0];
        let output_signal_data = [1.0; 4];
        let mut buffer = [0.0; 4];


        let tensor_descriptor = CuTensorDescriptor::<f32>::fully_packed(&[2, 2, 1]);
        let input = CuVector::<f32>::from_host_data(&[-0.75, -0.5, 0.0, 1.0]);
        let output_signal = CuVector::<f32>::from_host_data(&output_signal_data);

        let mut forward_output = CuVector::<f32>::zero(tensor_descriptor.data_len());
        activation.forward(&cudnn, &tensor_descriptor.link(&input), 1.0, &mut tensor_descriptor.link_mut(&mut forward_output), 0.0);
        input.dev_assert_equals(&input_data);

        let mut forward_inplace_output = input.clone();
        activation.forward_inplace(&cudnn, &mut tensor_descriptor.link_mut(&mut forward_inplace_output), 1.0, 0.0);
        forward_inplace_output.clone_to_host(&mut buffer);
        forward_output.dev_assert_equals(&buffer);

        println!("{} forward : Output[{:?}]", name, forward_output);


        let mut backward_output = CuVector::<f32>::zero(tensor_descriptor.data_len());
        activation.backward(&cudnn, 1.0, 0.0,
                            &tensor_descriptor.link(&input),
                            &tensor_descriptor.link(&forward_output),
                            &tensor_descriptor.link(&output_signal),
                            &mut tensor_descriptor.link_mut(&mut backward_output));
        input.dev_assert_equals(&input_data);
        output_signal.dev_assert_equals(&output_signal_data);
        forward_inplace_output.clone_to_host(&mut buffer);
        forward_output.dev_assert_equals(&buffer);

        let mut backward_inplace_output = output_signal.clone();
        activation.backward_inplace(&cudnn, 1.0, 0.0,
                                    &tensor_descriptor.link(&input),
                                    &tensor_descriptor.link(&forward_output),
                                    &mut tensor_descriptor.link_mut(&mut backward_inplace_output));
        input.dev_assert_equals(&input_data);
        output_signal.dev_assert_equals(&output_signal_data);
        backward_inplace_output.clone_to_host(&mut buffer);
        println!("{:?}", backward_output);
        println!("{:?}", backward_inplace_output);
        backward_output.dev_assert_equals(&buffer);

        println!("{} backward : Output[{:?}]", name, backward_output);
    }

    #[test]
    fn sigmoid_forward() {
        test_activation("sigmoid", CuActivationDescriptor::sigmoid());
    }

    #[test]
    fn relu_forward() {
        test_activation("relu", CuActivationDescriptor::relu());
    }

    #[test]
    fn tanh_forward() {
        test_activation("tanh", CuActivationDescriptor::tanh());
    }

    #[test]
    fn clipped_relu_forward() {
        test_activation("clippedRelu", CuActivationDescriptor::clipped_relu(0.5));
    }

    #[test]
    fn elu_forward() {
        test_activation("elu", CuActivationDescriptor::elu(0.5));
    }

    /*#[test]
    fn identity_forward() {
        test_forward("identity", CuActivationDescriptor::identity());
    }*/

    #[test]
    #[ignore]
    fn sigmoid_forward_benchmarch() {
        use std::time::Instant;

        let mut vector = CuVector::<f32>::zero(500);

        let t0 = Instant::now();
        for _ in 0..10000 {
            vector.fast_sigmoid(&DEFAULT_STREAM)
        }
        let dt = t0.elapsed();
        println!("Finished in {}.{}", dt.as_secs(), dt.subsec_nanos());


        let cudnn = Cudnn::new();
        let activation = CuActivationDescriptor::sigmoid();
        let tensor_descriptor = CuTensorDescriptor::<f32>::fully_packed(&[2, 3, 4, 5]);
        let mut vector = CuVector::<f32>::zero(tensor_descriptor.data_len());
        println!("Input = {:?}", vector);
        let t0 = Instant::now();
        for _ in 0..10000 {
            activation.forward_inplace(&cudnn, &mut tensor_descriptor.link_mut(&mut vector), 1.0, 0.0);
        }
        let dt = t0.elapsed();
        println!("Finished in {}.{}", dt.as_secs(), dt.subsec_nanos());
    }

}