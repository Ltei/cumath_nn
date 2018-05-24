
use std::{ptr, mem::size_of};
use std::marker::PhantomData;
use std::os::raw::c_void;
use std::fmt::{self, Debug};
use cumath::*;
use super::ffi::*;
use super::*;



pub struct CuConvolutionDescriptor<T: CuDataType> {
    _phantom: PhantomData<T>,
    data: *mut _ConvolutionDescriptorStruct,
    array_len: i32,
}

impl<T: CuDataType> Drop for CuConvolutionDescriptor<T> {
    fn drop(&mut self) {
        cudnn_destroy_convolution_descriptor(self.data)
    }
}

impl<T: CuDataType> CuConvolutionDescriptor<T> {

    pub fn get_forward_workspace_size(&self, cudnn: &Cudnn, input_desc: &CuTensorDescriptor<f32>, kernel_desc: &CuFilterDescriptor<f32>,
                                      output_desc: &CuTensorDescriptor<f32>, algo: CudnnConvolutionFwdAlgo) -> usize {
        let mut output = 0;
        cudnn_get_convolution_forward_workspace_size(cudnn.handle, input_desc.data, kernel_desc.data,
                                                     self.data, output_desc.data, algo, &mut output);
        output
    }

    pub fn forward(&self, cudnn: &mut Cudnn, alpha: f32, beta: f32, input: &CuTensorDeref<f32>, kernel_desc: &CuFilterDescriptor<f32>, kernel_data: &CuVectorDeref<f32>,
                   workspace: &mut CuVectorDeref<f32>, output: &mut CuTensorDeref<f32>, algo: CudnnConvolutionFwdAlgo) {
        cudnn_convolution_forward(cudnn.handle,
                                  &alpha as *const f32 as *const c_void,
                                  input.descriptor.data, input.data as *const c_void,
                                  kernel_desc.data, kernel_data.as_ptr() as *const c_void,
                                  self.data, algo,
                                  workspace.as_mut_ptr() as *mut c_void, workspace.len() * size_of::<f32>(),
                                  &beta as *const f32 as *const c_void,
                                  output.descriptor.data, output.data as *mut c_void);
    }

    pub fn backward_data(&self, cudnn: &mut Cudnn,
                         alpha: f32, beta: f32, output: &CuTensorDeref<f32>,
                         kernel_desc: &CuFilterDescriptor<f32>, kernel_data: &mut CuVectorDeref<f32>,
                         workspace: &mut CuVectorDeref<f32>, input: &mut CuTensorDeref<f32>, algo: CudnnConvolutionBwdDataAlgo) {
        unimplemented!()
    }

}

impl CuConvolutionDescriptor<f32> {

    pub fn new(paddings: &[i32], filters_stride: &[i32], dilatations: &[i32], mode: CudnnConvolutionMode, group_count: i32, math_type: CudnnMathType) -> CuConvolutionDescriptor<f32> {
        let len = paddings.len();
        assert_eq!(len, filters_stride.len());
        assert_eq!(len, dilatations.len());
        let mut data = ptr::null_mut();
        cudnn_create_convolution_descriptor(&mut data);
        cudnn_set_convolution_nd_descriptor(data, len as i32, paddings.as_ptr(),
                                            filters_stride.as_ptr(), dilatations.as_ptr(), mode, CudnnDataType::Float);
        //cudnn_set_convolution_group_count(data, group_count);
        //cudnn_set_convolution_math_type(data, math_type);
        CuConvolutionDescriptor { _phantom: PhantomData, data, array_len: len as i32 }
    }

    pub fn new_2d(pad_h: i32, pad_w: i32, u: i32, v: i32, dilatation_h: i32, dilatation_w: i32, mode: CudnnConvolutionMode) -> CuConvolutionDescriptor<f32> {
        let mut data = ptr::null_mut();
        cudnn_create_convolution_descriptor(&mut data);
        cudnn_set_convolution2d_descriptor(data, pad_h, pad_w, u, v, dilatation_h, dilatation_w, mode, CudnnDataType::Float);
        //cudnn_set_convolution_group_count(data, group_count); TODO
        //cudnn_set_convolution_math_type(data, math_type);
        CuConvolutionDescriptor { _phantom: PhantomData, data, array_len: 2 }
    }

    pub fn get_info(&self) -> CuConvolutionDescriptorInfo {
        let len = self.array_len as usize;
        let mut array_length = -1;
        let mut pads = vec![-1; len];
        let mut filter_strides = vec![-1; len];
        let mut dilatations = vec![-1; len];
        let mut mode = CudnnConvolutionMode::Convolution;
        let mut data_type = CudnnDataType::Int8x4;
        cudnn_get_convolution_nd_descriptor(self.data, self.array_len, &mut array_length, pads.as_mut_ptr(), filter_strides.as_mut_ptr(), dilatations.as_mut_ptr(), &mut mode, &mut data_type);
        CuConvolutionDescriptorInfo {
            array_length,
            pads,
            filter_strides,
            dilatations,
            mode,
            data_type,
        }
    }

}


#[derive(Debug, PartialEq)]
pub struct CuConvolutionDescriptorInfo {
    pub array_length: i32,
    pub pads: Vec<i32>,
    pub filter_strides: Vec<i32>,
    pub dilatations: Vec<i32>,
    pub mode: CudnnConvolutionMode,
    pub data_type: CudnnDataType,
}




#[cfg(test)]
mod tests {

    use super::*;

    #[test]
    fn compare_2d_nd() {
        let convolution_2d = CuConvolutionDescriptor::<f32>::new_2d(
            1, 1,
            1, 1,
            1, 1,
            CudnnConvolutionMode::CrossCorrelation
        );
        let convolution_nd = CuConvolutionDescriptor::<f32>::new(
            &[1, 1], &[1, 1], &[1, 1],
            CudnnConvolutionMode::CrossCorrelation, 1,
            CudnnMathType::Default);
        assert!(convolution_2d.get_info().eq(&convolution_nd.get_info()))
    }

    #[test]
    fn convolution() {
        let mut cudnn = Cudnn::new();

        let convolution = CuConvolutionDescriptor::<f32>::new(&[1, 1], &[1, 1], &[1, 1],
                                                      CudnnConvolutionMode::CrossCorrelation, 1, CudnnMathType::Default);
        let input_desc = CuTensorDescriptor::<f32>::new(&[1, 3, 4, 2], &[24, 1, 6, 3]);
        let kernel_desc = CuFilterDescriptor::<f32>::new(CudnnTensorFormat::Nchw, &[3, 3, 3, 3]);

        //println!("conv_desc = {:?}", convolution.get_info());
        //println!("input_desc = {:?}", input_desc.get_info());
        //println!("kernel_desc = {:?}", kernel_desc.get_info(4));

        let mut input_data = CuVector::<f32>::zero(input_desc.data_len());
        let mut output_data = CuVector::<f32>::zero(input_desc.data_len());
        let kernel_data = CuVector::<f32>::zero(9);
        let algo = CudnnConvolutionFwdAlgo::Gemm;

        let workspace_size = convolution.get_forward_workspace_size(&cudnn, &input_desc,
                                                                   &kernel_desc,
                                                                   &input_desc, algo);
        let mut workspace = CuVector::<f32>::zero(workspace_size);

        convolution.forward(&mut cudnn, 1.0, 1.0, &mut input_desc.link_mut(&mut input_data), &kernel_desc, &kernel_data,
                           &mut workspace, &mut input_desc.link_mut(&mut output_data), algo);

    }


    #[test]
    fn convolution2d() {
        let mut cudnn = Cudnn::new();

        let width = 2;
        let height = 4;

        let convolution = CuConvolutionDescriptor::<f32>::new_2d(
            1, 1,
            1, 1,
            1, 1,
            CudnnConvolutionMode::CrossCorrelation
        );
        let kernel_desc = CuFilterDescriptor::<f32>::new_4d(
            CudnnTensorFormat::Nchw,
            3, 3, 3, 3
        );
        let input_desc = CuTensorDescriptor::<f32>::new_4d(
            CudnnTensorFormat::Nhwc,
            1, 3,
            height, width,
        );
        let output_desc = CuTensorDescriptor::<f32>::new_4d(
            CudnnTensorFormat::Nhwc,
            1,3,
            height, width,
        );

        //println!("conv_desc = {:?}", convolution.get_info());
        //println!("input_desc = {:?}", input_desc.get_info());
        //println!("output_desc = {:?}", output_desc.get_info());
        //println!("kernel_desc = {:?}", kernel_desc.get_info(4));


        let mut input_data = CuVector::<f32>::new(1.0, input_desc.data_len());
        let mut output_data = CuVector::<f32>::zero(output_desc.data_len());
        let kernel_data = CuVector::<f32>::new(2.0, 9);

        let algo = CudnnConvolutionFwdAlgo::Gemm;

        let workspace_size = convolution.get_forward_workspace_size(&cudnn, &input_desc,
                                                                    &kernel_desc,
                                                                    &output_desc, algo);
        let mut workspace = CuVector::<f32>::zero(workspace_size);

        convolution.forward(&mut cudnn,
                            1.0, 1.0,
                            &mut input_desc.link_mut(&mut input_data),
                            &kernel_desc, &kernel_data,
                            &mut workspace,
                            &mut output_desc.link_mut(&mut output_data), algo);

        //println!("Input = {:?}", input_data);
        //println!("Output = {:?}", output_data);

    }

}