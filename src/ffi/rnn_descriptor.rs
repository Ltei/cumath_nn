
use std::os::raw::c_void;
use super::{CudnnStatus, CudnnRNNInputMode, CudnnRNNAlgo, CudnnRNNMode, CudnnDirectionMode, CudnnDataType};
use super::cudnn::_CudnnStruct;
use super::tensor_descriptor::_TensorDescriptorStruct;
use super::filter_descriptor::_FilterDescriptorStruct;
use super::dropout_descriptor::_DropoutDescriptorStruct;



pub enum _RNNDescriptorStruct {}


#[allow(non_snake_case)]
extern {

    fn cudnnCreateRNNDescriptor(rnnDesc: *mut*mut _RNNDescriptorStruct) -> CudnnStatus;

    fn cudnnDestroyRNNDescriptor(rnnDesc: *mut _RNNDescriptorStruct) -> CudnnStatus;

    fn cudnnSetRNNDescriptor(
        handle: *mut _CudnnStruct,
        rnnDesc: *mut _RNNDescriptorStruct,
        hiddenSize: i32,
        numLayers: i32,
        dropoutDesc: *const _DropoutDescriptorStruct,
        inputMode: CudnnRNNInputMode,
        direction: CudnnDirectionMode,
        mode: CudnnRNNMode,
        algo: CudnnRNNAlgo,
        dataType: CudnnDataType,
    ) -> CudnnStatus;

    fn cudnnGetRNNDescriptor(
        handle: *mut _CudnnStruct,
        rnnDesc: *const _RNNDescriptorStruct,
        hiddenSize: *mut i32,
        numLayers: *mut i32,
        dropoutDesc: *mut _DropoutDescriptorStruct,
        inputMode: *mut CudnnRNNInputMode,
        direction: *mut CudnnDirectionMode,
        mode: *mut CudnnRNNMode,
        algo: *mut CudnnRNNAlgo,
        dataType: *mut CudnnDataType,
    ) -> CudnnStatus;

    fn cudnnGetRNNWorkspaceSize(
        handle: *mut _CudnnStruct,
        rnnDesc: *const _RNNDescriptorStruct,
        seqLength: i32,
        xDesc: *const*const _TensorDescriptorStruct,
        sizeInBytes: *mut usize,
    ) -> CudnnStatus;

    fn cudnnGetRNNTrainingReserveSize(
        handle: *mut _CudnnStruct,
        rnnDesc: *const _RNNDescriptorStruct,
        seqLength: i32,
        xDesc: *const*const _TensorDescriptorStruct,
        sizeInBytes: *mut usize,
    ) -> CudnnStatus;

    fn cudnnRNNForwardInference(
        handle: *mut _CudnnStruct,
        rnnDesc: *const _RNNDescriptorStruct,
        seqLength: i32,
        xDesc: *const*const _TensorDescriptorStruct,
        x: *const c_void,
        hxDesc: *const _TensorDescriptorStruct,
        hx: *const c_void,
        cxDesc: *const _TensorDescriptorStruct,
        cx: *const c_void,
        wDesc: *const _FilterDescriptorStruct,
        w: *const c_void,
        yDesc: *const*const _TensorDescriptorStruct,
        y: *mut c_void,
        hyDesc: *const _TensorDescriptorStruct,
        hy: *mut c_void,
        cyDesc: *const _TensorDescriptorStruct,
        cy: *mut c_void,
        workspace: *mut c_void,
        workspaceSizeInBytes: usize,
    ) -> CudnnStatus;

    fn cudnnRNNForwardTraining(
        handle: *mut _CudnnStruct,
        rnnDesc: *const _RNNDescriptorStruct,
        seqLength: i32,
        xDesc: *const*const _TensorDescriptorStruct,
        x: *const c_void,
        hxDesc: *const _TensorDescriptorStruct,
        hx: *const c_void,
        cxDesc: *const _TensorDescriptorStruct,
        cx: *const c_void,
        wDesc: *const _FilterDescriptorStruct,
        w: *const c_void,
        yDesc: *const*const _TensorDescriptorStruct,
        y: *mut c_void,
        hyDesc: *const _TensorDescriptorStruct,
        hy: *mut c_void,
        cyDesc: *const _TensorDescriptorStruct,
        cy: *mut c_void,
        workspace: *mut c_void,
        workspaceSizeInBytes: usize,
        reserveSpace: *mut c_void,
        reserveSpaceSizeInBytes: usize,
    ) -> CudnnStatus;

    fn cudnnRNNBackwardData(
        handle: *mut _CudnnStruct,
        rnnDesc: *const _RNNDescriptorStruct,
        seqLength: i32,
        yDesc: *const*const _TensorDescriptorStruct,
        y: *const c_void,
        dyDesc: *const*const _TensorDescriptorStruct,
        dy: *const c_void,
        dhyDesc: *const _TensorDescriptorStruct,
        dhy: *const c_void,
        dcyDesc: *const _TensorDescriptorStruct,
        dcy: *const c_void,
        wDesc: *const _FilterDescriptorStruct,
        w: *const c_void,
        hxDesc: *const _TensorDescriptorStruct,
        hx: *const c_void,
        cxDesc: *const _TensorDescriptorStruct,
        cx: *const c_void,
        dxDesc: *const*const _TensorDescriptorStruct,
        dx: *mut c_void,
        dhxDesc: *const _TensorDescriptorStruct,
        dhx: *mut c_void,
        dcxDesc: *const _TensorDescriptorStruct,
        dcx: *mut c_void,
        workspace: *mut c_void,
        workspaceSizeInBytes: usize,
        reserveSpace: *mut c_void,
        reserveSpaceSizeInBytes: usize,
    ) -> CudnnStatus;

    fn cudnnRNNBackwardWeights(
        handle: *mut _CudnnStruct,
        rnnDesc: *const _RNNDescriptorStruct,
        seqLength: i32,
        xDesc: *const*const _TensorDescriptorStruct,
        x: *const c_void,
        hxDesc: *const _TensorDescriptorStruct,
        hx: *const c_void,
        yDesc: *const*const _TensorDescriptorStruct,
        y: *const c_void,
        workspace: *mut c_void,
        workspaceSizeInBytes: usize,
        dwDesc: *const _FilterDescriptorStruct,
        dw: *const c_void,
        reserveSpace: *mut c_void,
        reserveSpaceSizeInBytes: usize,
    ) -> CudnnStatus;

}





#[inline]
pub fn cudnn_create_rnn_descriptor(rnn_desc: *mut*mut _RNNDescriptorStruct) {
    #[cfg(not(feature = "disable_checks"))] {
        unsafe { cudnnCreateRNNDescriptor(rnn_desc) }.assert_success();
    }
    #[cfg(feature = "disable_checks")] {
        unsafe { cudnnCreateRNNDescriptor(rnn_desc) };
    }
}

#[inline]
pub fn cudnn_destroy_rnn_descriptor(rnn_desc: *mut _RNNDescriptorStruct) {
    #[cfg(not(feature = "disable_checks"))] {
        unsafe { cudnnDestroyRNNDescriptor(rnn_desc) }.assert_success();
    }
    #[cfg(feature = "disable_checks")] {
        unsafe { cudnnDestroyRNNDescriptor(rnn_desc) };
    }
}

#[inline]
pub fn cudnn_set_rnn_descriptor(handle: *mut _CudnnStruct, rnn_desc: *mut _RNNDescriptorStruct, hidden_size: i32, num_layers: i32, dropout_desc: *const _DropoutDescriptorStruct, input_mode: CudnnRNNInputMode, direction: CudnnDirectionMode, mode: CudnnRNNMode, algo: CudnnRNNAlgo, data_type: CudnnDataType) {
    #[cfg(not(feature = "disable_checks"))] {
        unsafe { cudnnSetRNNDescriptor(handle, rnn_desc, hidden_size, num_layers, dropout_desc, input_mode, direction, mode, algo, data_type) }.assert_success();
    }
    #[cfg(feature = "disable_checks")] {
        unsafe { cudnnSetRNNDescriptor(handle, rnn_desc, hidden_size, num_layers, dropout_desc, input_mode, direction, mode, algo, data_type) };
    }
}

#[inline]
pub fn cudnn_get_rnn_descriptor(handle: *mut _CudnnStruct, rnn_desc: *const _RNNDescriptorStruct, hidden_size: *mut i32, num_layers: *mut i32, dropout_desc: *mut _DropoutDescriptorStruct, input_mode: *mut CudnnRNNInputMode, direction: *mut CudnnDirectionMode, mode: *mut CudnnRNNMode, algo: *mut CudnnRNNAlgo, data_type: *mut CudnnDataType) {
    #[cfg(not(feature = "disable_checks"))] {
        unsafe { cudnnGetRNNDescriptor(handle, rnn_desc, hidden_size, num_layers, dropout_desc, input_mode, direction, mode, algo, data_type) }.assert_success();
    }
    #[cfg(feature = "disable_checks")] {
        unsafe { cudnnGetRNNDescriptor(handle, rnn_desc, hidden_size, num_layers, dropout_desc, input_mode, direction, mode, algo, data_type) };
    }
}

#[inline]
pub fn cudnn_get_rnn_workspace_size(handle: *mut _CudnnStruct, rnn_desc: *const _RNNDescriptorStruct, seq_length: i32, x_desc: *const*const _TensorDescriptorStruct, size_in_bytes: *mut usize) {
    #[cfg(not(feature = "disable_checks"))] {
        unsafe { cudnnGetRNNWorkspaceSize(handle, rnn_desc, seq_length, x_desc, size_in_bytes) }.assert_success();
    }
    #[cfg(feature = "disable_checks")] {
        unsafe { cudnnGetRNNWorkspaceSize(handle, rnn_desc, seq_length, x_desc, size_in_bytes) };
    }
}

#[inline]
pub fn cudnn_get_rnn_training_reserve_size(handle: *mut _CudnnStruct, rnn_desc: *const _RNNDescriptorStruct, seq_length: i32, x_desc: *const*const _TensorDescriptorStruct, size_in_bytes: *mut usize) {
    #[cfg(not(feature = "disable_checks"))] {
        unsafe { cudnnGetRNNTrainingReserveSize(handle, rnn_desc, seq_length, x_desc, size_in_bytes) }.assert_success();
    }
    #[cfg(feature = "disable_checks")] {
        unsafe { cudnnGetRNNTrainingReserveSize(handle, rnn_desc, seq_length, x_desc, size_in_bytes) };
    }
}

#[inline]
pub fn cudnn_rnn_forward_inference(handle: *mut _CudnnStruct, rnn_desc: *const _RNNDescriptorStruct, seq_length: i32, x_desc: *const*const _TensorDescriptorStruct, x: *const c_void, hx_desc: *const _TensorDescriptorStruct, hx: *const c_void, cx_desc: *const _TensorDescriptorStruct, cx: *const c_void, w_desc: *const _FilterDescriptorStruct, w: *const c_void, y_desc: *const*const _TensorDescriptorStruct, y: *mut c_void, hy_desc: *const _TensorDescriptorStruct, hy: *mut c_void, cy_desc: *const _TensorDescriptorStruct, cy: *mut c_void, workspace: *mut c_void, workspace_size_in_bytes: usize) {
    #[cfg(not(feature = "disable_checks"))] {
        unsafe { cudnnRNNForwardInference(handle, rnn_desc, seq_length, x_desc, x, hx_desc, hx, cx_desc, cx, w_desc, w, y_desc, y, hy_desc, hy, cy_desc, cy, workspace, workspace_size_in_bytes) }.assert_success();
    }
    #[cfg(feature = "disable_checks")] {
        unsafe { cudnnRNNForwardInference(handle, rnn_desc, seq_length, x_desc, x, hx_desc, hx, cx_desc, cx, w_desc, w, y_desc, y, hy_desc, hy, cy_desc, cy, workspace, workspace_size_in_bytes) };
    }
}

#[inline]
pub fn cudnn_rnn_forward_training(handle: *mut _CudnnStruct, rnn_desc: *const _RNNDescriptorStruct, seq_length: i32, x_desc: *const*const _TensorDescriptorStruct, x: *const c_void, hx_desc: *const _TensorDescriptorStruct, hx: *const c_void, cx_desc: *const _TensorDescriptorStruct, cx: *const c_void, w_desc: *const _FilterDescriptorStruct, w: *const c_void, y_desc: *const*const _TensorDescriptorStruct, y: *mut c_void, hy_desc: *const _TensorDescriptorStruct, hy: *mut c_void, cy_desc: *const _TensorDescriptorStruct, cy: *mut c_void, workspace: *mut c_void, workspace_size_in_bytes: usize, reserve_space: *mut c_void, reserve_space_size_in_bytes: usize) {
    #[cfg(not(feature = "disable_checks"))] {
        unsafe { cudnnRNNForwardTraining(handle, rnn_desc, seq_length, x_desc, x, hx_desc, hx, cx_desc, cx, w_desc, w, y_desc, y, hy_desc, hy, cy_desc, cy, workspace, workspace_size_in_bytes, reserve_space, reserve_space_size_in_bytes) }.assert_success();
    }
    #[cfg(feature = "disable_checks")] {
        unsafe { cudnnRNNForwardTraining(handle, rnn_desc, seq_length, x_desc, x, hx_desc, hx, cx_desc, cx, w_desc, w, y_desc, y, hy_desc, hy, cy_desc, cy, workspace, workspace_size_in_bytes, reserve_space, reserve_space_size_in_bytes) };
    }
}

#[inline]
pub fn cudnn_rnn_backward_data(handle: *mut _CudnnStruct, rnn_desc: *const _RNNDescriptorStruct, seq_length: i32, y_desc: *const*const _TensorDescriptorStruct, y: *const c_void, dy_desc: *const*const _TensorDescriptorStruct, dy: *const c_void, dhy_desc: *const _TensorDescriptorStruct, dhy: *const c_void, dcy_desc: *const _TensorDescriptorStruct, dcy: *const c_void, w_desc: *const _FilterDescriptorStruct, w: *const c_void, hx_desc: *const _TensorDescriptorStruct, hx: *const c_void, cx_desc: *const _TensorDescriptorStruct, cx: *const c_void, dx_desc: *const*const _TensorDescriptorStruct, dx: *mut c_void, dhx_desc: *const _TensorDescriptorStruct, dhx: *mut c_void, dcx_desc: *const _TensorDescriptorStruct, dcx: *mut c_void, workspace: *mut c_void, workspace_size_in_bytes: usize, reserve_space: *mut c_void, reserve_space_size_in_bytes: usize) {
    #[cfg(not(feature = "disable_checks"))] {
        unsafe { cudnnRNNBackwardData(handle, rnn_desc, seq_length, y_desc, y, dy_desc, dy, dhy_desc, dhy, dcy_desc, dcy, w_desc, w, hx_desc, hx, cx_desc, cx, dx_desc, dx, dhx_desc, dhx, dcx_desc, dcx, workspace, workspace_size_in_bytes, reserve_space, reserve_space_size_in_bytes) }.assert_success();
    }
    #[cfg(feature = "disable_checks")] {
        unsafe { cudnnRNNBackwardData(handle, rnn_desc, seq_length, y_desc, y, dy_desc, dy, dhy_desc, dhy, dcy_desc, dcy, w_desc, w, hx_desc, hx, cx_desc, cx, dx_desc, dx, dhx_desc, dhx, dcx_desc, dcx, workspace, workspace_size_in_bytes, reserve_space, reserve_space_size_in_bytes) };
    }
}

#[inline]
pub fn cudnn_rnn_backward_weights(handle: *mut _CudnnStruct, rnn_desc: *const _RNNDescriptorStruct, seq_length: i32, x_desc: *const*const _TensorDescriptorStruct, x: *const c_void, hx_desc: *const _TensorDescriptorStruct, hx: *const c_void, y_desc: *const*const _TensorDescriptorStruct, y: *const c_void, workspace: *mut c_void, workspace_size_in_bytes: usize, dw_desc: *const _FilterDescriptorStruct, dw: *const c_void, reserve_space: *mut c_void, reserve_space_size_in_bytes: usize) {
    #[cfg(not(feature = "disable_checks"))] {
        unsafe { cudnnRNNBackwardWeights(handle, rnn_desc, seq_length, x_desc, x, hx_desc, hx, y_desc, y, workspace, workspace_size_in_bytes, dw_desc, dw, reserve_space, reserve_space_size_in_bytes) }.assert_success();
    }
    #[cfg(feature = "disable_checks")] {
        unsafe { cudnnRNNBackwardWeights(handle, rnn_desc, seq_length, x_desc, x, hx_desc, hx, y_desc, y, workspace, workspace_size_in_bytes, dw_desc, dw, reserve_space, reserve_space_size_in_bytes) };
    }
}



