



pub(crate) struct DimensionsStrides {
    len: i32,
    dimensions: Vec<i32>,
    strides: *const i32,
}

impl DimensionsStrides {

    pub fn new_packed(dimensions: &[i32]) -> DimensionsStrides {
        DimensionsStrides {
            len: dimensions.len() as i32,
            dimensions
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