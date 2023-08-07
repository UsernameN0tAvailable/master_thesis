#include <torch/extension.h>

at::Tensor backward_weight(
    const at::Tensor& grad_output,
    const at::Tensor& input,
    const at::Tensor& weight,
    c10::IntArrayRef stride,
    c10::IntArrayRef padding,
    c10::IntArrayRef dilation,
    int64_t groups) {

    at::Tensor grad_input, grad_weight, grad_bias;
    std::tie(grad_input, grad_weight, grad_bias) = 
        at::convolution_backward(
            grad_output, 
            input, 
            weight, 
            at::nullopt, // bias is not being used, so use at::nullopt
            stride, 
            padding, 
            dilation, 
            false, // transposed
            {0, 0}, // output_padding
            groups,
            {true, true, false} // output_mask
        );
    return grad_weight;
}

at::Tensor backward_input(
    const at::Tensor& grad_output,
    const at::Tensor& weight,
    c10::IntArrayRef stride,
    c10::IntArrayRef padding,
    c10::IntArrayRef dilation,
    int64_t groups) {

    at::Tensor grad_input, grad_weight, grad_bias;
    std::tie(grad_input, grad_weight, grad_bias) = 
        at::convolution_backward(
            grad_output, 
            at::Tensor(), // We don't need to calculate the gradient of weight in this case, so we pass an empty tensor
            weight, 
            at::nullopt, // bias is not being used, so use at::nullopt
            stride, 
            padding, 
            dilation, 
            false, // transposed
            {0, 0}, // output_padding
            groups,
            {true, false, false} // output_mask
        );
    return grad_input;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("backward_weight", &backward_weight, "Conv backward_weight 2D");
    m.def("backward_input", &backward_input, "Conv backward_input 2D");
}
