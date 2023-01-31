func.func @depthwise_conv_1d_nwc_wcm(%input: tensor<1x12x8xf32>, %filter: tensor<3x8x8xf32>)
// <- function.builtin
//        ^ function
//                                   ^ variable.parameter
//                                           ^ type
//                                                               ^ variable.parameter
//                                                                        ^ type
  -> tensor<1x10x8x8xf32> {
// ^ operator
//   ^ type
  %zero = arith.constant 0.000000e+00 : f32
// ^ variable
//        ^ function.builtin
//                       ^ number
//                                      ^ type
  %init = tensor.empty() : tensor<1x10x8x8xf32>
// ^ variable
//        ^ function.builtin
//                         ^ type
  %fill = linalg.fill ins(%zero : f32) outs(%init : tensor<1x10x8x8xf32>) -> tensor<1x10x8x8xf32>
// ^ variable
//        ^ function.builtin
//                    ^ keyword
//                        ^ variable
//                                ^ type
//                                     ^ keyword
  %0 = linalg.depthwise_conv_1d_nwc_wcm {dilations = dense<1> : tensor<1xi64>,
// ^ variable
//     ^ function.builtin
//                                       ^ property
//                                                   ^ constant.builtin
    strides = dense<1> : tensor<1xi64>}
//            ^ constant.builtin
    ins(%input, %filter : tensor<1x12x8xf32>, tensor<3x8x8xf32>)
//      ^ variable.parameter
//              ^ variable.parameter
    outs(%fill : tensor<1x10x8x8xf32>) -> tensor<1x10x8x8xf32>
//       ^ variable
  return %0 : tensor<1x10x8x8xf32>
// ^ function.builtin
//       ^ variable
}
