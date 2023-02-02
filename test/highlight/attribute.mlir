func.func @depthwise_conv_1d_nwc_wcm(%input: tensor<1x12x8xf32>, %filter: tensor<3x8x8xf32>)
// <- function.builtin
//        ^ function
//                                   ^ variable.parameter
//                                           ^ type.builtin
//                                                               ^ variable.parameter
//                                                                        ^ type.builtin
  -> tensor<1x10x8x8xf32> {
// ^ operator
//   ^ type.builtin
  %zero = arith.constant 0.000000e+00 : f32
// ^ variable
//        ^ function.builtin
//                       ^ number
//                                      ^ type.builtin
  %init = tensor.empty() : tensor<1x10x8x8xf32>
// ^ variable
//        ^ function.builtin
//                         ^ type.builtin
  %fill = linalg.fill ins(%zero : f32) outs(%init : tensor<1x10x8x8xf32>) -> tensor<1x10x8x8xf32>
// ^ variable
//        ^ function.builtin
//                    ^ keyword
//                        ^ variable
//                                ^ type.builtin
//                                     ^ keyword
  %0 = linalg.depthwise_conv_1d_nwc_wcm {dilations = dense<1> : tensor<1xi64>,
// ^ variable
//     ^ function.builtin
//                                       ^ attribute
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

func.func @fastmath(%arg0: f32, %arg1: f32) {
// <- function.builtin
//        ^ function
//                  ^ variable.parameter
//                         ^ type.builtin
//                              ^ variable.parameter
//                                     ^ type.builtin
  %5 = arith.negf %arg0 fastmath<fast> : f32
//     ^ function.builtin
//                      ^ attribute
  %6 = arith.addf %arg0, %arg1 fastmath<none> : f32
//     ^ function.builtin
//                             ^ attribute
  %8 = arith.mulf %arg0, %arg1 fastmath<reassoc,nnan,ninf,nsz,arcp,contract,afn> : f32
//     ^ function.builtin
//                             ^ attribute
  return
// ^ function.builtin
}
