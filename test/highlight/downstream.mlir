func.func @sort_memref(%input1: memref<?x?xf32>, %input2: memref<?x?xi32>,
// <- function.builtin
//        ^ function
//                     ^ variable.parameter
//                              ^ type.builtin
                       %init1: memref<?x?xf32>, %init2: memref<?x?xi32>) {
  thlo.sort
      ins(%input1: memref<?x?xf32>, %input2: memref<?x?xi32>)
//    ^ keyword
//                                  ^ variable.parameter
      outs(%init1: memref<?x?xf32>, %init2: memref<?x?xi32>)
//    ^ keyword
//                                  ^ variable
      { dimension = 0 : i64, is_stable = true }
//                                       ^ constant.builtin
      (%e11: f32, %e12: f32, %e21: i32, %e22: i32) {
        %gt = arith.cmpf ogt, %e11, %e12: f32
//            ^ function.builtin
//                       ^ keyword
//                            ^ variable
//                                  ^ variable
//                                        ^ type.builtin
        thlo.yield %gt : i1
      }
  func.return
// ^ function.builtin
}

module @arctorustadt {
// <- function.builtin
//     ^ function
      func.func @ok0(%in: !arc.adt<"i32">) -> () {
//    ^ function.builtin
//              ^ function
//                   ^ variable.parameter
//                        ^ type
            return
//          ^ function.builtin
      }

      func.func @ok2(%in: !arc.adt<"i32">) -> !arc.adt<"i32"> {
//    ^ function.builtin
//              ^ function
//                   ^ variable.parameter
//                        ^ type
//                                            ^ type
            return %in: !arc.adt<"i32">
//          ^ function.builtin
//                 ^ variable.parameter
//                      ^ type
      }

      func.func @ok4() -> !arc.adt<"i32"> {
//    ^ function.builtin
//              ^ function
//                        ^ type
            %out = arc.adt_constant "4711": !arc.adt<"i32">
            return %out: !arc.adt<"i32">
//          ^ function.builtin
//                       ^ type
      }
}
