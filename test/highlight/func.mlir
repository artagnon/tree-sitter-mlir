func.func @test_addi(%arg0 : i64, %arg1 : i64) -> i64 {
// <- function.builtin
//        ^ function
//                  ^ punctuation.bracket
//                   ^ variable.parameter
//                         ^ punctuation.delimeter
//                           ^ type
//                              ^ punctuation.delimeter
//                                ^ variable.parameter
//                                        ^ type
//                                           ^ punctuation.bracket
//                                             ^ operator
//                                                ^ type
//                                                    ^ punctuation.bracket
  %0 = arith.addi %arg0, %arg1 : i64
// ^ variable
//   ^ operator
//     ^ function.builtin
//                ^ variable.parameter
//                       ^ variable.parameter
//                               ^ type
  return %0 : i64
// ^ function.builtin
//       ^ variable
//            ^ type
}
// <- punctuation.bracket
