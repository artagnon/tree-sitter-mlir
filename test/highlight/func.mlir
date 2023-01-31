func.func @test_addi(%arg0 : i64, %arg1 : i64) -> i64 {
// <- keyword
//        ^ function
//                  ^ punctuation.bracket
//                   ^ variable.parameter
//                           ^ type
//                                ^ variable.parameter
//                                        ^ type
//                                           ^ punctuation.bracket
//                                                ^ type
//                                                    ^ punctuation.bracket
  %0 = arith.addi %arg0, %arg1 : i64
// ^ variable
//     ^ function
//                ^ variable.parameter
//                       ^ variable.parameter
//                               ^ type
  return %0 : i64
// ^ function
//       ^ variable
//            ^ type
}
// <- punctuation.bracket
