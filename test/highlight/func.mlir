func.func @test_addi(%arg0 : i64, %arg1 : i64) -> i64 {
// <- keyword
//        ^ function
//                   ^ variable.parameter
//                           ^ type
//                                ^ variable.parameter
//                                        ^ type
//                                                ^ type
  %0 = arith.addi %arg0, %arg1 : i64
//     ^ function
//                               ^ type
  return %0 : i64
// ^ function
//            ^ type
}
