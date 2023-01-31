func.func @simple(i64, i1) -> i64 {
// <- keyword
//        ^ function
//               ^ punctuation.bracket
//                ^ type
//                   ^ punctuation.delimeter
//                     ^ type
//                       ^ punctuation.bracket
//                         ^ operator
//                            ^ type
//                                ^ punctuation.bracket
^bb0(%a: i64, %cond: i1):
// <- tag
//   ^ variable.parameter
//       ^ type
//            ^ variable.parameter
//                   ^ type
  cf.cond_br %cond, ^bb1, ^bb2
// ^ function
//           ^ variable.parameter
//                  ^ tag
//                        ^ tag

^bb1:
// <- tag
  cf.br ^bb3(%a: i64)    // Branch passes %a as the argument
// ^ function
//      ^ tag
//           ^ variable.parameter
//               ^ type
//                       ^ comment

^bb2:
// <- tag
  %b = arith.addi %a, %a : i64
// ^ variable
//   ^ operator
//     ^ function
//                ^ variable.parameter
//                    ^ variable.parameter
//                         ^ type
  cf.br ^bb3(%b: i64)    // Branch passes %b as the argument
// ^ function
//      ^ tag
//           ^ variable
//               ^ type
//                       ^ comment
^bb3(%c: i64):
// <- tag
//   ^ variable.parameter
//        ^ type
  cf.br ^bb4(%c, %a : i64, i64)
// ^ function
//      ^ tag
//           ^ variable.parameter
//               ^ variable.parameter
//                    ^ type
//                         ^ type
^bb4(%d : i64, %e : i64):
// <- tag
//   ^ variable.parameter
//        ^ type
//             ^ variable.parameter
//                  ^ type
  %0 = arith.addi %d, %e : i64
// ^ variable
//   ^ operator
//     ^ function
//                ^ variable.parameter
//                    ^ variable.parameter
//                          ^ type
  return %0 : i64   // Return is also a terminator.
//       ^ variable
//            ^ type
//                  ^ comment
}
// <- punctuation.bracket
