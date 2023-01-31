[
  "func.func"
  "llvm.func"
] @keyword

[
  "return"
  "llvm.return"
  "arith.addi"
  "cf.br"
  "cf.cond_br"
] @function

(type) @type

[
  (integer_literal)
  (float_literal)
] @number

[
  "("
  ")"
  "{"
  "}"
] @punctuation.bracket

[
  ":"
  "->"
  ","
] @punctuation

(string_literal) @string

(func_dialect name: (symbol_ref_id) @function)
(llvm_dialect name: (symbol_ref_id) @function)

(function_arg_list (value_use) @variable.parameter)
(block_arg_list (value_use) @variable.parameter)

(caret_id) @tag
(value_use) @variable
(comment) @comment
