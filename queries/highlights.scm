[
  "func.func"
  "llvm.func"
  "ins"
  "outs"
  "else"
  "loc"
  "attributes"
] @keyword

[
  "return"
  "llvm.return"
  "arith.constant"
  "arith.addi"
  "cf.br"
  "cf.cond_br"
  "tensor.empty"
  "linalg.fill"
  "linalg.depthwise_conv_1d_nwc_wcm"
] @function

(type) @type

[
  (integer_literal)
  (float_literal)
] @number

[
  (bool_literal)
  (complex_literal)
  (tensor_literal)
  (unit_literal)
] @constant.builtin

(string_literal) @string

(attribute) @property

[
  "("
  ")"
  "{"
  "}"
  "["
  "]"
] @punctuation.bracket

[
  ":"
  ","
] @punctuation.delimeter

[
  "="
  "->"
] @operator

(func_dialect name: (symbol_ref_id) @function)
(llvm_dialect name: (symbol_ref_id) @function)

(function_arg_list (value_use) @variable.parameter)
(block_arg_list (value_use) @variable.parameter)

(caret_id) @tag
(value_use) @variable
(comment) @comment
