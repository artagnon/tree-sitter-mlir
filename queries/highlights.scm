"func.func" @keyword
"llvm.func" @keyword

"return" @function
"llvm.return" @function
"arith.addi" @function

(type) @type
(integer_literal) @number
(float_literal) @number

(func_dialect name: (symbol_ref_id) @function)
(llvm_dialect name: (symbol_ref_id) @function)

(function_arg_list (value_use) @variable.parameter)
(successor (value_use_list (value_use)) @variable.parameter)
