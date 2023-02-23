'use strict';

module.exports = {
  func_dialect: $ => prec.right(choice(
    // operation ::= `func.call_indirect` $callee `(` $callee_operands `)` attr-dict
    //               `:` type($callee)
    // operation ::= `func.call` $callee `(` $operands `)` attr-dict
    //               `:` functional-type($operands, results)
    seq(choice('func.call', 'call', 'func.call_indirect', 'call_indirect'),
      field('callee', $.symbol_ref_id),
      field('operands', $._value_use_list_parens),
      field('attributes', optional($.attribute)),
      field('return', $._function_type_annotation)),

    // operation ::= `func.constant` attr-dict $value `:` type(results)
    seq(choice('func.constant', 'constant'),
      field('attributes', optional($.attribute)),
      field('value', $.symbol_ref_id),
      field('return', $._function_type_annotation)),

    seq('func.func', $._op_func),

    seq(choice('func.return', 'return'),
      field('attributes', optional($.attribute)),
      field('results', optional($._value_use_type_list)))
  )),

  func_return: $ => seq(token('->'), $.type_list_attr_parens),
  func_arg_list: $ => seq('(', optional(choice($.variadic,
    $._value_id_and_type_attr_list)), ')'),
  _value_id_and_type_attr_list: $ => seq($._value_id_and_type_attr,
    repeat(seq(',', $._value_id_and_type_attr)), optional(seq(',', $.variadic))),
  _value_id_and_type_attr: $ => seq($._function_arg, optional($.attribute)),
  _function_arg: $ => choice(seq($.value_use, ':', $.type), $.value_use, $.type),
  type_list_attr_parens: $ => choice($.type, seq('(', $.type, optional($.attribute),
    repeat(seq(',', $.type, optional($.attribute))), ')'), seq('(', ')')),
  variadic: $ => token('...'),

  // (func.func|llvm.func) takes arguments, an optional return type, and and optional body
  _op_func: $ => seq(
    field('visibility', optional('private')),
    field('name', $.symbol_ref_id),
    field('arguments', $.func_arg_list),
    field('return', optional($.func_return)),
    field('attributes', optional(seq(token('attributes'), $.attribute))),
    field('body', optional($.region)))
}
