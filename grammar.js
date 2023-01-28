module.exports = grammar({
  name: 'mlir',
  extras: $ => [/\s/,
    $.comment,
  ],
  conflicts: $ => [],
  rules: {
    // Top level production:
    //   (operation | attribute-alias-def | type-alias-def)
    toplevel: $ => seq(choice(
      $.operation,
      $.attribute_alias_def,
      $.type_alias_def,
    )),

    // Common syntax (lang-ref)
    //  digit     ::= [0-9]
    //  hex_digit ::= [0-9a-fA-F]
    //  letter    ::= [a-zA-Z]
    //  id-punct  ::= [$._-]
    //
    //  integer-literal ::= decimal-literal | hexadecimal-literal
    //  decimal-literal ::= digit+
    //  hexadecimal-literal ::= `0x` hex_digit+
    //  float-literal ::= [-+]?[0-9]+[.][0-9]*([eE][-+]?[0-9]+)?
    //  string-literal  ::= `"` [^"\n\f\v\r]* `"`   TODO: define escaping rules
    //
    _digit: $ => /[0-9]/,
    _hex_digit: $ => /[0-9a-fA-F]/,
    integer_literal: $ => choice($._decimal_literal, $._hexadecimal_literal),
    _decimal_literal: $ => repeat1($._digit),
    _hexadecimal_literal: $ => seq('0x', repeat1($._hex_digit)),
    float_literal: $ => token(
      seq(optional(/[-+]/), repeat1(/[0_9]/),
        optional(seq('.', repeat(/[0-9]/),
          optional(seq(/[eE]/, optional(/[-+]/),
            repeat1(/[0-9]/))))))),
    string_literal: $ => seq(
      '"',
      repeat(token.immediate(prec(1, /[^\\"\n\f\v\r]+/))),
      '"',
    ),
    bool_literal: $ => choice('true', 'false'),
    literal: $ => choice($.integer_literal, $.float_literal,
      $.string_literal, $.bool_literal, 'unit'),

    // Identifiers
    //   bare-id ::= (letter|[_]) (letter|digit|[_$.])*
    //   bare-id-list ::= bare-id (`,` bare-id)*
    //   value-id ::= `%` suffix-id
    //   suffix-id ::= (digit+ | ((letter|id-punct) (letter|id-punct|digit)*))
    //   alias-name :: = bare-id
    //
    //   symbol-ref-id ::= `@` (suffix-id | string-literal) (`::`
    //   symbol-ref-id)?
    //   value-id-list ::= value-id (`,` value-id)*
    //
    //   // Uses of value, e.g. in an operand list to an operation.
    //   value-use ::= value-id
    //   value-use-list ::= value-use (`,` value-use)*
    bare_id: $ => seq(token(/[a-zA-Z_]/),
      token.immediate(repeat(/[a-zA-Z0-9_$.]/))),
    _alias_or_dialect_id: $ => seq(token(/[a-zA-Z_]/),
      token.immediate(repeat(/[a-zA-Z0-9_$]/))),
    bare_id_list: $ => seq($.bare_id, repeat(seq(',', $.bare_id))),
    value_id: $ => seq('%', $._suffix_id),
    _suffix_id: $ => choice(repeat1(/[0-9]/),
      seq(/[a-zA-Z_$.]/, repeat(/[a-zA-Z0-9_$.]/))),
    symbol_ref_id: $ => seq('@', choice($._suffix_id, $.string_literal),
      optional(seq('::', $.symbol_ref_id))),
    value_use: $ => $.value_id,
    value_use_list: $ => seq($.value_use, repeat(seq(',', $.value_use))),

    // Operations
    //   operation            ::= op-result-list? (generic-operation |
    //                            custom-operation)
    //                            trailing-location?
    //   generic-operation    ::= string-literal `(` value-use-list? `)`
    //   successor-list?
    //                            region-list? dictionary-attribute? `:`
    //                            function-type
    //   custom-operation     ::= bare-id custom-operation-format
    //   op-result-list       ::= op-result (`,` op-result)* `=`
    //   op-result            ::= value-id (`:` integer-literal)
    //   successor-list       ::= `[` successor (`,` successor)* `]`
    //   successor            ::= caret-id (`:` bb-arg-list)?
    //   region-list          ::= `(` region (`,` region)* `)`
    //   dictionary-attribute ::= `{` (attribute-entry (`,` attribute-entry)*)?
    //                            `}`
    //   trailing-location    ::= (`loc` `(` location `)`)?
    operation: $ => seq(optional($.op_result_list),
      choice($.generic_operation, $.custom_operation),
      optional($.trailing_location)),
    generic_operation: $ =>
      seq($.string_literal, '(', optional($.value_use_list),
        ')', optional($.successor_list),
        optional($.region_list),
        optional($.dictionary_attribute), ':',
        $.function_type),
    // custom-operation rule is defined later in the grammar, post the generic.
    op_result_list: $ => seq($.op_result, repeat(seq(',', $.op_result)), '='),
    op_result: $ => seq($.value_id, optional(seq(':', $.integer_literal))),
    successor_list: $ => seq('[', $.successor, repeat(seq(',', $.successor)),
      ']'),
    successor: $ => seq($.caret_id, optional($.block_arg_list)),
    region_list: $ => seq('(', $.region, repeat(seq(',', $.region)), ')'),
    dictionary_attribute: $ => seq(
      '{',
      optional(seq($.attribute_entry,
        repeat(seq(',', $.attribute_entry)))),
      '}'),
    trailing_location: $ => seq('loc(', $.location, ')'),
    // TODO: Complete location forms.
    location: $ => $.string_literal,

    // Blocks
    //   block           ::= block-label operation+
    //   block-label     ::= block-id block-arg-list? `:`
    //   block-id        ::= caret-id
    //   caret-id        ::= `^` suffix-id
    //   value-id-and-type ::= value-id `:` type
    //
    //   // Non-empty list of names and types.
    //   value-id-and-type-list ::= value-id-and-type (`,` value-id-and-type)*
    //
    //   block-arg-list ::= `(` value-id-and-type-list? `)`
    block: $ => seq($.block_label, repeat1($.operation)),
    block_label: $ => seq($._block_id, optional($.block_arg_list), ':'),
    _block_id: $ => $.caret_id,
    caret_id: $ => seq('^', $._suffix_id),
    value_id_and_type: $ => seq(choice(seq($.value_id, ':', $.type), $.value_id, $.type)),
    _value_id_and_type_list: $ => seq($.value_id_and_type,
      repeat(seq(',', $.value_id_and_type))),
    block_arg_list: $ => seq('(', optional($._value_id_and_type_list), ')'),

    // Regions
    //   region      ::= `{` entry-block? block* `}`
    //   entry-block ::= operation+
    region: $ => seq('{', optional($.entry_block), repeat($.block), '}'),
    entry_block: $ => repeat1($.operation),

    // Types
    //   type ::= type-alias | dialect-type | builtin-type
    //
    //   type-list-no-parens ::=  type (`,` type)*
    //   type-list-parens ::= `(` type-list-no-parens? `)`
    //
    //   // This is a common way to refer to a value with a specified type.
    //   ssa-use-and-type ::= ssa-use `:` type
    //   ssa-use ::= value-use
    //
    //   // Non-empty list of names and types.
    //   ssa-use-and-type-list ::= ssa-use-and-type (`,` ssa-use-and-type)*
    //
    //   function-type ::= (type | type-list-parens) `->` (type |
    //   type-list-parens)
    type: $ => choice($.type_alias, $.dialect_type, $.builtin_type),
    type_list_no_parens: $ => seq($.type, repeat(seq(',', $.type))),
    type_list_parens: $ => seq('(', optional($.type_list_no_parens), ')'),
    ssa_use_and_type: $ => seq($.ssa_use, ':', $.type),
    ssa_use: $ => $.value_use,
    ssa_use_and_type_list: $ => seq($.ssa_use_and_type,
      repeat(seq(',', $.ssa_use_and_type))),
    function_type: $ => seq(choice($.type, $.type_list_parens), '->',
      choice($.type, $.type_list_parens)),

    // Type aliases
    //   type-alias-def ::= '!' alias-name '=' type
    //   type-alias ::= '!' alias-name
    type_alias_def: $ => seq('!', $._alias_or_dialect_id, '=', $.type),
    type_alias: $ => seq('!', $._alias_or_dialect_id),

    // Dialect Types
    //   dialect-namespace ::= bare-id
    //
    //   opaque-dialect-item ::= dialect-namespace '<' string-literal '>'
    //
    //   pretty-dialect-item ::= dialect-namespace '.'
    //   pretty-dialect-item-lead-ident
    //                                                 pretty-dialect-item-body?
    //
    //   pretty-dialect-item-lead-ident ::= '[A-Za-z][A-Za-z0-9._]*'
    //   pretty-dialect-item-body ::= '<' pretty-dialect-item-contents+ '>'
    //   pretty-dialect-item-contents ::= pretty-dialect-item-body
    //                                 | '(' pretty-dialect-item-contents+ ')'
    //                                 | '[' pretty-dialect-item-contents+ ']'
    //                                 | '{' pretty-dialect-item-contents+ '}'
    //                                 | '[^[<({>\])}\0]+'
    //
    //   dialect-type ::= '!' (opaque-dialect-item | pretty-dialect-item)
    dialect_type: $ => seq(
      '!', choice($.opaque_dialect_item, $.pretty_dialect_item)),
    dialect_namespace: $ => $._alias_or_dialect_id,
    dialect_ident: $ => $._alias_or_dialect_id,
    opaque_dialect_item: $ => seq($.dialect_namespace, '<', $.string_literal,
      '>'),
    pretty_dialect_item: $ => seq($.dialect_namespace, '.', $.dialect_ident,
      optional(seq('<', $.pretty_dialect_item_contents, '>'))),
    pretty_dialect_item_contents: $ => prec.left(choice(
      repeat1(/[^()\[\]{}<>]/),
      seq('(', repeat1($.pretty_dialect_item_contents), ')'),
      seq('[', repeat1($.pretty_dialect_item_contents), ']'),
      seq('{', repeat1($.pretty_dialect_item_contents), '}'),
      seq('<', repeat1($.pretty_dialect_item_contents), '>'))),

    // Builtin types
    builtin_type: $ => choice(
      // TODO: Add opaque_type, function_type
      $.integer_type,
      $.float_type,
      $.complex_type,
      $.index_type,
      $.memref_type,
      $.none_type,
      $.tensor_type,
      $.vector_type,
      $.tuple_type),

    // signed-integer-type ::= `si`[1-9][0-9]*
    // unsigned-integer-type ::= `ui`[1-9][0-9]*
    // signless-integer-type ::= `i`[1-9][0-9]*
    // integer-type ::= signed-integer-type | unsigned-integer-type | signless-integer-type
    integer_type: $ => seq(choice('si', 'ui', 'i'), /[1-9]/, repeat(/[0-9]/)),
    float_type: $ => choice('f16', 'f32', 'f64', 'f80', 'f128', 'bf16', 'f8E4M3FN', 'f8E5M2'),
    index_type: $ => 'index',
    _primitive_type: $ => choice($.integer_type, $.float_type, $.index_type),
    none_type: $ => 'none',
    complex_type: $ => seq('complex<', $._primitive_type, '>'),

    // memref-type ::= `memref` `<` dimension-list-ranked type
    //                 (`,` layout-specification)? (`,` memory-space)? `>`
    // layout-specification ::= attribute-value
    // memory-space ::= attribute-value
    memref_type: $ => seq('memref<', $.dim_list,
      optional(seq(',', $.attribute_value)), optional(seq(',', $.attribute_value)), '>'),
    dim_list: $ => seq($._memref_dim, repeat(seq('x', $._memref_dim))),
    _memref_dim: $ => choice($._primitive_type, $._decimal_literal, '?', '*'),

    // tensor-type ::= `tensor` `<` dimension-list type (`,` encoding)? `>`
    // dimension-list ::= (dimension `x`)*
    // dimension ::= `?` | decimal-literal
    // encoding ::= attribute-value
    // tensor-type ::= `tensor` `<` `*` `x` type `>`
    tensor_type: $ => seq('tensor<', $.dim_list,
      optional(seq(',', $.tensor_encoding)), '>'),
    tensor_encoding: $ => $.attribute_value,

    // vector-type ::= `vector` `<` vector-dim-list vector-element-type `>`
    // vector-element-type ::= float-type | integer-type | index-type
    // vector-dim-list := (static-dim-list `x`)? (`[` static-dim-list `]` `x`)?
    // static-dim-list ::= decimal-literal (`x` decimal-literal)*
    vector_type: $ => seq('vector<', $.vector_dim_list, $._primitive_type, '>'),
    vector_dim_list: $ => choice(seq($._static_dim_list, 'x',
      optional(seq('[', $._static_dim_list, ']', 'x'))), seq('[', $._static_dim_list, ']', 'x')),
    _static_dim_list: $ => prec.left(seq($._decimal_literal, repeat(seq('x', $._decimal_literal)))),

    // tuple-type ::= `tuple` `<` (type ( `,` type)*)? `>`
    tuple_type: $ => seq('tuple<', $.tuple_dim, repeat(seq(',', $.tuple_dim)), '>'),
    tuple_dim: $ => choice($._primitive_type, $.none_type, $.complex_type,
      $.memref_type, $.tensor_type, $.vector_type),

    // Attributes
    //   attribute-entry ::= (bare-id | string-literal) `=` attribute-value
    //   attribute-value ::= attribute-alias | dialect-attribute |
    //   builtin-attribute
    attribute_entry: $ => seq(choice($.bare_id, $.string_literal), optional(seq(choice('=', ':'),
      $.attribute_value))),
    attribute_value: $ => choice($.attribute_alias, $.dialect_attribute,
      $.builtin_attribute),

    // Attribute Value Aliases
    //   attribute-alias-def ::= '#' alias-name '=' attribute-value
    //   attribute-alias ::= '#' alias-name
    attribute_alias_def: $ => seq('#', $._alias_or_dialect_id, '=', $.attribute_value),
    attribute_alias: $ => seq('#', $._alias_or_dialect_id),
    // Dialect Attribute Values
    dialect_attribute: $ => seq('#', choice($.opaque_dialect_item,
      $.pretty_dialect_item)),

    // Builtin Attribute Values
    builtin_attribute: $ => choice(
      // TODO
      $.type,
      seq($.literal, optional(seq(':', $.type))),
    ),

    // Comment (standard BCPL)
    comment: $ => token(seq('//', /.*/)),

    // TODO: complete
    custom_operation: $ => choice(
      $.func_dialect,
      $.llvm_dialect,
      $.cf_dialect,
      $.arith_dialect,
      $.scf_dialect,
    ),

    func_dialect: $ => prec.right(choice(
      seq('func.func', $._op_func),
      seq(choice('func.return', 'return'),
        field('attributes', optional($.dictionary_attribute)),
        field('results', optional($._value_id_and_type_list))),
    )),
    function_return: $ => seq('->', $.type_list_attr_parens),
    block_arg_attr_list: $ => seq('(', optional($._value_id_and_type_attr_list), ')'),
    _value_id_and_type_attr_list: $ => seq($.value_id_and_type_attr,
      repeat(seq(',', $.value_id_and_type_attr))),
    value_id_and_type_attr: $ => choice(seq($.value_id_and_type,
      optional($.dictionary_attribute)), $.variadic),
    type_list_attr_parens: $ => choice($.type, seq('(', $.type, optional($.dictionary_attribute),
      repeat(seq(',', $.type, optional($.dictionary_attribute))), ')')),
    variadic: $ => '...',

    // (func.func|llvm.func) takes arguments, an optional return type, and and optional body
    _op_func: $ => seq(
      field('name', $.symbol_ref_id),
      field('arguments', $.block_arg_attr_list),
      field('return', optional($.function_return)),
      field('attributes', optional(seq('attributes', $.dictionary_attribute))),
      field('body', optional($.region))),

    llvm_dialect: $ => prec.right(choice(
      seq('llvm.func', $._op_func),
      seq('llvm.return',
        field('attributes', optional($.dictionary_attribute)),
        field('results', optional($._value_id_and_type_list))))),

    cf_dialect: $ => choice(
      // operation ::= `cf.br` $dest (`(` $destOperands^ `:` type($destOperands) `)`)? attr-dict
      seq('cf.br', field('successor', $.successor),
        field('attributes', optional($.dictionary_attribute))),

      // operation ::= `cf.cond_br` $condition `,`
      // $trueDest(`(` $trueDestOperands ^ `:` type($trueDestOperands)`)`)? `,`
      // $falseDest(`(` $falseDestOperands ^ `:` type($falseDestOperands)`)`)? attr-dict
      seq('cf.cond_br', seq(
        field('condition', $.value_use), ',',
        field('trueDest', $.successor), ',',
        field('falseDest', $.successor)),
        optional($.dictionary_attribute)),

      // operation ::= `cf.switch` $flag `:` type($flag) `,` `[` `\n`
      //               custom<SwitchOpCases>(ref(type($flag)),$defaultDestination,
      //               $defaultOperands,
      //               type($defaultOperands),
      //               $case_values,
      //               $caseDestinations,
      //               $caseOperands,
      //               type($caseOperands))
      //               `]`
      //               attr-dict
      seq('cf.switch', field('flag', $.value_id_and_type), ',', '[',
        $.case_label, $.successor, repeat(seq(',', $.case_label, $.successor)), ']',
        field('attributes', optional($.dictionary_attribute))),
    ),
    case_label: $ => seq(choice($.integer_literal, 'default'), ':'),

    arith_dialect: $ => choice(
      // operation ::= `arith.constant` attr-dict $value
      seq('arith.constant', optional($.dictionary_attribute), $.literal_and_type),

      // operation ::= `arith.addi` $lhs `,` $rhs attr-dict `:` type($result)
      seq('arith.addi',
        field('lhs', $.value_use), ',',
        field('rhs', $.value_use),
        optional($.dictionary_attribute), ':',
        $.type),

      // operation ::= `arith.cmpi` $predicate `,` $lhs `,` $rhs attr-dict `:` type($lhs)
      // operation ::= `arith.cmpf` $predicate `,` $lhs `,` $rhs attr-dict `:` type($lhs)
      // operation ::= `arith.divsi` $lhs `,` $rhs attr-dict `:` type($result)
      // operation ::= `arith.divui` $lhs `,` $rhs attr-dict `:` type($result)
      seq(choice('arith.cmpi', 'arith.cmpf', 'arith.divsi', 'arith.divui'),
        field('predicate',
          choice('eq', 'oeq', 'ne', 'slt', 'sle', 'sgt', 'sge', 'ult', 'ule', 'ugt', 'uge')), ',',
        field('lhs', $.value_use), ',',
        field('rhs', $.value_use),
        field('attributes', optional($.dictionary_attribute)), ':', $.type),

      // operation ::= `arith.extf` $in attr-dict `:` type($in) `to` type($out)
      // operation ::= `arith.extsi` $in attr-dict `:` type($in) `to` type($out)
      // operation ::= `arith.extui` $in attr-dict `:` type($in) `to` type($out)
      // operation ::= `arith.fptosi` $in attr-dict `:` type($in) `to` type($out)
      // operation ::= `arith.fptoui` $in attr-dict `:` type($in) `to` type($out)
      // operation ::= `arith.index_cast` $in attr-dict `:` type($in) `to` type($out)
      // operation ::= `arith.index_castui` $in attr-dict `:` type($in) `to` type($out)
      // operation ::= `arith.sitofp` $in attr-dict `:` type($in) `to` type($out)
      // operation ::= `arith.uitofp` $in attr-dict `:` type($in) `to` type($out)
      // operation ::= `arith.bitcast` $in attr-dict `:` type($in) `to` type($out)
      seq(choice('arith.extf', 'arith.extsi', 'arith.extui', 'arith.fptosi', 'arith.fptoui',
        'arith.index_cast', 'arith.index_castui', 'arith.sitofp', 'arith.uitofp', 'arith.bitcast'),
        field('in', $.value_use),
        field('attributes', $.dictionary_attribute), ':',
        field('fromtype', $.type), 'to',
        field('totype', $.type))
    ),
    literal_and_type: $ => seq($.literal, ':', $.type),

    scf_dialect: $ => prec.right(choice(
      // scf.for %iv = %lb to %ub step %step {
      // ... // body
      // }
      seq('scf.for',
        field('iv', $.value_use), '=',
        field('lb', $.value_use), 'to',
        field('ub', $.value_use), 'step',
        field('step', $.value_use), 'iter_args', '(',
        field('iter_args', seq($.value_use, '=', $.value_use)), ')',
        field('return', $.function_return),
        field('body', $.region)),

      // operation ::= `scf.yield` attr-dict ($results^ `:` type($results))?
      seq('scf.yield',
        field('attributes', optional($.dictionary_attribute)),
        field('results', optional($._value_id_and_type_list))),
    ))
  }
});
