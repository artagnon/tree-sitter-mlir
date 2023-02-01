module.exports = grammar({
  name: 'mlir',
  extras: $ => [/\s/,
    $.comment
  ],
  conflicts: $ => [],
  rules: {
    // Top level production:
    //   (operation | attribute-alias-def | type-alias-def)
    toplevel: $ => seq(choice(
      $.operation,
      $.attribute_alias_def,
      $.type_alias_def
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
    integer_literal: $ => choice($._decimal_literal, $._hexadecimal_literal),
    _decimal_literal: $ => token(seq(optional(/[-+]/), repeat1(/[0-9]/))),
    _hexadecimal_literal: $ => token(seq('0x', repeat1(/[0-9a-fA-F]/))),
    float_literal: $ => token(seq(
      optional(/[-+]/), repeat1(/[0-9]/), '.', repeat(/[0-9]/),
      optional(seq(/[eE]/, optional(/[-+]/), repeat1(/[0-9]/))))),
    string_literal: $ => token(seq('"', repeat(/[^\\"\n\f\v\r]+/), '"')),
    bool_literal: $ => token(choice('true', 'false')),
    unit_literal: $ => token('unit'),
    complex_literal: $ => seq('(', choice($.integer_literal, $.float_literal), ',',
      choice($.integer_literal, $.float_literal), ')'),
    tensor_literal: $ => seq(choice('dense', 'sparse'), '<',
      choice(seq($.nested_idx_list, repeat(seq(',', $.nested_idx_list))),
        $._primitive_idx_literal), '>'),
    literal: $ => choice($.integer_literal, $.float_literal, $.string_literal, $.bool_literal,
      $.tensor_literal, $.complex_literal, $.unit_literal),

    nested_idx_list: $ => seq('[', optional(choice($.nested_idx_list, $._idx_list)),
      repeat(seq(',', $.nested_idx_list)), ']'),
    _idx_list: $ => prec.right(seq($._primitive_idx_literal,
      repeat(seq(',', $._primitive_idx_literal)))),
    _primitive_idx_literal: $ => choice($.integer_literal, $.float_literal,
      $.bool_literal, $.complex_literal),

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
    bare_id: $ => token(seq(/[a-zA-Z_]/, repeat(/[a-zA-Z0-9_$.]/))),
    _alias_or_dialect_id: $ => token(seq(/[a-zA-Z_]/, repeat(/[a-zA-Z0-9_$]/))),
    bare_id_list: $ => seq($.bare_id, repeat(seq(',', $.bare_id))),
    value_use: $ => seq('%', $._suffix_id),
    _suffix_id: $ => token(choice(repeat1(/[0-9]/), seq(/[a-zA-Z_$.]/,
      repeat(/[a-zA-Z0-9_$.]/)))),
    symbol_ref_id: $ => seq('@', choice($._suffix_id, $.string_literal),
      optional(seq('::', $.symbol_ref_id))),
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
    operation: $ => seq(
      field('lhs', optional($.op_result_list)),
      field('rhs', choice($.generic_operation, $.custom_operation)),
      field('location', optional($.trailing_location))),
    generic_operation: $ =>
      seq($.string_literal, '(', optional($.value_use_list),
        ')', optional($.successor_list),
        optional($.region_list),
        optional($.dictionary_attribute), ':',
        $.function_type),
    // custom-operation rule is defined later in the grammar, post the generic.
    op_result_list: $ => seq($.op_result, repeat(seq(',', $.op_result)), '='),
    op_result: $ => seq($.value_use, optional(seq(':', $.integer_literal))),
    successor_list: $ => seq('[', $.successor, repeat(seq(',', $.successor)),
      ']'),
    successor: $ => seq($.caret_id, optional($._value_arg_list)),
    region_list: $ => seq('(', $.region, repeat(seq(',', $.region)), ')'),
    dictionary_attribute: $ => seq(
      '{',
      optional(seq($.attribute_entry,
        repeat(seq(',', $.attribute_entry)))),
      '}'),
    trailing_location: $ => seq('loc', '(', $.location, ')'),
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
    _value_use_and_type: $ => seq($.value_use, optional(seq(':', $.type))),
    _value_use_and_type_list: $ => seq($._value_use_and_type,
      repeat(seq(',', $._value_use_and_type))),
    block_arg_list: $ => seq('(', optional($._value_use_and_type_list), ')'),
    _value_arg_list: $ => seq('(', optional($._value_use_type_list), ')'),
    _value_use_type_list: $ => seq($.value_use_list, ':', $.type_list_no_parens),

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
      repeat1(/[^<>]/),
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
    integer_type: $ => token(seq(choice('si', 'ui', 'i'), /[1-9]/, repeat(/[0-9]/))),
    float_type: $ => token(choice('f16', 'f32', 'f64', 'f80', 'f128', 'bf16',
      'f8E4M3FN', 'f8E5M2')),
    index_type: $ => token('index'),
    none_type: $ => token('none'),
    complex_type: $ => seq('complex<', $._prim_type, '>'),
    _prim_type: $ => choice($.integer_type, $.float_type, $.index_type,
      $.complex_type, $.none_type, $.memref_type),

    // memref-type ::= `memref` `<` dimension-list-ranked type
    //                 (`,` layout-specification)? (`,` memory-space)? `>`
    // layout-specification ::= attribute-value
    // memory-space ::= attribute-value
    memref_type: $ => seq('memref', '<',
      field('dimension_list', $.dim_list),
      optional(seq(',', $.attribute_value)),
      optional(seq(',', $.attribute_value)), '>'),
    dim_list: $ => seq($._dim_primitive, repeat(seq('x', $._dim_primitive))),
    _dim_primitive: $ => choice($._prim_type, repeat1($._digit), '?', '*'),

    // tensor-type ::= `tensor` `<` dimension-list type (`,` encoding)? `>`
    // dimension-list ::= (dimension `x`)*
    // dimension ::= `?` | decimal-literal
    // encoding ::= attribute-value
    // tensor-type ::= `tensor` `<` `*` `x` type `>`
    tensor_type: $ => seq('tensor', '<', $.dim_list,
      optional(seq(',', $.tensor_encoding)), '>'),
    tensor_encoding: $ => $.attribute_value,

    // vector-type ::= `vector` `<` vector-dim-list vector-element-type `>`
    // vector-element-type ::= float-type | integer-type | index-type
    // vector-dim-list := (static-dim-list `x`)? (`[` static-dim-list `]` `x`)?
    // static-dim-list ::= decimal-literal (`x` decimal-literal)*
    vector_type: $ => seq('vector', '<', $.vector_dim_list, $._prim_type, '>'),
    vector_dim_list: $ => choice(seq($._static_dim_list, 'x',
      optional(seq('[', $._static_dim_list, ']', 'x'))), seq('[', $._static_dim_list, ']', 'x')),
    _static_dim_list: $ => prec.left(seq(repeat1($._digit), repeat(seq('x', repeat1($._digit))))),

    // tuple-type ::= `tuple` `<` (type ( `,` type)*)? `>`
    tuple_type: $ => seq('tuple', '<', $.tuple_dim, repeat(seq(',', $.tuple_dim)), '>'),
    tuple_dim: $ => choice($._prim_type, $.tensor_type, $.vector_type),

    // Attributes
    //   attribute-entry ::= (bare-id | string-literal) `=` attribute-value
    //   attribute-value ::= attribute-alias | dialect-attribute |
    //   builtin-attribute
    attribute_entry: $ => seq(choice($.bare_id, $.string_literal),
      optional(seq(choice('=', ':'), $._attribute_value))),
    _attribute_value: $ => choice(seq($.literal, optional(seq(':', $.type))),
      $.type, $.attribute_value, seq('[', $.literal, repeat(seq(',', $.literal)), ']')),
    attribute_value: $ => choice($.attribute_alias, $.dialect_attribute, $.builtin_attribute),
    attribute: $ => choice($.attribute_value, $.dictionary_attribute),

    // Attribute Value Aliases
    //   attribute-alias-def ::= '#' alias-name '=' attribute-value
    //   attribute-alias ::= '#' alias-name
    attribute_alias_def: $ => seq('#', $._alias_or_dialect_id, '=', $.attribute_value),
    attribute_alias: $ => seq('#', $._alias_or_dialect_id),

    // Dialect Attribute Values
    dialect_attribute: $ => seq('#', choice($.opaque_dialect_item, $.pretty_dialect_item)),

    // Builtin Attribute Values
    builtin_attribute: $ => choice(
      // TODO
      $.strided_layout,
      $._affine_map_list,
    ),
    strided_layout: $ => seq('strided', '<', '[', $.dim_list, ']',
      ',', 'offset', ':', choice($.integer_literal, '?', '*'), '>'),
    _affine_map_list: $ => seq('[', $.affine_map, repeat(seq(',', $.affine_map)), ']'),
    affine_map: $ => seq('affine_map', '<', '(', $._loop_indices, ')',
      '->', '(', $._loop_indices, ')', '>'),
    _loop_indices: $ => seq($._loop_index, repeat(seq(',', $._loop_index))),
    _loop_index: $ => token(seq(/[a-zA-Z]/, repeat(/[a-zA-Z0-9]/))),

    // Comment (standard BCPL)
    comment: $ => token(seq('//', /.*/)),

    // TODO: complete
    custom_operation: $ => choice(
      $.builtin_dialect,
      $.func_dialect,
      $.llvm_dialect,
      $.cf_dialect,
      $.arith_dialect,
      $.scf_dialect,
      $.memref_dialect,
      $.tensor_dialect,
      $.linalg_dialect
    ),

    builtin_dialect: $ => choice(
      // operation ::= `builtin.module` ($sym_name^)? attr-dict-with-keyword $bodyRegion
      seq('module',
        field('name', optional($.bare_id)),
        field('attributes', optional($.attribute)),
        field('body', $.region)),

      // operation ::= `builtin.unrealized_conversion_cast` ($inputs^ `:` type($inputs))?
      //                `to` type($outputs) attr-dict
      seq('unrealized_cast_conversion',
        field('inputs', $._value_use_type_list), 'to',
        field('outputs', $.type_list_no_parens),
        field('attributes', optional($.attribute)))
    ),

    func_dialect: $ => prec.right(choice(
      seq('func.func', $._op_func),

      seq(choice('func.return', 'return'),
        field('attributes', optional($.attribute)),
        field('results', optional($._value_use_type_list)))
    )),

    function_return: $ => seq('->', $.type_list_attr_parens),
    function_arg_list: $ => seq('(', optional(choice($.variadic,
      $._value_id_and_type_attr_list)), ')'),
    _value_id_and_type_attr_list: $ => seq($._value_id_and_type_attr,
      repeat(seq(',', $._value_id_and_type_attr)), optional(seq(',', $.variadic))),
    _value_id_and_type_attr: $ => seq($._function_arg, optional($.attribute)),
    _function_arg: $ => choice(seq($.value_use, ':', $.type), $.value_use, $.type),
    type_list_attr_parens: $ => choice($.type, seq('(', $.type, optional($.attribute),
      repeat(seq(',', $.type, optional($.attribute))), ')')),
    variadic: $ => '...',

    // (func.func|llvm.func) takes arguments, an optional return type, and and optional body
    _op_func: $ => seq(
      field('name', $.symbol_ref_id),
      field('arguments', $.function_arg_list),
      field('return', optional($.function_return)),
      field('attributes', optional(seq('attributes', $.attribute))),
      field('body', optional($.region))),

    llvm_dialect: $ => prec.right(choice(
      seq('llvm.func', $._op_func),

      seq('llvm.return',
        field('attributes', optional($.attribute)),
        field('results', optional($._value_use_type_list))))),

    cf_dialect: $ => choice(
      // operation ::= `cf.br` $dest (`(` $destOperands^ `:` type($destOperands) `)`)? attr-dict
      seq('cf.br', field('successor', $.successor),
        field('attributes', optional($.attribute))),

      // operation ::= `cf.cond_br` $condition `,`
      // $trueDest(`(` $trueDestOperands ^ `:` type($trueDestOperands)`)`)? `,`
      // $falseDest(`(` $falseDestOperands ^ `:` type($falseDestOperands)`)`)? attr-dict
      seq('cf.cond_br', seq(
        field('condition', $.value_use), ',',
        field('trueDest', $.successor), ',',
        field('falseDest', $.successor)),
        optional($.attribute)),

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
      seq('cf.switch', field('flag', $._value_use_and_type), ',', '[',
        $.case_label, $.successor, repeat(seq(',', $.case_label, $.successor)), ']',
        field('attributes', optional($.attribute))),
    ),

    case_label: $ => seq(choice($.integer_literal, 'default'), ':'),

    arith_dialect: $ => choice(
      // operation ::= `arith.constant` attr-dict $value
      seq('arith.constant', optional($.attribute), $.literal_and_type),

      // operation ::= `arith.addi` $lhs `,` $rhs attr-dict `:` type($result)
      // operation ::= `arith.subi` $lhs `,` $rhs attr-dict `:` type($result)
      // operation ::= `arith.divsi` $lhs `,` $rhs attr-dict `:` type($result)
      // operation ::= `arith.divui` $lhs `,` $rhs attr-dict `:` type($result)
      // operation ::= `arith.ceildivsi` $lhs `,` $rhs attr-dict `:` type($result)
      // operation ::= `arith.ceildivui` $lhs `,` $rhs attr-dict `:` type($result)
      // operation ::= `arith.floordivsi` $lhs `,` $rhs attr-dict `:` type($result)
      // operation ::= `arith.remsi` $lhs `,` $rhs attr-dict `:` type($result)
      // operation ::= `arith.remui` $lhs `,` $rhs attr-dict `:` type($result)
      // operation ::= `arith.muli` $lhs `,` $rhs attr-dict `:` type($result)
      // operation ::= `arith.mulsi_extended` $lhs `,` $rhs attr-dict `:` type($lhs)
      // operation ::= `arith.andi` $lhs `,` $rhs attr-dict `:` type($result)
      // operation ::= `arith.ori` $lhs `,` $rhs attr-dict `:` type($result)
      // operation ::= `arith.xori` $lhs `,` $rhs attr-dict `:` type($result)
      // operation ::= `arith.maxsi` $lhs `,` $rhs attr-dict `:` type($result)
      // operation ::= `arith.maxui` $lhs `,` $rhs attr-dict `:` type($result)
      // operation ::= `arith.minsi` $lhs `,` $rhs attr-dict `:` type($result)
      // operation ::= `arith.minui` $lhs `,` $rhs attr-dict `:` type($result)
      // operation ::= `arith.shli` $lhs `,` $rhs attr-dict `:` type($result)
      // operation ::= `arith.shrsi` $lhs `,` $rhs attr-dict `:` type($result)
      // operation ::= `arith.shrui` $lhs `,` $rhs attr-dict `:` type($result)
      seq(choice('arith.addi', 'arith.subi', 'arith.divsi', 'arith.divui',
        'arith.ceildivsi', 'arith.ceildivui', 'arith.floordivsi',
        'arith.remsi', 'arith.remui', 'arith.muli', 'arith.mulsi_extended',
        'arith.andi', 'arith.ori', 'arith.xori',
        'arith.maxsi', 'arith.maxui', 'arith.minsi', 'arith.minui',
        'arith.shli', 'arith.shrsi', 'arith.shrui'),
        field('lhs', $.value_use), ',',
        field('rhs', $.value_use),
        field('attributes', optional($.attribute)), ':',
        $.type),

      // operation ::= `arith.addui_extended` $lhs `,` $rhs attr-dict `:` type($sum)
      //                `,` type($overflow)
      seq('arith.addui_extended',
        field('lhs', $.value_use), ',',
        field('rhs', $.value_use),
        field('attributes', optional($.attribute)),
        ':', $.type, ',', $.type),

      // operation ::= `arith.addf` $lhs `,` $rhs (`fastmath` `` $fastmath^)?
      //                attr-dict `:` type($result)
      // operation ::= `arith.divf` $lhs `,` $rhs (`fastmath` `` $fastmath^)?
      //                attr-dict `:` type($result)
      // operation ::= `arith.maxf` $lhs `,` $rhs (`fastmath` `` $fastmath^)?
      //                attr-dict `:` type($result)
      // operation ::= `arith.minf` $lhs `,` $rhs (`fastmath` `` $fastmath^)?
      //                attr-dict `:` type($result)
      // operation ::= `arith.mulf` $lhs `,` $rhs (`fastmath` `` $fastmath^)?
      //                attr-dict `:` type($result)
      // operation ::= `arith.remf` $lhs `,` $rhs (`fastmath` `` $fastmath^)?
      //                attr-dict `:` type($result)
      // operation ::= `arith.subf` $lhs `,` $rhs (`fastmath` `` $fastmath^)?
      //                attr-dict `:` type($result)
      seq(choice('arith.addf', 'arith.divf', 'arith.maxf', 'arith.minf', 'arith.mulf',
        'arith.remf', 'arith.subf'),
        field('lhs', $.value_use), ',',
        field('rhs', $.value_use),
        field('fastmath', optional($._fastmath_flags)),
        field('attributes', optional($.attribute)), ':',
        $.type),

      // operation ::= `arith.negf` $operand (`fastmath` `` $fastmath^)?
      //                attr-dict `:` type($result)
      seq(choice('arith.negf'),
        field('operand', $.value_use),
        field('fastmath', optional($._fastmath_flags)),
        field('attributes', optional($.attribute)), ':',
        $.type),

      // operation ::= `arith.cmpi` $predicate `,` $lhs `,` $rhs attr-dict `:` type($lhs)
      // operation ::= `arith.cmpf` $predicate `,` $lhs `,` $rhs attr-dict `:` type($lhs)
      seq(choice('arith.cmpi', 'arith.cmpf'),
        field('predicate',
          choice('eq', 'oeq', 'ne', 'slt', 'sle', 'sgt', 'sge', 'ult', 'ule', 'ugt', 'uge')), ',',
        field('lhs', $.value_use), ',',
        field('rhs', $.value_use),
        field('attributes', optional($.attribute)), ':', $.type),

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
      // operation ::= `arith.truncf` $in attr-dict `:` type($in) `to` type($out)
      seq(choice('arith.extf', 'arith.extsi', 'arith.extui', 'arith.fptosi', 'arith.fptoui',
        'arith.index_cast', 'arith.index_castui', 'arith.sitofp', 'arith.uitofp', 'arith.bitcast',
        'arith.truncf'),
        field('in', $.value_use),
        field('attributes', $.attribute),
        $._from_type_to_type),

      seq('arith.select',
        field('cond', $.value_use), ',',
        field('truebr', $.value_use), ',',
        field('falsebr', $.value_use),
        ':', $.type_list_no_parens)
    ),

    _fastmath_flags: $ => seq('fastmath', '<',
      seq($.fastmath_flag, repeat(seq(',', $.fastmath_flag))), '>'),
    fastmath_flag: $ => choice('none', 'reassoc', 'nnan', 'ninf', 'nsz', 'arcp',
      'contract', 'afn', 'fast'),

    literal_and_type: $ => seq($.literal, ':', $.type),
    _from_type_to_type: $ => seq(':',
      field('fromtype', $.type), 'to',
      field('totype', $.type)),

    scf_dialect: $ => prec.right(choice(
      seq('scf.if',
        field('condition', $.value_use),
        optional($.function_return),
        field('trueblk', $.region), 'else',
        field('falseblk', $.region)),

      // scf.for %iv = %lb to %ub step %step {
      // ... // body
      // }
      seq('scf.for',
        field('iv', $.value_use), '=',
        field('lb', $.value_use), 'to',
        field('ub', $.value_use), 'step',
        field('step', $.value_use),
        field('iter_args', optional(seq('iter_args', '(', $.value_use, '=', $.value_use, ')'))),
        field('return', optional($.function_return)),
        field('body', $.region)),

      // operation ::= `scf.yield` attr-dict ($results^ `:` type($results))?
      seq('scf.yield',
        field('attributes', optional($.attribute)),
        field('results', optional($._value_use_type_list))),
    )),

    memref_dialect: $ => choice(
      // operation ::= `memref.view` $source `[` $byte_shift `]` `` `[` $sizes `]` attr-dict
      //         `:` type($source) `to` type(results)
      seq('memref.view',
        field('source', $.value_use), '[',
        field('byte_shift', $.value_use), ']', '[',
        field('sizes', $.value_use_list), ']',
        field('attributes', optional($.attribute)),
        $._from_type_to_type)
    ),

    tensor_dialect: $ => choice(
      // operation ::= `tensor.empty` `(`$dynamicSizes`)` attr-dict `:` type($result)
      seq('tensor.empty', '(',
        field('size', optional($.value_use_list)), ')',
        field('attributes', optional($.attribute)), ':',
        field('return', $.type)),

      // operation ::= `tensor.cast` $source attr-dict `:` type($source) `to` type($dest)
      seq('tensor.cast',
        field('in', $.value_use),
        field('attributes', optional($.attribute)),
        $._from_type_to_type),

      // operation ::= `tensor.collapse_shape` $src $reassociation attr-dict `:` type($src)
      //                `into` type($result)
      seq(choice('tensor.collapse_shape', 'tensor.expand_shape'),
        field('tensor', $.value_use),
        field('reassociation', $.nested_idx_list),
        field('attributes', optional($.attribute)),
        ':', $.type, 'into', $.type),

      // operation ::= `tensor.extract` $tensor `[` $indices `]` attr-dict `:` type($tensor)
      seq('tensor.extract',
        field('tensor', $.value_use), '[',
        field('indices', optional($.value_use_list)), ']',
        field('attributes', optional($.attribute)), ':',
        $.type)
    ),

    linalg_dialect: $ => choice(
      seq(choice('linalg.batch_matmul', 'linalg.batch_matmul_transpose_b', 'linalg.batch_matvec',
        'linalg.batch_reduce_matmul', 'linalg.conv_1d_ncw_fcw', 'linalg.conv_1d_nwc_wcf',
        'linalg.conv_1d', 'linalg.conv_2d_nchw_fchw', 'linalg.conv_2d_ngchw_fgchw',
        'linalg.conv_2d_nhwc_fhwc', 'linalg.conv_2d_nhwc_hwcf', 'linalg.conv_2d_nhwc_hwcf_q',
        'linalg.conv_2d', 'linalg.conv_3d_ndhwc_dhwcf', 'linalg.conv_3d_ndhwc_dhwcf_q',
        'linalg.conv_3d', 'linalg.copy', 'linalg.depthwise_conv_1d_nwc_wcm',
        'linalg.depthwise_conv_2d_nchw_chw', 'linalg.depthwise_conv_2d_nhwc_hwc',
        'linalg.depthwise_conv_2d_nhwc_hwc_q', 'linalg.depthwise_conv_2d_nhwc_hwcm',
        'linalg.depthwise_conv_2d_nhwc_hwcm_q', 'linalg.depthwise_conv_3d_ndhwc_dhwc',
        'linalg.depthwise_conv_3d_ndhwc_dhwcm', 'linalg.dot', 'linalg.elemwise_binary',
        'linalg.elemwise_unary', 'linalg.fill', 'linalg.fill_rng_2d', 'linalg.matmul',
        'linalg.matmul_transpose_b', 'linalg.matmul_unsigned', 'linalg.matvec', 'linalg.mmt4d',
        'linalg.pooling_nchw_max', 'linalg.pooling_nchw_sum', 'linalg.pooling_ncw_max',
        'linalg.pooling_ncw_sum', 'linalg.pooling_ndhwc_max', 'linalg.pooling_ndhwc_min',
        'linalg.pooling_ndhwc_sum', 'linalg.pooling_nhwc_max', 'linalg.pooling_nhwc_max_unsigned',
        'linalg.pooling_nhwc_min', 'linalg.pooling_nhwc_min_unsigned', 'linalg.pooling_nhwc_sum',
        'linalg.pooling_nwc_max', 'linalg.pooling_nwc_max_unsigned', 'linalg.pooling_nwc_min',
        'linalg.pooling_nwc_min_unsigned', 'linalg.pooling_nwc_sum',
        'linalg.quantized_batch_matmul', 'linalg.quantized_matmul', 'linalg.vecmat'),
        field('attributes', optional($.attribute)),
        'ins', '(', field('ins', $._value_use_type_list), ')',
        'outs', '(', field('outs', $._value_use_type_list), ')',
        optional($.function_return)
      ),

      seq('linalg.generic',
        field('attributes', $.attribute),
        field('ins', seq('ins', '(', $._value_use_type_list, ')')),
        field('outs', seq('outs', '(', $._value_use_type_list, ')')),
        field('body', $.region), optional($.function_return)),

      seq('linalg.map',
        field('ins', optional(seq('ins', '(', $._value_use_type_list, ')'))),
        field('outs', seq('outs', '(', $._value_use_type_list, ')')),
        field('arguments', $.block_arg_list),
        field('body', $.region), optional($.function_return)),

      seq('linalg.yield', $._value_use_type_list)
    ),
  }
});
