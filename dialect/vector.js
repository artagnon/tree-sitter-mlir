'use strict';

module.exports = {
  vector_dialect: $ => prec.right(choice(
    // operation ::= `vector.bitcast` $source attr-dict `:` type($source) `to` type($result)
    // operation ::= `vector.broadcast` $source attr-dict `:` type($source) `to` type($vector)
    // operation ::= `vector.extract_strided_slice` $vector attr-dict
    //               `:` type($vector) `to` type(results)
    // operation ::= `vector.print` $source attr-dict `:` type($source)
    // operation ::= `vector.splat` $input attr-dict `:` type($aggregate)
    // operation ::= `vector.shape_cast` $source attr-dict `:` type($source) `to` type($result)
    // operation ::= `vector.type_cast` $memref attr-dict `:` type($memref) `to` type($result)
    seq(choice('vector.bitcast', 'vector.broadcast', 'vector.extract_strided_slice',
      'vector.print', 'vector.splat', 'vector.shape_cast', 'vector.type_cast'),
      field('in', $.value_use),
      field('attributes', optional($.attribute)),
      field('return', $._type_annotation)),

    // operation ::= `vector.constant_mask` $mask_dim_sizes attr-dict `:` type(results)
    seq('vector.constant_mask',
      field('mask', $._dense_idx_list),
      field('attributes', optional($.attribute)),
      field('return', $._type_annotation)),

    seq('vector.contract',
      field('attributes', optional($.attribute)),
      field('lhs', $.value_use),
      field('rhs', $.value_use),
      field('acc', $.value_use),
      field('masks', $._value_use_list)),

    // operation ::= `vector.create_mask` $operands attr-dict `:` type(results)
    seq(choice('vector.create_mask', 'vector.outerproduct'),
      field('operands', $._value_use_list),
      field('attributes', optional($.attribute)),
      field('return', $._type_annotation)),

    // operation ::= `vector.expandload` $base `[` $indices `]` `,` $mask `,` $pass_thru
    //               attr-dict
    //               `:` type($base) `,` type($mask) `,` type($pass_thru) `into` type($result)
    // operation ::= `vector.maskedload` $base `[` $indices `]` `,` $mask `,` $pass_thru
    //               attr-dict
    //               `:` type($base) `,` type($mask) `,` type($pass_thru) `into` type($result)
    // operation ::= `vector.maskedstore` $base `[` $indices `]` `,` $mask `,` $valueToStore
    //                attr-dict
    //                `:` type($base) `,` type($mask) `,` type($valueToStore)
    seq(choice('vector.expandload', 'vector.maskedload', 'vector.maskedstore'),
      field('base', seq($.value_use, $._dense_idx_list)), ',',
      field('mask', $.value_use), ',',
      field('pass_thru', $.value_use),
      field('attributes', optional($.attribute)),
      field('return', $._type_annotation)),

    // operation ::= `vector.extract` $vector `` $position attr-dict `:` type($vector)
    // operation ::= `vector.extractelement` $vector `[` ($position^ `:` type($position))? `]`
    //                attr-dict `:` type($vector)
    // operation ::= `vector.load` $base `[` $indices `]` attr-dict
    //               `:` type($base) `,` type($nresult)
    // operation ::= `vector.scalable.extract` $source `[` $pos `]` attr-dict
    //               `:` type($res) `from` type($source)
    seq(choice('vector.extract', 'vector.extractelement', 'vector.load', 'vector.scalable.extract'),
      field('operand', seq($.value_use, $._dense_idx_list)),
      field('attributes', optional($.attribute)),
      field('return', $._type_annotation)),

    // operation ::= `vector.fma` $lhs `,` $rhs `,` $acc attr-dict `:` type($lhs)
    seq('vector.fma',
      field('lhs', $.value_use), ',',
      field('rhs', $.value_use), ',',
      field('acc', $.value_use),
      field('attributes', optional($.attribute)),
      field('return', $._type_annotation)),

    // operation ::= `vector.flat_transpose` $matrix attr-dict `:` type($matrix) `->` type($res)
    seq('vector.flat_transpose',
      field('matrix', $.value_use),
      field('attributes', optional($.attribute)),
      field('return', $._function_type_annotation)),

    // operation ::= `vector.gather` $base `[` $indices `]` `[` $index_vec `]` `,` $mask `,`
    //               $pass_thru attr-dict
    //               `:` type($base) `,` type($index_vec)  `,` type($mask) `,` type($pass_thru)
    //               `into` type($result)
    // operation ::= `vector.scatter` $base `[` $indices `]` `[` $index_vec `]` `,` $mask `,`
    //               $valueToStore attr-dict
    //               `:` type($base) `,` type($index_vec)  `,` type($mask) `,` type($valueToStore)
    seq(choice('vector.gather', 'vector.scatter'),
      field('base', seq($.value_use, $._dense_idx_list, $._dense_idx_list)), ',',
      field('mask', $.value_use), ',',
      field('pass_thru', $.value_use),
      field('attributes', optional($.attribute)),
      field('return', $._type_annotation)),

    // operation ::= `vector.insert` $source `,` $dest $position attr-dict
    //               `:` type($source) `into` type($dest)
    // operation ::= `vector.insertelement` $source `,` $dest
    //               `[` ($position^ `:` type($position))? `]`  attr-dict
    //               `:` type($result)
    // operation ::= `vector.scalable.insert` $source `,` $dest `[` $pos `]` attr-dict
    //               `:` type($source) `into` type($dest)
    // operation ::= `vector.shuffle` operands $mask attr-dict `:` type(operands)
    // operation ::= `vector.store` $valueToStore `,` $base `[` $indices `]` attr-dict
    //               `:` type($base) `,` type($valueToStore)
    seq(choice('vector.insert', 'vector.insertelement', 'vector.scalable.insert', 'vector.shuffle',
      'vector.store'),
      field('source', $.value_use), ',',
      field('destination', $.value_use),
      field('position', $._dense_idx_list),
      field('attributes', optional($.attribute)),
      field('return', $._type_annotation)),

    // operation ::= `vector.insert_strided_slice` $source `,` $dest attr-dict
    //               `:` type($source) `into` type($dest)
    seq('vector.insert_strided_slice',
      field('source', $.value_use), ',',
      field('destination', $.value_use),
      field('attributes', optional($.attribute)),
      field('return', $._type_annotation)),

    // operation ::= `vector.matrix_multiply` $lhs `,` $rhs attr-dict
    //                `:` `(` type($lhs) `,` type($rhs) `)` `->` type($res)
    seq('vector.matrix_multiply',
      field('lhs', $.value_use), ',',
      field('rhs', $.value_use),
      field('attributes', optional($.attribute)),
      field('return', $._function_type_annotation)),

    seq('vector.transfer_read',
      field('source', seq($.value_use, $._dense_idx_list)),
      field('paddingMask', optional(seq(',', $._value_use_list))),
      field('attributes', optional($.attribute)),
      field('return', $._type_annotation)),

    seq('vector.transfer_write',
      field('vector', $.value_use), ',',
      field('source', seq($.value_use, $._dense_idx_list)),
      field('mask', optional(seq(',', $.value_use))),
      field('attributes', optional($.attribute)),
      field('return', $._type_annotation)),

    // operation ::= `vector.transpose` $vector `,` $transp attr-dict
    //               `:` type($vector) `to` type($result)
    seq('vector.transpose',
      field('vector', $.value_use), ',',
      field('indices', $._dense_idx_list),
      field('attributes', optional($.attribute)),
      field('return', $._type_annotation)),

    // operation ::= `vector.vscale` attr-dict
    seq('vector.vscale',
      field('attributes', optional($.attribute))),

    // operation ::= `vector.yield` attr-dict ($operands^ `:` type($operands))?
    seq('vector.yield',
      field('attributes', optional($.attribute)),
      field('results', optional($._value_use_type_list)))
  ))
}
