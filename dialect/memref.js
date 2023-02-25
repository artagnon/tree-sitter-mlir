'use strict';

module.exports = {
  memref_dialect: $ => prec.right(choice(
    // operation ::= `memref.assume_alignment` $memref `,` $alignment attr-dict `:` type($memref)
    seq('memref.assume_alignment',
      field('memref', $.value_use), ',',
      field('alignment', $.integer_literal),
      field('attributes', optional($.attribute)),
      field('return', $._type_annotation)),

    // operation ::= `memref.alloc` `(`$dynamicSizes`)` (`` `[` $symbolOperands^ `]`)? attr-dict
    //               `:` type($memref)
    // operation ::= `memref.alloca` `(`$dynamicSizes`)` (`` `[` $symbolOperands^ `]`)? attr-dict
    //               `:` type($memref)
    seq(choice('memref.alloc', 'memref.alloca'),
      field('dynamicSizes', $._value_use_list_parens),
      field('symbolOperands', optional($._dense_idx_list)),
      field('attributes', optional($.attribute)),
      field('return', $._type_annotation)),

    seq('memref.alloca_scope',
      field('body', $.region),
      field('attributes', optional($.attribute))),

    // operation ::= `memref.alloca_scope.return` attr-dict ($results^ `:` type($results))?
    seq('memref.alloca_scope.return',
      field('attributes', optional($.attribute)),
      field('results', optional($._value_use_type_list))),

    // operation ::= `memref.atomic_rmw` $kind $value `,` $memref `[` $indices `]` attr-dict
    //               `:` `(` type($value) `,` type($memref) `)` `->` type($result)
    seq('memref.atomic_rmw',
      field('kind', choice($.atomic_rmw_attr, $.string_literal)),
      field('value', $.value_use), ',',
      field('memref', seq($.value_use, optional($._dense_idx_list))),
      field('attributes', optional($.attribute)),
      field('return', $._function_type_annotation)),

    // operation ::= `memref.atomic_yield` $result attr-dict `:` type($result)
    // operation ::= `memref.cast` $source attr-dict `:` type($source) `to` type($dest)
    // operation ::= `memref.dealloc` $memref attr-dict `:` type($memref)
    // operation ::= `memref.rank` $memref attr-dict `:` type($memref)
    // operation ::= `memref.memory_space_cast` $source attr-dict `:` type($source) `to` type($dest)
    seq(choice('memref.atomic_yield', 'memref.cast', 'memref.dealloc', 'memref.rank',
      'memref.memory_space_cast'),
      field('in', $.value_use),
      field('attributes', optional($.attribute)),
      field('return', $._type_annotation)),

    // operation ::= `memref.copy` $source `,` $target attr-dict
    //               `:` type($source) `to` type($target)
    seq('memref.copy',
      field('source', $.value_use), ',',
      field('target', $.value_use),
      field('attributes', optional($.attribute)),
      field('return', $._type_annotation)),

    // operation ::= `memref.collapse_shape` $src $reassociation attr-dict
    //               `:` type($src) `into` type($result)
    // operation ::= `memref.expand_shape` $src $reassociation attr-dict
    //               `:` type($src) `into` type($result)
    seq(choice('memref.collapse_shape', 'memref.expand_shape'),
      field('source', $.value_use),
      field('reassociation', $.nested_idx_list),
      field('attributes', optional($.attribute)),
      field('return', $._type_annotation)),

    // operation ::= `memref.dim` attr-dict $source `,` $index `:` type($source)
    seq('memref.dim',
      field('attributes', optional($.attribute)),
      field('source', $.value_use), ',',
      field('index', $.value_use),
      field('return', $._type_annotation)),

    // operation ::= `memref.dma_start` ssa-use`[`ssa-use-list`]` `,`
    //                ssa-use`[`ssa-use-list`]` `,` ssa-use `,`
    //                ssa-use`[`ssa-use-list`]` (`,` ssa-use `,` ssa-use)?
    //               `:` memref-type `,` memref-type `,` memref-type
    // operation ::= `memref.dma_wait` ssa-use`[`ssa-use-list`]` `,` ssa-use `:` memref-type
    seq(choice('memref.dma_start', 'memref.dma_wait'),
      field('operands', seq($.value_use, optional($._dense_idx_list),
        repeat(seq(',', $.value_use, optional($._dense_idx_list))))), ',',
      field('return', $._type_annotation)),

    // operation ::= `memref.extract_aligned_pointer_as_index` $source
    //               `:` type($source) `->` type(results) attr-dict
    // operation ::= `memref.extract_strided_metadata` $source
    //               `:` type($source) `->` type(results) attr-dict
    seq(choice('memref.extract_aligned_pointer_as_index', 'memref.extract_strided_metadata'),
      field('source', $.value_use),
      field('return', seq($._function_type_annotation, optional(seq(',', $._type_list_no_parens)))),
      field('attributes', optional($.attribute))),

    seq('memref.generic_atomic_rmw',
      field('operand', seq($.value_use, $._dense_idx_list)),
      field('return', optional($._type_annotation)),
      field('body', $.region),
      field('attributes', optional($.attribute))),

    // operation ::= `memref.get_global` $name `:` type($result) attr-dict
    seq('memref.get_global',
      field('name', $.symbol_ref_id),
      field('return', $._type_annotation),
      field('attributes', optional($.attribute))),

    // operation ::= `memref.global` ($sym_visibility^)?
    //               (`constant` $constant^)?
    //               $sym_name `:`
    //               custom<GlobalMemrefOpTypeAndInitialValue>($type, $initial_value)
    //               attr-dict
    seq('memref.global',
      field('visibility', optional($.string_literal)),
      field('constant_attr', optional($.constant_attr)),
      field('name', $.symbol_ref_id),
      field('type', $._type_annotation),
      field('initializer', optional(seq('=', $._literal))),
      field('attributes', optional($.attribute))),

    // operation ::= `memref.load` $memref `[` $indices `]` attr-dict `:` type($memref)
    seq('memref.load',
      field('memref', seq($.value_use, $._dense_idx_list)),
      field('attributes', optional($.attribute)),
      field('return', $._type_annotation)),

    seq('memref.prefetch',
      field('source', $.value_use),
      field('indices', optional($._dense_idx_list)), ',',
      field('isWrite', $.isWrite_attr), ',',
      field('localityHint', $.localityHint_attr), ',',
      field('isDataCache', $.isDataCache_attr),
      field('attributes', optional($.attribute)),
      field('return', $._type_annotation)),

    // operation ::= `memref.realloc` $source (`(` $dynamicResultSize^ `)`)? attr-dict
    //               `:` type($source) `to` type(results)
    seq('memref.realloc',
      field('source', $.value_use),
      field('dynamicResultSize', optional($._value_use_list_parens)),
      field('attributes', optional($.attribute)),
      field('return', $._type_annotation)),

    // operation ::= `memref.reinterpret_cast` $source `to` `offset` `` `:`
    //               custom<DynamicIndexList>($offsets, $static_offsets)
    //               `` `,` `sizes` `` `:`
    //               custom<DynamicIndexList>($sizes, $static_sizes)
    //               `` `,` `strides` `` `:`
    //               custom<DynamicIndexList>($strides, $static_strides)
    //               attr-dict `:` type($source) `to` type($result)
    seq('memref.reinterpret_cast',
      field('source', $.value_use), token('to'),
      field('offset', seq(token('offset'), ':', $._dense_idx_list, ',')),
      field('sizes', seq(token('sizes'), ':', $._dense_idx_list, ',')),
      field('strides', seq(token('strides'), ':', $._dense_idx_list)),
      field('attributes', optional($.attribute)),
      field('return', $._type_annotation)),

    // operation ::= `memref.reshape` $source `(` $shape `)` attr-dict
    //               `:` functional-type(operands, results)
    seq('memref.reshape',
      field('source', $.value_use),
      field('shape', seq('(', $.value_use, ')')),
      field('attributes', optional($.attribute)),
      field('return', $._function_type_annotation)),

    // operation ::= `memref.store` $value `,` $memref `[` $indices `]` attr-dict
    //                `:` type($memref)
    seq('memref.store',
      field('source', $.value_use), ',',
      field('destination', $.value_use),
      field('indices', $._dense_idx_list),
      field('attributes', optional($.attribute)),
      field('return', $._type_annotation)),

    // operation ::= `memref.subview` $source ``
    //               custom<DynamicIndexList>($offsets, $static_offsets)
    //               custom<DynamicIndexList>($sizes, $static_sizes)
    //               custom<DynamicIndexList>($strides, $static_strides)
    //               attr-dict `:` type($source) `to` type($result)
    seq('memref.subview',
      field('source', $.value_use),
      field('offsets', $._dense_idx_list),
      field('sizes', $._dense_idx_list),
      field('strides', $._dense_idx_list),
      field('attributes', optional($.attribute)),
      field('return', $._type_annotation)),

    // operation ::= `memref.tensor_store` $tensor `,` $memref attr-dict `:` type($memref)
    seq('memref.tensor_store',
      field('tensor', $.value_use), ',',
      field('memref', $.value_use),
      field('attributes', optional($.attribute)),
      field('return', $._type_annotation)),

    seq('memref.transpose',
      field('in', $.value_use),
      field('permutation', seq($._multi_dim_affine_expr_parens, token('->'),
        $._multi_dim_affine_expr_parens)),
      field('return', $._type_annotation)),

    // operation ::= `memref.view` $source `[` $byte_shift `]` `` `[` $sizes `]` attr-dict
    //         `:` type($source) `to` type(results)
    seq('memref.view',
      field('source', $.value_use),
      field('byte_shift', $._dense_idx_list),
      field('sizes', $._dense_idx_list),
      field('attributes', optional($.attribute)),
      field('return', $._type_annotation))
  )),

  atomic_rmw_attr: $ => token(choice('addf', 'addi', 'assign', 'maxf', 'maxs', 'maxu', 'minf',
    'mins', 'minu', 'mulf', 'muli', 'ori', 'andi'))
}
