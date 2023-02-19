[
  "ins"
  "outs"
  "else"
  "do"
  "loc"
  "attributes"
  "into"
  "to"
  "step"
  "low"
  "high"
  "iter_args"
  "padding_value"
  "inner_tiles"
  "gather_dims"
  "scatter_dims"
  "outer_dims_perm"
  "inner_dims_pos"
  "shared_outs"
  "default"
] @keyword

[
  "module"
  "unrealized_cast_conversion"

  "func.func"
  "func.return"
  "return"

  "llvm.func"
  "llvm.return"

  "cf.br"
  "cf.cond_br"
  "cf.switch"

  "arith.constant"
  "arith.addi"
  "arith.subi"
  "arith.divsi"
  "arith.divui"
  "arith.ceildivsi"
  "arith.ceildivui"
  "arith.floordivsi"
  "arith.remsi"
  "arith.remui"
  "arith.muli"
  "arith.mulsi_extended"
  "arith.andi"
  "arith.ori"
  "arith.xori"
  "arith.maxsi"
  "arith.maxui"
  "arith.minsi"
  "arith.minui"
  "arith.shli"
  "arith.shrsi"
  "arith.shrui"
  "arith.addui_extended"
  "arith.addf"
  "arith.divf"
  "arith.maxf"
  "arith.minf"
  "arith.mulf"
  "arith.remf"
  "arith.subf"
  "arith.negf"
  "arith.cmpi"
  "arith.cmpf"
  "arith.extf"
  "arith.extsi"
  "arith.extui"
  "arith.fptosi"
  "arith.fptoui"
  "arith.index_cast"
  "arith.index_castui"
  "arith.sitofp"
  "arith.uitofp"
  "arith.bitcast"
  "arith.truncf"
  "arith.select"

  "scf.condition"
  "scf.execute_region"
  "scf.if"
  "scf.index_switch"
  "scf.for"
  "scf.forall"
  "scf.forall.in_parallel"
  "scf.parallel"
  "scf.reduce"
  "scf.reduce.return"
  "scf.while"
  "scf.yield"

  "memref.alloc"
  "memref.cast"
  "memref.copy"
  "memref.collapse_shape"
  "memref.expand_shape"
  "memref.prefetch"
  "memref.rank"
  "memref.realloc"
  "memref.view"

  "tensor.empty"
  "tensor.cast"
  "tensor.dim"
  "tensor.collapse_shape"
  "tensor.expand_shape"
  "tensor.extract"
  "tensor.insert"
  "tensor.extract_slice"
  "tensor.insert_slice"
  "tensor.parallel_insert_slice"
  "tensor.from_elements"
  "tensor.gather"
  "tensor.scatter"
  "tensor.pad"
  "tensor.reshape"
  "tensor.splat"
  "tensor.pack"
  "tensor.unpack"
  "tensor.generate"
  "tensor.rank"
  "tensor.yield"

  "linalg.batch_matmul"
  "linalg.batch_matmul_transpose_b"
  "linalg.batch_matvec"
  "linalg.batch_reduce_matmul"
  "linalg.broadcast"
  "linalg.conv_1d_ncw_fcw"
  "linalg.conv_1d_nwc_wcf"
  "linalg.conv_1d"
  "linalg.conv_2d_nchw_fchw"
  "linalg.conv_2d_ngchw_fgchw"
  "linalg.conv_2d_nhwc_fhwc"
  "linalg.conv_2d_nhwc_hwcf"
  "linalg.conv_2d_nhwc_hwcf_q"
  "linalg.conv_2d"
  "linalg.conv_3d_ndhwc_dhwcf"
  "linalg.conv_3d_ndhwc_dhwcf_q"
  "linalg.conv_3d"
  "linalg.copy"
  "linalg.depthwise_conv_1d_nwc_wcm"
  "linalg.depthwise_conv_2d_nchw_chw"
  "linalg.depthwise_conv_2d_nhwc_hwc"
  "linalg.depthwise_conv_2d_nhwc_hwc_q"
  "linalg.depthwise_conv_2d_nhwc_hwcm"
  "linalg.depthwise_conv_2d_nhwc_hwcm_q"
  "linalg.depthwise_conv_3d_ndhwc_dhwc"
  "linalg.depthwise_conv_3d_ndhwc_dhwcm"
  "linalg.dot"
  "linalg.elemwise_binary"
  "linalg.elemwise_unary"
  "linalg.fill"
  "linalg.fill_rng_2d"
  "linalg.matmul"
  "linalg.matmul_transpose_b"
  "linalg.matmul_unsigned"
  "linalg.matvec"
  "linalg.mmt4d"
  "linalg.pooling_nchw_max"
  "linalg.pooling_nchw_sum"
  "linalg.pooling_ncw_max"
  "linalg.pooling_ncw_sum"
  "linalg.pooling_ndhwc_max"
  "linalg.pooling_ndhwc_min"
  "linalg.pooling_ndhwc_sum"
  "linalg.pooling_nhwc_max"
  "linalg.pooling_nhwc_max_unsigned"
  "linalg.pooling_nhwc_min"
  "linalg.pooling_nhwc_min_unsigned"
  "linalg.pooling_nhwc_sum"
  "linalg.pooling_nwc_max"
  "linalg.pooling_nwc_max_unsigned"
  "linalg.pooling_nwc_min"
  "linalg.pooling_nwc_min_unsigned"
  "linalg.pooling_nwc_sum"
  "linalg.quantized_batch_matmul"
  "linalg.quantized_matmul"
  "linalg.vecmat"
  "linalg.generic"
  "linalg.index"
  "linalg.map"
  "linalg.yield"
] @function.builtin

(generic_operation) @function

(builtin_type) @type.builtin

[
  (type_alias)
  (dialect_type)
  (type_alias_def)
] @type

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

[
  (attribute_alias_def)
  (attribute_alias)
  (bare_attribute_entry)
  (attribute)
  (fastmath_attr)
  (scatter_dims_attr)
  (gather_dims_attr)
  (outer_dims_perm_attr)
  (inner_dims_pos_attr)
  (unique_attr)
  (nofold_attr)
  (isWrite_attr)
  (localityHint_attr)
  (isDataCache_attr)
] @attribute

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

(func_arg_list (value_use) @variable.parameter)
(block_arg_list (value_use) @variable.parameter)

(caret_id) @tag
(value_use) @variable
(comment) @comment
