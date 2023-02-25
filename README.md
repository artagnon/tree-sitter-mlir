# tree-sitter-mlir

[MLIR](https://mlir.llvm.org) grammar for [tree-sitter](https://github.com/tree-sitter/tree-sitter). The parser is incomplete, and the bench statistics on the test files in the MLIR tree are as follows:

```
Math, 100% passed
Builtin, 100% passed
Func, 100% passed
ControlFlow, 100% passed
Memref, 90.91% passed
Tensor, 93.33% passed
Arith, 83.33% passed
SCF, 88% passed
Affine, 76.92% passed
Linalg, 51.11% passed
```
