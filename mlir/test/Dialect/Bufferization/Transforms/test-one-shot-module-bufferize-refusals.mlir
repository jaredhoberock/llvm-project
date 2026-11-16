// RUN: mlir-opt %s -test-one-shot-module-bufferize -split-input-file -verify-diagnostics

// A new buffer-like value with a dynamic extent and nothing to copy has no
// extent to be allocated with: its type refuses.

func.func @custom_types_alloc_refused() -> !test.test_tensor<[-9223372036854775808], f64> {
  // expected-error @+2 {{allocating a buffer of type '!test.test_memref<[-9223372036854775808], f64>' is not supported}}
  // expected-error @+1 {{failed to bufferize op}}
  %e = bufferization.alloc_tensor() : !test.test_tensor<[-9223372036854775808], f64>
  %0 = "test.dummy_tensor_op"(%e)
    : (!test.test_tensor<[-9223372036854775808], f64>)
    -> !test.test_tensor<[-9223372036854775808], f64>
  return %0 : !test.test_tensor<[-9223372036854775808], f64>
}
