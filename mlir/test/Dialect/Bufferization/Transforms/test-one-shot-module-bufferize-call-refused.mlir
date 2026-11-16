// RUN: mlir-opt %s -test-one-shot-module-bufferize="default-cast test-tensor-to-dynamic-memref" -verify-diagnostics

// A call's operand of a buffer type that cannot be cast to the callee's
// parameter type, which cannot be allocated like it, is refused.

func.func private @callee(!test.test_tensor<[4], f64>)

func.func @call_operand_refused(%b: !test.test_memref<[4], f64>) {
  %t = bufferization.to_tensor %b restrict writable
    : !test.test_memref<[4], f64> to !test.test_tensor<[4], f64>
  // expected-error @+2 {{cannot cast or copy operand #0 of type '!test.test_memref<[4], f64>' to the callee's parameter type 'memref<?xf64>'}}
  // expected-error @+1 {{failed to bufferize op}}
  func.call @callee(%t) : (!test.test_tensor<[4], f64>) -> ()
  return
}
