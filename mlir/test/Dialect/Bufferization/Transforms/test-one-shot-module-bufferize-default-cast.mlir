// RUN: mlir-opt %s -test-one-shot-module-bufferize="default-cast" -split-input-file -verify-diagnostics | FileCheck %s

// A call's operand whose buffer type differs from the callee's parameter is
// cast where the cast holds for every value of its type, and copied
// otherwise: a dynamic stride is not cast to a static one.

func.func private @custom_types_callee(!test.test_tensor<[4], f64>)

// CHECK-LABEL: func.func @custom_types_call_operand_cast_or_copy(
// CHECK-SAME: %[[a:[^:]+]]: !test.test_memref<[4], f64, strided<[1]>>,
// CHECK-SAME: %[[b:[^:]+]]: !test.test_memref<[4], f64, strided<[2]>>,
// CHECK-SAME: %[[c:[^:]+]]: !test.test_memref<[4], f64, strided<[?], offset: ?>>
// CHECK: %[[am:.+]] = builtin.unrealized_conversion_cast %[[a]] : !test.test_memref<[4], f64, strided<[1]>> to memref<4xf64, strided<[1]>>
// CHECK: %[[ac:.+]] = memref.cast %[[am]] : memref<4xf64, strided<[1]>> to memref<4xf64>
// CHECK: %[[acast:.+]] = builtin.unrealized_conversion_cast %[[ac]] : memref<4xf64> to !test.test_memref<[4], f64>
// CHECK: call @custom_types_callee(%[[acast]])
// CHECK: %[[balloc:.+]] = memref.alloc() {alignment = 64 : i64} : memref<4xf64>
// CHECK: %[[bcopy:.+]] = builtin.unrealized_conversion_cast %[[balloc]] : memref<4xf64> to !test.test_memref<[4], f64>
// CHECK: %[[bm:.+]] = builtin.unrealized_conversion_cast %[[b]] : !test.test_memref<[4], f64, strided<[2]>> to memref<4xf64, strided<[2]>>
// CHECK: memref.copy %[[bm]],
// CHECK: call @custom_types_callee(%[[bcopy]])
// CHECK: %[[calloc:.+]] = memref.alloc() {alignment = 64 : i64} : memref<4xf64>
// CHECK: %[[ccopy:.+]] = builtin.unrealized_conversion_cast %[[calloc]] : memref<4xf64> to !test.test_memref<[4], f64>
// CHECK: %[[cm:.+]] = builtin.unrealized_conversion_cast %[[c]] : !test.test_memref<[4], f64, strided<[?], offset: ?>> to memref<4xf64, strided<[?], offset: ?>>
// CHECK: memref.copy %[[cm]],
// CHECK: call @custom_types_callee(%[[ccopy]])
func.func @custom_types_call_operand_cast_or_copy(
    %a: !test.test_memref<[4], f64, strided<[1]>>,
    %b: !test.test_memref<[4], f64, strided<[2]>>,
    %c: !test.test_memref<[4], f64, strided<[?], offset: ?>>) {
  %ta = bufferization.to_tensor %a restrict writable
    : !test.test_memref<[4], f64, strided<[1]>> to !test.test_tensor<[4], f64>
  func.call @custom_types_callee(%ta) : (!test.test_tensor<[4], f64>) -> ()
  %tb = bufferization.to_tensor %b restrict writable
    : !test.test_memref<[4], f64, strided<[2]>> to !test.test_tensor<[4], f64>
  func.call @custom_types_callee(%tb) : (!test.test_tensor<[4], f64>) -> ()
  %tc = bufferization.to_tensor %c restrict writable
    : !test.test_memref<[4], f64, strided<[?], offset: ?>>
      to !test.test_tensor<[4], f64>
  func.call @custom_types_callee(%tc) : (!test.test_tensor<[4], f64>) -> ()
  return
}

// -----

// Where two buffers meet and the default cast refuses one of them, the join
// says so.

func.func @custom_types_select_cast_refused(
    %a: !test.test_memref<[4], f64, strided<[1]>>,
    %b: !test.test_memref<[4], f64, strided<[2]>>, %cond: i1)
    -> !test.test_tensor<[4], f64> {
  %ta = bufferization.to_tensor %a restrict
    : !test.test_memref<[4], f64, strided<[1]>> to !test.test_tensor<[4], f64>
  %tb = bufferization.to_tensor %b restrict
    : !test.test_memref<[4], f64, strided<[2]>> to !test.test_tensor<[4], f64>
  // expected-error @+2 {{cannot cast a buffer of type '!test.test_memref<[4], f64, strided<[2]>>' to '!test.test_memref<[4], f64, strided<[1]>>'}}
  // expected-error @+1 {{failed to bufferize op}}
  %r = arith.select %cond, %ta, %tb : !test.test_tensor<[4], f64>
  return %r : !test.test_tensor<[4], f64>
}
