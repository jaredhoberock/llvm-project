// RUN: mlir-opt -verify-diagnostics -ownership-based-buffer-deallocation \
// RUN:   -split-input-file %s | FileCheck %s
// RUN: mlir-opt -verify-diagnostics -ownership-based-buffer-deallocation \
// RUN:   --buffer-deallocation-simplification -split-input-file %s \
// RUN:   | FileCheck %s --check-prefix=SIMPLIFIED

// A value of a buffer-like type that is not a memref is deallocated as it is
// when it dies.

// CHECK-LABEL: func @buffer_like_dies_in_its_block(
//       CHECK:   %[[true:.*]] = arith.constant true
//       CHECK:   %[[buffer:.*]] = "test.alloc_buffer_like"()
//   CHECK-NOT:   memref.extract_strided_metadata
//       CHECK:   bufferization.dealloc (%[[buffer]] : !test.test_memref<[4], f32>) if (%[[true]])
//  CHECK-NEXT:   return
func.func @buffer_like_dies_in_its_block() {
  %0 = "test.alloc_buffer_like"() : () -> !test.test_memref<[4], f32>
  return
}

// -----

// The ownership of a buffer-like value forwarded to a successor block is
// forwarded with it, and the value is deallocated in the block where it dies.

// CHECK-LABEL: func @buffer_like_forwarded(
//       CHECK:   %[[true:.*]] = arith.constant true
//       CHECK:   %[[buffer:.*]] = "test.alloc_buffer_like"()
//       CHECK:   %[[owned:.*]] = bufferization.dealloc (%[[buffer]] : !test.test_memref<[4], f32>) if (%[[true]]) retain (%[[buffer]] : !test.test_memref<[4], f32>)
//       CHECK:   cf.br ^[[bb1:.*]](%[[buffer]], %[[owned]] : !test.test_memref<[4], f32>, i1)
//       CHECK: ^[[bb1]](%[[arg:.*]]: !test.test_memref<[4], f32>, %[[cond:.*]]: i1):
//       CHECK:   bufferization.dealloc (%[[arg]] : !test.test_memref<[4], f32>) if (%[[cond]])
//  CHECK-NEXT:   return

// The simplification proves that the retained value is the deallocated one.

// SIMPLIFIED-LABEL: func @buffer_like_forwarded(
//       SIMPLIFIED:   %[[true:.*]] = arith.constant true
//       SIMPLIFIED:   %[[buffer:.*]] = "test.alloc_buffer_like"()
//  SIMPLIFIED-NEXT:   cf.br ^[[bb1:.*]](%[[buffer]], %[[true]] : !test.test_memref<[4], f32>, i1)
//       SIMPLIFIED: ^[[bb1]](%[[arg:.*]]: !test.test_memref<[4], f32>, %[[cond:.*]]: i1):
//       SIMPLIFIED:   bufferization.dealloc (%[[arg]] : !test.test_memref<[4], f32>) if (%[[cond]])
func.func @buffer_like_forwarded() {
  %0 = "test.alloc_buffer_like"() : () -> !test.test_memref<[4], f32>
  cf.br ^bb1(%0 : !test.test_memref<[4], f32>)
^bb1(%1: !test.test_memref<[4], f32>):
  return
}

// -----

// A public function returns buffers it owns: a buffer-like argument, which it
// does not own, is cloned before it is returned.

// CHECK-LABEL: func @buffer_like_argument_returned(
//  CHECK-SAME:     %[[arg:.*]]: !test.test_memref<[4], f32>)
//       CHECK:   bufferization.clone %[[arg]] : !test.test_memref<[4], f32> to !test.test_memref<[4], f32>
func.func @buffer_like_argument_returned(%arg0: !test.test_memref<[4], f32>) -> !test.test_memref<[4], f32> {
  return %arg0 : !test.test_memref<[4], f32>
}
