// RUN: mlir-opt %s -inline="default-pipeline=" -split-input-file | FileCheck %s

// A call inside a gpu.launch body is inlined.

func.func @callee(%arg0: f32) {
  %0 = memref.alloca() : memref<4xf32>
  %c0 = arith.constant 0 : index
  memref.store %arg0, %0[%c0] : memref<4xf32>
  return
}

// CHECK-LABEL: func @call_in_launch
//       CHECK:   gpu.launch
//   CHECK-NOT:     call
//       CHECK:     %[[alloca:.*]] = memref.alloca() : memref<4xf32>
//       CHECK:     memref.store %{{.*}}, %[[alloca]][%{{.*}}] : memref<4xf32>
//   CHECK-NOT:     call
//       CHECK:     gpu.terminator
func.func @call_in_launch(%arg0: f32) {
  %c1 = arith.constant 1 : index
  gpu.launch blocks(%bx, %by, %bz) in (%gx = %c1, %gy = %c1, %gz = %c1)
             threads(%tx, %ty, %tz) in (%sx = %c1, %sy = %c1, %sz = %c1) {
    func.call @callee(%arg0) : (f32) -> ()
    gpu.terminator
  }
  return
}

// -----

// A call inside a gpu.func is inlined.

gpu.module @kernels {
  func.func @callee(%arg0: f32) {
    %0 = memref.alloca() : memref<4xf32>
    %c0 = arith.constant 0 : index
    memref.store %arg0, %0[%c0] : memref<4xf32>
    return
  }

  // CHECK-LABEL: gpu.func @call_in_gpu_func
  //   CHECK-NOT:   call
  //       CHECK:   %[[alloca:.*]] = memref.alloca() : memref<4xf32>
  //       CHECK:   memref.store %{{.*}}, %[[alloca]][%{{.*}}] : memref<4xf32>
  //   CHECK-NOT:   call
  //       CHECK:   gpu.return
  gpu.func @call_in_gpu_func(%arg0: f32) kernel {
    func.call @callee(%arg0) : (f32) -> ()
    gpu.return
  }
}
