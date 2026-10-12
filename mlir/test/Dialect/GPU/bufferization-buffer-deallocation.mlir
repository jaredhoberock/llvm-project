// RUN: mlir-opt %s -buffer-deallocation-pipeline -split-input-file --allow-unregistered-dialect | FileCheck %s

func.func @gpu_launch() {
  %c1 = arith.constant 1 : index
  gpu.launch blocks(%arg0, %arg1, %arg2) in (%arg6 = %c1, %arg7 = %c1, %arg8 = %c1)
    threads(%arg3, %arg4, %arg5) in (%arg9 = %c1, %arg10 = %c1, %arg11 = %c1) {
    %alloc = memref.alloc() : memref<2xf32>
    "test.read_buffer"(%alloc) : (memref<2xf32>) -> ()
    gpu.terminator
  }
  return
}

// CHECK-LABEL: func @gpu_launch
//       CHECK:   gpu.launch
//       CHECK:     [[ALLOC:%.+]] = memref.alloc(
//       CHECK:     memref.dealloc [[ALLOC]]
//       CHECK:     gpu.terminator

// -----

// A synchronous launch: the buffer it reads is freed after it. An asynchronous
// launch is not covered; its buffers' lifetime runs to the await of its token.
module attributes {gpu.container_module} {
  gpu.module @kernels {
    gpu.func @read_buffer(%arg0: memref<2xf32>) kernel {
      gpu.return
    }
  }

  func.func @gpu_launch_func() {
    %c1 = arith.constant 1 : index
    %alloc = memref.alloc() : memref<2xf32>
    gpu.launch_func @kernels::@read_buffer
        blocks in (%c1, %c1, %c1) threads in (%c1, %c1, %c1)
        args(%alloc : memref<2xf32>)
    return
  }
}

// CHECK-LABEL: func @gpu_launch_func
//       CHECK:   [[ALLOC:%.+]] = memref.alloc(
//       CHECK:   gpu.launch_func
//       CHECK:   memref.dealloc [[ALLOC]]
//       CHECK:   return
