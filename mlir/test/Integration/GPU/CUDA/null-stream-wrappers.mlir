// RUN: mlir-runner %s \
// RUN:   --shared-libs=%mlir_cuda_runtime \
// RUN:   --shared-libs=%mlir_c_runner_utils \
// RUN:   --entry-point-result=void 2>&1 \
// RUN: | FileCheck %s

// Each wrapper that hands a stream to the driver is given the null stream,
// which is the default stream of the current context, while no context is
// current. None of them may fail for want of one.

// CHECK-NOT: failed with
// CHECK: {{^}}24681357{{$}}
module {
  llvm.func @mgpuMemAlloc(i64, !llvm.ptr, i8) -> !llvm.ptr
  llvm.func @mgpuMemFree(!llvm.ptr, !llvm.ptr)
  llvm.func @mgpuMemcpy(!llvm.ptr, !llvm.ptr, i64, !llvm.ptr)
  llvm.func @mgpuMemset32(!llvm.ptr, i32, i64, !llvm.ptr)
  llvm.func @mgpuMemset16(!llvm.ptr, i16, i64, !llvm.ptr)
  llvm.func @mgpuEventCreate() -> !llvm.ptr
  llvm.func @mgpuEventRecord(!llvm.ptr, !llvm.ptr)
  llvm.func @mgpuStreamWaitEvent(!llvm.ptr, !llvm.ptr)
  llvm.func @mgpuStreamSynchronize(!llvm.ptr)
  llvm.func @mgpuEventDestroy(!llvm.ptr)
  llvm.func @printI64(i64)
  llvm.func @printNewline()

  llvm.func @main() {
    %stream = llvm.mlir.zero : !llvm.ptr
    %bytes = llvm.mlir.constant(64 : i64) : i64
    %count = llvm.mlir.constant(16 : i64) : i64
    %host_shared = llvm.mlir.constant(0 : i8) : i8
    %zero32 = llvm.mlir.constant(0 : i32) : i32
    %zero16 = llvm.mlir.constant(0 : i16) : i16
    %a = llvm.call @mgpuMemAlloc(%bytes, %stream, %host_shared) : (i64, !llvm.ptr, i8) -> !llvm.ptr
    %b = llvm.call @mgpuMemAlloc(%bytes, %stream, %host_shared) : (i64, !llvm.ptr, i8) -> !llvm.ptr
    llvm.call @mgpuMemset32(%a, %zero32, %count, %stream) : (!llvm.ptr, i32, i64, !llvm.ptr) -> ()
    llvm.call @mgpuMemset16(%b, %zero16, %count, %stream) : (!llvm.ptr, i16, i64, !llvm.ptr) -> ()
    llvm.call @mgpuMemcpy(%b, %a, %bytes, %stream) : (!llvm.ptr, !llvm.ptr, i64, !llvm.ptr) -> ()
    %event = llvm.call @mgpuEventCreate() : () -> !llvm.ptr
    llvm.call @mgpuEventRecord(%event, %stream) : (!llvm.ptr, !llvm.ptr) -> ()
    llvm.call @mgpuStreamWaitEvent(%stream, %event) : (!llvm.ptr, !llvm.ptr) -> ()
    llvm.call @mgpuStreamSynchronize(%stream) : (!llvm.ptr) -> ()
    llvm.call @mgpuEventDestroy(%event) : (!llvm.ptr) -> ()
    llvm.call @mgpuMemFree(%a, %stream) : (!llvm.ptr, !llvm.ptr) -> ()
    llvm.call @mgpuMemFree(%b, %stream) : (!llvm.ptr, !llvm.ptr) -> ()
    %sentinel = llvm.mlir.constant(24681357 : i64) : i64
    llvm.call @printI64(%sentinel) : (i64) -> ()
    llvm.call @printNewline() : () -> ()
    llvm.return
  }
}
