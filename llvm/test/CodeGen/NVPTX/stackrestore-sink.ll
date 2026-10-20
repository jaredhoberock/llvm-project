; RUN: llc < %s -mtriple=nvptx64 -mcpu=sm_52 -mattr=+ptx73 | FileCheck %s
; RUN: %if ptxas-isa-7.3 %{ llc < %s -mtriple=nvptx64 -mcpu=sm_52 -mattr=+ptx73 | %ptxas-verify %}

; A load of a dynamic alloca made after a stacksave, then the stackrestore that
; frees it, then a second dynamic alloca of the same size that may reuse the
; freed bytes. The first load's only use is in the successor that makes the
; second alloca. The stackrestore has unmodeled side effects and no memory
; operands, so a load must not be moved across it: MachineSink keeps the load
; ahead of the restore instead of sinking it into the successor, where it would
; read the second alloca's bytes.

define ptx_kernel void @k(ptr addrspace(1) %out, i64 %n, i64 %i, i64 %j, i1 %c) {
; CHECK-LABEL: k(
; CHECK:       stacksave.u64
; CHECK:       ld.local.b64
; CHECK:       stackrestore.u64
; CHECK:       ld.local.b64
; CHECK-NOT:   ld.local
; CHECK:       ret;
entry:
  %s = call ptr @llvm.stacksave.p0()
  %a = alloca i64, i64 %n, align 16
  %ai = getelementptr i64, ptr %a, i64 %i
  store i64 1, ptr %ai
  %aj = getelementptr i64, ptr %a, i64 %j
  %v = load i64, ptr %aj
  call void @llvm.stackrestore.p0(ptr %s)
  br i1 %c, label %then, label %else

then:
  %b = alloca i64, i64 %n, align 16
  %bi = getelementptr i64, ptr %b, i64 %i
  store i64 2, ptr %bi
  %bj = getelementptr i64, ptr %b, i64 %j
  %w = load i64, ptr %bj
  %sum = add i64 %v, %w
  store i64 %sum, ptr addrspace(1) %out
  ret void

else:
  store i64 0, ptr addrspace(1) %out
  ret void
}

declare ptr @llvm.stacksave.p0()
declare void @llvm.stackrestore.p0(ptr)
