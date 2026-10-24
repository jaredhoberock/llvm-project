// RUN: mlir-opt %s -canonicalize | FileCheck %s

// Inlining an scf.execute_region builds cf.br ops, so the scf dialect must
// load the cf dialect: this file holds no cf op, so nothing else loads it.

// CHECK-LABEL: func @inline_execute_region
// CHECK-NOT:     scf.execute_region
// CHECK:         arith.addi
func.func @inline_execute_region(%x: i64) -> i64 {
  %r = scf.execute_region -> i64 {
    %one = arith.constant 1 : i64
    %sum = arith.addi %x, %one : i64
    scf.yield %sum : i64
  }
  return %r : i64
}
