//===- BufferLikeTypeTest.cpp - unit tests for buffer-like types ----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "gtest/gtest.h"

using namespace mlir;
using namespace mlir::bufferization;

namespace {
/// A function with one argument of `argType`, to cast in its body.
struct CastFixture {
  CastFixture(MLIRContext &context, Type argType)
      : builder(&context), module(ModuleOp::create(UnknownLoc::get(&context))) {
    context.loadDialect<BufferizationDialect, func::FuncDialect,
                        memref::MemRefDialect>();
    builder.setInsertionPointToEnd(module->getBody());
    auto func = func::FuncOp::create(builder, builder.getUnknownLoc(), "f",
                                     builder.getFunctionType({argType}, {}));
    builder.setInsertionPointToEnd(func.addEntryBlock());
    arg = func.getArgument(0);
  }

  /// Asks the argument's type to cast it to `resultType`.
  bool casts(Type resultType) {
    return succeeded(
        cast<BufferLikeType>(arg.getType())
            .genCast(builder, builder.getUnknownLoc(), arg, resultType));
  }

  OpBuilder builder;
  OwningOpRef<ModuleOp> module;
  Value arg;
};
} // namespace

// A memref's cast holds for every value: a dynamic extent stays dynamic.
TEST(BufferLikeTypeTest, MemRefCastKeepsDynamicExtents) {
  MLIRContext context;
  Type f32 = Float32Type::get(&context);
  CastFixture fixture(context, MemRefType::get({ShapedType::kDynamic}, f32));
  EXPECT_FALSE(fixture.casts(MemRefType::get({4}, f32)));
  EXPECT_TRUE(fixture.casts(UnrankedMemRefType::get(f32, Attribute())));
}

TEST(BufferLikeTypeTest, MemRefCastToDynamicExtent) {
  MLIRContext context;
  Type f32 = Float32Type::get(&context);
  CastFixture fixture(context, MemRefType::get({4}, f32));
  EXPECT_TRUE(fixture.casts(MemRefType::get({ShapedType::kDynamic}, f32)));
}
