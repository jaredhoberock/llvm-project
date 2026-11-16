//===- BufferizationTypeInterfaces.td - Type Interfaces ----*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This is the definition file for type interfaces used in Bufferization.
//
//===----------------------------------------------------------------------===//

#ifndef BUFFERIZATION_TYPE_INTERFACES
#define BUFFERIZATION_TYPE_INTERFACES

include "mlir/IR/OpBase.td"

def Bufferization_TensorLikeTypeInterface
    : TypeInterface<"TensorLikeType"> {
  let cppNamespace = "::mlir::bufferization";
  let description = [{
    Indicates that this type is a tensor type (similarly to a MLIR builtin
    tensor) for bufferization purposes.

    `isTensorLike` refines whether a particular instance of the type is
    tensor-like; `isa<TensorLikeType>` requires both a registered model and
    this predicate.
  }];

  let methods = [
    InterfaceMethod<[{
        Returns whether a BufferLike type is compatible to this TensorLike type.
        The BufferLike type is assumed to be created by unknown type converter.
      }],
      /*retTy=*/"::mlir::LogicalResult",
      /*methodName=*/"verifyCompatibleBufferType",
      /*args=*/(ins
        "::mlir::bufferization::BufferLikeType":$bufferType,
        "::llvm::function_ref<::mlir::InFlightDiagnostic()>":$emitError)
    >,
    InterfaceMethod<[{
        Returns whether this instance of the type is a tensor type for
        bufferization purposes. A type overrides this when only some of its
        instances are, for example when that depends on its parameters.
      }],
      /*retTy=*/"bool",
      /*methodName=*/"isTensorLike",
      /*args=*/(ins),
      /*methodBody=*/"",
      /*defaultImplementation=*/"return true;"
    >
  ];

  let extraClassOf = [{
    return $_type.isTensorLike();
  }];
}

def Bufferization_BufferLikeTypeInterface
    : TypeInterface<"BufferLikeType"> {
  let cppNamespace = "::mlir::bufferization";
  let description = [{
    Indicates that this type is a buffer type (similarly to a MLIR builtin
    memref) for bufferization purposes.

    `isBufferLike` refines whether a particular instance of the type is
    buffer-like; `isa<BufferLikeType>` requires both a registered model and
    this predicate.

    The `gen` methods build at the builder's insertion point, which a caller
    places in a block of the IR, so that a method may look at what stands
    around it. `genAllocate`, `genCopy` and `genCast` fail where the type has
    no answer, which is their default; a method that fails may have built
    operations, and a caller that must leave the IR unchanged removes them.
  }];

  let methods = [
    InterfaceMethod<[{
        Returns whether this instance of the type is a buffer type for
        bufferization purposes. A type overrides this when only some of its
        instances are, for example when that depends on its parameters.
      }],
      /*retTy=*/"bool",
      /*methodName=*/"isBufferLike",
      /*args=*/(ins),
      /*methodBody=*/"",
      /*defaultImplementation=*/"return true;"
    >,
    InterfaceMethod<[{
        Returns whether this instance of the type says how its values are
        deallocated and identified, by `genFree` and `genIdentity`, which may
        be called only where this holds.
      }],
      /*retTy=*/"bool",
      /*methodName=*/"supportsDeallocation",
      /*args=*/(ins),
      /*methodBody=*/"",
      /*defaultImplementation=*/"return false;"
    >,
    InterfaceMethod<[{
        Builds the deallocation of the allocation `buffer`, a value of this
        type, belongs to. Where the type's values can be parts of an
        allocation, `buffer` may be a part, and the whole allocation is
        deallocated.
      }],
      /*retTy=*/"void",
      /*methodName=*/"genFree",
      /*args=*/(ins "::mlir::OpBuilder &":$builder, "::mlir::Location":$loc,
                    "::mlir::Value":$buffer),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        llvm_unreachable("genFree not implemented");
      }]
    >,
    InterfaceMethod<[{
        Builds the identity of the allocation `buffer`, a value of this type,
        belongs to: an `index`. Two values that one deallocation would release
        have equal identities; values of different allocations have different
        identities, except where deallocating either does nothing. Identities
        are compared across buffer types, so a type whose values can belong to
        an allocation of another type gives that type's identity; a builtin
        memref's identity is its aligned pointer.
      }],
      /*retTy=*/"::mlir::Value",
      /*methodName=*/"genIdentity",
      /*args=*/(ins "::mlir::OpBuilder &":$builder, "::mlir::Location":$loc,
                    "::mlir::Value":$buffer),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        llvm_unreachable("genIdentity not implemented");
      }]
    >,
    InterfaceMethod<[{
        Builds a new buffer of this type shaped as `like`, which may be of
        another buffer type; with no `like`, only for an instance whose
        extents are all static. `alignment` is in bytes, 0 for none. The
        contents are not set, and the buffer is deallocated by `genFree`.
        Fails where the type cannot read the extents of `like`.
      }],
      /*retTy=*/"::mlir::FailureOr<::mlir::Value>",
      /*methodName=*/"genAllocate",
      /*args=*/(ins "::mlir::OpBuilder &":$builder, "::mlir::Location":$loc,
                    "::mlir::Value":$like, "unsigned":$alignment),
      /*methodBody=*/"",
      /*defaultImplementation=*/"return ::mlir::failure();"
    >,
    InterfaceMethod<[{
        Builds the copy of the contents of the buffer `source` into
        `destination`, a value of this type. Fails where the type cannot copy
        from the type of `source`.
      }],
      /*retTy=*/"::mlir::LogicalResult",
      /*methodName=*/"genCopy",
      /*args=*/(ins "::mlir::OpBuilder &":$builder, "::mlir::Location":$loc,
                    "::mlir::Value":$source, "::mlir::Value":$destination),
      /*methodBody=*/"",
      /*defaultImplementation=*/"return ::mlir::failure();"
    >,
    InterfaceMethod<[{
        Builds `buffer`, a value of this type, as a value of `resultType` that
        is the same buffer. Succeeds only where the cast holds for every value
        of this type, so that it cannot fail when the program runs; a caller
        may copy the buffer where it fails.
      }],
      /*retTy=*/"::mlir::FailureOr<::mlir::Value>",
      /*methodName=*/"genCast",
      /*args=*/(ins "::mlir::OpBuilder &":$builder, "::mlir::Location":$loc,
                    "::mlir::Value":$buffer, "::mlir::Type":$resultType),
      /*methodBody=*/"",
      /*defaultImplementation=*/"return ::mlir::failure();"
    >
  ];

  let extraClassOf = [{
    return $_type.isBufferLike();
  }];
}

#endif // BUFFERIZATION_TYPE_INTERFACES
