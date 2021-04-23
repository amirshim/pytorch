#pragma once

#ifdef TORCH_ENABLE_LLVM

namespace torch {
namespace jit {
namespace tensorexpr {

struct SymbolAddressIf {
  const char* symbol;
  void* address;
};

SymbolAddressIf* getSymbols();

} // namespace tensorexpr
} // namespace jit
} // namespace torch
#endif // TORCH_ENABLE_LLVM
