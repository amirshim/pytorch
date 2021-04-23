#include <torch/csrc/jit/tensorexpr/intrinsic_symbols_if.h>
#include <torch/csrc/jit/tensorexpr/intrinsic_symbols.h>
#include <torch/csrc/jit/tensorexpr/external_functions_registry.h>

using namespace torch::jit::tensorexpr;

SymbolAddressIf* torch::jit::tensorexpr::getSymbols() {
  auto const& intrinsics = getIntrinsicSymbols();
  auto const& externals = getNNCFunctionRegistry();
  SymbolAddressIf* syms = new SymbolAddressIf[intrinsics.size() + externals.size() + 1];
  int i = 0;
  for (auto const& sym : intrinsics) {
    syms[i].symbol = sym.symbol;
    syms[i].address = (void*)sym.address;
    i++;
  }
  for (auto const& kv : externals) {
    syms[i].symbol = kv.first.c_str();
    syms[i].address = (void*)kv.second;
    i++;
  }
  syms[i].symbol = nullptr;
  syms[i].address = nullptr;
  return syms;
}
