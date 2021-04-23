#pragma once

#ifdef TORCH_ENABLE_LLVM
#include <c10/util/Exception.h>
#include <torch/csrc/WindowsTorchApiMacro.h>

#include <llvm/ExecutionEngine/JITSymbol.h>
#include <llvm/ExecutionEngine/Orc/Core.h>
#include <llvm/ExecutionEngine/Orc/ThreadSafeModule.h>
#include <llvm/Target/TargetMachine.h>

#include <memory>
#include <string>

namespace torch {
namespace jit {
namespace tensorexpr {

void DispatchParallel(int8_t* func, int start, int stop, int8_t* packed_data);

} // namespace tensorexpr
} // namespace jit
} // namespace torch

namespace llvm {
namespace orc {

class PytorchLLVMJITImpl;

class TORCH_API PytorchLLVMJIT {
 public:
  PytorchLLVMJIT();
  ~PytorchLLVMJIT();

  void addModule(std::unique_ptr<Module> M, std::unique_ptr<LLVMContext> C);

  JITSymbol findSymbol(const char* Name);

  bool hasSymbol(const char* Name);

  TargetMachine& getTargetMachine();

  const DataLayout& getDataLayout();

 private:
  // Use the PImpl idiom here to hide the no-rtti parts of the JIT structure.
  std::unique_ptr<PytorchLLVMJITImpl> impl_;
};

} // end namespace orc
} // end namespace llvm

#endif // ENABLE LLVM
