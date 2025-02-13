#include "mlir/IR/MLIRContext.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/ExecutionEngine/ExecutionEngine.h"
#include "mlir/InitAllDialects.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"

#include "llvm/Support/CommandLine.h"

#include "lib/Passes.h"

#include <string>
#include <fstream>

using namespace mlir;
namespace cl = llvm::cl;

static cl::opt<int> tileSize ("tile-size",
                              cl::desc("Option to set tile size for matmul (default: 32)"),
                              cl::init(32));

static cl::opt<std::string> inputFilename(cl::Positional,
                                          cl::desc("<input file>"),
                                          cl::init("-"),
                                          cl::value_desc("filename"));

static cl::opt<bool> printInput("printInput",
                                cl::desc("Print input file"),
                                cl::init(false));

void printInputFile(const std::string &inputFileName)
{
  std::ifstream file(inputFileName);
  if (!file.is_open()) {
    llvm::errs() << "Failed to open input file: " << inputFileName << "\n";
    return;
  }

  llvm::outs() << "Input file \n";
  std::string line;
  while (std::getline(file, line)) {
    llvm::outs() << line << "\n";
  }

  llvm::outs() << "\n";
  
  file.close();                                     
}

int main(int argc, char **argv) {
  
  cl::ParseCommandLineOptions(argc, argv, "SuPraS' First Pass to tile matmul\n");

  if (printInput) {
    printInputFile(inputFilename);
  }
  
  mlir::DialectRegistry registry;
  registry.insert<affine::AffineDialect, arith::ArithDialect, func::FuncDialect, memref::MemRefDialect, tosa::TosaDialect>();

  MLIRContext context(registry);

  OwningOpRef<ModuleOp> module (parseSourceFile<ModuleOp>(inputFilename, &context));
  if (!module) {
    llvm::errs() << "Failed to parse input file\n";
    return 1;
  }

  PassManager pm(&context);
  
  pm.addPass(createSuprasFirstPass(tileSize));

  if (failed(pm.run(*module))) {
    llvm::errs() << "Failed to run pass\n";
    return 1;
  }
  
  module->print(llvm::outs());
  return 0;
}
