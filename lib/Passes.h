#ifndef LIB__PASSES_H_
#define LIB__PASSES_H_

#include "mlir/Pass/Pass.h"
#include <memory>

namespace mlir {
std::unique_ptr<Pass> createSuprasFirstPass();
std::unique_ptr<Pass> createSuprasFirstPass(int);
  
#define GEN_PASS_DECL_SUPRASFIRSTPASS
#define GEN_PASS_REGISTRATION
#include "lib/Passes.h.inc"
}

#endif  // LIB_PASSES_H

