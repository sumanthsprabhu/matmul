#ifndef LIB_PASSES_TD_
#define LIB_PASSES_TD_

include "mlir/Pass/PassBase.td"

def SuprasFirstPass : Pass<"supras-first-pass"> {
  let summary = "Loop Invariant Code Motion";
  let constructor = "mlir::createSuprasFirstPass()";
  let options = [
      Option<"tileSize", "tileSize", "int", "32",
           "Use this option to set the tile size for matmul (default: 32)">,
           ];

}

#endif  // LIB_PASSES_TD_
