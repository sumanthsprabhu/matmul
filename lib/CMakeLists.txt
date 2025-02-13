add_mlir_library(SuprasFirstPass
    SuprasFirstPass.cpp

    ADDITIONAL_HEADER_DIRS
    ${PROJECT_SOURCE_DIR}/lib/     

    DEPENDS
    MLIRSuprasFirstPassIncGen


    LINK_LIBS PUBLIC
    MLIRIR
    MLIRArithDialect
    MLIRLinalgDialect
    MLIRAffineDialect
    MLIRMemRefDialect
    MLIRTosaDialect
    MLIRTensorDialect
    MLIRTensorUtils
    MLIRTransformUtils
    MLIRFuncDialect
    MLIRBufferizationDialect                
)

set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls -name lib)
add_public_tablegen_target(MLIRSuprasFirstPassIncGen)
add_mlir_doc(Passes LibPasses ./ -gen-pass-doc)
