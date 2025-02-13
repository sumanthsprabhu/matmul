get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)

set (LIBS
    ${dialect_libs}
    ${conversion_libs}
    SuprasFirstPass
    MLIROptLib
    MLIRPass
)

add_llvm_executable(suprasfirstpass-opt suprasfirstpass-opt.cpp)

llvm_update_compile_flags(suprasfirstpass-opt)
target_link_libraries(suprasfirstpass-opt PRIVATE ${LIBS})

mlir_check_all_link_libraries(suprasfirstpass-opt)
