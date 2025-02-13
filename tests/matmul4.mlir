func.func @example(%arg0: tensor<1x4x4xf32>, %arg1: tensor<1x4x4xf32>) -> tensor<1x4x4xf32> {
  %0 = tosa.matmul %arg0, %arg1 : (tensor<1x4x4xf32>, tensor<1x4x4xf32>) -> tensor<1x4x4xf32>
  return %0 : tensor<1x4x4xf32>
}

  func.func private @printMemrefF32(tensor<*xf32>)
  func.func @main() {
    %cst = arith.constant dense<1.000000e+00> : tensor<1x4x4xf32>
    %cst_0 = arith.constant dense<1.000000e+00> : tensor<1x4x4xf32>
    %0 = call @example(%cst, %cst_0) : (tensor<1x4x4xf32>, tensor<1x4x4xf32>) -> tensor<1x4x4xf32>
    %1 = call @example(%cst, %0) : (tensor<1x4x4xf32>, tensor<1x4x4xf32>) -> tensor<1x4x4xf32>
    %2 = call @example(%1, %0) : (tensor<1x4x4xf32>, tensor<1x4x4xf32>) -> tensor<1x4x4xf32>
    %cast = tensor.cast %2 : tensor<1x4x4xf32> to tensor<*xf32>    
    call @printMemrefF32(%cast) : (tensor<*xf32>) -> ()
    return
  } 
