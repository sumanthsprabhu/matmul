# Problem
Consider matrix multiplication of square dense matrices of dimension `m x m`. Suppose, we have `p` processors, each of which can do a matrix multiplciation of `t x t` matrices, where `m` is a multiple of `t`.  How can we partition the multiplication into `p` processors _efficiently_? Assume memory access penatly is absent. 

Matrix multiplication is an important operation in machine learning models. Many hardware designs and compiler optimizations focus on making this operation more [efficient](https://arxiv.org/pdf/2002.03794). For instance, Google's TPU has a Matrix Multiplier Unit (MXU) with matrix as primitive. In this context, efficient matrix-matrix multiplication is vital, and parallelization is a common technique to achieve this.

## MLIR
Assume that the input problem is given in MLIR with the `tosa.matmul` operation and dense 3-d tensors with fixed dimensions. The output is a 3-d tensor, with `tosa.matmul` parallelized to the `linalg.batch_matmul` operation that uses `bufferization`.


# Solution Sketch
I have used [tiling](https://alvinwan.com/how-to-tile-matrix-multiplication/) to parallelize matrix multiplication, assuming the size of the tile is `t`. This will result in `(m/t)^3` matrix multiplications that can be scheduled on `p` processors in parallel.

# Installation
* [Build](https://mvinay.github.io/mlir/tosa-lowerings.html) llvm-mlir in a directory
** You need to replace `mlir-cpu-runner` by `mlir-runner` in the last command
** Our tool was built using LLVM with commit `9cd83d6ea`. You can use the same commit by executing following command after `git clone`
```
git checkout 9cd83d6ea
```


* Clone and build our tool
```
git clone https://github.com/sumanthsprabhu/matmul.git
cd matmul
mkdir build
cd build
export MY_LLVM_SRC=<your_llvm-project_directory_from_previous_step>
cmake ..
make
```

* Running (from `build` directory)
After build is successful, the tool be present in `./tool/suprasfirstpass-opt`. You can use mllir files in `tests` directory. The size of the tile can be set using the option `--tileSize`, whose default value is 32.
```
./tool/suprasfirstpass-opt --tileSize <tile-size> <input-file>
```

For example,
```
./tool/suprasfirstpass-opt --tileSize 2 ../tests/matmul1.mlir

module {
  func.func @example(%arg0: tensor<1x2x2xf32>, %arg1: tensor<1x2x2xf32>) -> tensor<1x2x2xf32> {
    %alloc = memref.alloc() : memref<1x2x2xf32>
    %cst = arith.constant 0.000000e+00 : f32
    affine.parallel (%arg2, %arg3, %arg4) = (0, 0, 0) to (1, 2, 2) {
      affine.store %cst, %alloc[%arg2, %arg3, %arg4] : memref<1x2x2xf32>
    }
    %0 = bufferization.to_memref %arg0 : tensor<1x2x2xf32> to memref<1x2x2xf32, strided<[?, ?, ?], offset: ?>>
    %1 = bufferization.to_memref %arg1 : tensor<1x2x2xf32> to memref<1x2x2xf32, strided<[?, ?, ?], offset: ?>>
    affine.parallel (%arg2, %arg3, %arg4, %arg5) = (0, 0, 0, 0) to (1, 2, 2, 2) step (1, 2, 2, 2) {
      %subview = memref.subview %0[%arg2, %arg3, %arg4] [1, 2, 2] [1, 1, 1] : memref<1x2x2xf32, strided<[?, ?, ?], offset: ?>> to memref<1x2x2xf32, strided<[?, ?, ?], offset: ?>>
      %subview_0 = memref.subview %1[%arg2, %arg4, %arg5] [1, 2, 2] [1, 1, 1] : memref<1x2x2xf32, strided<[?, ?, ?], offset: ?>> to memref<1x2x2xf32, strided<[?, ?, ?], offset: ?>>
      %subview_1 = memref.subview %alloc[%arg2, %arg3, %arg5] [1, 2, 2] [1, 1, 1] : memref<1x2x2xf32> to memref<1x2x2xf32, strided<[4, 2, 1], offset: ?>>
      linalg.batch_matmul ins(%subview, %subview_0 : memref<1x2x2xf32, strided<[?, ?, ?], offset: ?>>, memref<1x2x2xf32, strided<[?, ?, ?], offset: ?>>) outs(%subview_1 : memref<1x2x2xf32, strided<[4, 2, 1], offset: ?>>)
    }
    %2 = bufferization.to_tensor %alloc restrict : memref<1x2x2xf32> to tensor<1x2x2xf32>
    return %2 : tensor<1x2x2xf32>
  }
  func.func private @printMemrefF32(tensor<*xf32>)
  func.func @main() {
    %cst = arith.constant dense<1.000000e+00> : tensor<1x2x2xf32>
    %cst_0 = arith.constant dense<1.000000e+00> : tensor<1x2x2xf32>
    %0 = call @example(%cst, %cst_0) : (tensor<1x2x2xf32>, tensor<1x2x2xf32>) -> tensor<1x2x2xf32>
    %cast = tensor.cast %0 : tensor<1x2x2xf32> to tensor<*xf32>
    call @printMemrefF32(%cast) : (tensor<*xf32>) -> ()
    return
  }
}

```

# Testing
You may use the files in `tests` directory. To print the result of matrix multiplication, run the following command in `build` directory of our tool and `$MY_LLVM_SRC` set to `llvm-project` directory
```
./tool/suprasfirstpass-opt --tileSize 2 ../tests/matmul1.mlir | $MY_LLVM_SRC/build/bin/mlir-opt -pass-pipeline="builtin.module(one-shot-bufferize{bufferize-function-boundaries function-boundary-type-conversion=identity-layout-map}, convert-linalg-to-affine-loops, test-lower-to-llvm)" | $MY_LLVM_SRC/build/bin/mlir-runner -O3 -e main -entry-point-result=void -shared-libs=$MY_LLVM_SRC/build/lib/libmlir_runner_utils.so

Unranked Memref base@ = 0x55daf4c6a850 rank = 3 offset = 0 sizes = [1, 2, 2] strides = [4, 2, 1] data = 
[[[2,    2], 
  [2,    2]]]

```

# References
[1](https://github.com/j2kun/mlir-tutorial) For out of tree building
[2](https://github.com/llvm/llvm-project/tree/main/mlir) For APIs


