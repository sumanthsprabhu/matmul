#include "lib/Passes.h"

#include "mlir/IR/PatternMatch.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tensor/Utils/Utils.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/IR/Builders.h"
#include "mlir/Dialect/Tosa/Utils/ConversionUtils.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"

namespace mlir {
#define GEN_PASS_DEF_SUPRASFIRSTPASS
#include "lib/Passes.h.inc"
}


using namespace mlir;

namespace {
  struct SuprasFirstPass
    : public impl::SuprasFirstPassBase<SuprasFirstPass> {
    SuprasFirstPass(int _tileSize) {
      tileSize = _tileSize;
    }
    
    void getDependentDialects(DialectRegistry &registry) const override {
      registry
        .insert<affine::AffineDialect, arith::ArithDialect, linalg::LinalgDialect, tensor::TensorDialect, bufferization::BufferizationDialect>();
    }

    void runOnOperation() override;
  };
  
  class TosaMatmulToAffineLoopConverter : public OpConversionPattern<tosa::MatMulOp> {
    public:
    int tileSize;
    using OpConversionPattern<tosa::MatMulOp>::OpConversionPattern;
    TosaMatmulToAffineLoopConverter(int _tileSize, MLIRContext *ctx) : TosaMatmulToAffineLoopConverter(ctx){
      tileSize = _tileSize;
    }
    
    
    LogicalResult matchAndRewrite(tosa::MatMulOp op, OpAdaptor adaptor, ConversionPatternRewriter &rewriter) const final {

      Location loc = op.getLoc();

      //create a memref to store the result
      auto outputType = cast<ShapedType>(op.getType());
      auto outputElementType = outputType.getElementType();            
      auto outputMemrefType = MemRefType::get(outputType.getShape(), outputType.getElementType());
      auto outputMemref = rewriter.create<memref::AllocOp>(loc,outputMemrefType);


      //Initialize it to zero
      Value zero = rewriter.create<arith::ConstantOp>(loc,
                                                      rewriter.getZeroAttr(outputElementType));
      auto outputDimSize = outputMemrefType.getShape().size();
      SmallVector<AffineMap> outputUpperBounds = {rewriter.getConstantAffineMap(outputMemrefType.getShape()[0]),
                                                  rewriter.getConstantAffineMap(outputMemrefType.getShape()[1]),
                                                  rewriter.getConstantAffineMap(outputMemrefType.getShape()[2])};

      
      auto outputInitLoop = rewriter.create<affine::AffineParallelOp>(loc,
                                                                      /*resultTypes=*/TypeRange(),
                                                                      /*reductions=*/llvm::ArrayRef<arith::AtomicRMWKind>(),
                                                                      llvm::ArrayRef(SmallVector<AffineMap>(outputDimSize, rewriter.getConstantAffineMap(0))),
                                                                      /*lbArgs*=*/ValueRange(),
                                                                      llvm::ArrayRef(outputUpperBounds),
                                                                      /*ubArgs*=*/ValueRange(),
                                                                      SmallVector<int64_t>(outputDimSize, 1));
      rewriter.setInsertionPointToStart(outputInitLoop.getBody());

      auto initOp = rewriter.create<affine::AffineStoreOp>(loc,
                                                           zero,
                                                           outputMemref,
                                                           outputInitLoop.getIVs());
                                                           
      rewriter.setInsertionPointAfter(outputInitLoop);         



      //Get memrefs of input matrices
      auto lhs = adaptor.getA();
      auto lhsType = cast<TensorType>(lhs.getType());
      auto lhsMemref = rewriter.create<bufferization::ToMemrefOp>(loc,
                                                                  bufferization::getMemRefTypeWithFullyDynamicLayout(lhsType),
                                                                  lhs);      
      auto rhs = adaptor.getB();
      auto rhsType = cast<TensorType>(rhs.getType());
      auto rhsMemref = rewriter.create<bufferization::ToMemrefOp>(loc,
                                                                  bufferization::getMemRefTypeWithFullyDynamicLayout(rhsType),
                                                                  rhs);

      
      // Assume input matrices are of size batchSize x m x k and bathchSize x k x n 
      auto batchSize = lhsType.getDimSize(0);
      auto m = lhsType.getDimSize(1);
      auto k = lhsType.getDimSize(2);
      auto n = rhsType.getDimSize(2);

      //Create a parallel loop to run matrix multiplication
      auto ub = SmallVector<AffineMap>{rewriter.getConstantAffineMap(batchSize),
                                       rewriter.getConstantAffineMap(m),
                                       rewriter.getConstantAffineMap(k),
                                       rewriter.getConstantAffineMap(n)};
      auto lb = SmallVector<AffineMap>(ub.size(), rewriter.getConstantAffineMap(0));
      SmallVector<int64_t> steps = {/*tile size for batch*/1, tileSize, tileSize, tileSize};

      auto parallelOp = rewriter.create<affine::AffineParallelOp>(loc,
                                                                  /*resultTypes=*/TypeRange(),
                                                                  /*reductions=*/llvm::ArrayRef<arith::AtomicRMWKind>(),
                                                                  llvm::ArrayRef(lb), /*lbArgs*=*/ValueRange(),
                                                                  llvm::ArrayRef(ub), /*ubArgs*=*/ValueRange(),
                                                                  steps);
      rewriter.setInsertionPointToStart(parallelOp.getBody());

      
      //Inside loop add matrix multiplication; first get subviews of input and output matrices
      auto batchIdx = parallelOp.getIVs()[0];
      auto mIdx = parallelOp.getIVs()[1];
      auto kIdx = parallelOp.getIVs()[2];
      auto nIdx = parallelOp.getIVs()[3];
      auto tileSizeValue = OpFoldResult(rewriter.getIndexAttr(tileSize));
      auto oneValue = OpFoldResult(rewriter.getIndexAttr(1));

      auto lhsMemrefSubview = rewriter.create<memref::SubViewOp>(loc,
                                                                 lhsMemref,
                                                                 ArrayRef<OpFoldResult>{batchIdx, mIdx, kIdx},
                                                                 ArrayRef<OpFoldResult>{oneValue, tileSizeValue, tileSizeValue},
                                                                 ArrayRef<OpFoldResult>{oneValue, oneValue, oneValue});
      auto rhsMemrefSubview = rewriter.create<memref::SubViewOp>(loc,
                                                                 rhsMemref,
                                                                 ArrayRef<OpFoldResult>{batchIdx, kIdx, nIdx},
                                                                 ArrayRef<OpFoldResult>{oneValue, tileSizeValue, tileSizeValue},
                                                                 ArrayRef<OpFoldResult>{oneValue, oneValue, oneValue});
      auto outputMemrefSubview = rewriter.create<memref::SubViewOp>(loc,
                                                                    outputMemref,
                                                                    ArrayRef<OpFoldResult>{batchIdx, mIdx, nIdx},
                                                                    ArrayRef<OpFoldResult>{oneValue, tileSizeValue, tileSizeValue},
                                                                    ArrayRef<OpFoldResult>{oneValue, oneValue, oneValue});

      auto batchMatmulOp = rewriter.create<linalg::BatchMatmulOp>(loc,
                                                                  ValueRange{lhsMemrefSubview, rhsMemrefSubview},
                                                                  ValueRange{outputMemrefSubview});
            

      rewriter.setInsertionPointAfter(parallelOp);
      

      //convert the outputmeref to tensor so that whoever is using this operation will typecheck
      auto outputTensor = rewriter.create<bufferization::ToTensorOp>(loc, outputMemref, /*restrict=*/true);
                                                                     

      rewriter.replaceOp(op, outputTensor);

      return success();

    }

    
  };

}


void SuprasFirstPass::runOnOperation() {
  
  getOperation()->walk([&](func::FuncOp func) {
                         ConversionTarget target(getContext());
                         target.addLegalDialect<affine::AffineDialect, linalg::LinalgDialect,
                                                tosa::TosaDialect, tensor::TensorDialect, bufferization::BufferizationDialect>();
                         target.addIllegalOp<tosa::MatMulOp>();
                         target.markUnknownOpDynamicallyLegal([](Operation *) { return true; });
                         
                         RewritePatternSet patterns(&getContext());
                         patterns.add<TosaMatmulToAffineLoopConverter>(tileSize, patterns.getContext());
                         (void)applyFullConversion(func, target, std::move(patterns));
                       });
}

std::unique_ptr<Pass> mlir::createSuprasFirstPass(int tileSize = 32) {
  return std::make_unique<SuprasFirstPass>(tileSize);
}

