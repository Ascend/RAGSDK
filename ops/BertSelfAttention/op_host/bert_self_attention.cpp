/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
 * Description: BertSelfAttention融合算子的host侧tiling实现
 * Author: s00667951
 * Create: 2024-06-05
 */

#include <cmath>
#include "bert_self_attention_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/tiling_api.h"
#include "tiling/softmax/softmax_tiling.h"

template <typename U, typename V>
inline auto DivUp(U a, V b) -> decltype(a + b)
{
    return ((a + b - 1) / b);
}

using half = uint16_t;

namespace {
constexpr uint32_t UB_BLOCK_BYTE_SIZE = 32;
constexpr uint32_t BASESEQLENGTH = 16;
constexpr uint32_t USERSPACE_QKVPROJECTED_NUM = 6;
constexpr uint32_t TRAVSERSE_TYPE_THRESHOLD = 2;
constexpr uint32_t TENSOR_DIM_ONE = 0;
constexpr uint32_t TENSOR_DIM_TWO = 1;
constexpr uint32_t TENSOR_DIM_THR = 2;

template <typename T>
uint32_t BaseSeqHeadCompute(uint32_t &baseSeqLength, const uint32_t sequenceLength, uint64_t ub_size)
{
    std::vector<int64_t> softMaxShapeVec = { sequenceLength, sequenceLength };
    ge::Shape srcShape(softMaxShapeVec);
    uint32_t softMaxWorkSpaceSize = AscendC::GetSoftMaxMinTmpSize(srcShape, sizeof(T), false);
    uint32_t realUbsize = ub_size - softMaxWorkSpaceSize;
    uint32_t typeSize = sizeof(T);
    baseSeqLength = BASESEQLENGTH;
    uint32_t tmpBaseSeqLength;
    uint32_t usedSpace = baseSeqLength * sequenceLength * typeSize + sequenceLength * typeSize +
                         baseSeqLength * sequenceLength / 8;
    do {
        tmpBaseSeqLength = baseSeqLength + BASESEQLENGTH;
        if (tmpBaseSeqLength <= sequenceLength) {
            uint32_t tmpUsedSpace = tmpBaseSeqLength * sequenceLength * typeSize + sequenceLength * typeSize +
                                    tmpBaseSeqLength * sequenceLength / 8;
            if (tmpUsedSpace <= realUbsize) {
                usedSpace = tmpUsedSpace;
                baseSeqLength = tmpBaseSeqLength;
            }
        }
    } while (tmpBaseSeqLength == baseSeqLength);
    return usedSpace;
}
}  // namespace

namespace optiling {
template <typename T>
static ge::graphStatus TilingBasic(gert::TilingContext *context, BertSelfAttentionTilingData &tilingData)
{
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    const gert::StorageShape *input_shape = context->GetInputShape(0);
    auto attrs = context->GetAttrs();
    if (attrs == nullptr) {
        printf("attrs null\n");
        return ge::GRAPH_FAILED;
    }
    auto batchSize = input_shape->GetStorageShape().GetDim(0);
    auto sequenceLength = input_shape->GetStorageShape().GetDim(1);
    auto featuresSize = input_shape->GetStorageShape().GetDim(2);
    auto headNum = *(attrs->GetAttrPointer<int>(0));
    auto headSize = *(attrs->GetAttrPointer<int>(1));
    auto dropOutKeepProb = *(attrs->GetAttrPointer<float>(2));

    float sqrtHeadSize = sqrt(headSize);

    // features byte size must 32 byte mutiple
    uint32_t featuresByteSize = featuresSize * sizeof(T);
    if ((featuresByteSize % UB_BLOCK_BYTE_SIZE) != 0) {
        printf("featuresByteSize is not 32 byte mutiple\n");
        return ge::GRAPH_FAILED;
    }
    // part2 begin
    uint32_t baseSeqLength;
    uint64_t ub_size;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ub_size);
    uint32_t usedSpace = BaseSeqHeadCompute<T>(baseSeqLength, sequenceLength, ub_size);
    uint32_t softMaxWorkSpaceSize = ub_size - usedSpace;
    uint32_t tailSeqLength = sequenceLength % baseSeqLength;
    if (tailSeqLength == 0) {
        tailSeqLength = baseSeqLength;
    }

    std::vector<int64_t> softMaxShapeVec = { baseSeqLength, sequenceLength };
    ge::Shape srcShape(softMaxShapeVec);
    AscendC::SoftMaxTilingFunc(srcShape, sizeof(T), softMaxWorkSpaceSize, tilingData.softMaxTilingData);

    tilingData.set_baseSeqLength(baseSeqLength);
    tilingData.set_tailSeqLength(tailSeqLength);
    tilingData.set_softMaxWorkSpaceSize(softMaxWorkSpaceSize);
    // part2 end todo

    tilingData.set_batchSize(batchSize);
    tilingData.set_sequenceLength(sequenceLength);
    tilingData.set_featuresSize(featuresSize);
    tilingData.set_headNum(headNum);
    tilingData.set_headSize(headSize);

    tilingData.set_sqrtHeadSize(sqrtHeadSize);
    tilingData.set_dropOutKeepProb(dropOutKeepProb);

    return ge::GRAPH_SUCCESS;
}

template <typename T>
static ge::graphStatus TilingCore(gert::TilingContext *context, BertSelfAttentionTilingData &tilingData)
{
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    auto aicNum = ascendcPlatform.GetCoreNumAic();
    auto aivNum = ascendcPlatform.GetCoreNumAiv();
    auto B = tilingData.get_batchSize();
    auto N = tilingData.get_headNum();
    auto S = tilingData.get_sequenceLength();
    auto h = tilingData.get_headSize();

    tilingData.set_aicNum(aicNum);
    tilingData.set_aivNum(aivNum);
    auto usedAivCoreNum = (B * N >= aivNum) ? aivNum : B * N;
    tilingData.set_usedAivCoreNum(usedAivCoreNum);
    auto fomerBatchNum = B * N % aivNum;
    auto perBlockBatchFormer = DivUp(B * N, aivNum);
    auto perBlockBatchLatter = B * N / aivNum;
    tilingData.set_fomerBatchNum(fomerBatchNum);
    tilingData.set_perBlockBatchFormer(perBlockBatchFormer);
    tilingData.set_perBlockBatchLatter(perBlockBatchLatter);

    return ge::GRAPH_SUCCESS;
}

template <typename T>
static ge::graphStatus TilingCube_Projection(gert::TilingContext *context, BertSelfAttentionTilingData &tilingData)
{
    using namespace matmul_tiling;
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    auto aicNum = ascendcPlatform.GetCoreNumAic();
    auto aivNum = ascendcPlatform.GetCoreNumAiv();
    context->SetBlockDim(aicNum);

    MultiCoreMatmulTiling cubeTilingEachCore(ascendcPlatform);
    auto M = tilingData.get_sequenceLength() * tilingData.get_batchSize();
    auto N = tilingData.get_featuresSize();
    auto K = tilingData.get_featuresSize();
    cubeTilingEachCore.SetDim(aivNum);
    if (std::is_same<T, float>::value) {
        cubeTilingEachCore.SetAType(TPosition::GM, CubeFormat::ND, DataType::DT_FLOAT);
        cubeTilingEachCore.SetBType(TPosition::GM, CubeFormat::ND, DataType::DT_FLOAT);
        cubeTilingEachCore.SetCType(TPosition::GM, CubeFormat::ND, DataType::DT_FLOAT);
        cubeTilingEachCore.SetBiasType(TPosition::GM, CubeFormat::ND, DataType::DT_FLOAT);
    } else {
        cubeTilingEachCore.SetAType(TPosition::GM, CubeFormat::ND, DataType::DT_FLOAT16);
        cubeTilingEachCore.SetBType(TPosition::GM, CubeFormat::ND, DataType::DT_FLOAT16);
        cubeTilingEachCore.SetCType(TPosition::GM, CubeFormat::ND, DataType::DT_FLOAT16);
        cubeTilingEachCore.SetBiasType(TPosition::GM, CubeFormat::ND, DataType::DT_FLOAT16);
    }
    cubeTilingEachCore.SetShape(M, N, K);
    cubeTilingEachCore.SetOrgShape(M, N, K);
    cubeTilingEachCore.SetBufferSpace(-1, -1, -1);
    cubeTilingEachCore.SetBias(true);
    if (tilingData.get_batchSize() < TRAVSERSE_TYPE_THRESHOLD) {
        cubeTilingEachCore.SetTraverse(MatrixTraverse::FIRSTN);
    } else {
        cubeTilingEachCore.SetTraverse(MatrixTraverse::FIRSTM);
    }
    int retTrans = cubeTilingEachCore.GetTiling(tilingData.projectionTiling);
    if (retTrans == -1) {
        printf("cube tiling each core error\n");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

template <typename T>
static ge::graphStatus TilingCube(gert::TilingContext *context, BertSelfAttentionTilingData &tilingData)
{
    using namespace matmul_tiling;
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    auto M = tilingData.get_sequenceLength() * tilingData.get_batchSize();
    auto N = tilingData.get_featuresSize();
    auto K = tilingData.get_featuresSize();

    TilingCube_Projection<T>(context, tilingData);

    auto B = tilingData.get_batchSize();
    auto S = tilingData.get_sequenceLength();
    auto h = tilingData.get_headSize();
    MatmulApiTiling cubeTilingScores(ascendcPlatform);  // 为score部分做batch matmul的tiling
    if (std::is_same<T, float>::value) {
        cubeTilingScores.SetAType(TPosition::GM, CubeFormat::ND, DataType::DT_FLOAT);
        cubeTilingScores.SetBType(TPosition::GM, CubeFormat::ND, DataType::DT_FLOAT, true);
        cubeTilingScores.SetCType(TPosition::GM, CubeFormat::ND, DataType::DT_FLOAT);
    } else {
        cubeTilingScores.SetAType(TPosition::GM, CubeFormat::ND, DataType::DT_FLOAT16);
        cubeTilingScores.SetBType(TPosition::GM, CubeFormat::ND, DataType::DT_FLOAT16, true);
        cubeTilingScores.SetCType(TPosition::GM, CubeFormat::ND, DataType::DT_FLOAT16);
    }
    cubeTilingScores.SetShape(S, S, h);
    cubeTilingScores.SetOrgShape(S, S, h);
    cubeTilingScores.SetBufferSpace(-1, -1, -1);
    cubeTilingScores.SetBias(false);
    int ret = cubeTilingScores.GetTiling(tilingData.scoresCubeTiling);
    if (ret == -1) {
        printf("cube tiling each core error\n");
        return ge::GRAPH_FAILED;
    }

    M = tilingData.get_baseSeqLength();
    N = tilingData.get_headSize();
    K = tilingData.get_sequenceLength();
    MatmulApiTiling cubeTilingProbs(ascendcPlatform);
    if (std::is_same<T, float>::value) {
        cubeTilingProbs.SetAType(TPosition::GM, CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT);
        cubeTilingProbs.SetBType(TPosition::GM, CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT);
        cubeTilingProbs.SetCType(TPosition::VECIN, CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT);
    } else {
        cubeTilingProbs.SetAType(TPosition::GM, CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT16);
        cubeTilingProbs.SetBType(TPosition::GM, CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT16);
        cubeTilingProbs.SetCType(TPosition::VECIN, CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT16);
    }
    cubeTilingProbs.SetShape(M, N, K);
    cubeTilingProbs.SetOrgShape(M, N, K);
    cubeTilingProbs.SetBias(false);
    cubeTilingProbs.SetBufferSpace(-1, -1, -1);
    if (cubeTilingProbs.GetTiling(tilingData.probsCubeTiling) == -1) {
        return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingFuncFP32(gert::TilingContext *context, BertSelfAttentionTilingData &tiling)
{
    context->SetTilingKey(0);
    auto ret = TilingBasic<float>(context, tiling);
    if (ret != ge::GRAPH_SUCCESS) {
        printf("TilingBasic error\n");
        return ret;
    }

    ret = TilingCore<float>(context, tiling);
    if (ret != ge::GRAPH_SUCCESS) {
        printf("TilingCore error\n");
        return ret;
    }

    ret = TilingCube<float>(context, tiling);
    if (ret != ge::GRAPH_SUCCESS) {
        printf("TilingCube error\n");
        return ret;
    }

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingFuncFP16(gert::TilingContext *context, BertSelfAttentionTilingData &tiling)
{
    context->SetTilingKey(1);
    auto ret = TilingBasic<half>(context, tiling);
    if (ret != ge::GRAPH_SUCCESS) {
        printf("TilingBasic error\n");
        return ret;
    }

    ret = TilingCore<half>(context, tiling);
    if (ret != ge::GRAPH_SUCCESS) {
        printf("TilingCore error\n");
        return ret;
    }

    ret = TilingCube<half>(context, tiling);
    if (ret != ge::GRAPH_SUCCESS) {
        printf("TilingCube error\n");
        return ret;
    }

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingFunc(gert::TilingContext *context)
{
    if (context == nullptr) {
        printf("context can't be nullptr\n");
        return ge::GRAPH_FAILED;
    }
    auto aTensor = context->GetInputTensor(0);
    if (aTensor == nullptr) {
        printf("aTensor is nullptr\n");
        return ge::GRAPH_FAILED;
    }
    ge::DataType dtype = aTensor->GetDataType();
    size_t usrSize = 1;
    BertSelfAttentionTilingData tiling;
    if (dtype == ge::DT_FLOAT) {
        auto ret = TilingFuncFP32(context, tiling);
        if (ret != ge::GRAPH_SUCCESS) {
            printf("TilingFuncFP32 error\n");
            return ret;
        }
        usrSize *= sizeof(float);
    } else if (dtype == ge::DT_FLOAT16) {
        auto ret = TilingFuncFP16(context, tiling);
        if (ret != ge::GRAPH_SUCCESS) {
            printf("TilingFuncFP16 error\n");
            return ret;
        }
        usrSize *= sizeof(half);
    } else {
        printf("dtype error\n");
        return ge::GRAPH_FAILED;
    }

    if (context->GetRawTilingData() == nullptr) {
        printf("RawTilingData can't be nullptr\n");
        return ge::GRAPH_FAILED;
    }
    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());

    auto B = tiling.get_batchSize();
    auto S = tiling.get_sequenceLength();
    auto H = tiling.get_featuresSize();
    auto N = tiling.get_headNum();
    usrSize *= USERSPACE_QKVPROJECTED_NUM * B * S * H + B * N * S * S;  // 设置用户需要使用的workspace大小为QKV投影及其transpose结果存储空间。
    // 如需要使用系统workspace需要调用GetLibApiWorkSpaceSize获取系统workspace的大小。
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    uint32_t sysWorkspaceSize = ascendcPlatform.GetLibApiWorkSpaceSize();
    size_t *currentWorkspace = context->GetWorkspaceSizes(1);
    if (currentWorkspace == nullptr) {
        printf("work space null\n");
        return ge::GRAPH_FAILED;
    }
    currentWorkspace[0] = sysWorkspaceSize + usrSize;

    return ge::GRAPH_SUCCESS;
}
}  // namespace optiling

namespace ge {
static ge::graphStatus InferShape(gert::InferShapeContext *context)
{
    auto attrs = context->GetAttrs();
    if (attrs == nullptr) {
        printf("attrs null\n");
        return ge::GRAPH_FAILED;
    }
    auto headNum = *(attrs->GetAttrPointer<int>(0));
    auto headSize = *(attrs->GetAttrPointer<int>(1));

    const gert::Shape *x1_shape = context->GetInputShape(0);
    auto B = x1_shape->GetDim(0);
    auto S = x1_shape->GetDim(1);
    auto H = x1_shape->GetDim(2);
    if (headNum * headSize != H) {
        printf("headNum * headSize != featureSize\n");
        return ge::GRAPH_FAILED;
    }

    gert::Shape *outValue_shape = context->GetOutputShape(0);
    outValue_shape->SetDim(TENSOR_DIM_ONE, B);
    outValue_shape->SetDim(TENSOR_DIM_TWO, S);
    outValue_shape->SetDim(TENSOR_DIM_THR, H);

    return GRAPH_SUCCESS;
}
}  // namespace ge

namespace ops {
class BertSelfAttention : public OpDef {
public:
    explicit BertSelfAttention(const char *name) : OpDef(name)
    {
        this->Input("input")
            .ParamType(REQUIRED)
            .DataType({ ge::DT_FLOAT, ge::DT_FLOAT16 })
            .Format({ ge::FORMAT_ND, ge::FORMAT_ND })
            .UnknownShapeFormat({ ge::FORMAT_ND, ge::FORMAT_ND });
        this->Input("queryW")
            .ParamType(REQUIRED)
            .DataType({ ge::DT_FLOAT, ge::DT_FLOAT16 })
            .Format({ ge::FORMAT_ND, ge::FORMAT_ND })
            .UnknownShapeFormat({ ge::FORMAT_ND, ge::FORMAT_ND });
        this->Input("queryBias")
            .ParamType(REQUIRED)
            .DataType({ ge::DT_FLOAT, ge::DT_FLOAT16 })
            .Format({ ge::FORMAT_ND, ge::FORMAT_ND })
            .UnknownShapeFormat({ ge::FORMAT_ND, ge::FORMAT_ND });
        this->Input("keyW")
            .ParamType(REQUIRED)
            .DataType({ ge::DT_FLOAT, ge::DT_FLOAT16 })
            .Format({ ge::FORMAT_ND, ge::FORMAT_ND })
            .UnknownShapeFormat({ ge::FORMAT_ND, ge::FORMAT_ND });
        this->Input("keyBias")
            .ParamType(REQUIRED)
            .DataType({ ge::DT_FLOAT, ge::DT_FLOAT16 })
            .Format({ ge::FORMAT_ND, ge::FORMAT_ND })
            .UnknownShapeFormat({ ge::FORMAT_ND, ge::FORMAT_ND });
        this->Input("valueW")
            .ParamType(REQUIRED)
            .DataType({ ge::DT_FLOAT, ge::DT_FLOAT16 })
            .Format({ ge::FORMAT_ND, ge::FORMAT_ND })
            .UnknownShapeFormat({ ge::FORMAT_ND, ge::FORMAT_ND });
        this->Input("valueBias")
            .ParamType(REQUIRED)
            .DataType({ ge::DT_FLOAT, ge::DT_FLOAT16 })
            .Format({ ge::FORMAT_ND, ge::FORMAT_ND })
            .UnknownShapeFormat({ ge::FORMAT_ND, ge::FORMAT_ND });
        this->Input("attentionMask")
            .ParamType(REQUIRED)
            .DataType({ ge::DT_FLOAT, ge::DT_FLOAT16 })
            .Format({ ge::FORMAT_ND, ge::FORMAT_ND })
            .UnknownShapeFormat({ ge::FORMAT_ND, ge::FORMAT_ND });
        this->Input("dropOutMask")
            .ParamType(REQUIRED)
            .DataType({ ge::DT_UINT8, ge::DT_UINT8 })
            .Format({ ge::FORMAT_ND, ge::FORMAT_ND })
            .UnknownShapeFormat({ ge::FORMAT_ND, ge::FORMAT_ND });
        this->Input("headMask")
            .ParamType(REQUIRED)
            .DataType({ ge::DT_FLOAT, ge::DT_FLOAT16 })
            .Format({ ge::FORMAT_ND, ge::FORMAT_ND })
            .UnknownShapeFormat({ ge::FORMAT_ND, ge::FORMAT_ND });
        this->Output("output")
            .ParamType(REQUIRED)
            .DataType({ ge::DT_FLOAT, ge::DT_FLOAT16 })
            .Format({ ge::FORMAT_ND, ge::FORMAT_ND })
            .UnknownShapeFormat({ ge::FORMAT_ND, ge::FORMAT_ND });
        this->Attr("numAttentionHeads").Int();
        this->Attr("attentionHeadSize").Int();
        this->Attr("dropOutKeepProb").Float();

        this->SetInferShape(ge::InferShape);

        this->AICore().SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend910b");
    }
};

OP_ADD(BertSelfAttention);
}  // namespace ops
