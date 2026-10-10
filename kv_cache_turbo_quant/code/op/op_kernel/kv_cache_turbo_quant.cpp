#include "kernel_operator.h"
#include "adv_api/matmul_intf.h"
#include "kv_cache_turbo_quant_tiling.h"

using namespace AscendC;

namespace {
// head_dim is runtime-parameterized (128 or 256, from tiling). Each AIV always
// processes HALF_ELEMS fp32 values per tile-half, so rows-per-tile shrink as the
// dim grows (128 -> 64 rows/AIV, 256 -> 32 rows/AIV) and every UB/GM footprint
// below is dimension-invariant.
constexpr uint32_t HALF_ELEMS = 8192;            // fp32 elems per AIV per tile
constexpr uint32_t TILE_ELEMS = 2 * HALF_ELEMS;  // fp32 elems per tile
constexpr uint32_t MAX_HALF_ROWS = 64;           // rows per AIV at head_dim=128
constexpr uint32_t MAX_LEVELS = 16;  // 2/3/4-bit -> 4/8/16 levels
constexpr float MIN_NORM = 1e-30f;

constexpr uint32_t WS_BUF_BYTES = TILE_ELEMS * sizeof(float);             // 64KB per buf
constexpr uint32_t WS_PAIR_STRIDE = 8 * WS_BUF_BYTES;                     // U0 U1 Y0 Y1 R0 R1 P0 P1

// Lloyd-Max centroids for standard normal, identical to the golden reference.
constexpr float CENTROIDS_2BIT[4] = {
    -0.1335033178f, -0.04002048075f, 0.04002048075f, 0.1335033178f,
};
constexpr float CENTROIDS_3BIT[8] = {
    -0.19020693f, -0.1187859178f, -0.06682205945f, -0.02166347019f,
    0.02166347019f, 0.06682205945f, 0.1187859178f, 0.19020693f,
};
constexpr float CENTROIDS_4BIT[16] = {
    -0.2414890379f, -0.1828317791f, -0.1429702938f, -0.1109927073f,
    -0.08325428516f, -0.05802082643f, -0.03428063914f, -0.01134236995f,
    0.01134236995f, 0.03428063914f, 0.05802082643f, 0.08325428516f,
    0.1109927073f, 0.1429702938f, 0.1828317791f, 0.2414890379f,
};

using MmF32 = Matmul<MatmulType<TPosition::GM, CubeFormat::ND, float>,
                     MatmulType<TPosition::GM, CubeFormat::ND, float, true>,
                     MatmulType<TPosition::GM, CubeFormat::ND, float>>;
} // namespace

// ------------------------- AIV (vector) side -------------------------
class KvTqVec {
public:
    __aicore__ inline KvTqVec() {}

    __aicore__ inline void Init(TPipe *pipe, MmF32 *mm1, MmF32 *mm2, GM_ADDR kvVectors, GM_ADDR rot,
        GM_ADDR qjl, GM_ADDR quantIdx, GM_ADDR quantQjl, GM_ADDR quantNorm, GM_ADDR quantGamma,
        GM_ADDR workspace, const KvCacheTurboQuantTilingData *td)
    {
        pipe_ = pipe;
        mm1_ = mm1;
        mm2_ = mm2;
        td_ = td;
        totalRows_ = td->totalRows;
        headDim_ = td->headDim;
        halfRows_ = HALF_ELEMS / headDim_;
        tileRows_ = 2 * halfRows_;
        qjlDim_ = td->qjlDim;
        mseBits_ = td->mseBits;
        levels_ = 1U << mseBits_;
        idxBytesPerRow_ = td->idxBytesPerRow;
        qjlBytesPerRow_ = td->qjlBytesPerRow;
        usedPairs_ = td->usedPairs;
        numTiles_ = td->numTiles;

        const uint32_t aivIdx = GetBlockIdx();
        pairId_ = aivIdx >> 1;
        subId_ = aivIdx & 1U;

        xGm_.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(kvVectors),
            static_cast<uint64_t>(totalRows_) * headDim_);
        idxGm_.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t *>(quantIdx),
            static_cast<uint64_t>(totalRows_) * idxBytesPerRow_);
        qjlOutGm_.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t *>(quantQjl),
            static_cast<uint64_t>(totalRows_) * qjlBytesPerRow_);
        normGm_.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(quantNorm), totalRows_);
        gammaGm_.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(quantGamma), totalRows_);
        rotGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(rot),
            static_cast<uint64_t>(headDim_) * headDim_);
        qjlGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(qjl),
            static_cast<uint64_t>(td->qjlDim) * headDim_);

        // All user regions must be based on GetUserWorkspace(): the first
        // RESERVED_WORKSPACE (16MB on dav_c220) of the op workspace belongs to the
        // runtime/libApi (FFTS+ dispatch, KFC system messages via GetSysWorkSpacePtr()).
        __gm__ uint8_t *usrWs = reinterpret_cast<__gm__ uint8_t *>(GetUserWorkspace(workspace));
        __gm__ uint8_t *base = usrWs + td->wsPairOffset + static_cast<uint64_t>(pairId_) * WS_PAIR_STRIDE;
        for (uint32_t db = 0; db < 2; ++db) {
            uGm_[db].SetGlobalBuffer(reinterpret_cast<__gm__ float *>(base + (0 + db) * WS_BUF_BYTES), TILE_ELEMS);
            yGm_[db].SetGlobalBuffer(reinterpret_cast<__gm__ float *>(base + (2 + db) * WS_BUF_BYTES), TILE_ELEMS);
            rGm_[db].SetGlobalBuffer(reinterpret_cast<__gm__ float *>(base + (4 + db) * WS_BUF_BYTES), TILE_ELEMS);
            pGm_[db].SetGlobalBuffer(reinterpret_cast<__gm__ float *>(base + (6 + db) * WS_BUF_BYTES),
                static_cast<uint64_t>(tileRows_) * td->qjlDim);
        }

        pipe_->InitBuffer(bufA_, HALF_ELEMS * sizeof(float));
        pipe_->InitBuffer(bufB_, HALF_ELEMS * sizeof(float));
        pipe_->InitBuffer(bufC_, HALF_ELEMS * sizeof(float));
        pipe_->InitBuffer(bufD_, HALF_ELEMS * sizeof(float));
        pipe_->InitBuffer(bufE_, HALF_ELEMS * sizeof(float));
        pipe_->InitBuffer(bufX_, HALF_ELEMS * sizeof(bfloat16_t));
        pipe_->InitBuffer(bufOffG_, 1024 * sizeof(uint32_t));
        pipe_->InitBuffer(bufOffM2_, 512 * sizeof(uint32_t));
        pipe_->InitBuffer(bufOffQ2_, 256 * sizeof(uint32_t));
        pipe_->InitBuffer(bufOffEO_, 256 * sizeof(uint32_t));
        pipe_->InitBuffer(bufPart_, 256 * sizeof(float));
        pipe_->InitBuffer(bufNorm_, MAX_HALF_ROWS * sizeof(float));
        pipe_->InitBuffer(bufRes_, MAX_HALF_ROWS * sizeof(float));
        pipe_->InitBuffer(bufNC_, MAX_HALF_ROWS * sizeof(float));
        pipe_->InitBuffer(bufGamma_, MAX_HALF_ROWS * sizeof(float));
        pipe_->InitBuffer(bufNormBf_, MAX_HALF_ROWS * sizeof(bfloat16_t));
        pipe_->InitBuffer(bufGammaBf_, MAX_HALF_ROWS * sizeof(bfloat16_t));
        pipe_->InitBuffer(bufCent_, MAX_LEVELS * sizeof(float));
        pipe_->InitBuffer(bufBound_, MAX_LEVELS * sizeof(float));
        pipe_->InitBuffer(bufMask_, HALF_ELEMS / 8);


        myTiles_ = (pairId_ < numTiles_) ? (numTiles_ - pairId_ + usedPairs_ - 1) / usedPairs_ : 0;
        if (myTiles_ == 0) {
            return;
        }
        InitTables();
    }

    __aicore__ inline void Process()
    {
        if (myTiles_ == 0) {
            return;
        }
        for (uint32_t i = 0; i < myTiles_; ++i) {
            const uint32_t db = i & 1U;
            Stage1(TileIdx(i), db);
            // fence: U write (MTE3) must be visible before the kfc client request
            SetFlag<HardEvent::MTE3_S>(7);
            WaitFlag<HardEvent::MTE3_S>(7);
            mm1_->SetTensorA(uGm_[db][subId_ * HALF_ELEMS], false);
            mm1_->SetTensorB(rotGm_, true);
            mm1_->IterateAll(yGm_[db][subId_ * HALF_ELEMS]);
            Stage3(TileIdx(i), db);
            SetFlag<HardEvent::MTE3_S>(7);
            WaitFlag<HardEvent::MTE3_S>(7);
            mm2_->SetTensorA(rGm_[db][subId_ * HALF_ELEMS], false);
            mm2_->SetTensorB(qjlGm_, true);
            mm2_->IterateAll(pGm_[db][subId_ * HALF_ELEMS]);
            Stage5(TileIdx(i), db);
        }
        mm1_->End();
        mm2_->End();
    }

private:
    __aicore__ inline uint32_t TileIdx(uint32_t i) const { return pairId_ + i * usedPairs_; }

    // The count-form Or on uint32/int32 silently processes only HALF the elements on
    // CANN 9.1 dav_c220 (OrImpl Level-2 reinterprets to int16 but keeps the element
    // count), and Level-0 Or only supports 16-bit dtypes. Bitwise-or is width-agnostic,
    // so run Level-0 Or on int16 views with doubled element count.
    __aicore__ inline void OrU32(const LocalTensor<uint32_t> &dst, const LocalTensor<uint32_t> &src0,
        const LocalTensor<uint32_t> &src1, uint32_t count)
    {
        Or(dst.ReinterpretCast<int16_t>(), src0.ReinterpretCast<int16_t>(), src1.ReinterpretCast<int16_t>(),
            static_cast<uint64_t>(128), static_cast<uint8_t>(count / 64), BinaryRepeatParams{1, 1, 1, 8, 8, 8});
    }

    __aicore__ inline uint32_t ValidRows(uint32_t tileIdx) const
    {
        const uint32_t rowBase = tileIdx * tileRows_ + subId_ * halfRows_;
        if (rowBase >= totalRows_) {
            return 0;
        }
        const uint32_t left = totalRows_ - rowBase;
        return left < halfRows_ ? left : halfRows_;
    }

    __aicore__ inline void InitTables()
    {
        auto cent = bufCent_.Get<float>();
        auto bound = bufBound_.Get<float>();
        const float *cs = mseBits_ == 2 ? CENTROIDS_2BIT : (mseBits_ == 3 ? CENTROIDS_3BIT : CENTROIDS_4BIT);
        for (uint32_t k = 0; k < levels_; ++k) {
            cent.SetValue(k, cs[k]);
        }
        for (uint32_t k = 0; k + 1 < levels_; ++k) {
            bound.SetValue(k, (cs[k] + cs[k + 1]) * 0.5f);
        }
        // NOTE (verified on CANN 9.1 dav_c220): vgather offsets and srcBaseOffset are BYTE-granular,
        // so every offset table below stores byte offsets (element offset * sizeof(uint32_t)).
        // level-1 deinterleave offsets: g_j[k, m] = src[k * headDim + 8 * m + j] (j via srcBaseOffset)
        // 1024 entries for both dims: halfRows * (headDim/8) = 64*16 = 32*32.
        auto offG = bufOffG_.Get<uint32_t>();
        const uint32_t mPerRow = headDim_ / 8;
        for (uint32_t i = 0; i < 1024; ++i) {
            offG.SetValue(i, ((i / mPerRow) * headDim_ + (i % mPerRow) * 8) * sizeof(uint32_t));
        }
        // mse2 level-2 offsets: z[k, m] from w[k * (headDim/8) + 2 * m + t]
        // 512 entries for both dims: halfRows * (headDim/16) = 64*8 = 32*16.
        auto offM2 = bufOffM2_.Get<uint32_t>();
        const uint32_t wPerRow = headDim_ / 8;
        const uint32_t m2 = headDim_ / 16;
        for (uint32_t i = 0; i < 512; ++i) {
            offM2.SetValue(i, ((i / m2) * wPerRow + (i % m2) * 2) * sizeof(uint32_t));
        }
        // qjl level-2 offsets: z[k, m] from b[k * (qjlDim/8) + 4 * m + t]
        const uint32_t w1 = qjlDim_ / 8;
        auto offQ2 = bufOffQ2_.Get<uint32_t>();
        const uint32_t q2 = qjlDim_ / 32;
        for (uint32_t i = 0; i < halfRows_ * q2; ++i) {
            offQ2.SetValue(i, ((i / q2) * w1 + (i % q2) * 4) * sizeof(uint32_t));
        }
        // even / odd offsets for partial-sum pairwise combine
        auto offEO = bufOffEO_.Get<uint32_t>();
        for (uint32_t i = 0; i < halfRows_; ++i) {
            offEO.SetValue(i, 2 * i * sizeof(uint32_t));
            offEO.SetValue(halfRows_ + i, (2 * i + 1) * sizeof(uint32_t));
        }
        // stride-4 offsets at [128, 128+4*halfRows) for the head_dim=256 row-sum tree
        // (4 partials per row: gather lanes 4i+p, p = 0..3)
        for (uint32_t p = 0; p < 4; ++p) {
            for (uint32_t i = 0; i < halfRows_; ++i) {
                offEO.SetValue(128 + p * halfRows_ + i, (4 * i + p) * sizeof(uint32_t));
            }
        }
        SetFlag<HardEvent::S_V>(0);
        WaitFlag<HardEvent::S_V>(0);
    }

    // per-row sum of src[halfRows,headDim] fp32 with the same reduction structure as
    // ReduceSum per row: tree within each 64-elem repeat, then pairwise over the
    // headDim/64 partials ((p0+p1) for 128 dims, (p0+p1)+(p2+p3) for 256 dims).
    __aicore__ inline void RowSumsF32(const LocalTensor<float> &dst64, const LocalTensor<float> &src)
    {
        auto part = bufPart_.Get<float>();
        const uint32_t nPartial = headDim_ / 64;  // 64-elem repeats per row: 2 or 4
        ReduceRepeat<ReduceType::SUM, float, float>(part, src, 64, nPartial * halfRows_, 1, 1, 8);
        auto offEO = bufOffEO_.Get<uint32_t>();
        if (nPartial == 2) {
            auto ev = part[halfRows_ * 2];
            auto od = part[halfRows_ * 3];
            Gather(ev, part, offEO, 0, static_cast<uint64_t>(halfRows_), 1, 8);
            Gather(od, part, offEO[halfRows_], 0, static_cast<uint64_t>(halfRows_), 1, 8);
            Add(dst64, ev, od, halfRows_);
        } else {
            auto a = part[halfRows_ * 4];
            auto b = part[halfRows_ * 5];
            auto t = part[halfRows_ * 6];
            Gather(a, part, offEO[128], 0, static_cast<uint64_t>(halfRows_), 1, 8);
            Gather(t, part, offEO[128 + halfRows_], 0, static_cast<uint64_t>(halfRows_), 1, 8);
            Add(a, a, t, halfRows_);
            Gather(b, part, offEO[128 + 2 * halfRows_], 0, static_cast<uint64_t>(halfRows_), 1, 8);
            Gather(t, part, offEO[128 + 3 * halfRows_], 0, static_cast<uint64_t>(halfRows_), 1, 8);
            Add(b, b, t, halfRows_);
            Add(dst64, a, b, halfRows_);
        }
    }

    // broadcast vals[halfRows] into D[halfRows,headDim] fp32 (each row filled with vals[r])
    __aicore__ inline void BroadcastRows(const LocalTensor<float> &dst, const LocalTensor<float> &vals,
        const LocalTensor<float> &bcTmp)
    {
        Brcb(bcTmp, vals, halfRows_ / 8, BrcbRepeatParams{1, 8});
        for (uint32_t j = 0; j < headDim_ / 8; ++j) {
            Copy(dst[j * 8], bcTmp, static_cast<uint64_t>(8), halfRows_,
                CopyRepeatParams{0, 0, static_cast<uint16_t>(headDim_ / 8), 1});
        }
    }

    // stage 1: x -> norm out, u -> GM ws, signal U ready
    __aicore__ inline void Stage1(uint32_t tileIdx, uint32_t db)
    {
        const uint32_t valid = ValidRows(tileIdx);
        const uint32_t rowBase = tileIdx * tileRows_ + subId_ * halfRows_;
        auto xBf = bufX_.Get<bfloat16_t>();
        auto xf = bufA_.Get<float>();
        auto x2 = bufB_.Get<float>();
        auto u = bufC_.Get<float>();
        auto norms = bufNorm_.Get<float>();

        if (valid > 0) {
            DataCopy(xBf, xGm_[static_cast<uint64_t>(rowBase) * headDim_], valid * headDim_);
        }
        if (valid < halfRows_) {
            Duplicate(xBf[static_cast<uint64_t>(valid) * headDim_], static_cast<bfloat16_t>(0.0f),
                (halfRows_ - valid) * headDim_);
        }
        SetFlag<HardEvent::MTE2_V>(0);
        WaitFlag<HardEvent::MTE2_V>(0);

        Cast(xf, xBf, RoundMode::CAST_NONE, HALF_ELEMS);
        Mul(x2, xf, xf, HALF_ELEMS);
        RowSumsF32(norms, x2);

        // TEMPORARY debug dump: stage1 intermediates for tile 0 / sub 0

        Sqrt(norms, norms, halfRows_);

        // quant_norm output (bf16 of unclamped norm)
        auto normBf = bufNormBf_.Get<bfloat16_t>();
        Cast(normBf, norms, RoundMode::CAST_RINT, halfRows_);
        SetFlag<HardEvent::V_MTE3>(2);
        WaitFlag<HardEvent::V_MTE3>(2);
        if (valid > 0) {
            DataCopyPad(normGm_[rowBase], normBf,
                DataCopyExtParams{1, static_cast<uint32_t>(valid * sizeof(bfloat16_t)), 0, 0, 0});
        }
        SetFlag<HardEvent::MTE3_V>(2);
        WaitFlag<HardEvent::MTE3_V>(2);

        auto nc = bufNC_.Get<float>();
        Maxs(nc, norms, MIN_NORM, halfRows_);
        BroadcastRows(bufD_.Get<float>(), nc, bufB_.Get<float>());
        Div(u, xf, bufD_.Get<float>(), HALF_ELEMS);

        SetFlag<HardEvent::V_MTE3>(0);
        WaitFlag<HardEvent::V_MTE3>(0);
        DataCopy(uGm_[db][subId_ * HALF_ELEMS], u, HALF_ELEMS);
    }

    // stage 3: y -> idx out, gamma out, rn -> GM ws, signal R ready
    __aicore__ inline void Stage3(uint32_t tileIdx, uint32_t db)
    {
        const uint32_t valid = ValidRows(tileIdx);
        const uint32_t rowBase = tileIdx * tileRows_ + subId_ * halfRows_;

        auto y = bufA_.Get<float>();
        DataCopy(y, yGm_[db][subId_ * HALF_ELEMS], HALF_ELEMS);
        SetFlag<HardEvent::MTE2_V>(1);
        WaitFlag<HardEvent::MTE2_V>(1);

        // quantization levels: idx = sum_k (y > bound_k), identical to golden
        auto idxF = bufE_.Get<float>();
        auto tmp = bufB_.Get<float>();
        auto mask = bufMask_.Get<uint8_t>();
        auto bound = bufBound_.Get<float>();
        Duplicate(idxF, 0.0f, HALF_ELEMS);
        for (uint32_t k = 0; k + 1 < levels_; ++k) {
            Compares(mask, y, bound.GetValue(k), CMPMODE::GT, HALF_ELEMS);
            Adds(tmp, idxF, 1.0f, HALF_ELEMS);
            Select(idxF, mask, tmp, idxF, SELMODE::VSEL_TENSOR_TENSOR_MODE, HALF_ELEMS);
        }
        auto idxI = bufC_.Get<int32_t>();
        Cast(idxI, idxF, RoundMode::CAST_RINT, HALF_ELEMS);

        // yhat = centroids[idx] via gather (vgather offsets are byte-granular: scale idx by 4)
        auto cent = bufCent_.Get<float>();
        auto offC = bufD_.Get<uint32_t>();
        ShiftLeft(offC, idxI.ReinterpretCast<uint32_t>(), static_cast<uint32_t>(2), HALF_ELEMS);
        Gather(idxF, cent, offC, 0, static_cast<uint64_t>(64),
            HALF_ELEMS / 64, 8);

        auto r = bufB_.Get<float>();
        Sub(r, y, idxF, HALF_ELEMS);
        auto r2 = bufE_.Get<float>();
        Mul(r2, r, r, HALF_ELEMS);
        auto resNorms = bufRes_.Get<float>();
        RowSumsF32(resNorms, r2);
        Sqrt(resNorms, resNorms, halfRows_);

        // gamma = norm * residual_norm (bf16)
        auto gammaF = bufGamma_.Get<float>();
        Mul(gammaF, bufNorm_.Get<float>(), resNorms, halfRows_);
        auto gammaBf = bufGammaBf_.Get<bfloat16_t>();
        Cast(gammaBf, gammaF, RoundMode::CAST_RINT, halfRows_);
        SetFlag<HardEvent::V_MTE3>(3);
        WaitFlag<HardEvent::V_MTE3>(3);
        if (valid > 0) {
            DataCopyPad(gammaGm_[rowBase], gammaBf,
                DataCopyExtParams{1, static_cast<uint32_t>(valid * sizeof(bfloat16_t)), 0, 0, 0});
        }
        SetFlag<HardEvent::MTE3_V>(3);
        WaitFlag<HardEvent::MTE3_V>(3);

        // residual unit vector -> R ws
        auto nc = bufNC_.Get<float>();
        Maxs(nc, resNorms, MIN_NORM, halfRows_);
        BroadcastRows(bufD_.Get<float>(), nc, bufA_.Get<float>());
        Div(r2, r, bufD_.Get<float>(), HALF_ELEMS);
        SetFlag<HardEvent::V_MTE3>(1);
        WaitFlag<HardEvent::V_MTE3>(1);
        DataCopy(rGm_[db][subId_ * HALF_ELEMS], r2, HALF_ELEMS);

        PackIdx(rowBase, valid);
    }

    // bit-pack idxI[halfRows,headDim] (int32 levels) into idxBytesPerRow bytes per row
    __aicore__ inline void PackIdx(uint32_t rowBase, uint32_t valid)
    {
        auto idxU = bufC_.Get<uint32_t>();
        auto offG = bufOffG_.Get<uint32_t>();
        auto g = bufA_.Get<uint32_t>();   // 8 x [64,16] uint32
        // level-1 deinterleave: g_j[k, m] = idx[k * 128 + 8 * m + j]
        for (uint32_t j = 0; j < 8; ++j) {
            Gather(g[j * 1024], idxU, offG, j * sizeof(uint32_t), static_cast<uint64_t>(64), 16, 8);
        }
        // combine into packed words
        auto w = bufC_.Get<uint32_t>();
        auto t = bufC_.Get<uint32_t>()[1024];
        const uint32_t bits = mseBits_;
        ShiftLeft(t, g[1024], bits, 1024);
        OrU32(w, g, t, 1024);
        for (uint32_t j = 2; j < 8; ++j) {
            ShiftLeft(t, g[j * 1024], j * bits, 1024);
            OrU32(w, w, t, 1024);
        }

        if (valid == 0) {
            return;
        }
        const uint64_t dstBase = static_cast<uint64_t>(rowBase) * idxBytesPerRow_;
        if (mseBits_ == 4) {
            // w[64,16] uint32 is already the packed 64B per row
            SetFlag<HardEvent::V_MTE3>(4);
            WaitFlag<HardEvent::V_MTE3>(4);
            DataCopyPad(idxGm_[dstBase], bufC_.Get<uint8_t>(),
                DataCopyExtParams{1, static_cast<uint32_t>(valid * idxBytesPerRow_), 0, 0, 0});
            SetFlag<HardEvent::MTE3_V>(4);
            WaitFlag<HardEvent::MTE3_V>(4);
        } else if (mseBits_ == 2) {
            // level-2: z[k, m] = w[16k+2m] | w[16k+2m+1] << 16 -> [64,8] uint32 = 32B/row
            auto offM2 = bufOffM2_.Get<uint32_t>();
            auto g2 = bufA_.Get<uint32_t>();
            Gather(g2, w, offM2, 0, static_cast<uint64_t>(64), 8, 8);
            Gather(g2[512], w, offM2, sizeof(uint32_t), static_cast<uint64_t>(64), 8, 8);
            ShiftLeft(g2[1024], g2[512], static_cast<uint32_t>(16), 512);
            auto z = bufB_.Get<uint32_t>();
            OrU32(z, g2, g2[1024], 512);
            SetFlag<HardEvent::V_MTE3>(4);
            WaitFlag<HardEvent::V_MTE3>(4);
            DataCopyPad(idxGm_[dstBase], bufB_.Get<uint8_t>(),
                DataCopyExtParams{1, static_cast<uint32_t>(valid * idxBytesPerRow_), 0, 0, 0});
            SetFlag<HardEvent::MTE3_V>(4);
            WaitFlag<HardEvent::MTE3_V>(4);
        } else {
            // mse3: 24-bit words, scalar compaction to 48B per row
            SetFlag<HardEvent::V_S>(4);
            WaitFlag<HardEvent::V_S>(4);
            auto stage = bufX_.Get<uint32_t>();
            auto wi = bufC_.Get<uint32_t>();
            const uint32_t wRow = headDim_ / 8;          // packed 24-bit words per row (16 / 32)
            const uint32_t outWords = idxBytesPerRow_ / 4;  // 12 / 24
            for (uint32_t rr = 0; rr < halfRows_; ++rr) {
                for (uint32_t j = 0; j < outWords; ++j) {
                    const uint32_t grp = (4 * j) / 3;
                    const uint32_t rem = (4 * j) % 3;
                    const uint32_t w0 = wi.GetValue(rr * wRow + grp);
                    const uint32_t w1v = wi.GetValue(rr * wRow + grp + 1);
                    uint32_t z;
                    if (rem == 0) {
                        z = w0 | ((w1v & 0xffU) << 24);
                    } else if (rem == 1) {
                        z = (w0 >> 8) | ((w1v & 0xffffU) << 16);
                    } else {
                        z = (w0 >> 16) | (w1v << 8);
                    }
                    stage.SetValue(rr * outWords + j, z);
                }
            }
            SetFlag<HardEvent::S_MTE3>(5);
            WaitFlag<HardEvent::S_MTE3>(5);
            DataCopyPad(idxGm_[dstBase], bufX_.Get<uint8_t>(),
                DataCopyExtParams{1, static_cast<uint32_t>(valid * idxBytesPerRow_), 0, 0, 0});
            SetFlag<HardEvent::MTE3_S>(5);
            WaitFlag<HardEvent::MTE3_S>(5);
        }
    }

    // stage 5: p -> sign bits -> qjl out
    __aicore__ inline void Stage5(uint32_t tileIdx, uint32_t db)
    {
        const uint32_t valid = ValidRows(tileIdx);
        const uint32_t rowBase = tileIdx * tileRows_ + subId_ * halfRows_;

        const uint32_t elems = halfRows_ * qjlDim_;
        auto p = bufA_.Get<float>();
        DataCopy(p, pGm_[db][subId_ * elems], elems);
        SetFlag<HardEvent::MTE2_V>(2);
        WaitFlag<HardEvent::MTE2_V>(2);

        // s = (p >= 0) ? 1 : 0  (matches golden ">= 0", incl. -0.0 -> 1)
        auto zeros = bufE_.Get<float>();
        auto ones = bufD_.Get<float>();
        auto sF = bufB_.Get<float>();
        auto mask = bufMask_.Get<uint8_t>();
        Duplicate(zeros, 0.0f, elems);
        Duplicate(ones, 1.0f, elems);
        Compares(mask, p, 0.0f, CMPMODE::LT, elems);
        Select(sF, mask, zeros, ones, SELMODE::VSEL_TENSOR_TENSOR_MODE, elems);
        auto sI = bufC_.Get<uint32_t>();
        Cast(sI.ReinterpretCast<int32_t>(), sF, RoundMode::CAST_RINT, elems);

        // level-1: g_j[k, m] = s[k * qjlDim + 8 * m + j], byte words b = sum_j g_j << j
        auto offG = bufOffG_.Get<uint32_t>();
        auto g = bufA_.Get<uint32_t>();
        const uint32_t w1 = qjlDim_ / 8;
        const uint32_t gCount = halfRows_ * w1;
        for (uint32_t j = 0; j < 8; ++j) {
            Gather(g[j * gCount], sI, offG, j * sizeof(uint32_t), static_cast<uint64_t>(64),
                static_cast<uint8_t>(16), 8);
        }
        auto b = bufC_.Get<uint32_t>();
        auto t = bufC_.Get<uint32_t>()[gCount];
        ShiftLeft(t, g[gCount], static_cast<uint32_t>(1), gCount);
        OrU32(b, g, t, gCount);
        for (uint32_t j = 2; j < 8; ++j) {
            ShiftLeft(t, g[j * gCount], j, gCount);
            OrU32(b, b, t, gCount);
        }
        // level-2: z[k, m] = b[4m] | b[4m+1]<<8 | b[4m+2]<<16 | b[4m+3]<<24 -> [64, qjlDim/32]
        auto offQ2 = bufOffQ2_.Get<uint32_t>();
        const uint32_t q2 = qjlDim_ / 32;
        const uint32_t zCount = halfRows_ * q2;
        for (uint32_t tt = 0; tt < 4; ++tt) {
            Gather(g[tt * zCount], b, offQ2, tt * sizeof(uint32_t), static_cast<uint64_t>(64),
                static_cast<uint8_t>(4), 8);
        }
        auto z = bufB_.Get<uint32_t>();
        ShiftLeft(t, g[zCount], static_cast<uint32_t>(8), zCount);
        OrU32(z, g, t, zCount);
        ShiftLeft(t, g[2 * zCount], static_cast<uint32_t>(16), zCount);
        OrU32(z, z, t, zCount);
        ShiftLeft(t, g[3 * zCount], static_cast<uint32_t>(24), zCount);
        OrU32(z, z, t, zCount);

        if (valid > 0) {
            SetFlag<HardEvent::V_MTE3>(4);
            WaitFlag<HardEvent::V_MTE3>(4);
            DataCopyPad(qjlOutGm_[static_cast<uint64_t>(rowBase) * qjlBytesPerRow_], bufB_.Get<uint8_t>(),
                DataCopyExtParams{1, static_cast<uint32_t>(valid * qjlBytesPerRow_), 0, 0, 0});
            SetFlag<HardEvent::MTE3_V>(4);
            WaitFlag<HardEvent::MTE3_V>(4);
        }
    }

    const KvCacheTurboQuantTilingData *td_ = nullptr;
    TPipe *pipe_ = nullptr;
    MmF32 *mm1_ = nullptr;
    MmF32 *mm2_ = nullptr;
    GlobalTensor<float> rotGm_;
    GlobalTensor<float> qjlGm_;
    GlobalTensor<bfloat16_t> xGm_;
    GlobalTensor<uint8_t> idxGm_;
    GlobalTensor<uint8_t> qjlOutGm_;
    GlobalTensor<bfloat16_t> normGm_;
    GlobalTensor<bfloat16_t> gammaGm_;
    GlobalTensor<float> uGm_[2];
    GlobalTensor<float> yGm_[2];
    GlobalTensor<float> rGm_[2];
    GlobalTensor<float> pGm_[2];

    TBuf<TPosition::VECCALC> bufA_;
    TBuf<TPosition::VECCALC> bufB_;
    TBuf<TPosition::VECCALC> bufC_;
    TBuf<TPosition::VECCALC> bufD_;
    TBuf<TPosition::VECCALC> bufE_;
    TBuf<TPosition::VECCALC> bufX_;
    TBuf<TPosition::VECCALC> bufOffG_;
    TBuf<TPosition::VECCALC> bufOffM2_;
    TBuf<TPosition::VECCALC> bufOffQ2_;
    TBuf<TPosition::VECCALC> bufOffEO_;
    TBuf<TPosition::VECCALC> bufPart_;
    TBuf<TPosition::VECCALC> bufNorm_;
    TBuf<TPosition::VECCALC> bufRes_;
    TBuf<TPosition::VECCALC> bufNC_;
    TBuf<TPosition::VECCALC> bufGamma_;
    TBuf<TPosition::VECCALC> bufNormBf_;
    TBuf<TPosition::VECCALC> bufGammaBf_;
    TBuf<TPosition::VECCALC> bufCent_;
    TBuf<TPosition::VECCALC> bufBound_;
    TBuf<TPosition::VECCALC> bufMask_;

    uint32_t totalRows_ = 0;
    uint32_t headDim_ = 0;
    uint32_t halfRows_ = 0;
    uint32_t tileRows_ = 0;
    uint32_t qjlDim_ = 0;
    uint32_t mseBits_ = 0;
    uint32_t levels_ = 0;
    uint32_t idxBytesPerRow_ = 0;
    uint32_t qjlBytesPerRow_ = 0;
    uint32_t usedPairs_ = 0;
    uint32_t numTiles_ = 0;
    uint32_t pairId_ = 0;
    uint32_t subId_ = 0;
    uint32_t myTiles_ = 0;
};

extern "C" __global__ __aicore__ void kv_cache_turbo_quant(GM_ADDR kv_vectors, GM_ADDR rotation_matrix,
    GM_ADDR qjl_matrix, GM_ADDR quant_idx, GM_ADDR quant_qjl, GM_ADDR quant_norm, GM_ADDR quant_gamma,
    GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(KvCacheTurboQuantTilingData);
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    // The compiler-generated GET_TILING_DATA_WITH_STRUCT mis-copies this composite
    // struct (device observed garbage fields while the raw GM tiling buffer held the
    // correct bytes). Copy the tiling bytes manually instead.
    KvCacheTurboQuantTilingData tilingData;
    {
        const __gm__ uint32_t *src = reinterpret_cast<const __gm__ uint32_t *>(tiling);
        uint32_t *dst = reinterpret_cast<uint32_t *>(&tilingData);
        for (uint32_t i = 0; i < sizeof(KvCacheTurboQuantTilingData) / sizeof(uint32_t); ++i) {
            dst[i] = src[i];
        }
    }

    // 910B high-level Matmul runs as a kfc client/server pair: the AIC executes the
    // server loop inside REGIST_MATMUL_OBJ and never reaches the code below; each AIV
    // acts as a client and drives mm1/mm2 synchronously via IterateAll.
    TPipe pipe;
    MmF32 mm1;
    MmF32 mm2;
    REGIST_MATMUL_OBJ(&pipe, GetSysWorkSpacePtr(), mm1, &tilingData.mm1Tiling, mm2, &tilingData.mm2Tiling);
    if ASCEND_IS_AIV {
        KvTqVec op;
        op.Init(&pipe, &mm1, &mm2, kv_vectors, rotation_matrix, qjl_matrix, quant_idx, quant_qjl,
            quant_norm, quant_gamma, workspace, &tilingData);
        op.Process();
    }
}
