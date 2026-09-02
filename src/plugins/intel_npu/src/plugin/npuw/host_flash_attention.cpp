// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// Configuration: Enable loop-based Q@K computation to avoid materialized K/V broadcast
// Set to 1 to enable grouped computation, 0 to use traditional broadcast
// Disabled by default because NPU compiler optimizations are suboptimal
#define ENABLE_HFA_LOOP_BASED_COMPUTATION 0

#include "host_flash_attention.hpp"

#include <algorithm>
#include <cctype>
#include <string>
#include <utility>

#include "intel_npu/ops/flash_attention_tile.hpp"
#include "logging.hpp"
#include "openvino/core/validation_util.hpp"
#include "openvino/op/ops.hpp"
#include "openvino/openvino.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "util.hpp"

namespace ov {
namespace npuw {
namespace function {

namespace opp = ov::pass::pattern;

// Helper struct: Holds all input parameter nodes for HFA tile model creation
// Contains 7 parameters: past_acc, past_max, past_d, k_tile, v_tile, q, mask_tile
struct HFATileInputs {
    std::shared_ptr<ov::op::v0::Parameter> past_acc;
    std::shared_ptr<ov::op::v0::Parameter> past_max;
    std::shared_ptr<ov::op::v0::Parameter> past_d;
    std::shared_ptr<ov::op::v0::Parameter> k_tile;
    std::shared_ptr<ov::op::v0::Parameter> v_tile;
    std::shared_ptr<ov::op::v0::Parameter> q;
    std::shared_ptr<ov::op::v0::Parameter> mask_tile;
};

// Helper struct: Holds f32-converted nodes from input parameters for computation
// All computations are performed in f32 for numerical stability
struct HFATileF32Nodes {
    std::shared_ptr<ov::Node> past_acc_f32;
    std::shared_ptr<ov::Node> past_max_f32;
    std::shared_ptr<ov::Node> past_d_f32;
    std::shared_ptr<ov::Node> k_tile_f32;
    std::shared_ptr<ov::Node> v_tile_f32;
    std::shared_ptr<ov::Node> q_f32;
    std::shared_ptr<ov::Node> mask_tile_f32;
};

// Helper struct: Flash attention computation results (all in f32 precision)
// Contains: acc (accumulator), maxx (maximum values), d (normalization denominator)
struct FlashAttentionResults {
    ov::Output<ov::Node> acc;
    ov::Output<ov::Node> maxx;
    ov::Output<ov::Node> d;
};

// ============================================================================
// Helper function: Create input parameters for HFA tile model
// ============================================================================
static HFATileInputs create_hfa_tile_inputs(const ov::Shape& q_shape,
                                            const ov::element::Type& input_dtype,
                                            const ov::element::Type& mask_dtype,
                                            int64_t tile_size,
                                            size_t kv_num_heads,
                                            bool v_transposed = true) {
    auto batch = q_shape[0];
    auto num_heads = q_shape[1];
    auto seq_len = q_shape[2];
    auto head_dim = q_shape[3];

    HFATileInputs inputs;

    auto set_param_name = [](std::shared_ptr<ov::op::v0::Parameter>& param, HFATileInputId id) {
        const char* name = hfa_tile_input_id_to_string(id);
        param->set_friendly_name(name);
        param->output(0).get_tensor().set_names({name});
    };

    // past_acc: [batch, num_heads, seq_len, head_dim]
    inputs.past_acc =
        std::make_shared<ov::op::v0::Parameter>(input_dtype, ov::Shape{batch, num_heads, seq_len, head_dim});
    set_param_name(inputs.past_acc, HFATileInputId::PAST_ACC);

    // past_max: [batch, num_heads, seq_len, 1]
    inputs.past_max = std::make_shared<ov::op::v0::Parameter>(input_dtype, ov::Shape{batch, num_heads, seq_len, 1});
    set_param_name(inputs.past_max, HFATileInputId::PAST_MAX);

    // past_d: [batch, num_heads, seq_len, 1]
    inputs.past_d = std::make_shared<ov::op::v0::Parameter>(input_dtype, ov::Shape{batch, num_heads, seq_len, 1});
    set_param_name(inputs.past_d, HFATileInputId::PAST_D);

    // k_tile: [batch, kv_num_heads, tile_size, head_dim]
    inputs.k_tile = std::make_shared<ov::op::v0::Parameter>(
        input_dtype,
        ov::Shape{batch, kv_num_heads, static_cast<size_t>(tile_size), head_dim});
    set_param_name(inputs.k_tile, HFATileInputId::K_TILE);

    // v_tile: [batch, kv_num_heads, head_dim, tile_size] when V is pre-transposed by OptimizeValueTensors,
    //          [batch, kv_num_heads, tile_size, head_dim] when V is in normal (non-transposed) layout.
    if (v_transposed) {
        inputs.v_tile = std::make_shared<ov::op::v0::Parameter>(
            input_dtype,
            ov::Shape{batch, kv_num_heads, head_dim, static_cast<size_t>(tile_size)});
    } else {
        inputs.v_tile = std::make_shared<ov::op::v0::Parameter>(
            input_dtype,
            ov::Shape{batch, kv_num_heads, static_cast<size_t>(tile_size), head_dim});
    }
    set_param_name(inputs.v_tile, HFATileInputId::V_TILE);

    // q: [batch, num_heads, seq_len, head_dim]
    inputs.q = std::make_shared<ov::op::v0::Parameter>(input_dtype, ov::Shape{batch, num_heads, seq_len, head_dim});
    set_param_name(inputs.q, HFATileInputId::Q);

    // mask_tile: [batch, 1, seq_len, tile_size] - use mask's original dtype
    inputs.mask_tile =
        std::make_shared<ov::op::v0::Parameter>(mask_dtype,
                                                ov::Shape{batch, 1, seq_len, static_cast<size_t>(tile_size)});
    set_param_name(inputs.mask_tile, HFATileInputId::MASK_TILE);

    return inputs;
}

// ============================================================================
// Helper function: Convert input parameters to f32
// ============================================================================
static HFATileF32Nodes convert_inputs_to_f32(const HFATileInputs& inputs,
                                             const ov::element::Type& mask_dtype,
                                             const ov::element::Type& compute_dtype,
                                             bool use_mask = true) {
    HFATileF32Nodes f32_nodes;

    f32_nodes.past_acc_f32 = std::make_shared<ov::op::v0::Convert>(inputs.past_acc, compute_dtype);
    f32_nodes.past_acc_f32->set_friendly_name("past_acc_f32");

    f32_nodes.past_max_f32 = std::make_shared<ov::op::v0::Convert>(inputs.past_max, compute_dtype);
    f32_nodes.past_max_f32->set_friendly_name("past_max_f32");

    f32_nodes.past_d_f32 = std::make_shared<ov::op::v0::Convert>(inputs.past_d, compute_dtype);
    f32_nodes.past_d_f32->set_friendly_name("past_d_f32");

    f32_nodes.k_tile_f32 = std::make_shared<ov::op::v0::Convert>(inputs.k_tile, compute_dtype);
    f32_nodes.k_tile_f32->set_friendly_name("k_tile_f32");

    f32_nodes.v_tile_f32 = std::make_shared<ov::op::v0::Convert>(inputs.v_tile, compute_dtype);
    f32_nodes.v_tile_f32->set_friendly_name("v_tile_f32");

    f32_nodes.q_f32 = std::make_shared<ov::op::v0::Convert>(inputs.q, compute_dtype);
    f32_nodes.q_f32->set_friendly_name("q_f32");

    if (use_mask) {
        // Convert mask to f32 if needed
        if (mask_dtype == compute_dtype) {
            f32_nodes.mask_tile_f32 = inputs.mask_tile;
        } else {
            f32_nodes.mask_tile_f32 = std::make_shared<ov::op::v0::Convert>(inputs.mask_tile, compute_dtype);
            f32_nodes.mask_tile_f32->set_friendly_name("mask_tile_f32");
        }
    }

    return f32_nodes;
}

// ============================================================================
// Helper function: Execute flash attention tile implementation using NPU fused op
// ============================================================================
static FlashAttentionResults execute_fused_flash_attention(const HFATileF32Nodes& f32_nodes,
                                                           const std::shared_ptr<ov::Node>& q_input,
                                                           const std::shared_ptr<ov::Node>& k_input,
                                                           const std::shared_ptr<ov::Node>& v_input,
                                                           bool is_last_tile = false,
                                                           bool is_first_tile = false,
                                                           bool v_transposed = true) {
    ov::intel_npu::op::FlashAttentionTile::Config config;
    config.is_head = is_first_tile;
    config.is_tail = is_last_tile;

    auto v_shape = v_input->get_output_partial_shape(0);
    auto rank = v_shape.rank().get_length();
    if (rank != 4)
        OPENVINO_THROW("v_input rank must be 4 for flash attention");

    // When V is pre-transposed by OptimizeValueTensors, the tile parameter is [B,H,head_dim,tile_size];
    // we transpose it back to [B,H,tile_size,head_dim] before feeding FlashAttentionTile (which expects
    // normal [B,H,seq_len,head_dim] layout).  When V was NOT transposed, the tile parameter already
    // holds normal layout [B,H,tile_size,head_dim] and no transpose is needed.
    std::shared_ptr<ov::Node> v_for_attn;
    if (v_transposed) {
        std::vector<int64_t> transpose_v_order({0, 1, 3, 2});
        auto transpose_order = std::make_shared<ov::op::v0::Constant>(ov::element::i64,
                                                                      ov::Shape{static_cast<size_t>(rank)},
                                                                      transpose_v_order);
        auto v_transpose = std::make_shared<ov::op::v1::Transpose>(v_input, transpose_order);
        v_transpose->set_friendly_name("v_input_transposed");
        v_for_attn = v_transpose;
    } else {
        // V is already in [B,H,tile_size,head_dim] — pass directly.
        v_for_attn = v_input;
    }

    auto squeeze = std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{1}, std::vector<int64_t>{-1});

    auto past_max_squeezed = std::make_shared<ov::op::v0::Squeeze>(f32_nodes.past_max_f32, squeeze);
    past_max_squeezed->set_friendly_name("past_max_squeezed");

    auto past_sum_squeezed = std::make_shared<ov::op::v0::Squeeze>(f32_nodes.past_d_f32, squeeze);
    past_sum_squeezed->set_friendly_name("past_sum_squeezed");

    std::shared_ptr<ov::intel_npu::op::FlashAttentionTile> flash_attn_tile;
    if (is_last_tile) {
        // Use mask for final tile to ensure proper masking of the last KV block
        flash_attn_tile = std::make_shared<ov::intel_npu::op::FlashAttentionTile>(q_input,
                                                                                  k_input,
                                                                                  v_for_attn,
                                                                                  f32_nodes.past_acc_f32,
                                                                                  past_max_squeezed,
                                                                                  past_sum_squeezed,
                                                                                  f32_nodes.mask_tile_f32,
                                                                                  config);
    } else {
        flash_attn_tile = std::make_shared<ov::intel_npu::op::FlashAttentionTile>(q_input,
                                                                                  k_input,
                                                                                  v_for_attn,
                                                                                  f32_nodes.past_acc_f32,
                                                                                  past_max_squeezed,
                                                                                  past_sum_squeezed,
                                                                                  config);
    }

    flash_attn_tile->set_friendly_name("npu_op_flash_attention_tile");
    FlashAttentionResults results;
    results.acc = flash_attn_tile->output(0);
    results.maxx = flash_attn_tile->output(1);
    results.d = flash_attn_tile->output(2);
    return results;
}

// ============================================================================
// Helper function: Execute flash attention algorithm (unified implementation)
// Supports both traditional broadcast and loop-based grouped computation
// ============================================================================
// Parameters:
//   use_grouped: If true, uses loop-based grouped computation (Q/P reshape)
//                If false, uses traditional broadcast K/V approach
static FlashAttentionResults execute_host_flash_attention(const HFATileF32Nodes& f32_nodes,
                                                          const std::shared_ptr<ov::Node>& q_input,
                                                          const std::shared_ptr<ov::Node>& k_input,
                                                          const std::shared_ptr<ov::Node>& v_input,
                                                          size_t batch,
                                                          size_t num_heads,
                                                          size_t kv_num_heads,
                                                          size_t seq_len,
                                                          size_t tile_size,
                                                          size_t head_dim,
                                                          bool use_grouped = false) {
    FlashAttentionResults results;

    // ========================================================================
    // Step 1: Compute QK (method differs based on use_grouped flag)
    // ========================================================================
    std::shared_ptr<ov::Node> qk;

    if (use_grouped) {
        // Loop-based grouped computation: Q and K are grouped format
        // Q_input:  [batch, kv_num_heads, factor * seq_len, head_dim]
        // K_input:  [batch, kv_num_heads, tile_size, head_dim]
        // QK_grouped: [batch, kv_num_heads, factor * seq_len, tile_size]
        auto qk_grouped = std::make_shared<ov::op::v0::MatMul>(q_input, k_input, false, true);
        qk_grouped->set_friendly_name("qk_grouped");

        // Reshape QK back: [batch, kv_num_heads, factor * seq_len, tile_size] -> [batch, num_heads, seq_len, tile_size]
        auto qk_reshape_pattern =
            std::make_shared<ov::op::v0::Constant>(ov::element::i64,
                                                   ov::Shape{4},
                                                   std::vector<int64_t>{static_cast<int64_t>(batch),
                                                                        static_cast<int64_t>(num_heads),
                                                                        static_cast<int64_t>(seq_len),
                                                                        static_cast<int64_t>(tile_size)});
        qk = std::make_shared<ov::op::v1::Reshape>(qk_grouped, qk_reshape_pattern, false);
        qk->set_friendly_name("qk");
    } else {
        // Traditional broadcast computation: use broadcast K directly
        // Q_input:  [batch, num_heads, seq_len, head_dim]
        // K_input:  [batch, num_heads, tile_size, head_dim] (already broadcast)
        // QK:       [batch, num_heads, seq_len, tile_size]
        qk = std::make_shared<ov::op::v0::MatMul>(q_input, k_input, false, true);
        qk->set_friendly_name("qk");
    }

    // ========================================================================
    // Step 2: Flash Attention core algorithm (same for both methods)
    // ========================================================================

    // qkm = qk + mask
    auto qkm = std::make_shared<ov::op::v1::Add>(qk, f32_nodes.mask_tile_f32);
    qkm->set_friendly_name("qkm");

    // maxx = max(past_max, reduce_max(qkm, axis=-1, keepdims=True))
    auto axes_const = std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{1}, std::vector<int64_t>{-1});
    auto qkm_max = std::make_shared<ov::op::v1::ReduceMax>(qkm, axes_const, true);
    qkm_max->set_friendly_name("qkm_max");

    auto maxx_node = std::make_shared<ov::op::v1::Maximum>(qkm_max, f32_nodes.past_max_f32);
    maxx_node->set_friendly_name("maxx");
    results.maxx = maxx_node->output(0);

    // p = exp(qkm - maxx)
    auto qkm_sub_maxx = std::make_shared<ov::op::v1::Subtract>(qkm, results.maxx);
    auto p = std::make_shared<ov::op::v0::Exp>(qkm_sub_maxx);
    p->set_friendly_name("p");

    // l = reduce_sum(p, axis=-1, keepdims=True)
    auto l = std::make_shared<ov::op::v1::ReduceSum>(p, axes_const, true);
    l->set_friendly_name("l");

    // alpha = exp(past_max - maxx)
    auto past_max_sub_maxx = std::make_shared<ov::op::v1::Subtract>(f32_nodes.past_max_f32, results.maxx);
    auto alpha = std::make_shared<ov::op::v0::Exp>(past_max_sub_maxx);
    alpha->set_friendly_name("alpha");

    // d = past_d * alpha + l
    auto past_d_alpha = std::make_shared<ov::op::v1::Multiply>(f32_nodes.past_d_f32, alpha);
    auto d_node = std::make_shared<ov::op::v1::Add>(past_d_alpha, l);
    d_node->set_friendly_name("d");
    results.d = d_node->output(0);

    // ========================================================================
    // Step 3: Compute PV and final accumulator (method differs based on use_grouped flag)
    // ========================================================================

    auto past_acc_alpha = std::make_shared<ov::op::v1::Multiply>(f32_nodes.past_acc_f32, alpha);
    std::shared_ptr<ov::Node> pv;

    if (use_grouped) {
        // Loop-based grouped computation: reshape P, multiply with V, reshape back
        size_t factor = num_heads / kv_num_heads;

        // Reshape P for grouped V multiplication: [batch, num_heads, seq_len, tile_size]
        //                                      -> [batch, kv_num_heads, factor * seq_len, tile_size]
        auto p_reshape_pattern =
            std::make_shared<ov::op::v0::Constant>(ov::element::i64,
                                                   ov::Shape{4},
                                                   std::vector<int64_t>{static_cast<int64_t>(batch),
                                                                        static_cast<int64_t>(kv_num_heads),
                                                                        static_cast<int64_t>(factor * seq_len),
                                                                        static_cast<int64_t>(tile_size)});
        auto p_grouped = std::make_shared<ov::op::v1::Reshape>(p, p_reshape_pattern, false);
        p_grouped->set_friendly_name("p_grouped");

        // pv_grouped = matmul(p_grouped, v^T)
        // P_grouped: [batch, kv_num_heads, factor * seq_len, tile_size]
        // V_input:   [batch, kv_num_heads, head_dim, tile_size]
        // PV_grouped: [batch, kv_num_heads, factor * seq_len, head_dim]
        auto pv_grouped = std::make_shared<ov::op::v0::MatMul>(p_grouped, v_input, false, true);
        pv_grouped->set_friendly_name("pv_grouped");

        // Reshape PV back: [batch, kv_num_heads, factor * seq_len, head_dim] -> [batch, num_heads, seq_len, head_dim]
        auto pv_reshape_pattern =
            std::make_shared<ov::op::v0::Constant>(ov::element::i64,
                                                   ov::Shape{4},
                                                   std::vector<int64_t>{static_cast<int64_t>(batch),
                                                                        static_cast<int64_t>(num_heads),
                                                                        static_cast<int64_t>(seq_len),
                                                                        static_cast<int64_t>(head_dim)});
        pv = std::make_shared<ov::op::v1::Reshape>(pv_grouped, pv_reshape_pattern, false);
        pv->set_friendly_name("pv");
    } else {
        // Traditional broadcast computation: use broadcast V directly
        // P:        [batch, num_heads, seq_len, tile_size]
        // V_input:  [batch, num_heads, head_dim, tile_size] (already broadcast)
        // PV:       [batch, num_heads, seq_len, head_dim]
        pv = std::make_shared<ov::op::v0::MatMul>(p, v_input, false, true);
        pv->set_friendly_name("pv");
    }

    // acc = past_acc * alpha + pv
    auto acc_node = std::make_shared<ov::op::v1::Add>(past_acc_alpha, pv);
    acc_node->set_friendly_name("acc");
    results.acc = acc_node->output(0);

    return results;
}

// ============================================================================
// Helper function: Broadcast KV from kv_num_heads to num_heads
// ============================================================================
static std::pair<std::shared_ptr<ov::Node>, std::shared_ptr<ov::Node>> broadcast_kv_tiles(
    const std::shared_ptr<ov::Node>& k_tile_f32,
    const std::shared_ptr<ov::Node>& v_tile_f32,
    size_t batch,
    size_t num_heads,
    size_t kv_num_heads,
    size_t tile_size,
    size_t head_dim) {
    size_t head_expansion = num_heads / kv_num_heads;

    // Broadcast K: [batch, kv_num_heads, tile_size, head_dim] -> [batch, num_heads, tile_size, head_dim]
    auto unsqueeze_axes_k =
        std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{1}, std::vector<int64_t>{2});
    auto k_unsqueezed = std::make_shared<ov::op::v0::Unsqueeze>(k_tile_f32, unsqueeze_axes_k);

    auto repeats_k =
        std::make_shared<ov::op::v0::Constant>(ov::element::i64,
                                               ov::Shape{5},
                                               std::vector<int64_t>{1, 1, static_cast<int64_t>(head_expansion), 1, 1});
    auto k_tiled = std::make_shared<ov::op::v0::Tile>(k_unsqueezed, repeats_k);

    auto k_reshape_pattern =
        std::make_shared<ov::op::v0::Constant>(ov::element::i64,
                                               ov::Shape{4},
                                               std::vector<int64_t>{static_cast<int64_t>(batch),
                                                                    static_cast<int64_t>(num_heads),
                                                                    static_cast<int64_t>(tile_size),
                                                                    static_cast<int64_t>(head_dim)});
    auto k_tile_broadcast = std::make_shared<ov::op::v1::Reshape>(k_tiled, k_reshape_pattern, false);
    k_tile_broadcast->set_friendly_name("k_tile_broadcast");

    // Broadcast V: [batch, kv_num_heads, head_dim, tile_size] -> [batch, num_heads, head_dim, tile_size]
    auto unsqueeze_axes_v =
        std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{1}, std::vector<int64_t>{2});
    auto v_unsqueezed = std::make_shared<ov::op::v0::Unsqueeze>(v_tile_f32, unsqueeze_axes_v);

    auto repeats_v =
        std::make_shared<ov::op::v0::Constant>(ov::element::i64,
                                               ov::Shape{5},
                                               std::vector<int64_t>{1, 1, static_cast<int64_t>(head_expansion), 1, 1});
    auto v_tiled = std::make_shared<ov::op::v0::Tile>(v_unsqueezed, repeats_v);

    auto v_reshape_pattern =
        std::make_shared<ov::op::v0::Constant>(ov::element::i64,
                                               ov::Shape{4},
                                               std::vector<int64_t>{static_cast<int64_t>(batch),
                                                                    static_cast<int64_t>(num_heads),
                                                                    static_cast<int64_t>(head_dim),
                                                                    static_cast<int64_t>(tile_size)});
    auto v_tile_broadcast = std::make_shared<ov::op::v1::Reshape>(v_tiled, v_reshape_pattern, false);
    v_tile_broadcast->set_friendly_name("v_tile_broadcast");

    return {k_tile_broadcast, v_tile_broadcast};
}

#if ENABLE_HFA_LOOP_BASED_COMPUTATION
// ============================================================================
// Helper function: Reshape Q for grouped computation (loop-based approach)
// Avoids materializing broadcasted K/V tensors by reshaping Q to match KV heads
// ============================================================================
// Q: [batch, num_heads, seq_len, head_dim] -> [batch, kv_num_heads, factor * seq_len, head_dim]
// where factor = num_heads / kv_num_heads
static std::shared_ptr<ov::Node> reshape_q_for_groups(const std::shared_ptr<ov::Node>& q_f32,
                                                      size_t batch,
                                                      size_t num_heads,
                                                      size_t kv_num_heads,
                                                      size_t seq_len,
                                                      size_t head_dim) {
    size_t factor = num_heads / kv_num_heads;

    // Reshape Q: [batch, num_heads, seq_len, head_dim] -> [batch, kv_num_heads, factor * seq_len, head_dim]
    auto q_reshape_pattern =
        std::make_shared<ov::op::v0::Constant>(ov::element::i64,
                                               ov::Shape{4},
                                               std::vector<int64_t>{static_cast<int64_t>(batch),
                                                                    static_cast<int64_t>(kv_num_heads),
                                                                    static_cast<int64_t>(factor * seq_len),
                                                                    static_cast<int64_t>(head_dim)});
    auto q_grouped = std::make_shared<ov::op::v1::Reshape>(q_f32, q_reshape_pattern, false);
    q_grouped->set_friendly_name("q_grouped");

    return q_grouped;
}
#endif  // ENABLE_HFA_LOOP_BASED_COMPUTATION

// ============================================================================
// Helper: the layout chain between the attention output and the block's Result
// ============================================================================
// The tile models compute the attention output in the SDPA's own [B,H,S,D] layout, but what the
// block has to hand back is whatever its Result carries. GenAI's decomposed blocks end in
// Transpose(0,2,1,3) -> Reshape([B,S,H*D]), which is exactly the pair create_final_tile_outputs
// hardcodes. LiteRT's fused blocks do not: Q folds each GQA group's query heads into the query
// axis, so the block ends in Reshape([B,H*G,q,D]) -> Transpose(0,2,1,3) = [B,q,H*G,D], and the
// head-flattening Reshape sits *outside* the block, feeding the O projection.
//
// Emitting the hardcoded pair there builds a final tile model whose Result is a different shape
// from the funcall result the consuming subgraph reads, and nothing checks it until the first
// inference does set_tensor and throws "The input tensor size is not equal to the model input
// type". So read the block's real chain and clone it into the tile model instead.
//
// Only pure layout ops are accepted, and only as a single unforked chain: anything else means the
// caller cannot reproduce the block and should say so rather than guess.
struct PostAttentionChain {
    std::vector<std::shared_ptr<ov::Node>> nodes;  // attention-output side first
    ov::Shape result_shape;
    bool found_result = false;
};

static PostAttentionChain collect_post_attention_chain(const std::shared_ptr<ov::Node>& attn_out) {
    PostAttentionChain chain;
    if (!attn_out || attn_out->get_output_size() != 1) {
        LOG_DEBUG("Post-attention chain: no single-output attention node to start from");
        return chain;
    }
    ov::Output<ov::Node> cur = attn_out->output(0);
    LOG_DEBUG("Post-attention chain: walking down from " << attn_out->get_type_name() << " '"
                                                         << attn_out->get_friendly_name() << "' "
                                                         << cur.get_partial_shape());
    LOG_BLOCK();
    while (true) {
        const auto consumers = cur.get_target_inputs();
        if (consumers.size() != 1) {
            LOG_DEBUG("stop: output has " << consumers.size() << " consumers, not 1 - not a chain");
            return PostAttentionChain{};
        }
        const auto& consumer = *consumers.begin();
        auto next = consumer.get_node()->shared_from_this();
        LOG_DEBUG("-> " << next->get_type_name() << " '" << next->get_friendly_name() << "' (port "
                        << consumer.get_index() << ") " << next->get_output_partial_shape(0));
        if (ov::is_type<ov::op::v0::Result>(next)) {
            if (cur.get_partial_shape().is_dynamic()) {
                LOG_DEBUG("stop: the block's Result is dynamic");
                return PostAttentionChain{};
            }
            chain.result_shape = cur.get_shape();
            chain.found_result = true;
            return chain;
        }
        // A Convert is walked through but never cloned. The one that shows up here is NPUW's own
        // f16 interconnect cast on the block boundary, and the tile model already ends in
        // Convert(..., output_dtype) with output_dtype read off this very Result - cloning it would
        // just duplicate that. Layout ops carry no arithmetic, so moving the cast to the end of the
        // chain is numerically the same thing.
        const bool is_cast = ov::is_type<ov::op::v0::Convert>(next);
        const bool is_layout_op =
            ov::is_type<ov::op::v1::Reshape>(next) || ov::is_type<ov::op::v1::Transpose>(next) ||
            ov::is_type<ov::op::v0::Squeeze>(next) || ov::is_type<ov::op::v0::Unsqueeze>(next);
        if (!is_layout_op && !is_cast) {
            LOG_DEBUG("stop: " << next->get_type_name() << " is not a pure layout op");
            return PostAttentionChain{};
        }
        if (consumer.get_index() != 0 || next->get_output_size() != 1) {
            LOG_DEBUG("stop: consumed on port " << consumer.get_index() << " / " << next->get_output_size()
                                                << " outputs");
            return PostAttentionChain{};
        }
        // Everything but the data port must be a Constant, so the node can be cloned into a model
        // that has no other connection to this graph.
        for (std::size_t i = 1; i < next->get_input_size(); ++i) {
            const auto arg = next->get_input_node_shared_ptr(i);
            if (!ov::is_type<ov::op::v0::Constant>(arg)) {
                LOG_DEBUG("stop: input " << i << " is " << arg->get_type_name() << " '" << arg->get_friendly_name()
                                         << "', not a Constant - cannot be cloned into the tile model");
                return PostAttentionChain{};
            }
        }
        if (!is_cast) {
            chain.nodes.push_back(next);
        }
        cur = next->output(0);
    }
}

// ============================================================================
// Helper function: Create final tile model outputs (division, transpose, reshape)
// ============================================================================
static ov::ResultVector create_final_tile_outputs(const FlashAttentionResults& results,
                                                  const ov::element::Type& output_dtype,
                                                  size_t batch,
                                                  size_t seq_len,
                                                  size_t num_heads,
                                                  size_t head_dim,
                                                  bool fused_flash_attention = false,
                                                  const std::vector<std::shared_ptr<ov::Node>>& post_attn_chain = {}) {
    std::shared_ptr<ov::Node> final_result;
    if (fused_flash_attention) {
        // If using FlashAttentionTile node, the output is already normalized, so skip division
        final_result = results.acc.get_node_shared_ptr();
        final_result->set_friendly_name("final_result");

    } else {
        // Division: result = acc / d
        final_result =
            std::make_shared<ov::op::v1::Divide>(results.acc.get_node_shared_ptr(), results.d.get_node_shared_ptr());
        final_result->set_friendly_name("final_result");
    }
    std::shared_ptr<ov::Node> reshaped_result;
    if (!post_attn_chain.empty()) {
        // Replay the block's own layout chain. final_result is in the SDPA's output layout
        // [batch, num_heads, seq_len, head_dim], which is what the first node of the chain saw.
        std::shared_ptr<ov::Node> cur = final_result;
        for (const auto& node : post_attn_chain) {
            ov::OutputVector new_inputs{cur->output(0)};
            for (std::size_t i = 1; i < node->get_input_size(); ++i) {
                new_inputs.push_back(node->get_input_node_shared_ptr(i)->clone_with_new_inputs({})->output(0));
            }
            cur = node->clone_with_new_inputs(new_inputs);
            cur->set_friendly_name("post_attn_" + node->get_friendly_name());
        }
        reshaped_result = cur;
    } else {
        // Transpose (0,2,1,3): [batch, num_heads, seq_len, head_dim] -> [batch, seq_len, num_heads, head_dim]
        auto transpose_order =
            std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{4}, std::vector<int64_t>{0, 2, 1, 3});
        auto transposed_result = std::make_shared<ov::op::v1::Transpose>(final_result, transpose_order);
        transposed_result->set_friendly_name("transposed_result");

        // Reshape: [batch, seq_len, num_heads, head_dim] -> [batch, seq_len, num_heads*head_dim]
        auto reshape_pattern =
            std::make_shared<ov::op::v0::Constant>(ov::element::i64,
                                                   ov::Shape{3},
                                                   std::vector<int64_t>{static_cast<int64_t>(batch),
                                                                        static_cast<int64_t>(seq_len),
                                                                        static_cast<int64_t>(num_heads * head_dim)});
        reshaped_result = std::make_shared<ov::op::v1::Reshape>(transposed_result, reshape_pattern, false);
    }
    reshaped_result->set_friendly_name("reshaped_result");

    // Convert final output to original SDPA output dtype
    auto final_output = std::make_shared<ov::op::v0::Convert>(reshaped_result, output_dtype);
    final_output->set_friendly_name("final_output");
    final_output->output(0).get_tensor().set_names({"output"});

    // Create result - only ONE output
    auto out_result = std::make_shared<ov::op::v0::Result>(final_output);
    out_result->set_friendly_name("out_result");

    return {out_result};
}

// ============================================================================
// Helper function: Create regular tile model outputs (intermediate states: acc, max, d)
// ============================================================================
static ov::ResultVector create_regular_tile_outputs(const FlashAttentionResults& results,
                                                    const ov::element::Type& input_dtype) {
    // Convert outputs back to input_dtype (f16)
    auto acc_output = std::make_shared<ov::op::v0::Convert>(results.acc, input_dtype);
    acc_output->set_friendly_name("acc_output");
    acc_output->output(0).get_tensor().set_names({"acc"});

    auto maxx_output = std::make_shared<ov::op::v0::Convert>(results.maxx, input_dtype);
    maxx_output->set_friendly_name("maxx_output");
    maxx_output->output(0).get_tensor().set_names({"maxx"});

    auto d_output = std::make_shared<ov::op::v0::Convert>(results.d, input_dtype);
    d_output->set_friendly_name("d_output");
    d_output->output(0).get_tensor().set_names({"d"});

    // Create results
    auto out_acc = std::make_shared<ov::op::v0::Result>(acc_output);
    out_acc->set_friendly_name("out_acc");

    auto out_maxx = std::make_shared<ov::op::v0::Result>(maxx_output);
    out_maxx->set_friendly_name("out_maxx");

    auto out_d = std::make_shared<ov::op::v0::Result>(d_output);
    out_d->set_friendly_name("out_d");

    return {out_acc, out_maxx, out_d};
}

// ============================================================================
// Helper function: Create regular tile model outputs for single flash attention node (intermediate states: acc, max, d)
// ============================================================================
static ov::ResultVector create_regular_tile_outputs_fused(const FlashAttentionResults& results,
                                                          const ov::element::Type& input_dtype) {
    auto acc_output = std::make_shared<ov::op::v0::Convert>(results.acc, input_dtype);
    acc_output->set_friendly_name("acc_output");
    acc_output->output(0).get_tensor().set_names({"acc"});

    auto axes = std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{1}, std::vector<int64_t>{-1});

    auto maxx_unsqueezed = std::make_shared<ov::op::v0::Unsqueeze>(results.maxx, axes);
    maxx_unsqueezed->set_friendly_name("maxx_unsqueezed");
    auto maxx_output = std::make_shared<ov::op::v0::Convert>(maxx_unsqueezed, input_dtype);
    maxx_output->set_friendly_name("maxx_output");
    maxx_output->output(0).get_tensor().set_names({"maxx"});

    auto d_unsqueezed = std::make_shared<ov::op::v0::Unsqueeze>(results.d, axes);
    d_unsqueezed->set_friendly_name("d_unsqueezed");
    auto d_output = std::make_shared<ov::op::v0::Convert>(d_unsqueezed, input_dtype);
    d_output->set_friendly_name("d_output");
    d_output->output(0).get_tensor().set_names({"d"});

    // Create results
    auto out_acc = std::make_shared<ov::op::v0::Result>(acc_output);
    out_acc->set_friendly_name("out_acc");

    auto out_maxx = std::make_shared<ov::op::v0::Result>(maxx_output);
    out_maxx->set_friendly_name("out_maxx");

    auto out_d = std::make_shared<ov::op::v0::Result>(d_output);
    out_d->set_friendly_name("out_d");

    return {out_acc, out_maxx, out_d};
}

// ============================================================================
// Helper function: Create individual tile model (regular or final)
// ============================================================================
// Parameters:
//   is_final_tile: If true, creates final tile with division/transpose/reshape
//   output_dtype: Output data type (only used when is_final_tile=true)
static std::shared_ptr<ov::Model> create_hfa_tile_model(const ov::Shape& q_shape,
                                                        const ov::element::Type& input_dtype,
                                                        const ov::element::Type& mask_dtype,
                                                        int64_t tile_size,
                                                        size_t kv_num_heads,
                                                        bool is_final_tile = false,
                                                        bool fused_flash_attention = false,
                                                        bool enable_mask_skipping = false,
                                                        bool v_transposed = true,
                                                        const ov::element::Type& output_dtype = ov::element::f16,
                                                        const std::shared_ptr<ov::op::v0::Constant>& q_scale = nullptr,
                                                        const std::vector<std::shared_ptr<ov::Node>>& post_attn_chain = {}) {
    LOG_DEBUG("Creating HFA " << (is_final_tile ? "FINAL " : "") << "tile model with tile_size=" << tile_size
                              << ", kv_num_heads=" << kv_num_heads << ", mask_dtype=" << mask_dtype
                              << (is_final_tile ? ", output_dtype=" + output_dtype.get_type_name() : "")
                              << ", fused_flash_attention=" << fused_flash_attention);

    // Extract dimensions
    NPUW_ASSERT(q_shape.size() == 4);
    auto batch = q_shape[0];
    auto num_heads = q_shape[1];
    auto seq_len = q_shape[2];
    auto head_dim = q_shape[3];

    NPUW_ASSERT(num_heads % kv_num_heads == 0 && "Q heads must be divisible by KV heads");

    auto compute_dtype = ov::element::f32;
    LOG_DEBUG("Using compute_dtype=f32 for all operations to match mask type");

    // Create input parameters
    auto inputs = create_hfa_tile_inputs(q_shape, input_dtype, mask_dtype, tile_size, kv_num_heads, v_transposed);

    // Convert all inputs to f32.
    // For the fused operation only the final tile uses a mask, regular tiles skip mask for performance if
    // enable_mask_skipping is true (depending on the model mask type).
    // For the non-fused operation all tiles require mask
    const bool use_mask = is_final_tile || !fused_flash_attention || !enable_mask_skipping;
    auto f32_nodes = convert_inputs_to_f32(inputs, mask_dtype, compute_dtype, use_mask);

    // The tile models compute raw Q*K^T -- they carry no attention scale of their own, so they
    // assume Q arrives pre-scaled. That holds when the block was extracted in its decomposed form
    // (the producer's Multiply(q, scale) sits upstream of the block boundary), but NOT when it was
    // extracted from a fused SDPA, whose scale lives on the op's own port and would otherwise be
    // dropped. Re-apply it here. Nothing is inserted when q_scale is null, so decomposed-form
    // graphs build exactly the same tile models as before.
    if (q_scale) {
        std::shared_ptr<ov::Node> scale_f32 = q_scale;
        if (q_scale->get_output_element_type(0) != compute_dtype) {
            scale_f32 = std::make_shared<ov::op::v0::Convert>(q_scale, compute_dtype);
            scale_f32->set_friendly_name("q_scale_f32");
        }
        auto q_scaled = std::make_shared<ov::op::v1::Multiply>(f32_nodes.q_f32, scale_f32);
        q_scaled->set_friendly_name("q_f32_scaled");
        f32_nodes.q_f32 = q_scaled;
        LOG_DEBUG("Applying explicit attention scale inside the HFA tile model");
    }

    FlashAttentionResults results;

#if ENABLE_HFA_LOOP_BASED_COMPUTATION
    // ========================================================================
    // Loop-based computation: Reshape Q to avoid K/V broadcast materialization
    // ========================================================================
    LOG_DEBUG("Using loop-based grouped computation (ENABLED) - avoids K/V broadcast");

    // Reshape Q for grouped computation
    auto q_grouped = reshape_q_for_groups(f32_nodes.q_f32, batch, num_heads, kv_num_heads, seq_len, head_dim);

    // Execute flash attention with grouped computation (K and V remain 4D, no broadcast)
    results = execute_host_flash_attention(f32_nodes,
                                           q_grouped,             // Q: grouped format
                                           f32_nodes.k_tile_f32,  // K: original 4D
                                           f32_nodes.v_tile_f32,  // V: original 4D
                                           batch,
                                           num_heads,
                                           kv_num_heads,
                                           seq_len,
                                           tile_size,
                                           head_dim,
                                           true);  // use_grouped = true
#else
    // ========================================================================
    // Traditional broadcast-based computation: Materialize K/V broadcast
    // ========================================================================
    LOG_DEBUG("Using traditional broadcast computation (DISABLED loop-based) - materializes K/V broadcast");

    if (fused_flash_attention) {
        // Execute fused flash attention node MHA, GQA
        results = execute_fused_flash_attention(f32_nodes,
                                                f32_nodes.q_f32,
                                                f32_nodes.k_tile_f32,
                                                f32_nodes.v_tile_f32,
                                                is_final_tile,
                                                false,  // is_first_tile
                                                v_transposed);
    } else {
        // Broadcast K and V tiles from kv_num_heads to num_heads
        auto [k_broadcast, v_broadcast] = broadcast_kv_tiles(f32_nodes.k_tile_f32,
                                                             f32_nodes.v_tile_f32,
                                                             batch,
                                                             num_heads,
                                                             kv_num_heads,
                                                             tile_size,
                                                             head_dim);

        // Execute flash attention algorithm with broadcasted K/V
        results = execute_host_flash_attention(f32_nodes,
                                               f32_nodes.q_f32,  // Q: original 4D
                                               k_broadcast,      // K: broadcast to num_heads
                                               v_broadcast,      // V: broadcast to num_heads
                                               batch,
                                               num_heads,
                                               kv_num_heads,
                                               seq_len,
                                               tile_size,
                                               head_dim,
                                               false);  // use_grouped = false
    }
#endif  // ENABLE_HFA_LOOP_BASED_COMPUTATION

    // Create model outputs and name based on tile type
    ov::ResultVector model_results;
    std::string model_name;

    if (is_final_tile) {
        // === FINAL TILE: Add division, transpose and reshape for final output ===
        model_results = create_final_tile_outputs(results,
                                                  output_dtype,
                                                  batch,
                                                  seq_len,
                                                  num_heads,
                                                  head_dim,
                                                  fused_flash_attention,
                                                  post_attn_chain);
        model_name = "HFA_Final_Tile";
        LOG_DEBUG("HFA FINAL tile model created: inputs=" << input_dtype << ", compute=" << compute_dtype
                                                          << ", output=" << output_dtype);
    } else {
        // === REGULAR TILE: Output intermediate states (acc, max, d) ===
        if (fused_flash_attention) {
            LOG_DEBUG("Using fused flash attention implementation - outputs acc, max, d from separate nodes");
            model_results = create_regular_tile_outputs_fused(results, input_dtype);

        } else {
            LOG_DEBUG("Using host flash attention implementation - outputs acc, max, d from the same node");
            model_results = create_regular_tile_outputs(results, input_dtype);
        }
        model_name = "HFA_Tile";
        LOG_DEBUG("HFA tile model created: inputs=" << input_dtype << ", compute=" << compute_dtype
                                                    << ", outputs=" << input_dtype);
    }

    // Create model parameters
    ov::ParameterVector model_params =
        {inputs.past_acc, inputs.past_max, inputs.past_d, inputs.k_tile, inputs.v_tile, inputs.q};
    if (use_mask) {
        model_params.push_back(inputs.mask_tile);
    }

    // Create and return model
    return std::make_shared<ov::Model>(model_results, model_params, model_name);
}

// ============================================================================
// Helper function: Extract actual Parameter by skipping Convert nodes
// ============================================================================
static std::shared_ptr<ov::Node> skip_convert_nodes(const std::shared_ptr<ov::Node>& node) {
    auto current = node;
    while (current && ov::is_type<ov::op::v0::Convert>(current.get())) {
        if (current->get_input_size() > 0) {
            current = current->get_input_node_shared_ptr(0);
        } else {
            break;
        }
    }
    return current;
}

// ============================================================================
// Helper function: Build SDPA parameter index mapping
// ============================================================================
static void build_sdpa_param_mapping(HostFlashAttention& hfa,
                                     const std::shared_ptr<ov::Model>& model,
                                     const ov::npuw::util::SDPAPatternNodes& pattern_nodes) {
    LOG_INFO("Building SDPA input parameter index mapping...");

    // Helper lambda to safely extract parameter from node (skipping Convert ops)
    auto extract_param = [&](const std::shared_ptr<ov::Node>& node) -> std::shared_ptr<ov::op::v0::Parameter> {
        return ov::as_type_ptr<ov::op::v0::Parameter>(skip_convert_nodes(node));
    };

    // Extract Q (query) parameter - input 0 of MatMul1 (decomposed) or of the SDPA (fused)
    if (auto q_param = extract_param(pattern_nodes.query_source().get_node_shared_ptr())) {
        hfa._query_param_idx = model->get_parameter_index(q_param);
    } else {
        LOG_WARN("Q input is not a Parameter of this block - HFA would bind the wrong tensor");
    }

    // Extract past KV parameters from a Concat node: all inputs except the last are treated as
    // past (one entry in non-block mode, multiple entries in block mode); the last input is
    // the present key/value. Key and value follow identical logic.
    auto extract_kv_params = [&](const std::shared_ptr<ov::Node>& concat_node,
                                 std::vector<std::size_t>& block_indices,
                                 std::size_t& present_idx_out,
                                 const char* kv_name) {
        if (!concat_node)
            return;
        const size_t n = concat_node->get_input_size();
        block_indices.clear();
        block_indices.reserve(n - 1);
        for (size_t i = 0; i < n - 1; ++i) {
            if (auto param = extract_param(concat_node->get_input_node_shared_ptr(i))) {
                const std::size_t idx = model->get_parameter_index(param);
                block_indices.push_back(idx);
                LOG_DEBUG("  Found " << kv_name << " block[" << i << "] at parameter index " << idx);
            } else {
                LOG_WARN("Could not extract parameter from " << kv_name << " Concat input[" << i << "]");
            }
        }
        if (auto param = extract_param(concat_node->get_input_node_shared_ptr(n - 1))) {
            present_idx_out = model->get_parameter_index(param);
            LOG_DEBUG("  Found " << kv_name << "_present at parameter index " << present_idx_out);
        }
    };

    extract_kv_params(pattern_nodes.past_key_concat_node,
                      hfa._past_key_block_indices,
                      hfa._present_key_param_idx,
                      "past_key");
    extract_kv_params(pattern_nodes.past_value_concat_node,
                      hfa._past_value_block_indices,
                      hfa._present_value_param_idx,
                      "past_value");

    // Extract mask parameter - input 1 of add_node (decomposed) or input 3 of the SDPA (fused)
    if (auto add_param = extract_param(pattern_nodes.mask_source().get_node_shared_ptr())) {
        hfa._attention_mask_param_idx = model->get_parameter_index(add_param);
    }

    LOG_INFO("Built SDPA input mapping: query="
             << hfa._query_param_idx << ", present_key=" << hfa._present_key_param_idx
             << ", present_value=" << hfa._present_value_param_idx << ", mask=" << hfa._attention_mask_param_idx);
    LOG_INFO("  Past key blocks: " << hfa._past_key_block_indices.size());
    LOG_INFO("  Past value blocks: " << hfa._past_value_block_indices.size());

    // Print KV cache blocks
    LOG_DEBUG("Past key blocks (" << hfa._past_key_block_indices.size() << "):");
    for (size_t i = 0; i < hfa._past_key_block_indices.size(); ++i) {
        LOG_DEBUG("  block[" << i << "] -> parameter[" << hfa._past_key_block_indices[i] << "]");
    }

    LOG_DEBUG("Past value blocks (" << hfa._past_value_block_indices.size() << "):");
    for (size_t i = 0; i < hfa._past_value_block_indices.size(); ++i) {
        LOG_DEBUG("  block[" << i << "] -> parameter[" << hfa._past_value_block_indices[i] << "]");
    }

    LOG_DEBUG("=============================================");
}

// ============================================================================
// Helper function: Build tile model parameter index mapping
// ============================================================================
static void build_tile_param_mapping(HostFlashAttention& hfa, const std::shared_ptr<ov::Model>& tile_model) {
    LOG_INFO("Building HFA Tile Model input index mapping...");

    // Parse tile model inputs by their tensor names
    // Expected input order: [past_acc, past_max, past_d, k_tile, v_tile, q, mask_tile]
    const auto& tile_inputs = tile_model->inputs();
    for (std::size_t i = 0; i < tile_inputs.size(); ++i) {
        const auto& tensor_names = tile_inputs[i].get_names();
        if (tensor_names.empty()) {
            LOG_WARN("Tile model input[" << i << "] has no tensor name");
            continue;
        }

        const std::string& name = *tensor_names.begin();

        // Map tensor name to enum ID
        if (name == hfa_tile_input_id_to_string(HFATileInputId::PAST_ACC)) {
            hfa._tile_param_index_map[HFATileInputId::PAST_ACC] = i;
        } else if (name == hfa_tile_input_id_to_string(HFATileInputId::PAST_MAX)) {
            hfa._tile_param_index_map[HFATileInputId::PAST_MAX] = i;
        } else if (name == hfa_tile_input_id_to_string(HFATileInputId::PAST_D)) {
            hfa._tile_param_index_map[HFATileInputId::PAST_D] = i;
        } else if (name == hfa_tile_input_id_to_string(HFATileInputId::K_TILE)) {
            hfa._tile_param_index_map[HFATileInputId::K_TILE] = i;
        } else if (name == hfa_tile_input_id_to_string(HFATileInputId::V_TILE)) {
            hfa._tile_param_index_map[HFATileInputId::V_TILE] = i;
        } else if (name == hfa_tile_input_id_to_string(HFATileInputId::Q)) {
            hfa._tile_param_index_map[HFATileInputId::Q] = i;
        } else if (name == hfa_tile_input_id_to_string(HFATileInputId::MASK_TILE)) {
            hfa._tile_param_index_map[HFATileInputId::MASK_TILE] = i;
        } else {
            LOG_WARN("Unknown tile model input name: " << name);
        }
    }

    // Print the tile input mapping
    LOG_DEBUG("");
    LOG_DEBUG("========== HFA Tile Model Input Mapping ==========");
    LOG_DEBUG("Total entries: " << hfa._tile_param_index_map.size());

    for (const auto& [input_id, input_idx] : hfa._tile_param_index_map) {
        LOG_DEBUG("  " << hfa_tile_input_id_to_string(input_id) << " -> input[" << input_idx << "]");
    }
    LOG_DEBUG("==================================================");
}

// ============================================================================
// Helper function: Build tile model output index mapping
// ============================================================================
static void build_tile_output_mapping(HostFlashAttention& hfa, const std::shared_ptr<ov::Model>& tile_model) {
    LOG_INFO("Building HFA Tile Model output index mapping...");

    // Parse tile model outputs by their tensor names
    // Expected output order: [acc, maxx, d]
    const auto& tile_outputs = tile_model->outputs();
    for (std::size_t i = 0; i < tile_outputs.size(); ++i) {
        const auto& tensor_names = tile_outputs[i].get_names();
        if (tensor_names.empty()) {
            LOG_WARN("Tile model output[" << i << "] has no tensor name");
            continue;
        }

        const std::string& name = *tensor_names.begin();

        // Map tensor name to enum ID
        if (name == "acc") {
            hfa._tile_output_index_map[HFATileOutputId::ACC] = i;
        } else if (name == "maxx") {
            hfa._tile_output_index_map[HFATileOutputId::MAXX] = i;
        } else if (name == "d") {
            hfa._tile_output_index_map[HFATileOutputId::D] = i;
        } else {
            LOG_WARN("Unknown tile model output name: " << name);
        }
    }

    // Print the tile output mapping
    LOG_DEBUG("");
    LOG_DEBUG("========== HFA Tile Model Output Mapping ==========");
    LOG_DEBUG("Total entries: " << hfa._tile_output_index_map.size());

    for (const auto& [output_id, output_idx] : hfa._tile_output_index_map) {
        LOG_DEBUG("  " << hfa_tile_output_id_to_string(output_id) << " -> output[" << output_idx << "]");
    }
    LOG_DEBUG("==================================================");
}

// ============================================================================
// Helper function: Extract sequence dimension from Concat node
// ============================================================================
static std::optional<std::size_t> extract_sequence_dim_from_concat(const std::shared_ptr<ov::Node>& concat_node,
                                                                   const std::string& tensor_name) {
    if (!concat_node) {
        LOG_WARN("Failed to extract " << tensor_name << " concat node");
        return std::nullopt;
    }

    auto concat_op = std::dynamic_pointer_cast<ov::op::v0::Concat>(concat_node);
    if (!concat_op) {
        LOG_WARN("Failed to cast " << tensor_name << "_concat to Concat op");
        return std::nullopt;
    }

    const auto& concat_out_shape = concat_op->get_output_partial_shape(0);
    return ov::util::try_normalize_axis(concat_op->get_axis(), concat_out_shape.rank(), *concat_op);
}

std::optional<HostFlashAttention> HostFlashAttention::from(const std::shared_ptr<ov::Model>& model,
                                                           bool fused_flash_attention,
                                                           bool enable_mask_skipping) {
    LOG_INFO("Attempting to create HostFlashAttention"
             << (fused_flash_attention ? " with fused flash attention node" : ""));
    LOG_BLOCK();

    // ========================================================================
    // Step 1: Validate SDPA pattern and extract key nodes
    // ========================================================================
    auto pattern_nodes = ov::npuw::util::find_sdpa_pattern_nodes(model, /*allow_fused=*/true);
    if (!pattern_nodes.is_valid()) {
        LOG_WARN("Failed to re-find SDPA pattern nodes");
        return std::nullopt;
    }
    LOG_DEBUG("Matched attention block in " << (pattern_nodes.is_fused() ? "fused" : "decomposed") << " form");

    auto q_input = pattern_nodes.query_source().get_node_shared_ptr();
    auto k_concat = pattern_nodes.past_key_concat_node;

    // Skip Convert nodes to get to the actual Parameter/input
    q_input = skip_convert_nodes(q_input);

    if (!q_input || !k_concat) {
        LOG_WARN("Failed to extract Q input or K concat from pattern");
        return std::nullopt;
    }

    // ========================================================================
    // Step 2: Extract shape and data type information
    // ========================================================================
    auto q_shape = q_input->get_output_partial_shape(0);
    if (q_shape.is_dynamic()) {
        LOG_WARN("Dynamic shapes not yet supported for HFA");
        return std::nullopt;
    }

    auto q_shape_static = q_shape.to_shape();
    auto dtype = q_input->get_output_element_type(0);

    // Validate Q shape and extract query_size (seq_len dimension)
    if (q_shape_static.size() != 4) {
        LOG_WARN("Q shape must be 4D, got " << q_shape_static.size() << "D shape");
        return std::nullopt;
    }
    std::size_t query_size = q_shape_static[2];  // seq_len at index 2
    LOG_DEBUG("Extracted query_size (seq_len) from Q shape: " << query_size);

    auto mask_source = pattern_nodes.mask_source();
    if (!mask_source.get_node_shared_ptr()) {
        LOG_WARN("Attention block carries no additive mask - not supported by HFA");
        return std::nullopt;
    }
    auto mask_param = ov::npuw::util::find_mask_parameter(mask_source);
    if (!mask_param) {
        LOG_WARN("Could not find mask parameter in model");
        return std::nullopt;
    }
    auto mask_dtype = mask_param->get_output_element_type(0);

    // Explicit attention scale, present only in the fused form (see create_hfa_tile_model).
    std::shared_ptr<ov::op::v0::Constant> q_scale;
    if (auto scale_out = pattern_nodes.scale_source(); scale_out.get_node_shared_ptr()) {
        q_scale = ov::as_type_ptr<ov::op::v0::Constant>(scale_out.get_node_shared_ptr());
        if (!q_scale) {
            LOG_WARN("Attention scale is not a Constant - cannot fold it into the HFA tile model");
            return std::nullopt;
        }
        LOG_DEBUG("Found explicit attention scale: " << q_scale->cast_vector<float>().front());
    }

    auto output_dtype = ov::element::f16;  // Default fallback
    if (model->outputs().size() > 0) {
        output_dtype = model->output(0).get_element_type();
        LOG_DEBUG("Original SDPA output data type: " << output_dtype);
    } else {
        LOG_WARN("No outputs found in model, using default output dtype: " << output_dtype);
    }

    // ========================================================================
    // Step 3: Extract K/V sequence dimensions from Concat nodes
    // ========================================================================
    auto k_seq_dim_opt = extract_sequence_dim_from_concat(pattern_nodes.past_key_concat_node, "K");
    if (!k_seq_dim_opt) {
        return std::nullopt;
    }
    std::size_t k_seq_dim = k_seq_dim_opt.value();

    auto v_seq_dim_opt = extract_sequence_dim_from_concat(pattern_nodes.past_value_concat_node, "V");
    if (!v_seq_dim_opt) {
        return std::nullopt;
    }
    std::size_t v_seq_dim = v_seq_dim_opt.value();

    // ========================================================================
    // Step 4: Extract KV heads configuration and context size
    // ========================================================================
    size_t kv_num_heads = 0;
    size_t context_size = 0;
    if (!k_concat->get_output_partial_shape(0).is_static()) {
        return std::nullopt;
    }

    auto k_full_shape = k_concat->get_output_partial_shape(0).to_shape();
    // K shape after concat: [batch, kv_num_heads, kv_cache_size, head_dim]
    if (k_full_shape.size() != 4) {
        return std::nullopt;
    }

    kv_num_heads = k_full_shape[1];          // Extract kv_num_heads from K shape
    context_size = k_full_shape[k_seq_dim];  // Extract context size from sequence dimension

    if (kv_num_heads == 0) {
        LOG_WARN("Failed to determine KV num_heads");
        return std::nullopt;
    }

    if (context_size == 0) {
        LOG_WARN("Failed to determine context_size");
        return std::nullopt;
    }

    // ========================================================================
    // Step 5: Create tile models using query_size as tile_size
    // ========================================================================
    // V tensors are pre-transposed (stored as [B,H,head_dim,seq]) only when OptimizeValueTensors
    // succeeded, which is reflected by the V-concat axis being 3 instead of the default 2.
    const bool v_transposed = (v_seq_dim == 3);

    // Tile size. Historically this was just query_size (Q's axis-2 length). That is right only when
    // axis 2 carries the plain chunk length, which fails as soon as the producer folds each GQA
    // group's query heads into the query-sequence axis: axis 2 then holds G*q_len, so e.g.
    // 16512 % 2048 != 0 and context_size/query_size silently truncates the tile grid to 8 tiles
    // covering 16384 of 16512 positions.
    //
    // The runtime already tells us what the tile size has to be: the final tile must consume the
    // whole present KV slice in one inference (attn_subgraph.cpp asserts final_tile_length ==
    // tile_size), and every past block must be a whole number of tiles. So take it from the KV
    // Concat's last input -- the current-step K -- and check the rest lines up. For an unfolded
    // graph the present slice *is* query_size, so this reproduces the historical value exactly.
    const std::size_t num_k_inputs = k_concat->get_input_size();
    std::size_t tile_size = 0;
    if (num_k_inputs >= 2 && k_concat->get_input_partial_shape(num_k_inputs - 1).is_static()) {
        tile_size = k_concat->get_input_shape(num_k_inputs - 1)[k_seq_dim];
    }
    if (tile_size == 0 || context_size % tile_size != 0) {
        LOG_WARN("Unusable HFA tile size " << tile_size << ": it must be non-zero and divide context_size "
                                           << context_size);
        return std::nullopt;
    }
    // Every past block is walked in whole tiles by the runtime loop.
    for (std::size_t i = 0; i + 1 < num_k_inputs; ++i) {
        if (!k_concat->get_input_partial_shape(i).is_static() ||
            k_concat->get_input_shape(i)[k_seq_dim] % tile_size != 0) {
            LOG_WARN("Past KV block " << i << " is not a whole number of " << tile_size << "-position tiles");
            return std::nullopt;
        }
    }
    LOG_INFO("Creating HFA tile models with tile_size=" << tile_size << " (query_size=" << query_size
                                                        << ", context_size=" << context_size << ", "
                                                        << context_size / tile_size
                                                        << " tiles max), v_transposed=" << v_transposed);

    // ========================================================================
    // Step 5a: Make the final tile model produce the shape the block's Result has
    // ========================================================================
    // create_final_tile_outputs' hardcoded Transpose->Reshape yields [B, S, H*D]. That is right for
    // a GenAI-shaped block and wrong for a LiteRT one, where the block ends one op earlier (see
    // collect_post_attention_chain). Getting it wrong is not caught until the first inference, so
    // decide it here: reproduce the block's own chain when the shapes disagree, and refuse to build
    // the block at all if the chain cannot be reproduced.
    std::vector<std::shared_ptr<ov::Node>> post_attn_chain;
    {
        const auto attn_out =
            pattern_nodes.is_fused() ? pattern_nodes.fused_sdpa_node : pattern_nodes.matmul2_node;
        const ov::Shape tile_out_layout{q_shape_static[0], q_shape_static[1], query_size, q_shape_static[3]};
        const ov::Shape default_out_shape{q_shape_static[0], query_size, q_shape_static[1] * q_shape_static[3]};

        const auto chain = collect_post_attention_chain(attn_out);
        ov::Shape block_out_shape;
        bool block_out_known = false;
        if (chain.found_result) {
            block_out_shape = chain.result_shape;
            block_out_known = true;
        } else if (model->outputs().size() == 1 && model->output(0).get_partial_shape().is_static()) {
            block_out_shape = model->output(0).get_shape();
            block_out_known = true;
        }

        if (block_out_known && block_out_shape != default_out_shape) {
            const bool reproducible = chain.found_result && !chain.nodes.empty() && attn_out &&
                                      attn_out->get_output_partial_shape(0).is_static() &&
                                      attn_out->get_output_shape(0) == tile_out_layout;
            if (!reproducible) {
                LOG_WARN("The attention block has to produce "
                         << block_out_shape << " but HFA's final tile model produces " << default_out_shape
                         << ", and the block's post-attention chain cannot be replayed - refusing to build a"
                            " tile model whose Result does not match the block's");
                return std::nullopt;
            }
            post_attn_chain = chain.nodes;
            LOG_INFO("Attention block output is " << block_out_shape << ", not the default " << default_out_shape
                                                  << " - replaying the block's own " << post_attn_chain.size()
                                                  << "-node layout chain in the final tile model");
        }
    }

    auto tile_model = create_hfa_tile_model(q_shape_static,
                                            dtype,
                                            mask_dtype,
                                            static_cast<int64_t>(tile_size),
                                            kv_num_heads,
                                            false,
                                            fused_flash_attention,
                                            enable_mask_skipping,
                                            v_transposed,
                                            ov::element::f16,
                                            q_scale);
    if (!tile_model) {
        LOG_WARN("Failed to create HFA tile model");
        return std::nullopt;
    }

    auto final_tile_model = create_hfa_tile_model(q_shape_static,
                                                  dtype,
                                                  mask_dtype,
                                                  static_cast<int64_t>(tile_size),
                                                  kv_num_heads,
                                                  true,
                                                  fused_flash_attention,
                                                  enable_mask_skipping,
                                                  v_transposed,
                                                  output_dtype,
                                                  q_scale,
                                                  post_attn_chain);
    if (!final_tile_model) {
        LOG_WARN("Failed to create HFA final tile model");
        return std::nullopt;
    }

    // ========================================================================
    // Step 6: Create HostFlashAttention structure and set configuration
    // ========================================================================
    HostFlashAttention hfa;
    hfa._tile_model = tile_model;
    hfa._final_tile_model = final_tile_model;
    hfa._query_size = query_size;
    hfa._context_size = context_size;
    hfa._tile_size = tile_size;
    hfa._k_seq_dim = k_seq_dim;
    hfa._v_seq_dim = v_seq_dim;

    // ========================================================================
    // Step 7: Build SDPA parameter index mapping
    // ========================================================================
    build_sdpa_param_mapping(hfa, model, pattern_nodes);

    // ========================================================================
    // Step 8: Build tile model parameter index mapping
    // The first 6 input indices are identical in both models regular and final
    // final_tile_model has mask_tile (index 6)
    // ========================================================================
    build_tile_param_mapping(hfa, final_tile_model);

    // ========================================================================
    // Step 9: Build tile model output index mapping
    // ========================================================================
    build_tile_output_mapping(hfa, tile_model);

    LOG_INFO("Successfully created HostFlashAttention with query_size="
             << query_size << ", context_size=" << context_size << ", tile_size=" << tile_size);

    return hfa;
}

}  // namespace function

namespace compiled {

// Constructor implementation - extracts metadata
HostFlashAttention::HostFlashAttention(const function::HostFlashAttention& func_hfa) {
    LOG_INFO("Constructing compiled::HostFlashAttention");
    LOG_BLOCK();

    // Extract tile configuration from function HFA
    _tile_size = func_hfa._tile_size;

    // Store the tile models for later compilation
    _tile_model_to_compile = func_hfa._tile_model;
    _final_tile_model_to_compile = func_hfa._final_tile_model;

    // Copy query size, context size, and K/V sequence dimensions from function HFA
    _sdpa_attention_info._query_size = func_hfa._query_size;
    _sdpa_attention_info._context_size = func_hfa._context_size;
    _sdpa_attention_info._k_seq_dim = func_hfa._k_seq_dim;
    _sdpa_attention_info._v_seq_dim = func_hfa._v_seq_dim;

    // Pre-cache all indices from function HFA maps
    LOG_INFO("Pre-caching SDPA and tile indices...");

    // Pre-cache SDPA parameter indices (direct field access — no map lookup)
    _sdpa_attention_info._sdpa_indices.query = func_hfa._query_param_idx;

    // Copy all KV cache block indices
    _sdpa_attention_info._sdpa_indices.past_key_blocks = func_hfa._past_key_block_indices;
    _sdpa_attention_info._sdpa_indices.past_value_blocks = func_hfa._past_value_block_indices;

    _sdpa_attention_info._sdpa_indices.present_key = func_hfa._present_key_param_idx;
    _sdpa_attention_info._sdpa_indices.present_value = func_hfa._present_value_param_idx;
    _sdpa_attention_info._sdpa_indices.attention_mask = func_hfa._attention_mask_param_idx;

    // Pre-cache tile input indices
    auto get_tile_input_idx = [&](HFATileInputId input_id) -> std::size_t {
        auto it = func_hfa._tile_param_index_map.find(input_id);
        if (it == func_hfa._tile_param_index_map.end()) {
            OPENVINO_THROW("HFA: Tile input mapping not found for input ID: ", static_cast<uint8_t>(input_id));
        }
        return it->second;
    };

    auto get_tile_output_idx = [&](HFATileOutputId output_id) -> std::size_t {
        auto it = func_hfa._tile_output_index_map.find(output_id);
        if (it == func_hfa._tile_output_index_map.end()) {
            OPENVINO_THROW("HFA: Tile output mapping not found for output ID: ", static_cast<uint8_t>(output_id));
        }
        return it->second;
    };

    // Cache all tile input indices
    _sdpa_attention_info._tile_input_indices.q = get_tile_input_idx(HFATileInputId::Q);
    _sdpa_attention_info._tile_input_indices.k = get_tile_input_idx(HFATileInputId::K_TILE);
    _sdpa_attention_info._tile_input_indices.v = get_tile_input_idx(HFATileInputId::V_TILE);
    _sdpa_attention_info._tile_input_indices.mask = get_tile_input_idx(HFATileInputId::MASK_TILE);
    _sdpa_attention_info._tile_input_indices.acc = get_tile_input_idx(HFATileInputId::PAST_ACC);
    _sdpa_attention_info._tile_input_indices.max = get_tile_input_idx(HFATileInputId::PAST_MAX);
    _sdpa_attention_info._tile_input_indices.d = get_tile_input_idx(HFATileInputId::PAST_D);

    // Cache all tile output indices
    _sdpa_attention_info._tile_output_indices.acc = get_tile_output_idx(HFATileOutputId::ACC);
    _sdpa_attention_info._tile_output_indices.max = get_tile_output_idx(HFATileOutputId::MAXX);
    _sdpa_attention_info._tile_output_indices.d = get_tile_output_idx(HFATileOutputId::D);

    LOG_INFO("Pre-cached SDPA indices: [query="
             << _sdpa_attention_info._sdpa_indices.query
             << ", present_key=" << _sdpa_attention_info._sdpa_indices.present_key
             << ", present_value=" << _sdpa_attention_info._sdpa_indices.present_value
             << ", attention_mask=" << _sdpa_attention_info._sdpa_indices.attention_mask << "]");
    LOG_INFO("  Past key blocks: " << _sdpa_attention_info._sdpa_indices.past_key_blocks.size());
    LOG_INFO("  Past value blocks: " << _sdpa_attention_info._sdpa_indices.past_value_blocks.size());
    LOG_INFO("Attention configuration: query_size="
             << _sdpa_attention_info._query_size << ", context_size=" << _sdpa_attention_info._context_size
             << ", k_seq_dim=" << _sdpa_attention_info._k_seq_dim << ", v_seq_dim=" << _sdpa_attention_info._v_seq_dim);
    LOG_INFO("Pre-cached tile indices: inputs[q=" << _sdpa_attention_info._tile_input_indices.q
                                                  << ", k=" << _sdpa_attention_info._tile_input_indices.k
                                                  << ", v=" << _sdpa_attention_info._tile_input_indices.v
                                                  << ", mask=" << _sdpa_attention_info._tile_input_indices.mask
                                                  << ", acc=" << _sdpa_attention_info._tile_input_indices.acc
                                                  << ", max=" << _sdpa_attention_info._tile_input_indices.max
                                                  << ", d=" << _sdpa_attention_info._tile_input_indices.d
                                                  << "], outputs[acc=" << _sdpa_attention_info._tile_output_indices.acc
                                                  << ", max=" << _sdpa_attention_info._tile_output_indices.max
                                                  << ", d=" << _sdpa_attention_info._tile_output_indices.d << "]");

    // Note: _compiled_tile_model and _compiled_final_tile_model will be set later by
    // compile_host_flash_attention_model()
}
}  // namespace compiled

namespace runtime {
namespace host_flash_attention {

// PositionIDs constructor
PositionIDs::PositionIDs(std::size_t param_idx, std::size_t query_size, const ov::ISyncInferRequest& rq)
    : _position_ids_idx(param_idx),
      _query_size(query_size),
      _rq(rq) {
    // FIXME: speculative decode is indistinguishable at this point!
    _case = _query_size == 1 ? Case::GENERATE : Case::PREFILL;
}

Selector::Ptr PositionIDs::find(std::size_t query_size, const ov::ISyncInferRequest& rq) {
    auto is_position_ids = [](const ov::Output<const ov::Node>& p) {
        const auto& shape = p.get_shape();
        // FIXME: 2D/3D position IDs are not supported here YET
        return p.get_node()->get_friendly_name() == "position_ids" &&
               (shape.size() == 1 || (shape.size() == 2 && shape[0] == 1));
    };

    const auto& inputs = rq.get_inputs();
    auto pos_ids_iter = std::find_if(inputs.begin(), inputs.end(), is_position_ids);
    if (pos_ids_iter != inputs.end()) {
        const auto param_idx = std::distance(inputs.begin(), pos_ids_iter);
        return Selector::Ptr{new PositionIDs(param_idx, query_size, rq)};
    }
    return Selector::Ptr{};
}

void PositionIDs::prepare(int64_t past_len) {
    const auto& iport = _rq.get().get_compiled_model()->inputs()[_position_ids_idx];
    const auto in_tensor = _rq.get().get_tensor(iport);
    const auto in_dims = in_tensor->get_shape();

    // Same logic as regular attention PositionIDs
    auto* pos_data_ptr = in_tensor->data<int64_t>();
    for (int64_t idx = static_cast<int64_t>(in_dims.back()) - 1; idx >= 0; idx--) {
        if (pos_data_ptr[idx] > 0) {
            // Initialize fields
            _current_length = pos_data_ptr[idx];
            switch (_case) {
            case Case::GENERATE:
                // decode case, we have pos_id-1 past elements to take from kvcache
                _past_length = _current_length;
                break;
            case Case::PREFILL:
                // chunked prefill case. calculate the past_length in full chunks
                // FIXME: We know too much about chunking here
                _past_length = ((past_len + _query_size - 1) / _query_size) * _query_size;
                break;
            default:
                NPUW_ASSERT(false && "Reached the unreachable code");
            }
            return;
        }
    }
    LOG_WARN("Dynamic selector - no data found in the feature?");
    _current_length = -1;
}

int64_t PositionIDs::context_length() const {
    return _query_size + _past_length;
}

// ============================================================================
// MaskLength Selector
// ============================================================================

namespace {
// Measure the leading run of *visible* entries in the first scan_len entries of one row of an
// additive attention mask. The row itself is full_len long.
//
// The masked-out fill value is whatever the producer chose (-inf, -FLT_MAX, -65504, -1e9 ...), so
// instead of testing against a magic threshold we take it to be the row's minimum.
//
// That minimum is established over the WHOLE row even though only the first scan_len entries are
// measured, and the difference matters: the scanned region is the past-cache part of the merged KV,
// which on the very first prefill call is masked end to end. Judged on its own it is a constant
// region, indistinguishable from a fully visible one, and the first call would give up its entire
// speedup. The full row also spans the present slice, where the last query position is visible by
// construction - so as soon as anything is masked at all, the row shows both values.
//
// Returns {run_length, is_clean_prefix}. is_clean_prefix is false when a visible entry appears
// *after* the run - i.e. the occupied positions are not a contiguous prefix, which is what a
// right-aligned or wrapped (sliding-window) cache looks like. Callers must not optimize in that
// case: the run length says nothing useful about how much of the cache is live.
template <typename T>
std::pair<std::size_t, bool> leading_visible_run(const T* row, std::size_t full_len, std::size_t scan_len) {
    if (full_len == 0 || scan_len == 0) {
        return {0u, true};
    }
    auto lo = static_cast<float>(row[0]);
    auto hi = lo;
    for (std::size_t i = 1; i < full_len; ++i) {
        const auto v = static_cast<float>(row[i]);
        lo = std::min(lo, v);
        hi = std::max(hi, v);
    }
    if (lo == hi) {
        return {scan_len, true};  // nothing is masked anywhere in this row
    }
    std::size_t run = 0;
    while (run<scan_len&& static_cast<float>(row[run])> lo) {
        ++run;
    }
    for (std::size_t i = run; i < scan_len; ++i) {
        if (static_cast<float>(row[i]) > lo) {
            return {run, false};
        }
    }
    return {run, true};
}

// Is this top-level input plausibly the additive attention mask covering `context_size` KV
// positions, for a block whose tile is `tile_size`?
//
// Structure alone is not enough to be safe here. A [B,1,S,D] KV cache can pass a purely
// dimensional test - the global V cache is stored [1,1,512,16383] against a 16512 context, which
// clears "rank 4, dim1 == 1" and lands exactly one tile short of the context. So the name is
// required too, following the precedent PositionIDs::find sets by matching "position_ids"
// literally. A miss is harmless: the caller falls back to the full context, i.e. to precisely what
// the static graph does.
bool looks_like_attention_mask(const ov::Output<const ov::Node>& p, std::size_t tile_size, std::size_t context_size) {
    const auto& type = p.get_element_type();
    if (type != ov::element::f32 && type != ov::element::f16) {
        return false;
    }
    const auto& shape = p.get_shape();
    // [batch, 1, query_rows, kv_positions] - anything else and we cannot locate a query row.
    if (shape.size() != 4 || shape[1] != 1) {
        return false;
    }
    // The mask may be up to one tile short of the context: the KV length it describes is the
    // pre-alignment one (16511 vs a 16512 context here), since the padding the graph appends to
    // reach a whole number of tiles carries no mask of its own.
    const auto kv_positions = shape[3];
    if (kv_positions > context_size || context_size - kv_positions >= tile_size) {
        return false;
    }
    auto name = p.get_node()->get_friendly_name();
    std::transform(name.begin(), name.end(), name.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    return name.find("mask") != std::string::npos;
}
}  // anonymous namespace

MaskLength::MaskLength(std::size_t mask_idx,
                       std::size_t tile_size,
                       std::size_t context_size,
                       const ov::ISyncInferRequest& rq)
    : _mask_idx(mask_idx),
      _tile_size(tile_size),
      _context_size(context_size),
      _context_length(static_cast<int64_t>(context_size)),
      _rq(rq) {
    _case = Case::PREFILL;
}

Selector::Ptr MaskLength::find(std::size_t tile_size, std::size_t context_size, const ov::ISyncInferRequest& rq) {
    if (tile_size == 0 || context_size == 0 || tile_size > context_size) {
        return Selector::Ptr{};
    }
    // Search the top-level request's own inputs. The attention block's mask Parameter cannot be
    // used: it is the tiled (and padded) mask built outside the block, so it is an intermediate
    // tensor that is only bound once the subgraph runs - well after prepare() needs to read it.
    // The model-level mask is the same data before tiling, and its untiled query rows are in fact
    // the more direct thing to read.
    const auto& inputs = rq.get_inputs();
    std::size_t found = inputs.size();
    for (std::size_t i = 0; i < inputs.size(); ++i) {
        if (!looks_like_attention_mask(inputs[i], tile_size, context_size)) {
            continue;
        }
        // Prefer an exact context match over a padded one, then the longest candidate. With one
        // mask per attention family (sliding-window and global here) the context/tile pair already
        // picks out a single input, so this only breaks ties.
        if (found == inputs.size() || inputs[i].get_shape()[3] > inputs[found].get_shape()[3]) {
            found = i;
        }
    }
    if (found == inputs.size()) {
        LOG_WARN("HFA: no model input looks like a [B,1,Q,~" << context_size
                                                             << "] attention mask - cannot derive the KV length, "
                                                                "falling back to the full context");
        return Selector::Ptr{};
    }
    LOG_VERB("HFA: deriving the KV length from model input "
             << found << " ('" << inputs[found].get_node()->get_friendly_name() << "', " << inputs[found].get_shape()
             << ")");
    return Selector::Ptr{new MaskLength(found, tile_size, context_size, rq)};
}

void MaskLength::prepare(int64_t past_len) {
    // past_len is ignored on purpose: on this path it comes from history_size(), which the host
    // driving us does not maintain. The mask is the authority.
    const auto& iport = _rq.get().get_compiled_model()->inputs()[_mask_idx];
    const auto in_tensor = _rq.get().get_tensor(iport);
    const auto dims = in_tensor->get_shape();

    const std::size_t row_len = dims.back();
    const std::size_t num_rows = dims[dims.size() - 2];
    // The last query row sees the most: it is the newest token of this chunk.
    const std::size_t row_offset = (num_rows - 1) * row_len;

    // The merged KV is laid out [ past cache | present slice ], and the runtime's tile loop always
    // feeds the trailing present slice through the final tile. So only the past region is measured;
    // the present slice is added back below. The row can be shorter than the context (it describes
    // the KV length before tile alignment padding), hence the clamp.
    const std::size_t past_len_max = std::min(_context_size - _tile_size, row_len);

    std::pair<std::size_t, bool> run{past_len_max, true};
    if (in_tensor->get_element_type() == ov::element::f32) {
        run = leading_visible_run(in_tensor->data<float>() + row_offset, row_len, past_len_max);
    } else {
        run = leading_visible_run(in_tensor->data<ov::float16>() + row_offset, row_len, past_len_max);
    }

    if (!run.second) {
        // Occupied positions are not a contiguous prefix - fall back to the full context, which is
        // exactly what the static graph would have done.
        LOG_DEBUG("HFA: mask is not prefix-shaped, using the full context");
        _context_length = static_cast<int64_t>(_context_size);
        return;
    }

    // Round the past up to whole tiles (the extra positions are masked out anyway) and add the
    // present slice, which is always exactly one tile.
    const std::size_t past_tiles = (run.first + _tile_size - 1) / _tile_size;
    _context_length = static_cast<int64_t>(std::min((past_tiles + 1) * _tile_size, _context_size));
    LOG_DEBUG("HFA mask-derived KV length: past " << run.first << " -> " << _context_length << " total (tile "
                                                  << _tile_size << ")");
}

int64_t MaskLength::context_length() const {
    return _context_length;
}

// ============================================================================
// HFARuntimeContext Implementation
// ============================================================================

void HFARuntimeContext::reset() {
    m_mask_tile_cache.clear();
    m_mask_tile_buffers.clear();
    m_state_buffers.reset();
    m_current_buffer_idx = 0;
}

ov::SoPtr<ov::ITensor> HFARuntimeContext::find_cached_mask_tile(const ov::SoPtr<ov::ITensor>& mask_tensor,
                                                                int64_t mask_offset,
                                                                int64_t tile_length) const {
    HFATileMaskKey cache_key{mask_tensor, mask_offset, tile_length};
    auto it = m_mask_tile_cache.find(cache_key);
    if (it != m_mask_tile_cache.end()) {
        return it->second;
    }
    return {};
}

ov::SoPtr<ov::ITensor> HFARuntimeContext::get_mask_tile_buffer(size_t index) const {
    if (index >= m_mask_tile_buffers.size()) {
        throw std::out_of_range("HFA: mask tile buffer index " + std::to_string(index) + " out of range [0, " +
                                std::to_string(m_mask_tile_buffers.size()) + ")");
    }
    return m_mask_tile_buffers[index];
}

void HFARuntimeContext::cache_mask_tile(const ov::SoPtr<ov::ITensor>& mask_tensor,
                                        int64_t mask_offset,
                                        int64_t tile_length,
                                        const ov::SoPtr<ov::ITensor>& cached_tile) {
    HFATileMaskKey cache_key{mask_tensor, mask_offset, tile_length};
    m_mask_tile_cache[cache_key] = cached_tile;
}

void HFARuntimeContext::clear_mask_cache() {
    m_mask_tile_cache.clear();
}

void HFARuntimeContext::initialize_state_tensors(ov::SoPtr<ov::ITensor>& acc,
                                                 ov::SoPtr<ov::ITensor>& max,
                                                 ov::SoPtr<ov::ITensor>& sum) {
    const auto type = acc->get_element_type();
    if (type == ov::element::f16) {
        std::memset(acc->data<ov::float16>(), 0, acc->get_byte_size());
        std::fill_n(max->data<ov::float16>(), max->get_size(), std::numeric_limits<ov::float16>::lowest());
        std::memset(sum->data<ov::float16>(), 0, sum->get_byte_size());
    } else if (type == ov::element::f32) {
        std::memset(acc->data<float>(), 0, acc->get_byte_size());
        std::fill_n(max->data<float>(), max->get_size(), std::numeric_limits<float>::lowest());
        std::memset(sum->data<float>(), 0, sum->get_byte_size());
    } else {
        throw std::runtime_error("HFA: Unsupported state tensor type");
    }
}

void HFARuntimeContext::prepare_next_state_buffers() {
    if (!m_state_buffers.has_value()) {
        return;
    }
    size_t next_idx = 1 - m_current_buffer_idx;
    auto& next_buffer = (*m_state_buffers)[next_idx];
    initialize_state_tensors(next_buffer.acc, next_buffer.max, next_buffer.sum);
}

void HFARuntimeContext::switch_buffers() {
    if (m_state_buffers.has_value()) {
        m_current_buffer_idx = 1 - m_current_buffer_idx;
    }
}

}  // namespace host_flash_attention
}  // namespace runtime

}  // namespace npuw
}  // namespace ov
