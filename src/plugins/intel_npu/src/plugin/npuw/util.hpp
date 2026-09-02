// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <future>
#include <optional>
#include <random>
#include <string>

#include "llm_compiled_model_utils.hpp"
#include "logging.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/runtime/iplugin.hpp"
#include "openvino/runtime/itensor.hpp"
#include "openvino/runtime/so_ptr.hpp"

namespace ov {
namespace npuw {
namespace util {

bool is_set(const std::size_t sub_idx,
            const std::string& opt,
            const std::size_t real_idx = SIZE_MAX,
            const std::size_t end_idx = SIZE_MAX);

// Every great project has its own string class...
// NB: Newer C++ standards would allow to use string views or smt
ov::Tensor tensor_from_const(const std::shared_ptr<ov::Node>& node);

// In case of working with memory which will be detached later (Constant will be freed),
// we need to explicitly create a tensor which owns the memory during the execution.
ov::Tensor copy_tensor_from_const(const std::shared_ptr<ov::Node>& node);

bool starts_with(const std::string& str, const std::string& prefix);

std::string fmt(std::size_t number, std::size_t total);

// Matches the three DynamicQuantize decomposition implementations declared in
// kv_cache_compressed.hpp.
enum class DynamicQuantDecomposeMode {
    // V1: handcrafted symmetric-style path, i8 range [-127, 127].
    HandcraftedSymmetricI8 = 1,

    // V2: ONNX DynamicQuantizeLinear-style path, u8 range [0, 255].
    OnnxDynamicQuantizeLinear = 2,

    // V3: compiler pattern-style path. i8 requests are materialized as u8
    // storage for the asymmetric decomposition branch.
    CompilerPatternI8 = 3,
};

struct DynamicQuantStorageTypes {
    ov::element::Type quantized_data_type = ov::element::dynamic;
    ov::element::Type zero_point_type = ov::element::dynamic;
    ov::element::Type scale_type = ov::element::f32;
};

DynamicQuantStorageTypes resolve_dynamic_quant_storage_types(DynamicQuantDecomposeMode decompose_mode,
                                                             bool is_symmetric,
                                                             const ov::element::Type& quant_dt,
                                                             const ov::element::Type& scale_dt = ov::element::f32);

struct UnpackOptions {
    bool bUseOvParallelFor;
    size_t nPartitions;  // if 0 we use 64 elements step in parallel for, otherwise  target workload is dynamically
                         // calculated
    bool bStrictPartitioning;  // cannot reduce partitions in favor of speed
    explicit UnpackOptions(bool useParallelFor, size_t nPartitions, bool bStrictPartitioning)
        : bUseOvParallelFor(useParallelFor),
          nPartitions(nPartitions),
          bStrictPartitioning(bStrictPartitioning) {}
};

void unpack(const ov::SoPtr<ov::ITensor>& from,
            const ov::SoPtr<ov::ITensor>& to,
            const UnpackOptions& unpack_options = UnpackOptions{true, 16, false});

void unpack(const ov::SoPtr<ov::ITensor>& from,
            const ov::SoPtr<ov::ITensor>& scale,
            const ov::SoPtr<ov::ITensor>& to,
            const UnpackOptions& unpack_options = UnpackOptions{true, 16, false});

void unpack(const ov::SoPtr<ov::ITensor>& from,
            const ov::SoPtr<ov::ITensor>& zerop,
            const ov::SoPtr<ov::ITensor>& scale,
            const ov::SoPtr<ov::ITensor>& to,
            const UnpackOptions& unpack_options = UnpackOptions{true, 16, false});

void gather(const ov::SoPtr<ov::ITensor>& src, const ov::SoPtr<ov::ITensor>& idx, const ov::SoPtr<ov::ITensor>& dst);
void gather_cb4(const ov::SoPtr<ov::ITensor>& src,
                const ov::SoPtr<ov::ITensor>& idx,
                const ov::SoPtr<ov::ITensor>& dst);

using View = std::vector<std::size_t>;
ov::SoPtr<ov::ITensor> view(const ov::SoPtr<ov::ITensor>& src, const View& from, const View& to);

ov::SoPtr<ov::ITensor> view(const ov::SoPtr<ov::ITensor>& src, std::size_t dim, std::size_t offset, std::size_t len);

void to_f32(const ov::Tensor& in, ov::Tensor& out);
ov::Tensor to_f16(const ov::Tensor& t);
ov::Tensor transpose(const ov::Tensor& t);
ov::Tensor permute(const ov::Tensor& t, const std::vector<std::size_t>& axes);
ov::Tensor concat(const std::vector<ov::Tensor>& tt, std::size_t axis);

void permute_i4d(const ov::SoPtr<ov::ITensor>& src, ov::SoPtr<ov::ITensor>& dst, const std::array<int, 4> order);

// Start is inclusive, end is exclusive
using range_1d = std::pair<std::size_t, std::size_t>;
range_1d validMaskRange(const ov::SoPtr<ov::ITensor>& t);

namespace at {
template <class M_>
struct Impl {
    using M = typename std::decay<M_>::type;
    using V = typename M::mapped_type;

    M* m = nullptr;
    explicit Impl(M* pM) : m(pM) {}

    template <typename K>
    V& at(const K& k) {
        const auto iter = m->find(k);
        if (iter == m->end()) {
            std::stringstream ss;
            ss << "Key " << k << " is not found in a map of type " << typeid(m).name();
            const auto msg = ss.str();
            LOG_ERROR(msg);
            throw std::out_of_range(msg);
        }
        return iter->second;
    }

    template <typename K>
    V& at_or_at(const K& k1, const K& k2) {
        const auto iter = m->find(k1);
        if (iter == m->end()) {
            return at(k2);
        }
        return iter->second;
    }

    template <typename K>
    V& at_or_at_or_at(const K& k1, const K& k2, const K& k3) {
        const auto iter = m->find(k1);
        if (iter == m->end()) {
            return at_or_at(k2, k3);
        }
        return iter->second;
    }

    template <typename K>
    const V& at(const K& k) const {
        return const_cast<Impl*>(this)->at(k);
    }

    template <typename K>
    const V& at_or_at(const K& k1, const K& k2) const {
        return const_cast<Impl*>(this)->at_or_at(k1, k2);
    }

    template <typename K>
    const V& at_or_at_or_at(const K& k1, const K& k2, const K& k3) const {
        return const_cast<Impl*>(this)->at_or_at_or_at(k1, k2, k3);
    }

    template <typename K>
    V at_or(const K& k, const V& default_val) const {
        const auto iter = m->find(k);
        return iter != m->end() ? iter->second : default_val;
    }
};

template <typename M>
Impl<M> _(M* pM) {
    return Impl<M>(pM);
}

template <typename M>
Impl<M> _(M&& m) {
    return Impl<M>(&m);
}

template <typename M>
Impl<M> _(std::shared_ptr<M> pM) {
    return Impl<M>(pM.get());
}

}  // namespace at

// Written here to be a drop-in replacement for ov::parallel_for for the debug purposes
template <typename F>
void non_parallel_for(std::size_t count, F&& f) {
    for (std::size_t idx = 0u; idx < count; idx++) {
        f(idx);
    }
}

template <class CountedType>
struct Unique {
    static std::string name() {
        static std::size_t counter = 0u;
        return std::string(CountedType::name) + "_" + std::to_string(counter++);
    }
};

std::string generate_random_string(std::size_t size = 32);

using TensorPtr = ov::SoPtr<ov::ITensor>;
TensorPtr allocMem(const ov::element::Type type,
                   const ov::Shape& shape,
                   const std::string& device,
                   const std::shared_ptr<const ov::IPlugin>& plugin);

bool matchStringWithLoRAPattern(const std::string& input, const std::string& pattern_suffix);

bool matchLoRAMatMulAString(const std::string& input);

bool matchLoRAMatMulBString(const std::string& input);

bool matchLoRAMatMulAlphaString(const std::string& input);

bool matchLinCacheString(const std::string& input, const std::string& past_or_present = "past");

bool starts_with_past_lincache(const std::string& input_name);

// Structure to hold SDPA pattern nodes.
// After SplitKVCacheIntoBlocks the single past_key / past_value parameter is
// replaced by N block parameters, so both param-node fields are vectors.
// For the unmodified (non-block) case each vector contains exactly one element.
struct SDPAPatternNodes {
    std::shared_ptr<ov::Node> matmul1_node = nullptr;
    std::shared_ptr<ov::Node> matmul2_node = nullptr;
    std::shared_ptr<ov::Node> softmax_node = nullptr;
    std::shared_ptr<ov::Node> add_node = nullptr;
    // 1 (contiguous) or N (block-split) Parameter nodes feeding the KV Concat.
    std::vector<std::shared_ptr<ov::Node>> past_key_param_nodes;
    std::vector<std::shared_ptr<ov::Node>> past_value_param_nodes;
    std::shared_ptr<ov::Node> past_key_concat_node = nullptr;
    std::shared_ptr<ov::Node> past_value_concat_node = nullptr;
    // Set only when the attention block was matched in its *fused* form, i.e. as a single
    // ov::op::v13::ScaledDotProductAttention rather than MatMul->Add->Softmax->MatMul. In that
    // case matmul1/matmul2/softmax/add are all null and Q / mask / scale are read off the SDPA's
    // own ports. Only produced when find_sdpa_pattern_nodes() is asked for it explicitly.
    std::shared_ptr<ov::Node> fused_sdpa_node = nullptr;

    bool is_fused() const {
        return fused_sdpa_node != nullptr;
    }

    bool is_valid() const {
        const bool core = fused_sdpa_node || (matmul1_node && matmul2_node && softmax_node && add_node);
        return core && past_key_concat_node && past_value_concat_node;
    }

    // Q source: input 0 of MatMul1 (decomposed) or input 0 of the SDPA (fused).
    ov::Output<ov::Node> query_source() const {
        return fused_sdpa_node ? fused_sdpa_node->input_value(0) : matmul1_node->input_value(0);
    }

    // Additive attention mask: input 1 of the Add (decomposed) or input 3 of the SDPA (fused).
    // Returns an empty Output when the fused SDPA carries no mask (is_causal form).
    ov::Output<ov::Node> mask_source() const {
        if (fused_sdpa_node) {
            return fused_sdpa_node->get_input_size() > 3 ? fused_sdpa_node->input_value(3) : ov::Output<ov::Node>{};
        }
        return add_node->input_value(1);
    }

    // Explicit attention scale. Only a fused SDPA carries one; in the decomposed form the scale
    // has already been folded into Q by the producer, upstream of this block.
    ov::Output<ov::Node> scale_source() const {
        if (fused_sdpa_node && fused_sdpa_node->get_input_size() > 4) {
            return fused_sdpa_node->input_value(4);
        }
        return {};
    }

    // Log pattern information for debugging. prefix is optional.
    void log_pattern(const std::string& prefix = "") const {
        LOG_DEBUG("SDPA Pattern " << prefix << " nodes:");
        LOG_DEBUG("  MatMul1: " << (matmul1_node ? matmul1_node->get_friendly_name() : "null"));
        LOG_DEBUG("  Add: " << (add_node ? add_node->get_friendly_name() : "null"));
        LOG_DEBUG("  Softmax: " << (softmax_node ? softmax_node->get_friendly_name() : "null"));
        LOG_DEBUG("  MatMul2: " << (matmul2_node ? matmul2_node->get_friendly_name() : "null"));
        LOG_DEBUG("  Key Concat: " << (past_key_concat_node ? past_key_concat_node->get_friendly_name() : "null"));
        LOG_DEBUG(
            "  Value Concat: " << (past_value_concat_node ? past_value_concat_node->get_friendly_name() : "null"));
        LOG_DEBUG("  Fused SDPA: " << (fused_sdpa_node ? fused_sdpa_node->get_friendly_name() : "null"));
        LOG_DEBUG("  Past key params: " << past_key_param_nodes.size());
        LOG_DEBUG("  Past value params: " << past_value_param_nodes.size());
    }
};

// Find the decomposed SDPA sub-graph pattern (MatMul->Add->Softmax->MatMul) in a
// model and return all relevant nodes.  Returns an invalid result if not found.
// When allow_fused is set and no decomposed block is found, a lone fused
// ov::op::v13::ScaledDotProductAttention is accepted instead (see SDPAPatternNodes::is_fused).
// Callers that dereference matmul1/add/softmax/matmul2 directly must leave allow_fused false.
SDPAPatternNodes find_sdpa_pattern_nodes(const std::shared_ptr<ov::Model>& model, bool allow_fused = false);
std::vector<SDPAPatternNodes> find_all_sdpa_pattern_nodes(const std::shared_ptr<ov::Model>& model);

// Traverse upward from an Add node's mask input to find the attention-mask
// Parameter.  Only unary ops are traversed; returns nullptr on failure.
std::shared_ptr<ov::op::v0::Parameter> find_mask_parameter(const std::shared_ptr<ov::Node>& add_node);

// Same, but starting from an arbitrary output (e.g. the mask port of a fused SDPA).
std::shared_ptr<ov::op::v0::Parameter> find_mask_parameter(const ov::Output<ov::Node>& mask_source);

template <typename T>
void fill_tensor(ov::SoPtr<ov::ITensor> tensor, T fill_val, size_t offset = 0u) {
    T* tensor_data = tensor->data<T>();
    std::fill(tensor_data + offset, tensor_data + tensor->get_size(), fill_val);
}

void fill_tensor_bytes(ov::SoPtr<ov::ITensor> tensor, uint8_t fill_val);

template <class T>
typename std::underlying_type<T>::type _v(T&& t) {
    return static_cast<typename std::underlying_type<T>::type>(t);
}

template <class T>
class Delayed {
public:
    T& get() {
        return const_cast<T&>(get_impl());
    }
    // FIXME: since main purpose of this is to guard closure,
    // even const get should wait for the future to finish,
    // otherwise it's not ready yet (e.g. .size() method).
    const T& get() const {
        return get_impl();
    }
    T& unsafe_get() {
        return data;
    }
    void set_future(std::shared_future<void>& f) {
        future = f;
    }
    void wait() const {
        if (!done && future.valid()) {
            future.wait();
            done = true;
        }
    }

private:
    const T& get_impl() const {
        if (done)
            return data;
        if (future.valid()) {
            future.wait();
            done = true;
        }
        return data;
    }

    T data;
    std::shared_future<void> future;
    mutable bool done = false;
};

// Matches only the contiguous (non-block-split) past key param. Returns the layer index if matched.
std::optional<int> isPastKeyValuesKeyContiguous(const std::string& str);
// Matches only the contiguous (non-block-split) past value param. Returns the layer index if matched.
std::optional<int> isPastKeyValuesValueContiguous(const std::string& str);

// Backward compatibility: Alias for isPastKeyValuesKeyContiguous
inline std::optional<int> isPastKeyValuesKey(const std::string& str) {
    return isPastKeyValuesKeyContiguous(str);
}
// Backward compatibility: Alias for isPastKeyValuesValueContiguous
inline std::optional<int> isPastKeyValuesValue(const std::string& str) {
    return isPastKeyValuesValueContiguous(str);
}

std::optional<int> isPresentKeyValuesKey(const std::string& str);
std::optional<int> isPresentKeyValuesValue(const std::string& str);

// Matches any past key param: contiguous (past_key_values.N.key) or block-split (key_block_M).
bool isPastKeyParam(const std::string& str);
// Matches any past value param: contiguous or block-split.
bool isPastValueParam(const std::string& str);

// Detects DynamicQuantize scale/zp parameters for past KV cache.
// Returns true if the parameter name matches the DQ naming pattern.
bool isDQScaleOrZPKey(const std::string& str);
bool isDQScaleOrZPValue(const std::string& str);

// To remove input KV params that got badly matched in StatefulToStateless pass
// in Whisper model.
bool isRestoredPastKeyValueParam(const std::string& str);

}  // namespace util
}  // namespace npuw
}  // namespace ov
