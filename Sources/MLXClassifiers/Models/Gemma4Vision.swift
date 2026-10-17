// VENDORED — do not edit without updating this header (ADR-0004).
//
// Upstream: https://github.com/ml-explore/mlx-swift-lm
// Tag:      3.32.3 (commit 3b339ad6e3b3f44c8121ecff5131c7fd55e075e6)
// File:     Libraries/MLXVLM/Models/Gemma4.swift (the "// MARK: - Vision" section, lines 1531-1954,
//           and the helpers gemma4OneHot, gemma4RotateHalf, gemma4ApplyMultiDimensionalRoPE,
//           gemma4EnsureFusedSDPA, Gemma4RMSNormNoScale, Gemma4RMSNormZeroShift, Gemma4VisionConfiguration)
// sha256:   cc2567321563682f0ddccc8891ed66970f9ecc042580a12a8c606f91fc802fa0 (upstream file, before local changes)
// Vendored: 2026-10-17
// Licence:  MIT, Copyright (c) 2024 ml-explore (LICENSE-mlx-swift-lm beside this file). The golden
//           reference these numerics match is mlx-vlm 0.7.8 `mlx_vlm/models/gemma4/vision.py`
//           (MIT, Copyright (c) 2025 Prince Canuma, LICENSE-mlx-vlm beside this file).
//
// Local changes (story 4.1; numerics brought to golden `vision/encoder.py`):
//   - `private final class` -> internal `final class`, names unchanged; the text-side classes and the
//     `Gemma4` VLM model, processor and unified variants are not vendored
//   - `Gemma4VisionPatchEmbedder`: position embeddings summed as `hidden + (x + y)`, the order of
//     golden's one-hot matmul `.sum(axis=1)`, not upstream's `(hidden + x) + y`
//   - `Gemma4VisionConfiguration` decodes `rope_parameters.rope_theta` directly (no `StringOrNumber`)
//   - `gemma4ApplyMultiDimensionalRoPE`: only the multi-dimensional branch the tower uses
//   - `Gemma4VisionRMSNorm` / `Gemma4VisionRMSNormNoScale`: `pow(x, 2)` (golden `x_float**2`, MLX
//     `power`) instead of `square()`, which rounds differently in float32
//

import Foundation
import MLX
import MLXLMCommon
import MLXNN

// MARK: - Configuration

public struct Gemma4VisionConfiguration: Codable, Sendable {
    public let modelType: String
    public let hiddenLayers: Int
    public let hiddenSize: Int
    public let intermediateSize: Int
    public let attentionHeads: Int
    public let keyValueHeads: Int
    public let headDim: Int
    public let patchSize: Int
    public let rmsNormEps: Float
    public let defaultOutputLength: Int
    public let positionEmbeddingSize: Int
    public let poolingKernelSize: Int
    public let useClippedLinears: Bool
    public let standardize: Bool
    public let ropeTheta: Float

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case hiddenLayers = "num_hidden_layers"
        case hiddenSize = "hidden_size"
        case intermediateSize = "intermediate_size"
        case attentionHeads = "num_attention_heads"
        case keyValueHeads = "num_key_value_heads"
        case headDim = "head_dim"
        case patchSize = "patch_size"
        case rmsNormEps = "rms_norm_eps"
        case defaultOutputLength = "default_output_length"
        case positionEmbeddingSize = "position_embedding_size"
        case poolingKernelSize = "pooling_kernel_size"
        case useClippedLinears = "use_clipped_linears"
        case standardize
        case ropeParameters = "rope_parameters"
    }

    struct RopeParameters: Codable { var ropeTheta: Float?; enum CodingKeys: String, CodingKey { case ropeTheta = "rope_theta" } }

    public init(from decoder: any Swift.Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        modelType = try c.decodeIfPresent(String.self, forKey: .modelType) ?? "gemma4_vision"
        hiddenLayers = try c.decodeIfPresent(Int.self, forKey: .hiddenLayers) ?? 16
        hiddenSize = try c.decodeIfPresent(Int.self, forKey: .hiddenSize) ?? 768
        intermediateSize = try c.decodeIfPresent(Int.self, forKey: .intermediateSize) ?? 3072
        attentionHeads = try c.decodeIfPresent(Int.self, forKey: .attentionHeads) ?? 12
        keyValueHeads = try c.decodeIfPresent(Int.self, forKey: .keyValueHeads) ?? attentionHeads
        headDim = try c.decodeIfPresent(Int.self, forKey: .headDim) ?? 64
        patchSize = try c.decodeIfPresent(Int.self, forKey: .patchSize) ?? 16
        rmsNormEps = try c.decodeIfPresent(Float.self, forKey: .rmsNormEps) ?? 1e-6
        defaultOutputLength = try c.decodeIfPresent(Int.self, forKey: .defaultOutputLength) ?? 280
        positionEmbeddingSize = try c.decodeIfPresent(Int.self, forKey: .positionEmbeddingSize) ?? 10_240
        poolingKernelSize = try c.decodeIfPresent(Int.self, forKey: .poolingKernelSize) ?? 3
        useClippedLinears = try c.decodeIfPresent(Bool.self, forKey: .useClippedLinears) ?? false
        standardize = try c.decodeIfPresent(Bool.self, forKey: .standardize) ?? false
        ropeTheta = try c.decodeIfPresent(RopeParameters.self, forKey: .ropeParameters)?.ropeTheta ?? 100.0
    }

    public func encode(to encoder: any Swift.Encoder) throws {
        var c = encoder.container(keyedBy: CodingKeys.self)
        try c.encode(modelType, forKey: .modelType)
        try c.encode(hiddenSize, forKey: .hiddenSize)
    }
}

// MARK: - helpers

func gemma4OneHot(_ indices: MLXArray, numClasses: Int) -> MLXArray {
    expandedDimensions(indices, axis: -1) .== MLXArray(0 ..< numClasses)
}

func gemma4RotateHalf(_ x: MLXArray) -> MLXArray {
    let half = x.shape[x.shape.count - 1] / 2
    let x1 = x[.ellipsis, ..<half]
    let x2 = x[.ellipsis, half...]
    return concatenated([-x2, x1], axis: -1)
}

func gemma4ApplyMultiDimensionalRoPE(_ inputs: MLXArray, positions: MLXArray, baseFrequency: Float) -> MLXArray {
    let headDim = inputs.shape[inputs.ndim - 1]
    let numDimensions = positions.shape[positions.ndim - 1]
    let channelsPerDimension = 2 * (headDim / (2 * numDimensions))
    let halfPerDimension = channelsPerDimension / 2

    var parts: [MLXArray] = []
    parts.reserveCapacity(numDimensions)
    for d in 0 ..< numDimensions {
        let start = d * channelsPerDimension
        let end = start + channelsPerDimension
        let part = inputs[.ellipsis, start ..< end]

        let freqExponents = (2.0 / Float(channelsPerDimension)) * MLXArray(0 ..< halfPerDimension).asType(.float32)
        let timescale = MLX.pow(MLXArray(baseFrequency), freqExponents)
        let dimPositions = positions[.ellipsis, d ..< d + 1].asType(.float32)
        let sinusoid = dimPositions / timescale

        var cosValue = cos(sinusoid)
        var sinValue = sin(sinusoid)
        cosValue = concatenated([cosValue, cosValue], axis: -1).asType(inputs.dtype)
        sinValue = concatenated([sinValue, sinValue], axis: -1).asType(inputs.dtype)
        cosValue = expandedDimensions(cosValue, axis: 2)
        sinValue = expandedDimensions(sinValue, axis: 2)
        parts.append(part * cosValue + gemma4RotateHalf(part) * sinValue)
    }
    return concatenated(parts, axis: -1)
}

func gemma4EnsureFusedSDPA(queries: MLXArray, keys: MLXArray, values: MLXArray, scale: Float,
                           mask: MLXFast.ScaledDotProductAttentionMaskMode) -> MLXArray {
    let fusedDims = [64, 80, 128]
    let d = queries.dim(queries.ndim - 1)
    let target = fusedDims.first(where: { d <= $0 }) ?? d
    if target == d {
        return MLXFast.scaledDotProductAttention(queries: queries, keys: keys, values: values, scale: scale, mask: mask)
    }
    let pad: [IntOrPair] = [0, 0, 0, .init((0, target - d))]
    return MLXFast.scaledDotProductAttention(
        queries: MLX.padded(queries, widths: pad), keys: MLX.padded(keys, widths: pad),
        values: MLX.padded(values, widths: pad), scale: scale, mask: mask)[.ellipsis, ..<d]
}

final class Gemma4RMSNormNoScale: Module, UnaryLayer {
    let eps: Float
    init(eps: Float = 1e-6) { self.eps = eps; super.init() }
    func callAsFunction(_ x: MLXArray) -> MLXArray { MLXFast.rmsNorm(x, weight: MLXArray.mlxNone, eps: eps) }
}

final class Gemma4RMSNormZeroShift: Module, UnaryLayer {
    let eps: Float
    @ModuleInfo var weight: MLXArray
    init(dimensions: Int, eps: Float = 1e-6) {
        self.eps = eps
        self._weight.wrappedValue = MLXArray.ones([dimensions])
        super.init()
    }
    func callAsFunction(_ x: MLXArray) -> MLXArray { MLXFast.rmsNorm(x, weight: weight, eps: eps) }
}

// MARK: - Vision

final class Gemma4ClippableLinear: Module, UnaryLayer {
    let useClipping: Bool

    @ModuleInfo(key: "linear") var linear: Linear
    @ModuleInfo(key: "input_min") var inputMin: MLXArray?
    @ModuleInfo(key: "input_max") var inputMax: MLXArray?
    @ModuleInfo(key: "output_min") var outputMin: MLXArray?
    @ModuleInfo(key: "output_max") var outputMax: MLXArray?

    init(inFeatures: Int, outFeatures: Int, bias: Bool = false, useClipping: Bool) {
        self.useClipping = useClipping
        self._linear.wrappedValue = Linear(inFeatures, outFeatures, bias: bias)
        if useClipping {
            self._inputMin.wrappedValue = MLXArray(-Float.infinity)
            self._inputMax.wrappedValue = MLXArray(Float.infinity)
            self._outputMin.wrappedValue = MLXArray(-Float.infinity)
            self._outputMax.wrappedValue = MLXArray(Float.infinity)
        }
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let clippedInput = if let inputMin, let inputMax { clip(x, min: inputMin, max: inputMax) } else { x }
        let projected = linear(clippedInput)
        if let outputMin, let outputMax { return clip(projected, min: outputMin, max: outputMax) }
        return projected
    }
}

final class Gemma4VisionRMSNorm: Module, UnaryLayer {
    let eps: Float
    @ModuleInfo var weight: MLXArray

    init(dimensions: Int, eps: Float = 1e-6) {
        self.eps = eps
        self._weight.wrappedValue = MLXArray.ones([dimensions])
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let xFloat = x.asType(.float32)
        let variance = mean(MLX.pow(xFloat, MLXArray(Float(2))), axis: -1, keepDims: true)  // golden `x_float**2`
        let normalized = xFloat * rsqrt(variance + eps)
        return (normalized * weight.asType(.float32)).asType(x.dtype)
    }
}

final class Gemma4VisionRMSNormNoScale: Module, UnaryLayer {
    let eps: Float
    init(eps: Float = 1e-6) { self.eps = eps; super.init() }
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let xFloat = x.asType(.float32)
        let variance = mean(MLX.pow(xFloat, MLXArray(Float(2))), axis: -1, keepDims: true)  // golden `x_float**2`
        return (xFloat * rsqrt(variance + eps)).asType(x.dtype)
    }
}

final class Gemma4VisionAttention: Module {
    let numHeads: Int
    let numKVHeads: Int
    let headDim: Int
    let hiddenSize: Int
    let ropeBaseFrequency: Float

    @ModuleInfo(key: "q_proj") var qProj: Gemma4ClippableLinear
    @ModuleInfo(key: "k_proj") var kProj: Gemma4ClippableLinear
    @ModuleInfo(key: "v_proj") var vProj: Gemma4ClippableLinear
    @ModuleInfo(key: "o_proj") var oProj: Gemma4ClippableLinear
    @ModuleInfo(key: "q_norm") var qNorm: Gemma4VisionRMSNorm
    @ModuleInfo(key: "k_norm") var kNorm: Gemma4VisionRMSNorm
    @ModuleInfo(key: "_v_norm") var vNorm: Gemma4VisionRMSNormNoScale

    init(config: Gemma4VisionConfiguration) {
        self.numHeads = config.attentionHeads
        self.numKVHeads = config.keyValueHeads
        self.headDim = config.headDim
        self.hiddenSize = config.hiddenSize
        self.ropeBaseFrequency = config.ropeTheta
        let clip = config.useClippedLinears
        self._qProj.wrappedValue = Gemma4ClippableLinear(inFeatures: hiddenSize, outFeatures: numHeads * headDim, useClipping: clip)
        self._kProj.wrappedValue = Gemma4ClippableLinear(inFeatures: hiddenSize, outFeatures: numKVHeads * headDim, useClipping: clip)
        self._vProj.wrappedValue = Gemma4ClippableLinear(inFeatures: hiddenSize, outFeatures: numKVHeads * headDim, useClipping: clip)
        self._oProj.wrappedValue = Gemma4ClippableLinear(inFeatures: numHeads * headDim, outFeatures: hiddenSize, useClipping: clip)
        self._qNorm.wrappedValue = Gemma4VisionRMSNorm(dimensions: headDim, eps: config.rmsNormEps)
        self._kNorm.wrappedValue = Gemma4VisionRMSNorm(dimensions: headDim, eps: config.rmsNormEps)
        self._vNorm.wrappedValue = Gemma4VisionRMSNormNoScale(eps: config.rmsNormEps)
        super.init()
    }

    func callAsFunction(_ x: MLXArray, positions: MLXArray, mask: MLXArray? = nil) -> MLXArray {
        let (batch, length) = (x.dim(0), x.dim(1))
        var queries = qProj(x).reshaped(batch, length, numHeads, headDim)
        var keys = kProj(x).reshaped(batch, length, numKVHeads, headDim)
        var values = vProj(x).reshaped(batch, length, numKVHeads, headDim)

        queries = qNorm(queries)
        keys = kNorm(keys)
        values = vNorm(values)

        queries = gemma4ApplyMultiDimensionalRoPE(queries, positions: positions, baseFrequency: ropeBaseFrequency)
        keys = gemma4ApplyMultiDimensionalRoPE(keys, positions: positions, baseFrequency: ropeBaseFrequency)

        queries = queries.transposed(0, 2, 1, 3)
        keys = keys.transposed(0, 2, 1, 3)
        values = values.transposed(0, 2, 1, 3)

        let attentionMask: MLXFast.ScaledDotProductAttentionMaskMode = if let mask { .array(mask) } else { .none }
        let output = gemma4EnsureFusedSDPA(queries: queries, keys: keys, values: values, scale: 1.0, mask: attentionMask)
            .transposed(0, 2, 1, 3)
            .reshaped(batch, length, -1)
        return oProj(output)
    }
}

final class Gemma4VisionMLP: Module, UnaryLayer {
    @ModuleInfo(key: "gate_proj") var gateProj: Gemma4ClippableLinear
    @ModuleInfo(key: "up_proj") var upProj: Gemma4ClippableLinear
    @ModuleInfo(key: "down_proj") var downProj: Gemma4ClippableLinear

    init(config: Gemma4VisionConfiguration) {
        let clip = config.useClippedLinears
        self._gateProj.wrappedValue = Gemma4ClippableLinear(inFeatures: config.hiddenSize, outFeatures: config.intermediateSize, useClipping: clip)
        self._upProj.wrappedValue = Gemma4ClippableLinear(inFeatures: config.hiddenSize, outFeatures: config.intermediateSize, useClipping: clip)
        self._downProj.wrappedValue = Gemma4ClippableLinear(inFeatures: config.intermediateSize, outFeatures: config.hiddenSize, useClipping: clip)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        downProj(geluApproximate(gateProj(x)) * upProj(x))
    }
}

final class Gemma4VisionTransformerBlock: Module {
    @ModuleInfo(key: "self_attn") var selfAttention: Gemma4VisionAttention
    @ModuleInfo var mlp: Gemma4VisionMLP
    @ModuleInfo(key: "input_layernorm") var inputLayerNorm: Gemma4RMSNormZeroShift
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayerNorm: Gemma4RMSNormZeroShift
    @ModuleInfo(key: "pre_feedforward_layernorm") var preFeedforwardLayerNorm: Gemma4RMSNormZeroShift
    @ModuleInfo(key: "post_feedforward_layernorm") var postFeedforwardLayerNorm: Gemma4RMSNormZeroShift

    init(config: Gemma4VisionConfiguration) {
        self._selfAttention.wrappedValue = Gemma4VisionAttention(config: config)
        self._mlp.wrappedValue = Gemma4VisionMLP(config: config)
        self._inputLayerNorm.wrappedValue = Gemma4RMSNormZeroShift(dimensions: config.hiddenSize, eps: config.rmsNormEps)
        self._postAttentionLayerNorm.wrappedValue = Gemma4RMSNormZeroShift(dimensions: config.hiddenSize, eps: config.rmsNormEps)
        self._preFeedforwardLayerNorm.wrappedValue = Gemma4RMSNormZeroShift(dimensions: config.hiddenSize, eps: config.rmsNormEps)
        self._postFeedforwardLayerNorm.wrappedValue = Gemma4RMSNormZeroShift(dimensions: config.hiddenSize, eps: config.rmsNormEps)
        super.init()
    }

    func callAsFunction(_ x: MLXArray, positions: MLXArray, mask: MLXArray?) -> MLXArray {
        let normed = inputLayerNorm(x)
        let attentionOutput = selfAttention(normed, positions: positions, mask: mask)
        let h = x + postAttentionLayerNorm(attentionOutput)
        let ff = mlp(preFeedforwardLayerNorm(h))
        return h + postFeedforwardLayerNorm(ff)
    }
}

final class Gemma4VisionPatchEmbedder: Module {
    let patchSize: Int
    let hiddenSize: Int
    let positionEmbeddingSize: Int

    @ModuleInfo(key: "input_proj") var inputProjection: Linear
    @ModuleInfo(key: "position_embedding_table") var positionEmbeddingTable: MLXArray

    init(config: Gemma4VisionConfiguration) {
        self.patchSize = config.patchSize
        self.hiddenSize = config.hiddenSize
        self.positionEmbeddingSize = config.positionEmbeddingSize
        self._inputProjection.wrappedValue = Linear(3 * patchSize * patchSize, hiddenSize, bias: false)
        self._positionEmbeddingTable.wrappedValue = MLXArray.ones([2, positionEmbeddingSize, hiddenSize])
        super.init()
    }

    private func patchify(_ pixelValues: MLXArray) -> MLXArray {
        let (batch, channels, height, width) = (pixelValues.dim(0), pixelValues.dim(1), pixelValues.dim(2), pixelValues.dim(3))
        let patchesH = height / patchSize
        let patchesW = width / patchSize
        var patches = pixelValues.reshaped(batch, channels, patchesH, patchSize, patchesW, patchSize)
        patches = patches.transposed(0, 2, 4, 3, 5, 1)
        patches = patches.reshaped(batch, patchesH * patchesW, channels * patchSize * patchSize)
        patches = 2 * (patches - 0.5)
        return inputProjection(patches.asType(inputProjection.weight.dtype))
    }

    func callAsFunction(_ pixelValues: MLXArray, patchPositions: MLXArray) -> MLXArray {
        let hiddenStates = patchify(pixelValues)
        let batch = patchPositions.dim(0)
        let seqLen = patchPositions.dim(1)
        let xIndices = patchPositions[0..., 0..., 0].flattened().asType(.int32)
        let yIndices = patchPositions[0..., 0..., 1].flattened().asType(.int32)
        let xEmbeddings = take(positionEmbeddingTable[0], xIndices, axis: 0).reshaped(batch, seqLen, hiddenSize)
        let yEmbeddings = take(positionEmbeddingTable[1], yIndices, axis: 0).reshaped(batch, seqLen, hiddenSize)
        return hiddenStates + (xEmbeddings + yEmbeddings)  // MLXClassifiers: golden's summation order
    }
}

final class Gemma4VisionPooler: Module {
    let hiddenSize: Int
    let rootHiddenSize: Float

    init(config: Gemma4VisionConfiguration) {
        self.hiddenSize = config.hiddenSize
        self.rootHiddenSize = pow(Float(config.hiddenSize), 0.5)
        super.init()
    }

    func callAsFunction(_ hiddenStates: MLXArray, patchPositions: MLXArray, patchesW: Int, outputLength: Int) -> MLXArray {
        let scale = MLXArray(rootHiddenSize, dtype: hiddenStates.dtype)
        let numPatches = hiddenStates.dim(1)
        let length = max(outputLength, 1)
        if numPatches <= length {
            return hiddenStates * scale
        }
        let positions = patchPositions[0]
        let kernel = max(Int(sqrt(Double(numPatches / length))), 1)
        let divisor = kernel * kernel
        let kernelIndices = floor(positions.asType(.float32) / Float(kernel)).asType(.int32)
        let flatKernel = kernelIndices[0..., 0] + MLXArray(Int32(max(patchesW / kernel, 1))) * kernelIndices[0..., 1]
        let weights = gemma4OneHot(flatKernel, numClasses: length).asType(.float32) / Float(divisor)
        let output = einsum("lL,bld->bLd", weights, hiddenStates).asType(hiddenStates.dtype)
        return output * scale
    }
}

final class Gemma4VisionTransformerModel: Module {
    @ModuleInfo(key: "layers") var layers: [Gemma4VisionTransformerBlock]

    init(config: Gemma4VisionConfiguration) {
        self._layers.wrappedValue = (0 ..< config.hiddenLayers).map { _ in Gemma4VisionTransformerBlock(config: config) }
        super.init()
    }

    func callAsFunction(_ hiddenStates: MLXArray, positions: MLXArray, mask: MLXArray?) -> MLXArray {
        var h = hiddenStates
        for layer in layers { h = layer(h, positions: positions, mask: mask) }
        return h
    }
}

public final class Gemma4VisionModel: Module {
    let config: Gemma4VisionConfiguration
    let patchSize: Int
    let poolingKernelSize: Int

    @ModuleInfo(key: "patch_embedder") var patchEmbedder: Gemma4VisionPatchEmbedder
    @ModuleInfo(key: "encoder") var encoder: Gemma4VisionTransformerModel
    @ModuleInfo(key: "pooler") var pooler: Gemma4VisionPooler
    @ModuleInfo(key: "std_bias") var standardizationBias: MLXArray?
    @ModuleInfo(key: "std_scale") var standardizationScale: MLXArray?

    init(config: Gemma4VisionConfiguration) {
        self.config = config
        self.patchSize = config.patchSize
        self.poolingKernelSize = config.poolingKernelSize
        self._patchEmbedder.wrappedValue = Gemma4VisionPatchEmbedder(config: config)
        self._encoder.wrappedValue = Gemma4VisionTransformerModel(config: config)
        self._pooler.wrappedValue = Gemma4VisionPooler(config: config)
        if config.standardize {
            self._standardizationBias.wrappedValue = MLXArray.zeros([config.hiddenSize])
            self._standardizationScale.wrappedValue = MLXArray.ones([config.hiddenSize])
        }
        super.init()
    }

    private func patchPositions(batch: Int, patchesH: Int, patchesW: Int) -> MLXArray {
        let grids = meshGrid([MLXArray.arange(patchesW, dtype: .int32), MLXArray.arange(patchesH, dtype: .int32)])
        let positions = stacked([grids[0].flattened(), grids[1].flattened()], axis: 1).reshaped(1, patchesH * patchesW, 2)
        return batch == 1 ? positions : broadcast(positions, to: [batch, patchesH * patchesW, 2])
    }

    /// `[B, N, hidden]` soft tokens for `[B, 3, H, W]` pixels, `N = (H/16) * (W/16) / 9`.
    func callAsFunction(_ pixelValues: MLXArray) -> MLXArray {
        let pixels = pixelValues.ndim == 3 ? expandedDimensions(pixelValues, axis: 0) : pixelValues
        let batch = pixels.dim(0)
        let patchesH = pixels.dim(2) / patchSize
        let patchesW = pixels.dim(3) / patchSize
        let numPatches = patchesH * patchesW
        let outputLength = max(numPatches / (poolingKernelSize * poolingKernelSize), 1)

        let patchPositions = patchPositions(batch: batch, patchesH: patchesH, patchesW: patchesW)
        var hiddenStates = patchEmbedder(pixels, patchPositions: patchPositions)
        hiddenStates = encoder(hiddenStates, positions: patchPositions, mask: nil)  // golden's fast path: no all-zero mask
        hiddenStates = pooler(hiddenStates, patchPositions: patchPositions, patchesW: patchesW, outputLength: outputLength)
        if let standardizationBias, let standardizationScale {
            hiddenStates = (hiddenStates - standardizationBias) * standardizationScale
        }
        return hiddenStates
    }
}

public final class Gemma4MultimodalEmbedder: Module, UnaryLayer {
    @ModuleInfo(key: "embedding_projection") var embeddingProjection: Linear
    @ModuleInfo(key: "embedding_pre_projection_norm") var embeddingPreProjectionNorm: Gemma4RMSNormNoScale

    init(embeddingDim: Int, textHiddenSize: Int, eps: Float) {
        self._embeddingProjection.wrappedValue = Linear(embeddingDim, textHiddenSize, bias: false)
        self._embeddingPreProjectionNorm.wrappedValue = Gemma4RMSNormNoScale(eps: eps)
        super.init()
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        embeddingProjection(embeddingPreProjectionNorm(x))
    }
}
