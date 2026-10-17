// VENDORED — do not edit without updating this header (ADR-0004).
//
// Upstream: https://github.com/ml-explore/mlx-swift-lm
// Tag:      3.32.3 (commit 3b339ad6e3b3f44c8121ecff5131c7fd55e075e6)
// File:     Libraries/MLXLLM/Models/Gemma4.swift
// sha256:   91259b024bb5ac8f408327b6f9544d8327cc6e5d335702b8403b42500b159ad4 (upstream file, before local changes)
// Vendored: 2026-10-17
// Licence:  MIT, Copyright (c) 2024 ml-explore (LICENSE-mlx-swift-lm beside this file)
//
// Local changes (the minimum the runner needs, ADR-0004):
//   - dropped `@_spi(GemmaEncoder)` on `languageModel` (plain public)
//   - conforms to MLXLMCommon `BaseLanguageModel` instead of MLXLLM `LLMModel`, so MLXLMCommon `loadWeights` loads it
//   - LoRA and chat-convention extensions removed
//

//
//  Gemma4.swift
//  mlx-swift-lm
//
//  Port of https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/models/gemma4.py

import Foundation
import MLX
import MLXLMCommon
import MLXNN

// MARK: - Configuration

/// Configuration for the `"gemma4"` model_type.
/// This is a thin wrapper around `Gemma4TextConfiguration` that handles the
/// nested `text_config` structure from HuggingFace model configs.
public struct Gemma4Configuration: Codable, Sendable {
    var modelType: String = "gemma4"
    var textConfig: Gemma4TextConfiguration
    var vocabSize: Int = 262144

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case textConfig = "text_config"
        case vocabSize = "vocab_size"
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        self.modelType = try container.decodeIfPresent(String.self, forKey: .modelType) ?? "gemma4"
        self.vocabSize = try container.decodeIfPresent(Int.self, forKey: .vocabSize) ?? 262144

        // If text_config is present, decode from it; otherwise treat entire config as text config
        if let textConfig = try container.decodeIfPresent(
            Gemma4TextConfiguration.self, forKey: .textConfig)
        {
            self.textConfig = textConfig
            // Propagate vocab_size into text config
            self.textConfig.vocabSize = self.vocabSize
        } else {
            self.textConfig = try Gemma4TextConfiguration(from: decoder)
        }
    }
}

// MARK: - Model

public class Gemma4Model: Module, BaseLanguageModel, KVCacheDimensionProvider {
    public var vocabularySize: Int { languageModel.vocabularySize }
    public var kvHeads: [Int] { languageModel.kvHeads }

    /// The text model, exposed at `@_spi(GemmaEncoder)` scope.
    ///
    /// This is the type the model factory actually produces for `gemma4_unified`, so an
    /// encoder-style client tap cannot reach `Gemma4TextModel` without it — exposing only
    /// the inner types is not sufficient for a real load path.
    @ModuleInfo(key: "language_model") public var languageModel: Gemma4TextModel

    public init(_ config: Gemma4Configuration) {
        self._languageModel.wrappedValue = Gemma4TextModel(config.textConfig)
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        languageModel(inputs, cache: cache)
    }

    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        var sanitized = [String: MLXArray]()
        for (key, value) in weights {
            var k = key

            // Strip "model." prefix
            let startsWithModel = k.hasPrefix("model.")
            k = k.replacingOccurrences(of: "model.", with: "", options: .anchored)

            // Skip vision/audio weights. `vision_embedder` is the gemma4_unified
            // (12B) encoder-free vision module; without it, loading a multimodal
            // gemma4_unified checkpoint (e.g. mlx-community/gemma-4-12B-it-4bit)
            // through the text path fails with `Unhandled keys ["vision_embedder"]`.
            if k.hasPrefix("vision_tower") || k.hasPrefix("multi_modal_projector")
                || k.hasPrefix("audio_tower") || k.hasPrefix("embed_audio")
                || k.hasPrefix("embed_vision") || k.hasPrefix("vision_embedder")
            {
                continue
            }

            if !startsWithModel {
                sanitized[k] = value
                continue
            }

            // Remap language_model keys
            if k.hasPrefix("language_model") {
                k = k.replacingOccurrences(
                    of: "language_model.", with: "language_model.model.", options: .anchored)
            }

            sanitized[k] = value
        }

        return languageModel.sanitize(weights: sanitized)
    }

    public func newCache(parameters: GenerateParameters?) throws -> [any KVCache] {
        try languageModel.newCache(parameters: parameters)
    }
}
