// The MLX Runner over the vendored Gemma 4 text model (ADR-0004), mirroring
// golden `mlx_runner.py`'s split prefill (the port seam):
//
// Given `tokens`, an optional prefix cache and readout positions (all in the
// uncached suffix), with p0 = min(positions), p1 = max(positions):
//
// 1. Cache-only rows [nCached, p0): run the backbone and evaluate only the KV
//    cache state, never its output, so MLX's lazy graph drops the KV-consumer
//    layers (E2B: 15..34 reuse 0..14's K/V) and the vocab projection. Chunks of
//    `prefillStep` rows, the remainder chunk first.
// 2. Readout rows [p0, p1]: one forward through every layer, gather the hidden
//    rows at the readout positions and project only those through the tied
//    lm_head and the final soft-cap.
//
// First-token mode is therefore the cache-only prefill plus one L=1 pass.
// The text front end (tokenizer, chat template) is story 1.2's `ModelTokenizer`.
// Not yet ported: delta prefix cache (epic 3), images (epic 4), INT8/MoE (epic 5).

import Foundation
import MLX
import MLXLMCommon
import MLXNN

/// Not thread-safe: one decision at a time (the call records below are plain arrays).
public final class MLXRunner: Runner, @unchecked Sendable {
    public static let defaultModel = "mlx-community/gemma-4-E2B-it-qat-4bit"

    public let model: Gemma4Model
    public let modelId: String
    public let modelDirectory: URL
    /// The checkpoint's tokenizer and chat template (story 1.2).
    public let tokenizer: ModelTokenizer
    /// Chunk size of the cache-only rows (golden default 2048).
    public var prefillStep: Int = 2048
    /// Shapes seen by the vocab projection `[rows..., hidden]`, for tests (golden `head_calls`).
    public private(set) var headCalls: [[Int]] = []
    /// Each prefill as `(cachedTokens, computedTokens)` (golden `prefill_calls`).
    public private(set) var prefillCalls: [(cached: Int, computed: Int)] = []

    var text: Gemma4TextModel { model.languageModel }
    var backbone: Gemma4TextModelInner { model.languageModel.model }

    public init(model: Gemma4Model, modelId: String, modelDirectory: URL, tokenizer: ModelTokenizer) {
        self.model = model
        self.modelId = modelId
        self.modelDirectory = modelDirectory
        self.tokenizer = tokenizer
    }

    /// The snapshot directory of `repo` in the local Hugging Face cache
    /// (`$HF_HUB_CACHE`, `$HF_HOME/hub`, or `~/.cache/huggingface/hub`), never downloading.
    /// `revision` picks a snapshot; otherwise `refs/main`, else the only snapshot.
    public static func cachedSnapshot(repo: String = defaultModel, revision: String? = nil) -> URL? {
        let env = ProcessInfo.processInfo.environment
        let hub: URL
        if let c = env["HF_HUB_CACHE"] {
            hub = URL(fileURLWithPath: c)
        } else if let h = env["HF_HOME"] {
            hub = URL(fileURLWithPath: h).appendingPathComponent("hub")
        } else {
            hub = FileManager.default.homeDirectoryForCurrentUser
                .appendingPathComponent(".cache/huggingface/hub")
        }
        let repoDir = hub.appendingPathComponent("models--" + repo.replacingOccurrences(of: "/", with: "--"))
        let snapshots = repoDir.appendingPathComponent("snapshots")
        var rev = revision
        if rev == nil {
            rev = (try? String(contentsOf: repoDir.appendingPathComponent("refs/main"), encoding: .utf8))?
                .trimmingCharacters(in: .whitespacesAndNewlines)
        }
        if let rev {
            let dir = snapshots.appendingPathComponent(rev)
            return FileManager.default.fileExists(atPath: dir.appendingPathComponent("config.json").path) ? dir : nil
        }
        let all = (try? FileManager.default.contentsOfDirectory(at: snapshots, includingPropertiesForKeys: nil)) ?? []
        return all.count == 1 ? all[0] : nil
    }

    /// Load a Gemma 4 checkpoint from a local snapshot directory: config, the
    /// per-layer quantisation from `config.json`, and the safetensors weights
    /// (audio and vision tensors are dropped by `sanitize`), plus the tokenizer and
    /// chat template. With `expectedTokenizerSha256`, `tokenizer.json` must match it.
    public static func load(directory: URL, modelId: String = defaultModel,
                            expectedTokenizerSha256: String? = nil) async throws -> MLXRunner {
        let configURL = directory.appendingPathComponent("config.json")
        guard let data = try? Data(contentsOf: configURL) else {
            throw RunnerError.load("no config.json in \(directory.path)")
        }
        let decoder = JSONDecoder()
        let config = try decoder.decode(Gemma4Configuration.self, from: data)
        let base = try decoder.decode(BaseConfiguration.self, from: data)
        let model = Gemma4Model(config)
        try await loadWeights(
            modelDirectory: directory, model: model,
            perLayerQuantization: base.perLayerQuantization)
        let tokenizer = try await ModelTokenizer.load(directory: directory, expectedTokenizerSha256: expectedTokenizerSha256)
        return MLXRunner(model: model, modelId: modelId, modelDirectory: directory, tokenizer: tokenizer)
    }

    // MARK: - Runner

    /// Golden `tokenizer.encode(text, add_special_tokens=False)`.
    public func tokenize(_ text: String) -> [Int] { tokenizer.tokenize(text) }

    public var chatTemplate: ChatTemplate { tokenizer.chatTemplate }

    /// nil until the vision path lands (epic 4): a request with image blocks then
    /// fails to compile with golden's "no vision path" `PromptError`.
    public var imageTokens: ImageTokens? { nil }

    public func cachePrefix(_ tokens: [Int]) throws -> PrefixCache {
        let cache = try newCache()
        cacheOnly(tokens[...], cache: cache)
        return PrefixCache(tokens: tokens, state: cache)
    }

    public func prefill(_ tokens: [Int], prefixCache: PrefixCache?, readoutPositions: [Int]?) throws -> PrefillReadout {
        let (nCached, positions) = try checkPrefillArgs(
            tokens, prefixCache: prefixCache, readoutPositions: readoutPositions)
        let p0 = positions.min()!, p1 = positions.max()!
        let cache: [KVCache]
        if let prefixCache {
            guard let shared = prefixCache.state as? [KVCache] else {
                throw RunnerError.invalidArgument("prefix cache was not made by MLXRunner")
            }
            cache = shared.map { $0.copy() }  // a shared prefix is never mutated
        } else {
            cache = try newCache()
        }
        prefillCalls.append((nCached, p1 + 1 - nCached))

        cacheOnly(tokens[nCached ..< p0], cache: cache)
        let hidden = backbone(ids(tokens[p0 ... p1]), cache: cache)
        let rows = MLXArray(positions.map { Int32($0 - p0) })
        let picked = hidden[0..., rows, 0...]
        headCalls.append(picked.shape)
        let logits = text.head(picked)[0].asType(.float32)  // (k, vocab)
        let lse = logits.logSumExp(axis: -1)
        eval(logits, lse)

        let vocab = logits.dim(-1)
        let lses = lse.asArray(Float.self)
        var out: [LogitRow] = []
        for i in 0 ..< positions.count {
            let row = logits[i].asArray(Float.self)
            let total = Double(lses[i])
            out.append(LogitRow(count: vocab, lookup: { Double(row[$0]) }, logsumexp: { total }))
        }
        return PrefillReadout(positions: positions, logits: out, inputTokens: p1 + 1 - nCached, cachedTokens: nCached)
    }

    // MARK: - internals

    func newCache() throws -> [KVCache] {
        try model.newCache(parameters: nil)
    }

    private func ids(_ tokens: ArraySlice<Int>) -> MLXArray {
        MLXArray(tokens.map { Int32($0) })[.newAxis, 0...]
    }

    /// Advance `cache` over `tokens` evaluating only the KV state (golden `_cache_only`):
    /// chunks of `prefillStep`, the remainder first, so the last chunk is a full one.
    func cacheOnly(_ tokens: ArraySlice<Int>, cache: [KVCache]) {
        let n = tokens.count, step = prefillStep
        guard n > 0 else { return }
        let first = n % step == 0 ? step : n % step
        var bounds = [0]
        bounds.append(contentsOf: stride(from: first, to: n, by: step))
        bounds.append(n)
        let base = tokens.startIndex
        for (start, end) in zip(bounds, bounds.dropFirst()) {
            _ = backbone(ids(tokens[(base + start) ..< (base + end)]), cache: cache)  // output never evaluated
            eval(cache.flatMap { $0.state })
        }
    }
}
