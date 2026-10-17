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
//
// Delta prefix caching (story 3.1, golden story 3.4): `promptCache` (a
// `PromptCacheStore`, on by default; nil turns it off) keeps the KV state after
// step 1 of every prefill, keyed by `tokens[..<p0]`. The next `prefill` looks up
// the stored sequence sharing the longest common token prefix with its own tokens
// (capped at p0), rewinds a copy to that length (`rewindCache`), and runs step 1
// over the delta only. `cachedTokens` reports the hit length. A declared prefix
// cache still works and is used when it is the longer hit.
//
// Images (story 4.1, golden story 6.3): `prefill(..., images:)` / `cachePrefix(...,
// images:)` take the `ImageSlot` runs compilation recorded. A row range that
// overlaps an image runs on input embeddings: `embedTokens(ids) * embedScale` with
// the image's projected soft tokens in its rows (`VisionEncoder`, memoised per PNG
// sha256 and budget), per-layer inputs from the ids with image rows set to the pad
// id 0, then the same decoder layers, so the cache-only / readout split is
// unchanged. Rows without an image take the token-id path. The delta store keys on
// `keyedTokens` (image rows carry the image's hash and budget), and a hit is never
// left inside an image run.
// Gemma 4 26B-A4B (story 7.1): `use_bidirectional_attention: "vision"` makes each
// image block attend to itself both ways (golden `vision/merge.py`): image rows pass
// `blocks` to the backbone, and cache-only chunk bounds are moved off image runs.
// Not yet ported: the INT8 kernels on the MoE experts and golden's MoE gather kernel.

import Foundation
import MLX
import MLXLMCommon
import MLXNN

/// Not thread-safe: one decision at a time (the call records below are plain arrays).
public final class MLXRunner: DeltaCachingRunner, @unchecked Sendable {
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
    public internal(set) var prefillCalls: [(cached: Int, computed: Int)] = []
    /// Delta prefix cache (golden story 3.4); nil turns it off.
    public var promptCache: PromptCacheStore? = PromptCacheStore()
    /// The image placeholder convention from the snapshot's JSON (nil: no vision path).
    public let imageTokens: ImageTokens?
    /// The vision tower's encoder with its soft-token memo (nil: no vision path).
    public let vision: VisionEncoder?
    /// Image blocks attend bidirectionally (text config `use_bidirectional_attention: "vision"`, the 26B).
    public let visionBidirectional: Bool

    var text: Gemma4TextModel { model.languageModel }
    var backbone: Gemma4TextModelInner { model.languageModel.model }

    public init(model: Gemma4Model, modelId: String, modelDirectory: URL, tokenizer: ModelTokenizer,
                imageTokens: ImageTokens? = nil, visionBidirectional: Bool = false) {
        self.visionBidirectional = visionBidirectional
        self.model = model
        self.modelId = modelId
        self.modelDirectory = modelDirectory
        self.tokenizer = tokenizer
        self.imageTokens = model.visionTower == nil ? nil : imageTokens
        self.vision = model.visionTower == nil ? nil : VisionEncoder(model: model)
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
            hub = homeDirectory().appendingPathComponent(".cache/huggingface/hub")
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
    /// (audio tensors are dropped by `sanitize`; the vision tower loads with the rest),
    /// plus the tokenizer, chat template and image placeholder convention. With
    /// `expectedTokenizerSha256`, `tokenizer.json` must match it. `includeVision: false`
    /// builds the text model alone: the vision tower's weights are dropped by `sanitize`
    /// like the audio tower's, and the runner has no vision path (story 6.1, phone memory).
    public static func load(directory: URL, modelId: String = defaultModel,
                            expectedTokenizerSha256: String? = nil,
                            includeVision: Bool = true) async throws -> MLXRunner {
        let configURL = directory.appendingPathComponent("config.json")
        guard let data = try? Data(contentsOf: configURL) else {
            throw RunnerError.load("no config.json in \(directory.path)")
        }
        let decoder = JSONDecoder()
        var config = try decoder.decode(Gemma4Configuration.self, from: data)
        if !includeVision { config.visionConfig = nil }
        let base = try decoder.decode(BaseConfiguration.self, from: data)
        var bidirectional = false
        if case .string("vision") = try JSONValue.parse(data)["text_config"]?["use_bidirectional_attention"] {
            bidirectional = true
        }
        let model = Gemma4Model(config)
        try await loadWeights(
            modelDirectory: directory, model: model,
            perLayerQuantization: base.perLayerQuantization)
        let tokenizer = try await ModelTokenizer.load(directory: directory, expectedTokenizerSha256: expectedTokenizerSha256)
        return MLXRunner(model: model, modelId: modelId, modelDirectory: directory, tokenizer: tokenizer,
                         imageTokens: try ImageTokens.fromConfig(modelDirectory: directory),
                         visionBidirectional: bidirectional)
    }

    // MARK: - Runner

    /// Golden `tokenizer.encode(text, add_special_tokens=False)`.
    public func tokenize(_ text: String) -> [Int] { tokenizer.tokenize(text) }

    public var chatTemplate: ChatTemplate { tokenizer.chatTemplate }

    public func cachePrefix(_ tokens: [Int]) throws -> PrefixCache {
        try cachePrefix(tokens, images: [], delta: true)
    }

    public func prefill(_ tokens: [Int], prefixCache: PrefixCache?, readoutPositions: [Int]?) throws -> PrefillReadout {
        try prefill(tokens, prefixCache: prefixCache, readoutPositions: readoutPositions, images: [], delta: true)
    }

    public func cachePrefix(_ tokens: [Int], images: [ImageSlot]) throws -> PrefixCache {
        try cachePrefix(tokens, images: images, delta: true)
    }

    public func prefill(_ tokens: [Int], prefixCache: PrefixCache?, readoutPositions: [Int]?,
                        images: [ImageSlot]) throws -> PrefillReadout {
        try prefill(tokens, prefixCache: prefixCache, readoutPositions: readoutPositions, images: images, delta: true)
    }

    /// Golden `cache_prefix(tokens, delta=, images=)`: the delta store may serve part
    /// of `tokens`, and the result is stored when this call added rows.
    public func cachePrefix(_ tokens: [Int], images: [ImageSlot], delta: Bool) throws -> PrefixCache {
        let slots = try checkSlots(tokens, images)
        let key = try keyedTokens(tokens, slots)
        let (nCached, cache) = try start(key, maxLen: tokens.count, prefixCache: nil, delta: delta, slots: slots)
        try cacheOnly(tokens[nCached...], cache: cache, slots: slots, base: nCached)
        store(key, cache: cache, nCached: nCached, delta: delta)
        return PrefixCache(tokens: tokens, state: cache, images: slots)
    }

    /// Golden `prefill(..., delta=, images=)`.
    public func prefill(_ tokens: [Int], prefixCache: PrefixCache?, readoutPositions: [Int]?, images: [ImageSlot],
                        delta: Bool) throws -> PrefillReadout {
        let (_, positions) = try checkPrefillArgs(
            tokens, prefixCache: prefixCache, readoutPositions: readoutPositions, images: images)
        let slots = try checkSlots(tokens, images)
        let p0 = positions.min()!, p1 = positions.max()!
        if slots.contains(where: { ($0.start < p0 && p0 < $0.end) || (p0 <= $0.start && $0.start <= p1) }) {
            throw RunnerError.invalidArgument("readout positions must not fall inside or before an image run")
        }
        let key = try keyedTokens(tokens, slots)
        let (nCached, cache) = try start(key, maxLen: p0, prefixCache: prefixCache, delta: delta, slots: slots)
        prefillCalls.append((nCached, p1 + 1 - nCached))

        try cacheOnly(tokens[nCached ..< p0], cache: cache, slots: slots, base: nCached)
        store(Array(key[..<p0]), cache: cache, nCached: nCached, delta: delta)
        let hidden = try rows(tokens, p0, p1 + 1, cache: cache, slots: slots)
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

    /// Golden `_start`: `(nCached, working cache)`, the longer of the declared prefix
    /// and the delta store's hit (both copies), else an empty cache.
    func start(_ key: [Int], maxLen: Int, prefixCache: PrefixCache?, delta: Bool,
               slots: [ImageSlot] = []) throws -> (Int, [KVCache]) {
        let have = prefixCache?.count ?? 0
        if delta, let store = promptCache {
            let rewind: PromptCacheStore.Rewind = { state, stored, hit in
                rewindCache(state as! [KVCache], storedLen: stored, hitLen: hit)
            }
            var (hit, state) = store.lookup(key, rewind: rewind, maxLen: maxLen, minLen: have)
            if state != nil, let inside = slots.first(where: { $0.start < hit && hit < $0.end }) {
                (hit, state) = store.lookup(key, rewind: rewind, maxLen: inside.start, minLen: have)
            }
            if let state = state as? [KVCache] { return (hit, state) }
        }
        if let prefixCache {
            guard let shared = prefixCache.state as? [KVCache] else {
                throw RunnerError.invalidArgument("prefix cache was not made by MLXRunner")
            }
            return (have, shared.map { $0.copy() })  // a shared prefix is never mutated
        }
        return (0, try newCache())
    }

    /// Golden `_store`: keep a copy of `cache` (covering `key`) when this call added rows.
    func store(_ key: [Int], cache: [KVCache], nCached: Int, delta: Bool) {
        guard delta, let store = promptCache, key.count > nCached else { return }
        let snapshot = cache.map { $0.copy() }
        store.insert(key, state: snapshot, nbytes: cacheNbytes(snapshot))
    }

    private func ids(_ tokens: ArraySlice<Int>) -> MLXArray {
        MLXArray(tokens.map { Int32($0) })[.newAxis, 0...]
    }

    /// Advance `cache` over `tokens` evaluating only the KV state (golden `_cache_only`):
    /// chunks of `prefillStep`, the remainder first, so the last chunk is a full one.
    /// `slots` are image runs in absolute positions; `base` is the absolute position
    /// of `tokens.first` (the slice's own indices are absolute too).
    func cacheOnly(_ tokens: ArraySlice<Int>, cache: [KVCache], slots: [ImageSlot] = [], base: Int = 0) throws {
        let n = tokens.count, step = prefillStep
        guard n > 0 else { return }
        let first = n % step == 0 ? step : n % step
        var bounds = [0]
        bounds.append(contentsOf: stride(from: first, to: n, by: step))
        bounds.append(n)
        if visionBidirectional && !slots.isEmpty {
            bounds = boundsOffRuns(bounds, slots.map { ($0.start - base, $0.end - base) })
        }
        let all = Array(tokens), lo = tokens.startIndex
        let local = slots.filter { $0.end > base && $0.start < base + n }
        for (start, end) in zip(bounds, bounds.dropFirst()) {
            if local.isEmpty {
                _ = backbone(ids(tokens[(lo + start) ..< (lo + end)]), cache: cache)  // output never evaluated
            } else {
                _ = try rows(all, start, end, cache: cache, slots: local, base: base)
            }
            eval(cache.flatMap { $0.state })
        }
    }

    /// Golden `_check_images`: the slots sorted, and each a run of image-token ids with PNG bytes.
    func checkSlots(_ tokens: [Int], _ images: [ImageSlot]) throws -> [ImageSlot] {
        let slots = try checkImages(tokens, images)
        if slots.isEmpty { return slots }
        guard let spec = imageTokens else { throw RunnerError.invalidArgument("\(modelId) has no vision path") }
        for s in slots {
            if s.data.isEmpty { throw RunnerError.invalidArgument("image slot at \(s.start) carries no PNG bytes") }
            if tokens[s.start ..< s.end].contains(where: { $0 != spec.imageTokenId }) {
                throw RunnerError.invalidArgument("image run at \(s.start) is not all image-token ids")
            }
        }
        return slots
    }

    /// Golden `_image_features`: `[length, hidden]` soft tokens of `slot` (encoded once per sha256 and budget).
    func imageFeatures(_ slot: ImageSlot) throws -> MLXArray {
        guard let vision else { throw RunnerError.invalidArgument("\(modelId) has no vision path") }
        let budget = slot.maxSoftTokens ?? imageTokens?.maxSoftTokens ?? SoftTokenBudget.defaultBudget
        let feats = try vision.encodePNG(slot.data, sha256: slot.sha256, maxSoftTokens: budget)
        guard feats.dim(0) == slot.length else {
            throw RunnerError.invalidArgument("image encodes to \(feats.dim(0)) soft tokens, run has \(slot.length)")
        }
        return feats
    }

    /// Golden `_rows`: the backbone over `tokens[start ..< end]` (prompt rows `base + start ..`),
    /// token ids when no image overlaps, else merged input embeddings (golden `vision/merge.py`):
    /// text rows `embedTokens(ids) * embedScale`, image rows the soft tokens cast to that dtype,
    /// and the per-layer inputs from the ids with every image row set to the pad id 0.
    func rows(_ tokens: [Int], _ start: Int, _ end: Int, cache: [KVCache], slots: [ImageSlot],
              base: Int = 0) throws -> MLXArray {
        let segment = tokens[start ..< end]
        let lo = base + start, hi = base + end
        var pieces: [(off: Int, n: Int, feats: MLXArray)] = []
        for s in slots {
            let a = max(s.start, lo), b = min(s.end, hi)
            if a < b { pieces.append((a - lo, b - a, try imageFeatures(s)[(a - s.start) ..< (b - s.start)])) }
        }
        if pieces.isEmpty { return backbone(ids(segment), cache: cache) }
        pieces.sort { $0.off < $1.off }
        let text = backbone.embedTokens(ids(segment)) * backbone.embedScale
        var parts: [MLXArray] = [], at = 0
        var plIds = segment.map { Int32($0) }
        for p in pieces {
            if p.off > at { parts.append(text[0..., at ..< p.off]) }
            parts.append(p.feats.asType(text.dtype)[.newAxis])
            at = p.off + p.n
            for i in p.off ..< at { plIds[i] = 0 }
        }
        if at < text.dim(1) { parts.append(text[0..., at...]) }
        let h = concatenated(parts, axis: 1)
        var blocks: MLXArray? = nil
        if visionBidirectional {
            var b = [Int32](repeating: -1, count: segment.count)
            for (k, p) in pieces.enumerated() { for i in p.off ..< (p.off + p.n) { b[i] = Int32(k) } }
            blocks = MLXArray(b)
        }
        return backbone(MLXArray(plIds)[.newAxis, 0...], cache: cache, inputEmbeddings: h, blocks: blocks)
    }
}

/// Golden `_bounds_off_runs`: chunk boundaries with any that fall strictly inside a run
/// moved to its start (or its end, when the start is the previous boundary); duplicates dropped.
func boundsOffRuns(_ bounds: [Int], _ runs: [(Int, Int)]) -> [Int] {
    var out: [Int] = []
    for var b in bounds {
        for (a, e) in runs where a < b && b < e {
            b = (out.isEmpty || a > out.last!) ? a : e
            break
        }
        if out.isEmpty || b > out.last! { out.append(b) }
    }
    return out
}

/// The user's home directory (macOS), or the app container's home on iOS, where
/// `homeDirectoryForCurrentUser` is unavailable (story 6.1).
func homeDirectory() -> URL {
    #if os(macOS)
    FileManager.default.homeDirectoryForCurrentUser
    #else
    URL(fileURLWithPath: NSHomeDirectory())
    #endif
}
