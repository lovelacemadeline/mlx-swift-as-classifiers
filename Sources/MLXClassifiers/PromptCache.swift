// Delta prefix caching (epic 3), ported from golden `prompt_cache.py` (story 3.4)
// and the rewind half of golden `mlx_runner.py` (`rewind_cache`,
// `_rotating_rewind_plan`, `_rotating_rewind`, `cache_nbytes`).
//
// A `PromptCacheStore` keeps the KV state a runner had after prefilling some token
// sequence and answers `lookup(tokens)` with the longest prefix of `tokens` it can
// serve: the entry sharing the longest common token prefix with `tokens`, rewound
// to that length. A runner then prefills only the delta.
//
// The store knows nothing about KV layouts. Rewinding is the runner's business,
// passed in as `rewind(state, storedLen, hitLen) -> state?`: a *new* state holding
// exactly the first `hitLen` tokens (never mutating the stored one), or nil when
// that length is not representable (a sliding-window cache that already dropped
// the rows it would need). The store then tries the next-best entry, and finally
// reports no hit.
//
// Bounded LRU by entry count and by total bytes (`nbytes` per entry, given by the
// runner). A hit bumps recency. Thread-safe.
//
// Rewinding per layer (`rewindCache`): a `KVCacheSimple` trims to any length. A
// `RotatingKVCache` (window W) is first put in temporal order; if it never dropped
// a row it is sliced to the hit length. Once it has slid it holds rows
// `[start, offset)` and the next step only needs the last W - 1 of them, so a hit
// h is representable iff `h - start >= W - 1`: the rewound cache is those W - 1
// rows behind one zero placeholder row (`idx = W`), which both update paths drop
// or overwrite before attending. `RotatingKVCache.trim` is not used: past the wrap
// it is "merely consistent", not exact.

import Foundation
import MLX
import MLXLMCommon

/// Golden `DEFAULT_MAX_ENTRIES` / `DEFAULT_MAX_BYTES` (8 entries, 2 GiB).
public enum PromptCacheDefaults {
    public static let maxEntries = 8
    public static let maxBytes = 2 << 30
}

/// Golden `common_prefix_len`: length of the longest common prefix of two token sequences.
public func commonPrefixLen(_ a: [Int], _ b: [Int]) -> Int {
    let n = min(a.count, b.count)
    return a.withUnsafeBufferPointer { pa in
        b.withUnsafeBufferPointer { pb in
            var i = 0
            while i < n, pa[i] == pb[i] { i += 1 }
            return i
        }
    }
}

/// Golden `PromptCacheStore`: a bounded LRU of KV states keyed by the token sequence they cover.
public final class PromptCacheStore: @unchecked Sendable {
    /// `rewind(state, storedLen, hitLen) -> new state | nil` (nil: not representable).
    public typealias Rewind = (Any, Int, Int) -> Any?

    struct Entry {
        let state: Any
        let nbytes: Int
    }

    public let maxEntries: Int
    public let maxBytes: Int
    /// Keys, least recently used first (golden's `OrderedDict` order).
    private var order: [[Int]] = []
    private var entries: [[Int]: Entry] = [:]
    private let lock = NSLock()
    public private(set) var nbytes = 0
    public private(set) var hits = 0
    public private(set) var misses = 0
    /// Rewinds refused as not representable (sliding window already slid).
    public private(set) var unrepresentable = 0

    public init(maxEntries: Int = PromptCacheDefaults.maxEntries, maxBytes: Int = PromptCacheDefaults.maxBytes) {
        precondition(maxEntries >= 1, "max_entries must be at least 1")
        precondition(maxBytes >= 0, "max_bytes must be non-negative")
        self.maxEntries = maxEntries
        self.maxBytes = maxBytes
    }

    public var count: Int { lock.withLock { order.count } }

    public func contains(_ tokens: [Int]) -> Bool { lock.withLock { entries[tokens] != nil } }

    /// Stored sequences, least recently used first.
    public var keys: [[Int]] { lock.withLock { order } }

    public func clear() {
        lock.withLock {
            order.removeAll()
            entries.removeAll()
            nbytes = 0
        }
    }

    private func bump(_ key: [Int]) {
        if let i = order.firstIndex(of: key) { order.append(order.remove(at: i)) }
    }

    /// Golden `lookup`: `(hitLen, state)`, the longest representable cached prefix of
    /// `tokens`. `hitLen` is at most `maxLen` (default `tokens.count`); a hit no longer
    /// than `minLen` (what the caller already has) reports `(0, nil)`, as does no hit.
    /// `state` is `rewind`'s fresh copy, never a stored object.
    public func lookup(_ tokens: [Int], rewind: Rewind, maxLen: Int? = nil, minLen: Int = 0) -> (hit: Int, state: Any?) {
        let cap = maxLen.map { min($0, tokens.count) } ?? tokens.count
        return lock.withLock {
            // Python's list.sort is stable: equal hits keep insertion (LRU-first) order.
            let candidates = order.enumerated()
                .map { (hit: min(commonPrefixLen($0.element, tokens), cap), i: $0.offset, key: $0.element) }
                .sorted { $0.hit != $1.hit ? $0.hit > $1.hit : $0.i < $1.i }
            for c in candidates {
                if c.hit <= minLen { break }
                guard let state = rewind(entries[c.key]!.state, c.key.count, c.hit) else {
                    unrepresentable += 1
                    continue
                }
                bump(c.key)
                hits += 1
                return (c.hit, state)
            }
            misses += 1
            return (0, nil)
        }
    }

    /// Golden `insert`: store `state` (already the caller's own copy) as covering
    /// `tokens`. An existing key only has its recency bumped; an entry larger than
    /// `maxBytes` is not stored. Evicts least recently used entries until both
    /// bounds hold. Returns whether `state` was stored.
    @discardableResult
    public func insert(_ tokens: [Int], state: Any, nbytes n: Int) -> Bool {
        if tokens.isEmpty { return false }
        return lock.withLock {
            if entries[tokens] != nil {
                bump(tokens)
                return false
            }
            if n > maxBytes { return false }
            entries[tokens] = Entry(state: state, nbytes: n)
            order.append(tokens)
            nbytes += n
            while order.count > maxEntries || nbytes > maxBytes {
                let old = order.removeFirst()
                nbytes -= entries.removeValue(forKey: old)!.nbytes
            }
            return true
        }
    }
}

// MARK: - turning delta caching off per call

/// A runner with a delta store (golden: `MLXRunner.prompt_cache` and the `delta=`
/// keyword of `prefill` / `cache_prefix`).
public protocol DeltaCachingRunner: Runner {
    var promptCache: PromptCacheStore? { get }
    func cachePrefix(_ tokens: [Int], delta: Bool) throws -> PrefixCache
    func prefill(_ tokens: [Int], prefixCache: PrefixCache?, readoutPositions: [Int]?, delta: Bool) throws -> PrefillReadout
}

/// Golden `_NoDelta`: a runner proxy calling `prefill` / `cachePrefix` with `delta: false`.
struct NoDeltaRunner: Runner {
    let runner: DeltaCachingRunner

    func tokenize(_ text: String) -> [Int] { runner.tokenize(text) }
    var chatTemplate: ChatTemplate { runner.chatTemplate }
    var imageTokens: ImageTokens? { runner.imageTokens }
    var modelId: String { runner.modelId }

    func cachePrefix(_ tokens: [Int]) throws -> PrefixCache {
        try runner.cachePrefix(tokens, delta: false)
    }

    func prefill(_ tokens: [Int], prefixCache: PrefixCache?, readoutPositions: [Int]?) throws -> PrefillReadout {
        try runner.prefill(tokens, prefixCache: prefixCache, readoutPositions: readoutPositions, delta: false)
    }
}

/// Golden `without_delta`: `runner` with delta prefix caching off for every call
/// made through it (no lookup, no insert), or `runner` itself when it has no store.
public func withoutDelta(_ runner: Runner) -> Runner {
    guard let r = runner as? DeltaCachingRunner, r.promptCache != nil else { return runner }
    return NoDeltaRunner(runner: r)
}

// MARK: - exact rewind of an MLXLMCommon cache list

/// A rotating cache's layout read through `state` / `metaState` (its fields are private).
struct RotatingLayout {
    let keep: Int
    let window: Int
    let step: Int
    let offset: Int
    let idx: Int
    let origin: String
    let wrapped: Bool
    let keys: MLXArray
    let values: MLXArray

    init?(_ c: RotatingKVCache) {
        let m = c.metaState
        let s = c.state
        guard m.count >= 7, s.count == 2, let keep = Int(m[0]), let window = Int(m[1]), let step = Int(m[2]),
              let offset = Int(m[3]), let idx = Int(m[4]), let wrapped = Bool(m[6])
        else { return nil }
        (self.keep, self.window, self.step, self.offset, self.idx, self.origin, self.wrapped) =
            (keep, window, step, offset, idx, m[5], wrapped)
        (keys, values) = (s[0], s[1])
    }

    /// The rows in temporal order (upstream `temporalOrder`). `state` already slices
    /// an unwrapped buffer to `idx`; a wrapped ring is rolled (keep is 0 here).
    func temporal(_ a: MLXArray) -> MLXArray {
        if !wrapped || idx == a.dim(2) { return a }
        return concatenated([a[.ellipsis, idx..., 0...], a[.ellipsis, ..<idx, 0...]], axis: 2)
    }

    /// Rows `temporal` would return (golden `ordered_len`).
    var orderedLen: Int { keys.dim(2) }
}

/// Golden `_rotating_rewind_plan`: whether a `RotatingKVCache` can drop its last `n` tokens exactly.
func rotatingRewindPlan(_ c: RotatingKVCache, _ n: Int) -> Bool {
    guard let l = RotatingLayout(c), l.keep == 0, n <= l.offset else { return false }
    let start = l.offset - l.orderedLen
    return start == 0 || (l.offset - n) - start >= l.window - 1
}

/// Golden `_rotating_rewind`: a new `RotatingKVCache` holding `c` rewound by `n` tokens
/// (`rotatingRewindPlan(c, n)` must hold).
func rotatingRewound(_ c: RotatingKVCache, _ n: Int) -> RotatingKVCache {
    let l = RotatingLayout(c)!
    let keys = l.temporal(l.keys), values = l.temporal(l.values)
    let start = l.offset - keys.dim(2)
    let h = l.offset - n
    let out = RotatingKVCache(maxSize: l.window, keep: l.keep, step: l.step)
    let idx: Int
    if start == 0 {
        out.state = [keys[.ellipsis, ..<h, 0...], values[.ellipsis, ..<h, 0...]]
        idx = h
    } else {
        let w = l.window
        let lo = h - start - (w - 1), hi = h - start
        let padK = MLXArray.zeros([keys.dim(0), keys.dim(1), 1, keys.dim(3)], dtype: keys.dtype)
        let padV = MLXArray.zeros([values.dim(0), values.dim(1), 1, values.dim(3)], dtype: values.dtype)
        out.state = [concatenated([padK, keys[.ellipsis, lo ..< hi, 0...]], axis: 2),
                     concatenated([padV, values[.ellipsis, lo ..< hi, 0...]], axis: 2)]
        idx = w
    }
    out.metaState = [String(l.keep), String(l.window), String(l.step), String(h), String(idx), l.origin, "false"]
    return out
}

/// Golden `rewind_cache`: a copy of the cache list `cache` (covering `storedLen`
/// tokens) holding only the first `hitLen`, or nil if some layer cannot represent
/// it. `cache` itself is never mutated.
public func rewindCache(_ cache: [KVCache], storedLen: Int, hitLen: Int) -> [KVCache]? {
    let n = storedLen - hitLen
    precondition(n >= 0, "hit_len beyond the stored sequence")
    if n > 0 {
        for c in cache {
            if let r = c as? RotatingKVCache {
                if !rotatingRewindPlan(r, n) { return nil }
            } else if c is KVCacheSimple {
                if n > c.offset { return nil }
            } else if !c.isTrimmable {
                return nil
            }
        }
    }
    return cache.map { c -> KVCache in
        guard n > 0 else { return c.copy() }
        if let r = c as? RotatingKVCache { return rotatingRewound(r, n) }
        let out = c.copy()
        out.trim(n)
        return out.copy()  // compact to the trimmed rows, so the stored copy is exactly hitLen long
    }
}

/// Golden `cache_nbytes`: bytes held by a cache list's state arrays.
public func cacheNbytes(_ cache: [KVCache]) -> Int {
    cache.reduce(0) { $0 + $1.state.reduce(0) { $0 + $1.nbytes } }
}
