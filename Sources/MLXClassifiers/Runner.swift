// The Runner seam (golden ADR-0002 §4), mirroring golden `runner.py` one-to-one:
// `tokenize`, `chatTemplate`, `cachePrefix`, `prefill` with logits only at the
// readout positions. The image extension is epic 4 (`imageTokens` is nil until
// then); delta prefix caching is epic 3. Nothing here samples: the same tokens give
// the same logits.

import Foundation

/// One vocab-sized row of logits at a readout position (golden `LogitRow`).
/// `row[tokenId]` is a logit; `logsumexp()` the full-vocab normaliser (sequence mode).
public struct LogitRow: Sendable {
    public let count: Int
    private let lookup: @Sendable (Int) -> Double
    private let lse: @Sendable () -> Double

    public init(count: Int, lookup: @escaping @Sendable (Int) -> Double, logsumexp: @escaping @Sendable () -> Double) {
        self.count = count
        self.lookup = lookup
        self.lse = logsumexp
    }

    /// A row over plain values (tests, the fake runner); the normaliser is computed here.
    public static func fromList(_ values: [Double]) -> LogitRow {
        let m = values.max() ?? 0
        let total = m + Foundation.log(values.reduce(0) { $0 + Foundation.exp($1 - m) })
        return LogitRow(count: values.count, lookup: { values[$0] }, logsumexp: { total })
    }

    public subscript(tokenId: Int) -> Double {
        precondition(0 <= tokenId && tokenId < count, "token id \(tokenId) outside the vocab")
        return lookup(tokenId)
    }

    public func logsumexp() -> Double { lse() }
}

/// A prefilled prefix (golden `PrefixCache`). `state` is the runner's own KV
/// object, opaque to everything but that runner.
public struct PrefixCache {
    public let tokens: [Int]
    public let state: Any?

    public init(tokens: [Int], state: Any?) {
        self.tokens = tokens
        self.state = state
    }

    public var count: Int { tokens.count }
}

/// Logits at the requested positions only, plus token accounting (golden `runner.Readout`; `Readout` here is the wire readout mode).
/// `inputTokens` counts the rows this call computed, `cachedTokens` those served
/// from the prefix cache.
public struct PrefillReadout: Sendable {
    public let positions: [Int]
    public let logits: [LogitRow]
    public let inputTokens: Int
    public let cachedTokens: Int

    public func at(_ position: Int) throws -> LogitRow {
        guard let i = positions.firstIndex(of: position) else {
            throw RunnerError.invalidArgument("no logits at position \(position); have \(positions)")
        }
        return logits[i]
    }
}

public enum RunnerError: Error, CustomStringConvertible, Equatable {
    /// The golden runners raise ValueError for these.
    case invalidArgument(String)
    /// Loading the model failed (missing snapshot, bad config).
    case load(String)
    /// Not ported yet; names the story that ports it.
    case notImplemented(String)

    public var description: String {
        switch self {
        case .invalidArgument(let m): "invalid argument: \(m)"
        case .load(let m): "load: \(m)"
        case .notImplemented(let m): "not implemented: \(m)"
        }
    }
}

/// The model seam (golden `runner.Runner`).
///
/// - `tokenize(text)`: text to token ids, special-token markup parsed, nothing added.
/// - `chatTemplate`: the model's chat template (rendering is text only; `tokenize` makes the ids).
/// - `imageTokens`: the image placeholder convention, nil without a vision path (golden's
///   optional `image_tokens` attribute; default nil).
/// - `modelId`: what a response reports as `model` when the request names none (default "default").
/// - `cachePrefix(tokens)`: prefill `tokens` once and keep the KV state; no logits.
/// - `prefill(tokens, prefixCache, readoutPositions)`: prefill the *full* sequence
///   (only the suffix after `prefixCache.tokens` is computed); logits only at
///   `readoutPositions` (absolute, default the last one), the hidden state sliced
///   before the vocab projection.
public protocol Runner {
    func tokenize(_ text: String) -> [Int]
    var chatTemplate: ChatTemplate { get }
    var imageTokens: ImageTokens? { get }
    var modelId: String { get }
    func cachePrefix(_ tokens: [Int]) throws -> PrefixCache
    func prefill(_ tokens: [Int], prefixCache: PrefixCache?, readoutPositions: [Int]?) throws -> PrefillReadout
}

extension Runner {
    public var imageTokens: ImageTokens? { nil }
    public var modelId: String { "default" }

    public func prefill(_ tokens: [Int], prefixCache: PrefixCache? = nil) throws -> PrefillReadout {
        try prefill(tokens, prefixCache: prefixCache, readoutPositions: nil)
    }
}

/// Validate a `prefill` call; return `(cachedLen, positions)` (golden
/// `check_prefill_args`, images not yet ported). Shared so every Runner enforces
/// the same contract.
public func checkPrefillArgs(
    _ tokens: [Int], prefixCache: PrefixCache?, readoutPositions: [Int]?
) throws -> (cached: Int, positions: [Int]) {
    var nCached = 0
    if let prefixCache {
        nCached = prefixCache.count
        guard tokens.count >= nCached, Array(tokens[..<nCached]) == prefixCache.tokens else {
            throw RunnerError.invalidArgument("tokens do not start with the prefix cache's tokens")
        }
        if tokens.count == nCached {
            throw RunnerError.invalidArgument("nothing to prefill: tokens equal the cached prefix")
        }
    }
    if tokens.isEmpty { throw RunnerError.invalidArgument("cannot prefill an empty sequence") }
    let positions = readoutPositions ?? [tokens.count - 1]
    if positions.isEmpty { throw RunnerError.invalidArgument("readout_positions is empty") }
    for p in positions where !(nCached <= p && p < tokens.count) {
        throw RunnerError.invalidArgument(
            "readout position \(p) outside the prefilled suffix [\(nCached), \(tokens.count))"
                + " (a prefix cache holds KV, not logits)")
    }
    return (nCached, positions)
}
