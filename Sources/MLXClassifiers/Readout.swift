// First-token readout (golden ADR-0003, `readout.py` + `calibrate.py` + the
// first-token half of `decide.py`): option labels bound to single token ids,
// the logits gathered at the readout position, a masked softmax over them.
//
// The in-context label ids come from the tokenizer (golden `label_token_ids`,
// LabelBinding.swift); `bindLabelSequences` here takes those id sequences and
// applies the tokenizer-free rules: no shared first token, one token per label.

import Foundation

/// Golden `bind_labels` after tokenization (the tokenizer-free half; the in-context
/// id sequences come from `labelTokenIds` in LabelBinding.swift): one mask id per
/// label, or a `ClassifierError(.labelTokenization)` for duplicates, a shared first
/// token, or a label that is not one token, its message ending with `hint`.
public func bindLabelSequences(labels: [String], sequences: [[Int]], hint: String = BindingHint.sequence) throws
    -> (ids: [Int], tree: PrefixTree)
{
    guard Set(labels).count == labels.count else {
        throw ClassifierError(.labelTokenization, "duplicate labels [\(labels.map(pyRepr).joined(separator: ", "))]")
    }
    for (label, ids) in zip(labels, sequences) where ids.isEmpty {
        throw ClassifierError(.labelTokenization,
                              "label \(pyRepr(label)) merges with the token at the readout position; \(hint)")
    }
    let tree = PrefixTree(labels: labels, sequences: sequences)
    if !tree.collisions.isEmpty {
        let names = tree.collisions.map { $0.map(pyRepr).joined(separator: " and ") }.joined(separator: "; ")
        let whereText = tree.depth.map { "depth \($0)" } ?? "no depth (identical token ids)"
        throw ClassifierError(.labelTokenization,
            "labels \(names) share a first token; they first differ at \(whereText), "
            + "which first-token mode cannot read; \(hint)")
    }
    let multi = zip(labels, sequences).filter { $0.1.count != 1 }
    if !multi.isEmpty {
        let detail = multi.map { "\(pyRepr($0.0)) -> \($0.1.count) tokens [\($0.1.map(String.init).joined(separator: ", "))]" }
            .joined(separator: ", ")
        throw ClassifierError(.labelTokenization, "labels are not single tokens: \(detail); \(hint)")
    }
    return (sequences.map { $0[0] }, tree)
}

// MARK: - calibrate.py (the parts first-token mode needs)

/// The logits of `tokenIds` from a vocab row (the option mask).
public func gather(_ row: LogitRow, _ tokenIds: [Int]) -> [Double] {
    tokenIds.map { row[$0] }
}

/// Numerically stable softmax of `logits / temperature` (golden `softmax`).
public func softmax(_ logits: [Double], temperature: Double = 1.0) -> [Double] {
    precondition(temperature > 0, "temperature must be positive")
    let scaled = logits.map { $0 / temperature }
    let m = scaled.max()!
    let exps = scaled.map { Foundation.exp($0 - m) }
    let total = exactSum(exps)
    return exps.map { $0 / total }
}

/// Index of the first maximum (Python `max(range(n), key=...)`).
public func argmax(_ xs: [Double]) -> Int {
    var best = 0
    for i in xs.indices where xs[i] > xs[best] { best = i }
    return best
}

/// Neumaier-compensated sum, standing in for Python's `math.fsum` on a handful of terms.
func exactSum(_ xs: [Double]) -> Double {
    var sum = 0.0, c = 0.0
    for x in xs {
        let t = sum + x
        c += abs(sum) >= abs(x) ? (sum - t) + x : (x - t) + sum
        sum = t
    }
    return sum + c
}

// MARK: - decide.py, first-token scoring of one compiled prompt

/// One first-token decision over a compiled prompt (golden `decide._score` +
/// the raw softmax of `decide`/`fixtures.run_case`).
public struct FirstTokenScore: Sendable {
    public let options: [String]
    public let tokenIds: [Int]
    public let logits: [Double]
    public let probabilities: [Double]
    public let inputTokens: Int
    public let cachedTokens: Int
    public var answer: String { options[argmax(probabilities)] }
}

/// Score a compiled prompt's options from its token ids: bind the labels' in-context
/// id sequences, prefill (from the prefix cache over `tokens[..<prefixLen]` when
/// `prefixCached`), gather at `readoutPosition`, softmax. Throws
/// `ClassifierError(.labelTokenization)` before touching the model when the labels cannot be read.
public func scoreFirstToken(
    runner: Runner,
    tokens: [Int],
    readoutPosition: Int,
    options: [String],
    labels: [String],
    labelSequences: [[Int]],
    labelStyle: LabelStyle,
    prefixLen: Int,
    prefixCached: Bool
) throws -> FirstTokenScore {
    let hint = labelStyle == .optionText ? BindingHint.optionText : BindingHint.sequence
    let (ids, _) = try bindLabelSequences(labels: labels, sequences: labelSequences, hint: hint)
    let cache = prefixCached && prefixLen > 0 ? try runner.cachePrefix(Array(tokens[..<prefixLen])) : nil
    let r = try runner.prefill(tokens, prefixCache: cache, readoutPositions: [readoutPosition])
    let logits = gather(try r.at(readoutPosition), ids)
    return FirstTokenScore(
        options: options, tokenIds: ids, logits: logits, probabilities: softmax(logits),
        inputTokens: r.inputTokens, cachedTokens: r.cachedTokens)
}
