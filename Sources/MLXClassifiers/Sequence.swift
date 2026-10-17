// Sequence-scoring readout (`options.readout = "sequence"`, golden ADR-0003),
// ported from golden `sequence.py`. Each option's full text is teacher-forced
// after the prompt and scored by its length-normalised loglikelihood (mean
// log-probability per token). The prompt minus its last token is prefilled once
// and cached; each option then costs one small pass over
// `[last prompt token] + option tokens`, reading logits at exactly the positions
// that predict the option's tokens. Options may share first tokens or be many
// tokens long.

import Foundation

/// Per-option results in option order (golden `SequenceScores`). `scores` are the
/// length-normalised loglikelihoods used as the answer's logits.
public struct SequenceScores: Sendable {
    public let options: [String]
    public let tokenIds: [[Int]]
    public let loglikelihoods: [Double]
    public let scores: [Double]
    public let readoutPosition: Int
    public let inputTokens: Int
    public let cachedTokens: Int

    public var optionTokenIds: OrderedObject<[Int]> { OrderedObject(Array(zip(options, tokenIds))) }
}

/// Golden `score_sequences`.
public func scoreSequences(_ prompt: CompiledPrompt, runner: Runner, tokenize: Tokenize? = nil) throws -> SequenceScores {
    let tokenize = tokenize ?? runner.tokenize
    let tokens = prompt.tokens
    let n = tokens.count
    let conts = try sequenceOptionTokenIds(prompt, tokenize: tokenize).values  // identical texts are an error
    let cache = try runner.cachePrefix(Array(tokens[..<(n - 1)]), images: prompt.images)
    var lls: [Double] = [], scores: [Double] = []
    var inputTokens = 0, cachedTokens = 0
    for ids in conts {
        let positions = Array((n - 1) ..< (n - 1 + ids.count))  // position p predicts token p + 1
        let r = try runner.prefill(tokens + ids, prefixCache: cache, readoutPositions: positions, images: prompt.images)
        let ll = exactSum(try zip(positions, ids).map { p, t in tokenLogprob(try r.at(p), t) })
        lls.append(ll)
        scores.append(ll / Double(ids.count))
        inputTokens += r.inputTokens
        cachedTokens += r.cachedTokens
    }
    return SequenceScores(options: prompt.options, tokenIds: conts, loglikelihoods: lls, scores: scores,
                          readoutPosition: prompt.readoutPosition, inputTokens: inputTokens, cachedTokens: cachedTokens)
}
