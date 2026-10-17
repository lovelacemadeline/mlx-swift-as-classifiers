// Option binding in context, ported from golden `readout.py` (and the id half of
// `sequence.py`). Each option is bound to a label and the label is resolved to
// token ids in context: the ids it gets when emitted right after the compiled
// prompt. First-token mode needs depth one and one token per label; anything
// else is a `LabelTokenizationError`, never a silent fallback.

import Foundation

public enum BindingHint {
    public static let sequence = "use options.readout = \"sequence\", or labels that are single tokens"
    public static let optionText = "use options.readout = \"sequence\", or options.label_style = \"letters\" "
        + "(or option names that are single tokens)"
}

/// Python `repr` of a str, as golden's error messages print labels (the common
/// case: single quotes unless the text holds a single quote and no double quote).
func pyRepr(_ s: String) -> String {
    let q: Character = s.contains("'") && !s.contains("\"") ? "\"" : "'"
    var out = String(q)
    for u in s.unicodeScalars {
        switch u {
        case "\\": out += "\\\\"
        case "\n": out += "\\n"
        case "\r": out += "\\r"
        case "\t": out += "\\t"
        case _ where Character(u) == q: out += "\\\(q)"
        case _ where u.value < 0x20 || u.value == 0x7F: out += String(format: "\\x%02x", u.value)
        default: out.unicodeScalars.append(u)
        }
    }
    return out + String(q)
}

/// Golden `emitted_label`: the bare label after whitespace, else `" " + label`.
public func emittedLabel(context: String, label: String) -> String {
    context.isEmpty || context.pyEndsWithSpace ? label : " " + label
}

/// Golden `label_token_ids`: token ids of `label` right after `context`.
/// Throws `LabelTokenizationError` if appending the label re-tokenizes the context.
public func labelTokenIds(context: String, label: String, tokenize: Tokenize, contextTokens: [Int]? = nil,
                          hint: String = BindingHint.sequence) throws -> [Int] {
    let base = contextTokens ?? tokenize(context)
    let full = tokenize(context + emittedLabel(context: context, label: label))
    guard Array(full.prefix(base.count)) == base, full.count != base.count else {
        throw ClassifierError(.labelTokenization,
                              "label \(pyRepr(label)) merges with the token at the readout position; \(hint)")
    }
    return Array(full[base.count...])
}

/// Golden `PrefixTree`, summarised: `depth` is the first token depth at which
/// every label's sequence is distinct (nil if two are identical); `collisions`
/// groups labels sharing a first token, in first-seen order.
public struct PrefixTree: Sendable, Equatable {
    public let labels: [String]
    public let sequences: [[Int]]
    public let depth: Int?
    public let collisions: [[String]]

    public init(labels: [String], sequences: [[Int]]) {
        self.labels = labels
        self.sequences = sequences
        var firstOrder: [Int] = [], groups: [Int: [String]] = [:]
        for (label, ids) in zip(labels, sequences) {
            if groups[ids[0]] == nil { firstOrder.append(ids[0]) }
            groups[ids[0], default: []].append(label)
        }
        collisions = firstOrder.compactMap { groups[$0]!.count > 1 ? groups[$0] : nil }
        var depth: Int?
        for d in 1...(sequences.map(\.count).max() ?? 0) {
            let heads = sequences.map { Array($0.prefix(d)) }
            if Set(heads).count == heads.count { depth = d; break }
        }
        self.depth = depth
    }
}

/// Golden `OptionBinding`: options bound to single-token labels at one readout position.
public struct OptionBinding: Sendable, Equatable {
    public let options: [String]
    public let labels: [String]
    public let tokenIds: [Int]
    public let readoutPosition: Int
    public let tree: PrefixTree

    /// `{option: [token id]}`, the shape of `debug.option_token_ids`.
    public var optionTokenIds: OrderedObject<[Int]> { OrderedObject(zip(options, tokenIds.map { [$0] }).map { ($0, $1) }) }
}

/// Golden `bind_labels`: resolve labels in context and require first-token readout.
public func bindLabels(_ labels: [String], context: String, tokenize: Tokenize, contextTokens: [Int]? = nil,
                       hint: String = BindingHint.sequence) throws -> (ids: [Int], tree: PrefixTree) {
    guard Set(labels).count == labels.count else {
        throw ClassifierError(.labelTokenization, "duplicate labels [\(labels.map(pyRepr).joined(separator: ", "))]")
    }
    let base = contextTokens ?? tokenize(context)
    let seqs = try labels.map { try labelTokenIds(context: context, label: $0, tokenize: tokenize, contextTokens: base, hint: hint) }
    let tree = PrefixTree(labels: labels, sequences: seqs)
    if !tree.collisions.isEmpty {
        let names = tree.collisions.map { $0.map(pyRepr).joined(separator: " and ") }.joined(separator: "; ")
        let whereText = tree.depth.map { "depth \($0)" } ?? "no depth (identical token ids)"
        throw ClassifierError(.labelTokenization,
            "labels \(names) share a first token; they first differ at \(whereText), "
            + "which first-token mode cannot read; \(hint)")
    }
    let multi = zip(labels, seqs).filter { $0.1.count != 1 }
    if !multi.isEmpty {
        let detail = multi.map { "\(pyRepr($0.0)) -> \($0.1.count) tokens [\($0.1.map(String.init).joined(separator: ", "))]" }
            .joined(separator: ", ")
        throw ClassifierError(.labelTokenization, "labels are not single tokens: \(detail); \(hint)")
    }
    return (seqs.map { $0[0] }, tree)
}

/// Golden `bind_options`: bind `prompt`'s options to mask ids at its readout position.
public func bindOptions(_ prompt: CompiledPrompt, tokenize: Tokenize) throws -> OptionBinding {
    let labels = prompt.optionOrderLabels
    let hint = prompt.labelStyle == .optionText ? BindingHint.optionText : BindingHint.sequence
    let (ids, tree) = try bindLabels(labels, context: prompt.text, tokenize: tokenize, contextTokens: prompt.tokens, hint: hint)
    return OptionBinding(options: prompt.options, labels: labels, tokenIds: ids,
                         readoutPosition: prompt.readoutPosition, tree: tree)
}

/// The id half of golden `score_sequences`: each option's continuation ids
/// after the prompt, in option order; identical continuations are an error.
public func sequenceOptionTokenIds(_ prompt: CompiledPrompt, tokenize: Tokenize) throws -> OrderedObject<[Int]> {
    let conts = try prompt.options.indices.map {
        try labelTokenIds(context: prompt.text, label: prompt.label(for: $0), tokenize: tokenize, contextTokens: prompt.tokens)
    }
    guard Set(conts).count == conts.count else {
        throw ClassifierError(.labelTokenization,
            "options [\(prompt.options.map(pyRepr).joined(separator: ", "))] include identically tokenized texts")
    }
    return OrderedObject(zip(prompt.options, conts).map { ($0, $1) })
}

/// The token-id half of golden `fixtures.run_case`: everything a fixture's
/// `expected` block pins without a model. On a first-token binding error, ids
/// are the per-label in-context ids (empty where a label merges) and `error`
/// is set, as golden records them.
public struct TokenResult: Sendable, Equatable {
    public var tokens: [Int]
    public var prefixLen: Int
    public var readoutPosition: Int
    public var options: [String]
    public var labels: [String]
    public var images: [GoldenFixture.Image]?
    public var optionTokenIds: OrderedObject<[Int]>
    public var error: ClassifierError?
}

public func tokenResult(_ prompt: CompiledPrompt, tokenize: Tokenize) throws -> TokenResult {
    var r = TokenResult(tokens: prompt.tokens, prefixLen: prompt.prefixLen, readoutPosition: prompt.readoutPosition,
                        options: prompt.options, labels: prompt.optionOrderLabels,
                        images: prompt.images.isEmpty ? nil : prompt.images.map(\.described),
                        optionTokenIds: OrderedObject(), error: nil)
    if prompt.readout == .sequence {
        r.optionTokenIds = try sequenceOptionTokenIds(prompt, tokenize: tokenize)
        return r
    }
    do {
        r.optionTokenIds = try bindOptions(prompt, tokenize: tokenize).optionTokenIds
    } catch let e as ClassifierError {
        r.optionTokenIds = OrderedObject(zip(prompt.options, r.labels).map { o, lab in
            (o, (try? labelTokenIds(context: prompt.text, label: lab, tokenize: tokenize, contextTokens: prompt.tokens)) ?? [])
        })
        r.error = e
    }
    return r
}
