// The fixture runner: golden `fixtures.run_case` (one case through a Runner, the
// result shaped like a fixture's `expected` block) and `fixtures.compare` (the
// checker's comparison semantics). This is the reference a port mirrors:
// compile, bind in context, prefill (from the system-turn cache when
// `prefix_cached`), gather, raw softmax. Delta prefix caching is off, as golden
// runs cases (`without_delta`): a case pins its cache path.

import Foundation

/// Golden `_with_label_style`: `request` with `options.label_style` set, so the
/// case compiles through the same path as a wire request carrying that option.
public func withLabelStyle(_ request: SystemOneRequest, _ labelStyle: LabelStyle) -> SystemOneRequest {
    var opts = request.options ?? RequestOptions()
    if opts.resolvedLabelStyle == labelStyle { return request }
    opts.labelStyle = labelStyle
    var r = request
    r.options = opts
    return r
}

/// Golden `run_case`: run one fixture's `request` through `runner` and return the
/// result block. A first-token binding error is a result (`error`, with the
/// per-label in-context ids, empty where a label merges), not a throw.
public func runCase(_ fx: GoldenFixture, runner: Runner) throws -> GoldenFixture.Expected {
    let runner = withoutDelta(runner)
    let req = withLabelStyle(fx.request, fx.labelStyle)
    guard req.questions.count == 1, let name = req.questions.keys.first else {
        throw FixtureError(file: fx.id, message: "a fixture request holds exactly one question")
    }
    let readout = req.options?.resolvedReadout ?? .firstToken
    let prompt = try compileQuestion(req, name: name, template: runner.chatTemplate, tokenize: runner.tokenize,
                                     imageTokens: runner.imageTokens)
    var out = GoldenFixture.Expected(
        tokens: prompt.tokens, prefixLen: prompt.prefixLen, readoutPosition: prompt.readoutPosition,
        options: prompt.options, labels: prompt.optionOrderLabels,
        images: prompt.images.isEmpty ? nil : prompt.images.map(\.described),
        optionTokenIds: OrderedObject(), rawLogits: nil, rawProbabilities: nil, answer: nil, error: nil)
    let logits: [Double]
    if readout == .sequence {
        let s = try scoreSequences(prompt, runner: runner)
        out.optionTokenIds = s.optionTokenIds
        logits = s.scores
    } else {
        let binding: OptionBinding
        do {
            binding = try bindOptions(prompt, tokenize: runner.tokenize)
        } catch let e as ClassifierError {
            out.optionTokenIds = OrderedObject(zip(prompt.options, out.labels).map { o, lab in
                (o, (try? labelTokenIds(context: prompt.text, label: lab, tokenize: runner.tokenize,
                                        contextTokens: prompt.tokens)) ?? [])
            })
            out.error = GoldenFixture.ExpectedError(type: e.type, message: e.message)
            return out
        }
        let cache = fx.prefixCached ? try runner.cachePrefix(prompt.prefixTokens) : nil
        let r = try runner.prefill(prompt.tokens, prefixCache: cache, readoutPositions: [prompt.readoutPosition])
        out.optionTokenIds = binding.optionTokenIds
        logits = gather(try r.at(prompt.readoutPosition), binding.tokenIds)
    }
    let probs = softmax(logits)
    out.rawLogits = OrderedObject(Array(zip(prompt.options, logits)))
    out.rawProbabilities = OrderedObject(Array(zip(prompt.options, probs)))
    out.answer = prompt.options[argmax(probs)]
    return out
}

/// Golden `CaseReport`: the failures of one case and its distances.
public struct CaseReport: Sendable {
    public let caseId: String
    public var failures: [String] = []
    public var tokenIdsEqual = true
    public var maxProbDistance: Double?
    public var maxLogitDistance: Double?
    public var ok: Bool { failures.isEmpty }
}

/// Golden `_compare_ids`: the first divergence, then a length mismatch.
func compareIds(_ where_: String, _ want: [Int], _ got: [Int], _ f: inout [String]) -> Bool {
    var ok = true
    if let i = zip(want, got).enumerated().first(where: { $0.element.0 != $0.element.1 })?.offset {
        f.append("\(where_)[\(i)]: expected token id \(want[i]), got \(got[i])")
        ok = false
    }
    if want.count != got.count {
        f.append("\(where_): expected \(want.count) token ids, got \(got.count)")
        ok = false
    }
    return ok
}

/// Golden `_max_distance`: the largest absolute difference and where it is.
func maxDistance(_ where_: String, _ want: OrderedObject<Double>, _ got: OrderedObject<Double>,
                 _ f: inout [String]) -> (Double, String)? {
    guard Set(want.keys) == Set(got.keys), got.count == want.count else {
        f.append("\(where_): expected options \(want.keys.sorted()), got \(got.keys.sorted())")
        return nil
    }
    var worst = want.keys[0], d = -1.0
    for o in want.keys where abs(got[o]! - want[o]!) > d {
        d = abs(got[o]! - want[o]!)
        worst = o
    }
    return (d, worst)
}

/// Golden `compare`: exact on `tokens`, `prefix_len`, `readout_position`, `options`,
/// `labels`, `option_token_ids`, `images`, `answer` and `error.type`; within
/// `probTol` (max absolute difference) on `raw_probabilities`. `raw_logits`
/// distance is reported, and fails only when `logitTol` is given.
public func compareCase(expected e: GoldenFixture.Expected, result r: GoldenFixture.Expected, caseId: String = "?",
                        probTol: Double = FixtureFormat.defaultProbabilityTolerance,
                        logitTol: Double? = nil) -> CaseReport {
    var rep = CaseReport(caseId: caseId)
    var f: [String] = []
    if e.images != nil, r.images == nil { f.append("missing field 'images'") }
    if e.rawLogits != nil, r.rawLogits == nil { f.append("missing field 'raw_logits'") }
    if e.rawProbabilities != nil, r.rawProbabilities == nil { f.append("missing field 'raw_probabilities'") }
    if e.answer != nil, r.answer == nil { f.append("missing field 'answer'") }
    if e.error != nil, r.error == nil { f.append("missing field 'error'") }
    if let got = r.error, e.error == nil { f.append("unexpected error \(got)") }

    var tokOk = compareIds("tokens", e.tokens, r.tokens, &f)
    if r.prefixLen != e.prefixLen { f.append("prefix_len: expected \(e.prefixLen), got \(r.prefixLen)") }
    if r.readoutPosition != e.readoutPosition {
        f.append("readout_position: expected \(e.readoutPosition), got \(r.readoutPosition)")
        tokOk = false
    }
    if let wantImg = e.images, let gotImg = r.images, wantImg != gotImg {
        f.append("images: expected \(wantImg), got \(gotImg)")
        tokOk = false
    }
    if r.options != e.options { f.append("options: expected \(e.options), got \(r.options)") }
    if r.labels != e.labels { f.append("labels: expected \(e.labels), got \(r.labels)") }
    for entry in e.optionTokenIds.entries {
        if let got = r.optionTokenIds[entry.key] {
            tokOk = compareIds("option_token_ids['\(entry.key)']", entry.value, got, &f) && tokOk
        } else {
            f.append("option_token_ids: no ids for option '\(entry.key)'")
            tokOk = false
        }
    }
    rep.tokenIdsEqual = tokOk

    if let want = e.error {
        if let got = r.error, got.type != want.type { f.append("error.type: expected \(want.type), got \(got.type)") }
        rep.failures = f
        return rep
    }
    if let want = e.rawProbabilities, let got = r.rawProbabilities, let (d, o) = maxDistance("raw_probabilities", want, got, &f) {
        rep.maxProbDistance = d
        if !(d <= probTol) { f.append("raw_probabilities: max distance \(d) at '\(o)' exceeds \(probTol)") }
    }
    if let want = e.rawLogits, let got = r.rawLogits, let (d, o) = maxDistance("raw_logits", want, got, &f) {
        rep.maxLogitDistance = d
        if let logitTol, !(d <= logitTol) { f.append("raw_logits: max distance \(d) at '\(o)' exceeds \(logitTol)") }
    }
    if let want = e.answer, let got = r.answer, got != want { f.append("answer: expected '\(want)', got '\(got)'") }
    rep.failures = f
    return rep
}
