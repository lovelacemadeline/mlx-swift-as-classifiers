// `decide`: a wire request in, typed answers out, through any `Runner`. A port of
// golden `decide.py` and `answers.py::shape_answer`; nothing here knows about weights.
//
// First-token mode: the request `prefix` (the system turn) is KV-cached once per
// call and shared by every prompt; each prompt is one prefill with logits at its
// readout position. `options.permutations = N` compiles each `choice` question N
// times with the option order permuted (`permutationOrders`), labels re-bound per
// ordering, and averages the N distributions (raw and calibrated separately).
// `temperature` is the fitted calibration temperature, applied when
// `options.calibrated` (default true); without one calibrated equals raw.
//
// Delta prefix caching (golden story 3.4, here epic 3) is not ported: no runner has
// a prompt-cache store yet, so `options.delta_cache` changes nothing, as golden's
// `without_delta` is a no-op on a runner without a store. Sequence mode
// (Sequence.swift): each prompt caches itself and scores every option with one
// small pass.

import Foundation

/// Golden `permutation_orders`: `count` display orders over `n` options. Order 0
/// is the declared order, orders 1 ..< n the cyclic rotations, beyond that
/// shuffles seeded by `"\(seed):\(k)"` (CPython's `random.Random`).
public func permutationOrders(n: Int, count: Int, seed: String = "") -> [[Int]] {
    (0 ..< count).map { k in
        if k < n { return (0 ..< n).map { ($0 + k) % n } }
        var perm = Array(0 ..< n)
        var rng = PythonRandom(seed: "\(seed):\(k)")
        rng.shuffle(&perm)
        return perm
    }
}

/// One compiled ordering's per-option scores, in option (not display) order (golden `_Scored`).
struct Scored {
    let prompt: CompiledPrompt
    let logits: [Double]
    let tokenIds: OrderedObject<[Int]>
    let inputTokens: Int
    let cachedTokens: Int
}

/// Golden `decide._score`.
func scorePrompt(_ prompt: CompiledPrompt, runner: Runner, cache: PrefixCache?, readout: Readout) throws -> Scored {
    if readout == .sequence {
        let s = try scoreSequences(prompt, runner: runner)
        return Scored(prompt: prompt, logits: s.scores, tokenIds: s.optionTokenIds,
                      inputTokens: s.inputTokens, cachedTokens: s.cachedTokens)
    }
    let binding = try bindOptions(prompt, tokenize: runner.tokenize)
    let r = try runner.prefill(prompt.tokens, prefixCache: cache, readoutPositions: [prompt.readoutPosition])
    let logits = gather(try r.at(prompt.readoutPosition), binding.tokenIds)
    return Scored(prompt: prompt, logits: logits, tokenIds: binding.optionTokenIds,
                  inputTokens: r.inputTokens, cachedTokens: r.cachedTokens)
}

/// Golden `shape_answer`: option distributions (in option order) into the typed
/// answer. `raw` is the T = 1 distribution, `calibrated` the temperature-scaled
/// one; `confidence` is the calibrated probability of the returned option.
public func shapeAnswer(_ question: Question, options: [String], raw: [Double], calibrated cal: [Double],
                        debug: Debug? = nil) -> Answer {
    if case .noul = question {
        return .noul(NoulAnswer(noul: cal[0], rawNoul: raw[0], debug: debug))
    }
    let best = argmax(cal)
    let probabilities = OrderedObject(Array(zip(options, cal)))
    let rawProbabilities = OrderedObject(Array(zip(options, raw)))
    switch question {
    case .score(let q):
        // Python's sum(): left to right from 0.
        let expected = cal.enumerated().reduce(0.0) { $0 + Double($1.offset) * $1.element }
        return .score(ScoreAnswer(
            score: best, expectedScore: expected,
            legend: OrderedObject(q.criteria.enumerated().map { (String($0.offset), $0.element) }),
            probabilities: probabilities, rawProbabilities: rawProbabilities, confidence: cal[best], debug: debug))
    default:
        return .choice(ChoiceAnswer(choice: options[best], probabilities: probabilities,
                                    rawProbabilities: rawProbabilities, confidence: cal[best], debug: debug))
    }
}

/// Golden `decide`: answer every question in `request` with prefill only, no decode.
///
/// `usage.input_tokens` sums the tokens each prefill computed (suffixes only when
/// cached) and `cached_tokens` the cached tokens each reused; the one-off
/// `cachePrefix` cost is not counted. `timing_ms` is wall clock.
public func decide(_ request: SystemOneRequest, runner: Runner, temperature: Double? = nil) throws -> SystemOneResponse {
    let t0 = DispatchTime.now().uptimeNanoseconds
    func ms(since t: UInt64) -> Double { Double(DispatchTime.now().uptimeNanoseconds - t) / 1e6 }
    let opts = request.options ?? RequestOptions()
    let readout = opts.resolvedReadout
    let applied = opts.resolvedCalibrated ? (temperature ?? 1.0) : 1.0

    func compile(_ name: String, _ order: [Int]? = nil) throws -> CompiledPrompt {
        try compileQuestion(request, name: name, template: runner.chatTemplate, tokenize: runner.tokenize,
                            order: order, readout: readout, imageTokens: runner.imageTokens)
    }

    var plans: [(String, [CompiledPrompt])] = []
    for entry in request.questions.entries {
        if case .choice(let q) = entry.value, opts.resolvedPermutations > 1 {
            let orders = permutationOrders(n: q.criteria.count, count: opts.resolvedPermutations, seed: entry.key)
            plans.append((entry.key, try orders.map { try compile(entry.key, $0) }))
        } else {
            plans.append((entry.key, [try compile(entry.key)]))
        }
    }

    let first = plans[0].1[0]
    var cache: PrefixCache?
    if readout == .firstToken, first.prefixLen > 0 {
        cache = try runner.cachePrefix(first.prefixTokens)
    }

    var answers = OrderedObject<Answer>()
    var inputTokens = 0, cachedTokens = 0
    var prefillMs = 0.0
    for (name, prompts) in plans {
        let t1 = DispatchTime.now().uptimeNanoseconds
        let scored = try prompts.map { try scorePrompt($0, runner: runner, cache: cache, readout: readout) }
        prefillMs += ms(since: t1)
        inputTokens += scored.reduce(0) { $0 + $1.inputTokens }
        cachedTokens += scored.reduce(0) { $0 + $1.cachedTokens }

        let k = Double(scored.count)
        let options = scored[0].prompt.options
        let raws = scored.map { softmax($0.logits) }
        let cals = scored.map { softmax($0.logits, temperature: applied) }
        let raw = options.indices.map { i in exactSum(raws.map { $0[i] }) / k }
        let cal = options.indices.map { i in exactSum(cals.map { $0[i] }) / k }

        var dbg: Debug?
        if opts.resolvedDebug {
            let ids: OrderedObject<[Int]> = readout == .firstToken
                ? OrderedObject(options.map { o in (o, scored.flatMap { $0.tokenIds[o]! }) })  // one id per ordering
                : scored[0].tokenIds  // option texts tokenize identically in every ordering
            dbg = Debug(
                readoutPosition: scored[0].prompt.readoutPosition,
                optionTokenIds: ids,
                rawLogits: OrderedObject(options.enumerated().map { i, o in (o, exactSum(scored.map { $0.logits[i] }) / k) }),
                temperature: applied,
                permutationSpread: scored.count > 1
                    ? OrderedObject(options.enumerated().map { i, o in
                        (o, cals.map { $0[i] }.max()! - cals.map { $0[i] }.min()!)
                    })
                    : nil)
        }
        answers[name] = shapeAnswer(request.questions[name]!, options: options, raw: raw, calibrated: cal, debug: dbg)
    }

    let total = ms(since: t0)
    return SystemOneResponse(
        model: request.model ?? runner.modelId,
        answers: answers,
        usage: Usage(inputTokens: inputTokens, outputTokens: 0, cachedTokens: cachedTokens),
        timingMs: Timing(prefill: prefillMs, readout: total - prefillMs, total: total))
}
