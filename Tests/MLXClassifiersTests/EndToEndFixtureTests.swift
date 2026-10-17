import CryptoKit
import Foundation
import XCTest

@testable import MLXClassifiers

/// The golden fixtures end to end from their wire `request`: compile with the
/// checkpoint's tokenizer and chat template, bind in context, prefill, read out,
/// compared with golden `fixtures.compare` semantics (ids exact, raw
/// probabilities within 1e-3, answer exact, error type and message exact).
/// GPU tests: see `GPU.runner()` for when they skip.
final class EndToEndFixtureTests: XCTestCase {
    /// Runs every fixture `select` picks through `runCase`; returns `(id, max |Δp|)`.
    func runFixtures(_ select: (GoldenFixture) -> Bool) async throws -> [(String, Double?)] {
        let runner = try await GPU.runner()
        var seen: [(String, Double?)] = []
        for fx in try Fixtures.load(directory: fixturesDir) where select(fx) {
            let t0 = Date()
            let got = try runCase(fx, runner: runner)
            let ms = Date().timeIntervalSince(t0) * 1e3
            let rep = compareCase(expected: fx.expected, result: got, caseId: fx.id)
            XCTAssertTrue(rep.ok, "\(fx.id): \(rep.failures.joined(separator: "; "))")
            XCTAssertTrue(rep.tokenIdsEqual, fx.id)
            if let want = fx.expected.error {
                XCTAssertEqual(got.error?.message, want.message, fx.id)
            }
            let logitDeltas = got.rawLogits.map { r in r.entries.map { $0.value - fx.expected.rawLogits![$0.key]! } } ?? []
            print("E2E \(fx.id): max|dp| = \(rep.maxProbDistance.map { "\($0)" } ?? "n/a") answer=\(got.answer ?? got.error?.type ?? "?") "
                  + "logit deltas=\(logitDeltas) " + String(format: "%.1f ms", ms))
            seen.append((fx.id, rep.maxProbDistance))
        }
        return seen
    }

    /// Story 2.2: the five first-token text fixtures (four scored, one binding error).
    func testFirstTokenTextFixtures() async throws {
        let seen = try await runFixtures { $0.readout == .firstToken && !$0.request.hasImages }
        XCTAssertEqual(seen.map(\.0), [
            "choice-first-token-cached-list-state", "choice-first-token-multi-token-label-error",
            "choice-first-token-object-state", "noul-first-token", "score-first-token-no-prefix",
        ])
    }

    /// Story 2.3: every text fixture, first-token and sequence (all but the five image fixtures).
    func testAllTextFixtures() async throws {
        let seen = try await runFixtures { !$0.request.hasImages }
        XCTAssertEqual(seen.map(\.0), [
            "choice-first-token-cached-list-state", "choice-first-token-multi-token-label-error",
            "choice-first-token-object-state", "choice-sequence-multi-token-options", "noul-first-token",
            "noul-sequence", "score-first-token-no-prefix", "score-sequence-object-state",
        ])
    }

    /// Story 6.1: `includeVision: false` (the phone's text-only fallback) builds no vision
    /// path and still reproduces a text fixture.
    func testTextOnlyLoad() async throws {
        let shared = try await GPU.runner()
        let r = try await MLXRunner.load(directory: shared.modelDirectory, includeVision: false)
        r.promptCache = nil
        XCTAssertNil(r.vision)
        XCTAssertNil(r.imageTokens)
        let fx = try FixtureRun.load("noul-first-token")
        let rep = compareCase(expected: fx.expected, result: try runCase(fx, runner: r), caseId: fx.id)
        XCTAssertTrue(rep.ok, rep.failures.joined(separator: "; "))
    }

    /// The tokenizer stub is gone: the runner tokenizes like golden `MLXRunner.tokenize`.
    func testRunnerTokenizes() async throws {
        let runner = try await GPU.runner()
        let fx = try FixtureRun.load("noul-first-token")
        XCTAssertEqual(runner.tokenize("Yes"), fx.expected.optionTokenIds["yes"])
        XCTAssertEqual(runner.imageTokens?.imageTokenId, 258_880, "the vision path (story 4.1)")
    }
}

// MARK: - decide against golden decide

/// `Tests/MLXClassifiersTests/GoldenDecide/decide-responses.json`, written by
/// `scripts/golden_decide_reference.py` from golden `decide` on the real model.
struct DecideReference: Decodable {
    struct Case: Decodable {
        var id: String
        var request: SystemOneRequest
        var temperature: Double?
        var response: SystemOneResponse?
        var error: GoldenFixture.ExpectedError?
        /// sha256 of "id,id,..." of each prefill golden made -> readout position -> what golden read.
        var prefills: OrderedObject<OrderedObject<Row>>
    }

    struct Row: Codable, Sendable {
        /// {token id: logit}.
        var logits: OrderedObject<Double>
        /// The full-vocab logsumexp (sequence mode only).
        var lse: Double?
    }

    var format: String
    var cases: [Case]

    static func load() throws -> DecideReference {
        let url = repoRoot.appendingPathComponent("Tests/MLXClassifiersTests/GoldenDecide/decide-responses.json")
        return try SystemOneJSON.decode(DecideReference.self, from: try Data(contentsOf: url))
    }
}

/// The comparable parts of one answer: noul as `{"yes": p}`.
struct AnswerView {
    var type: String
    var pick: String
    var probabilities: OrderedObject<Double>
    var raw: OrderedObject<Double>?
    var confidence: Double?
    var expectedScore: Double?
    var legend: OrderedObject<String>?
    var debug: Debug?

    init(_ a: Answer) {
        switch a {
        case .noul(let n):
            type = "noul"; pick = "noul"
            probabilities = OrderedObject([("yes", n.noul)])
            raw = n.rawNoul.map { OrderedObject([("yes", $0)]) }
            debug = n.debug
        case .choice(let c):
            type = "choice"; pick = c.choice
            probabilities = c.probabilities; raw = c.rawProbabilities; confidence = c.confidence; debug = c.debug
        case .score(let s):
            type = "score"; pick = String(s.score)
            probabilities = s.probabilities; raw = s.rawProbabilities; confidence = s.confidence
            expectedScore = s.expectedScore; legend = s.legend; debug = s.debug
        }
    }
}

/// A Runner serving golden's recorded logits (`DecideReference.Case.prefills`)
/// with the real tokenizer and chat template: `decide`'s own arithmetic
/// (orderings, averaging, calibration, spread, answer shapes, usage) checked
/// with no model in the loop.
final class ReplayRunner: Runner, @unchecked Sendable {
    let front: ModelTokenizer
    let prefills: OrderedObject<OrderedObject<DecideReference.Row>>
    var missing: [String] = []

    init(front: ModelTokenizer, prefills: OrderedObject<OrderedObject<DecideReference.Row>>) {
        self.front = front
        self.prefills = prefills
    }

    var chatTemplate: ChatTemplate { front.chatTemplate }
    var modelId: String { MLXRunner.defaultModel }
    func tokenize(_ text: String) -> [Int] { front.tokenize(text) }

    static func key(_ tokens: [Int]) -> String {
        SHA256.hash(data: Data(tokens.map(String.init).joined(separator: ",").utf8))
            .map { String(format: "%02x", $0) }.joined()
    }

    func cachePrefix(_ tokens: [Int]) throws -> PrefixCache { PrefixCache(tokens: tokens, state: nil) }

    func prefill(_ tokens: [Int], prefixCache: PrefixCache?, readoutPositions: [Int]?) throws -> PrefillReadout {
        let (nCached, positions) = try checkPrefillArgs(tokens, prefixCache: prefixCache, readoutPositions: readoutPositions)
        let k = Self.key(tokens)
        if prefills[k] == nil { missing.append(k) }
        let logits = positions.map { p in
            let row = prefills[k]?[String(p)]
            return LogitRow(count: 262_144, lookup: { row?.logits[String($0)] ?? .nan }, logsumexp: { row?.lse ?? .nan })
        }
        return PrefillReadout(positions: positions, logits: logits, inputTokens: positions.max()! + 1 - nCached,
                              cachedTokens: nCached)
    }
}

final class DecideTests: XCTestCase {
    static let tol = FixtureFormat.defaultProbabilityTolerance

    /// Story 2.2: golden decide's arithmetic reproduced from golden's own logits,
    /// to 1e-12 (the only freedom is the order of float sums); ids, answers,
    /// usage exact. Tokenizer only, no GPU.
    func testDecideReplaysGoldenLogitsExactly() async throws {
        let front = try await TokenParityTests.sharedTokenizer()
        for c in try DecideReference.load().cases {
            let runner = ReplayRunner(front: front, prefills: c.prefills)
            if let want = c.error {
                XCTAssertThrowsError(try decide(c.request, runner: runner, temperature: c.temperature), c.id) { error in
                    XCTAssertEqual((error as? ClassifierError)?.type, want.type, c.id)
                    XCTAssertEqual((error as? ClassifierError)?.message, want.message, c.id)
                }
                continue
            }
            let want = try XCTUnwrap(c.response, c.id)
            let got = try decide(c.request, runner: runner, temperature: c.temperature)
            XCTAssertEqual(runner.missing, [], "\(c.id): a prompt golden never prefilled")
            let worst = compare(c.id, want, got, tol: 1e-12, spreadTol: 1e-12, expectedScoreTol: { _ in 1e-12 },
                                logitTol: 1e-12)
            print("REPLAY \(c.id): max|dp| = \(worst.p) max|dlogit| = \(worst.logit)")
        }
    }

    /// Compares one response to golden's; returns the worst probability and logit distances.
    @discardableResult
    func compare(_ id: String, _ want: SystemOneResponse, _ got: SystemOneResponse, tol: Double, spreadTol: Double,
                 expectedScoreTol: (Int) -> Double, logitTol: Double?) -> (p: Double, logit: Double) {
        XCTAssertEqual(got.model, want.model, id)
        XCTAssertEqual(got.usage, want.usage, id)
        XCTAssertEqual(got.answers.keys, want.answers.keys, id)
        var worstP = 0.0, worstLogit = 0.0
        for (name, w) in zip(want.answers.keys, want.answers.values) {
            let ctx = "\(id).\(name)"
            guard let g = got.answers[name] else { continue }
            let wa = AnswerView(w), ga = AnswerView(g)
            XCTAssertEqual(ga.type, wa.type, ctx)
            XCTAssertEqual(ga.pick, wa.pick, ctx)
            let dp = max(maxDelta(wa.probabilities, ga.probabilities, ctx), maxDelta(wa.raw, ga.raw, ctx),
                         abs((ga.confidence ?? 0) - (wa.confidence ?? 0)))
            XCTAssertLessThanOrEqual(dp, tol, ctx)
            worstP = max(worstP, dp)
            XCTAssertEqual(ga.legend, wa.legend, ctx)
            if let we = wa.expectedScore, let n = wa.legend?.count {
                XCTAssertLessThanOrEqual(abs((ga.expectedScore ?? .infinity) - we), expectedScoreTol(n), ctx)
            }
            guard let wd = wa.debug, let gd = ga.debug else {
                XCTFail("\(ctx): no debug block")
                continue
            }
            XCTAssertEqual(gd.readoutPosition, wd.readoutPosition, ctx)
            XCTAssertEqual(gd.optionTokenIds, wd.optionTokenIds, ctx)
            XCTAssertEqual(gd.temperature, wd.temperature, ctx)
            let dl = maxDelta(wd.rawLogits, gd.rawLogits, ctx)
            if let logitTol { XCTAssertLessThanOrEqual(dl, logitTol, ctx) }
            worstLogit = max(worstLogit, dl)
            XCTAssertEqual(gd.permutationSpread?.keys, wd.permutationSpread?.keys, ctx)
            XCTAssertLessThanOrEqual(maxDelta(wd.permutationSpread, gd.permutationSpread, ctx), spreadTol, ctx)
        }
        return (worstP, worstLogit)
    }

    func maxDelta(_ want: OrderedObject<Double>?, _ got: OrderedObject<Double>?, _ ctx: String) -> Double {
        guard let want, let got else {
            XCTAssertEqual(want == nil, got == nil, "\(ctx): presence")
            return 0
        }
        XCTAssertEqual(got.keys, want.keys, ctx)
        return want.entries.map { abs((got[$0.key] ?? .infinity) - $0.value) }.max() ?? 0
    }

    /// Story 2.2: golden `decide` responses reproduced from the wire requests on
    /// the GPU. Single-ordering cases (each a fixture's prompt, cached prefix as
    /// decide caches it, with and without calibration): probabilities within 1e-3.
    /// Permuted cases: answers, ids, temperature and usage exact; the probability
    /// distance is reported, not asserted, because their extra orderings are new
    /// prompts with near-tied labels where bf16 path noise is amplified (story 2.2
    /// records the measured deltas; `testDecideReplaysGoldenLogitsExactly` checks
    /// the same cases' arithmetic exactly).
    func testDecideMatchesGoldenDecide() async throws {
        let runner = try await GPU.runner()
        let ref = try DecideReference.load()
        XCTAssertEqual(ref.format, "mlx-classifiers-swift-decide-reference/1")
        XCTAssertEqual(ref.cases.count, 15)
        for c in ref.cases {
            if let want = c.error {
                XCTAssertThrowsError(try decide(c.request, runner: runner, temperature: c.temperature), c.id) { error in
                    let e = error as? ClassifierError
                    XCTAssertEqual(e?.type, want.type, c.id)
                    XCTAssertEqual(e?.message, want.message, c.id)
                }
                print("DECIDE \(c.id): error \(want.type) matches")
                continue
            }
            let want = try XCTUnwrap(c.response, c.id)
            let got = try decide(c.request, runner: runner, temperature: c.temperature)
            XCTAssertNotNil(got.timingMs?.total, c.id)
            let permuted = (c.request.options?.resolvedPermutations ?? 1) > 1
            let tol = permuted ? Double.infinity : Self.tol
            // sum(i * p_i) with each p_i within tol: |Δ| <= tol * sum(i).
            let worst = compare(c.id, want, got, tol: tol, spreadTol: permuted ? .infinity : 2 * Self.tol,
                                expectedScoreTol: { n in tol * Double(n * (n - 1) / 2) }, logitTol: nil)
            print("DECIDE \(c.id)\(permuted ? " (permuted, reported)" : ""): max|dp| = \(worst.p) "
                  + "max|dlogit| = \(worst.logit) usage=\(got.usage.inputTokens)/\(got.usage.cachedTokens ?? -1) "
                  + String(format: "%.1f ms", got.timingMs?.total ?? -1))
        }
    }
}

// MARK: - no GPU

final class DecideLogicTests: XCTestCase {
    /// Golden `permutation_orders`, values captured from the golden venv (CPython 3.12).
    func testPermutationOrdersMatchPython() {
        XCTAssertEqual(permutationOrders(n: 2, count: 8, seed: "q"),
                       [[0, 1], [1, 0], [1, 0], [0, 1], [0, 1], [1, 0], [0, 1], [0, 1]])
        XCTAssertEqual(permutationOrders(n: 3, count: 8, seed: "route"),
                       [[0, 1, 2], [1, 2, 0], [2, 0, 1], [1, 2, 0], [2, 1, 0], [1, 0, 2], [0, 2, 1], [1, 2, 0]])
        XCTAssertEqual(permutationOrders(n: 4, count: 8, seed: "level"),
                       [[0, 1, 2, 3], [1, 2, 3, 0], [2, 3, 0, 1], [3, 0, 1, 2], [1, 2, 0, 3], [0, 2, 1, 3],
                        [0, 3, 2, 1], [2, 1, 3, 0]])
        XCTAssertEqual(permutationOrders(n: 5, count: 8, seed: "x"),
                       [[0, 1, 2, 3, 4], [1, 2, 3, 4, 0], [2, 3, 4, 0, 1], [3, 4, 0, 1, 2], [4, 0, 1, 2, 3],
                        [3, 1, 4, 2, 0], [3, 4, 1, 0, 2], [3, 1, 4, 2, 0]])
    }

    /// Golden `calibrate.py` on a fixed synthetic set (values from the golden venv).
    func testTemperatureScalerMatchesGolden() throws {
        var rows: [[Double]] = [], labels: [Int] = []
        for i in 0 ..< 40 {
            var base = (0 ..< 3).map { Foundation.sin(Double(i) * 1.7 + Double($0)) * 4 }
            base[(i * 7) % 3] += 6.0
            rows.append(base)
            labels.append(i % 3)
        }
        var ts = TemperatureScaler()
        let r = try ts.fit(rows, labels)
        XCTAssertEqual(r.temperature, 0.6895193805365192, accuracy: 1e-12)
        XCTAssertEqual(r.nllBefore, 0.1893393287728386, accuracy: 1e-12)
        XCTAssertEqual(r.nllAfter, 0.1769201193584206, accuracy: 1e-12)
        XCTAssertEqual(r.eceBefore, 0.08976968308676084, accuracy: 1e-12)
        XCTAssertEqual(r.eceAfter, 0.06843086115083778, accuracy: 1e-12)
        XCTAssertEqual(ts.temperature, r.temperature)
        XCTAssertEqual(try nll(Array(rows[..<5]), Array(labels[..<5]), temperature: 2.5), 0.40895760893720745, accuracy: 1e-12)
        XCTAssertEqual(try expectedCalibrationError(rows[..<7].map { softmax($0) }, Array(labels[..<7]), nBins: 5),
                       0.21341657974335376, accuracy: 1e-12)
        XCTAssertThrowsError(try nll([], []))
        XCTAssertEqual(ts.apply([0, 0]), [0.5, 0.5])
    }

    func testTokenLogprob() {
        let row = LogitRow.fromList([0, Foundation.log(3)])
        XCTAssertEqual(tokenLogprob(row, 1), Foundation.log(0.75), accuracy: 1e-15)
    }

    func testShapeAnswer() {
        let score = Question.score(ScoreQuestion(instructions: "?", criteria: ["lo", "mid", "hi"]))
        guard case .score(let s) = shapeAnswer(score, options: ["0", "1", "2"], raw: [0.2, 0.5, 0.3],
                                                calibrated: [0.25, 0.25, 0.5]) else { return XCTFail() }
        XCTAssertEqual(s.score, 2)
        XCTAssertEqual(s.expectedScore!, 1.25, accuracy: 1e-15)
        XCTAssertEqual(s.legend, OrderedObject([("0", "lo"), ("1", "mid"), ("2", "hi")]))
        XCTAssertEqual(s.confidence, 0.5)
        let noul = Question.noul(NoulQuestion(instructions: "?"))
        guard case .noul(let n) = shapeAnswer(noul, options: ["yes", "no"], raw: [0.9, 0.1], calibrated: [0.8, 0.2])
        else { return XCTFail() }
        XCTAssertEqual(n.noul, 0.8)
        XCTAssertEqual(n.rawNoul, 0.9)
        let choice = Question.choice(ChoiceQuestion(instructions: "?", criteria: OrderedObject([("a", nil), ("b", nil)])))
        guard case .choice(let c) = shapeAnswer(choice, options: ["a", "b"], raw: [0.5, 0.5], calibrated: [0.5, 0.5])
        else { return XCTFail() }
        XCTAssertEqual(c.choice, "a", "first index wins ties")
    }
}
