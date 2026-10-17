import Foundation
import MLX
import MLXLMCommon
import XCTest

@testable import MLXClassifiers

// MARK: - no GPU: the store (golden tests/test_prompt_cache.py)

final class PromptCacheStoreTests: XCTestCase {
    let rewind: PromptCacheStore.Rewind = { state, _, hit in ["rewound", state, hit] as [Any] }

    func view(_ r: (hit: Int, state: Any?)) -> String {
        guard let s = r.state as? [Any] else { return "\(r.hit) nil" }
        return "\(r.hit) \(s.map { "\($0)" })"
    }

    func testCommonPrefixLen() {
        XCTAssertEqual(commonPrefixLen([1, 2, 3], [1, 2, 4]), 2)
        XCTAssertEqual(commonPrefixLen([1, 2], [1, 2, 3]), 2)
        XCTAssertEqual(commonPrefixLen([], [1]), 0)
        XCTAssertEqual(commonPrefixLen(Array(0 ..< 1000), Array(0 ..< 999) + [5]), 999)
    }

    func testLookupReturnsTheLongestCommonPrefixCapped() {
        let s = PromptCacheStore()
        s.insert([1, 2, 3, 4], state: "a", nbytes: 1)
        s.insert([1, 2, 9], state: "b", nbytes: 1)
        XCTAssertEqual(view(s.lookup([1, 2, 3, 7, 7], rewind: rewind)), "3 [\"rewound\", \"a\", \"3\"]")
        let tie = s.lookup([1, 2, 3, 4, 5], rewind: rewind, maxLen: 2)  // a and b tie at 2
        XCTAssertEqual(tie.hit, 2)
        // a was bumped by the hit above; ties go to the least recently used entry (Python's stable sort)
        XCTAssertEqual((tie.state as? [Any])?[1] as? String, "b")
        XCTAssertEqual(view(s.lookup([1, 2, 3, 4, 5], rewind: rewind, minLen: 4)), "0 nil")
        XCTAssertEqual(view(s.lookup([5], rewind: rewind)), "0 nil")
        XCTAssertEqual(s.hits, 2)
        XCTAssertEqual(s.misses, 2)
    }

    func testLookupFallsBackWhenARewindIsNotRepresentable() {
        let s = PromptCacheStore()
        s.insert([1, 2, 3, 4, 5], state: "deep", nbytes: 1)
        s.insert([1, 2, 3], state: "shallow", nbytes: 1)
        let r = s.lookup([1, 2, 3, 4, 9], rewind: { state, _, _ in (state as? String) == "deep" ? nil : state })
        XCTAssertEqual(r.hit, 3)
        XCTAssertEqual(r.state as? String, "shallow")
        XCTAssertEqual(s.unrepresentable, 1)
        XCTAssertEqual(view(s.lookup([1, 2, 3, 4, 9], rewind: { _, _, _ in nil })), "0 nil")
    }

    func testLRUEvictsByEntriesAndByBytes() {
        let s = PromptCacheStore(maxEntries: 2, maxBytes: 100)
        s.insert([1], state: "a", nbytes: 10)
        s.insert([2], state: "b", nbytes: 10)
        _ = s.lookup([1, 5], rewind: rewind)  // bumps [1]
        s.insert([3], state: "c", nbytes: 10)
        XCTAssertEqual(s.keys, [[1], [3]])  // [2] was least recently used
        XCTAssertFalse(s.insert([3], state: "c2", nbytes: 10))
        XCTAssertEqual(s.keys, [[1], [3]])
        s.insert([4], state: "d", nbytes: 95)  // over the byte budget: evicts until it fits
        XCTAssertEqual(s.keys, [[4]])
        XCTAssertEqual(s.nbytes, 95)
        XCTAssertFalse(s.insert([5], state: "huge", nbytes: 101))
        XCTAssertFalse(s.contains([5]))
        XCTAssertFalse(s.insert([], state: "empty", nbytes: 1))
    }

    func testWithoutDeltaIsTheRunnerItselfWithoutAStore() {
        let r = UnreachableRunner()
        XCTAssertTrue(withoutDelta(r) is UnreachableRunner)
    }
}

// MARK: - GPU (Metal only, no model): the exact rewind on MLXLMCommon caches

final class KVRewindTests: XCTestCase {
    override func setUpWithError() throws {
        if ProcessInfo.processInfo.environment["MLXCLASSIFIERS_SKIP_GPU"] == "1" { throw XCTSkip("MLXCLASSIFIERS_SKIP_GPU=1") }
        guard GPU.metallibAvailable() else { throw XCTSkip("no MLX metallib beside the test bundle; run under xcodebuild") }
    }

    /// Rows 0 ..< n of a deterministic (1, 2, n, 4) key tensor: row r holds r + 1 everywhere, offset by `salt`.
    func rows(_ range: Range<Int>, salt: Float = 0) -> MLXArray {
        let r = MLXArray(range.map { Float($0 + 1) + salt }).reshaped(1, 1, range.count, 1)
        return broadcast(r, to: [1, 2, range.count, 4]).asType(.bfloat16)
    }

    /// Feed `chunks` (row ranges) to `cache`; return the keys the last update presented to attention.
    @discardableResult
    func feed(_ cache: KVCache, _ chunks: [Range<Int>], salt: Float = 0) -> MLXArray {
        var presented = MLXArray(0)
        for c in chunks {
            let k = rows(c, salt: salt)
            presented = cache.update(keys: k, values: k * 2).0
        }
        eval(presented)
        return presented
    }

    /// The rewound cache presents exactly what a cache fed only the first `h` rows
    /// presents, for multi-row and single-row continuations; both cache kinds.
    func assertRewindExact(make: () -> KVCache, stored: [Range<Int>], h: Int, cold: [Range<Int>],
                           file: StaticString = #filePath, line: UInt = #line) throws {
        let entry = make()
        feed(entry, stored)
        let storedLen = stored.last!.upperBound
        let rewound = try XCTUnwrap(rewindCache([entry], storedLen: storedLen, hitLen: h), file: file, line: line)[0]
        XCTAssertEqual(rewound.offset, h, file: file, line: line)
        XCTAssertEqual(entry.offset, storedLen, "the stored entry is never mutated", file: file, line: line)
        for cont in [[h ..< h + 5], [h ..< h + 1, h + 1 ..< h + 2], [h ..< h + 1]] {
            let a = rewound.copy(), b = make()
            feed(b, cold)
            let ka = feed(a, cont, salt: 0.5), kb = feed(b, cont, salt: 0.5)
            XCTAssertEqual(ka.shape, kb.shape, "\(cont)", file: file, line: line)
            XCTAssertTrue(ka.shape == kb.shape && allClose(ka, kb, rtol: 0, atol: 0).item(Bool.self), "\(cont)",
                          file: file, line: line)
        }
    }

    func testSimpleCacheTrimsToAnyLength() throws {
        try assertRewindExact(make: { KVCacheSimple() }, stored: [0 ..< 30], h: 17, cold: [0 ..< 17])
        try assertRewindExact(make: { KVCacheSimple() }, stored: [0 ..< 10, 10 ..< 30], h: 3, cold: [0 ..< 3])
    }

    func testRotatingCacheBeforeTheWindowSlides() throws {
        // built from empty in one chunk: start == 0, rewinds anywhere (golden "retained entry")
        try assertRewindExact(make: { RotatingKVCache(maxSize: 8) }, stored: [0 ..< 30], h: 20, cold: [0 ..< 20])
        try assertRewindExact(make: { RotatingKVCache(maxSize: 8) }, stored: [0 ..< 30], h: 5, cold: [0 ..< 5])
    }

    func testRotatingCacheRewindsIntoItsLastChunkPastTheWindow() throws {
        // two chunks: holds W - 1 + 10 rows [13, 30); h = 25 keeps W - 1 rows behind it
        try assertRewindExact(make: { RotatingKVCache(maxSize: 8) }, stored: [0 ..< 20, 20 ..< 30], h: 25,
                              cold: [0 ..< 20, 20 ..< 25])
        try assertRewindExact(make: { RotatingKVCache(maxSize: 8) }, stored: [0 ..< 20, 20 ..< 30], h: 20,
                              cold: [0 ..< 20])
    }

    func testRotatingCacheRefusesARewindTheWindowCannotRepresent() {
        let c = RotatingKVCache(maxSize: 8)
        feed(c, [0 ..< 20, 20 ..< 30])  // rows [13, 30)
        XCTAssertNil(rewindCache([c], storedLen: 30, hitLen: 19))
        XCTAssertNotNil(rewindCache([c], storedLen: 30, hitLen: 20))
        let ring = RotatingKVCache(maxSize: 8)
        feed(ring, [0 ..< 8] + (8 ..< 12).map { $0 ..< $0 + 1 })  // single-row writes wrapped the ring
        XCTAssertNil(rewindCache([ring], storedLen: 12, hitLen: 10))
        let mixed: [KVCache] = [KVCacheSimple(), c]
        feed(mixed[0], [0 ..< 30])
        XCTAssertNil(rewindCache(mixed, storedLen: 30, hitLen: 19), "one layer refusing refuses the list")
    }
}

// MARK: - GPU: the delta store on Gemma 4 E2B against golden

/// `Tests/MLXClassifiersTests/GoldenDelta/delta-reference.json`, written by
/// `scripts/golden_delta_reference.py` from golden `decide` with delta caching on.
struct DeltaReference: Decodable {
    struct Stats: Decodable, Equatable {
        var entries: Int
        var hits: Int
        var misses: Int
        var unrepresentable: Int
        var nbytes: Int
    }

    struct Step: Decodable {
        var request: SystemOneRequest
        var response: SystemOneResponse
        var prefillCalls: [[Int]]
        var store: Stats
        var coldResponse: SystemOneResponse

        enum CodingKeys: String, CodingKey {
            case request, response, store
            case prefillCalls = "prefill_calls"
            case coldResponse = "cold_response"
        }
    }

    struct Scenario: Decodable {
        var id: String
        var maxEntries: Int
        var steps: [Step]

        enum CodingKeys: String, CodingKey {
            case id, steps
            case maxEntries = "max_entries"
        }
    }

    var format: String
    var scenarios: [Scenario]

    static func load() throws -> DeltaReference {
        let url = repoRoot.appendingPathComponent("Tests/MLXClassifiersTests/GoldenDelta/delta-reference.json")
        return try SystemOneJSON.decode(DeltaReference.self, from: try Data(contentsOf: url))
    }
}

/// Every answer's raw and calibrated probabilities (noul as P(yes)), flattened `question.option`.
func probabilityRows(_ r: SystemOneResponse) -> [(String, Double)] {
    r.answers.entries.flatMap { e -> [(String, Double)] in
        let a = AnswerView(e.value)
        return a.probabilities.entries.map { ("\(e.key).\($0.key)", $0.value) }
            + (a.raw?.entries.map { ("\(e.key).raw.\($0.key)", $0.value) } ?? [])
    }
}

func maxProbDelta(_ a: SystemOneResponse, _ b: SystemOneResponse) -> Double {
    let x = probabilityRows(a), y = probabilityRows(b)
    precondition(x.map(\.0) == y.map(\.0), "different answer shapes")
    return zip(x, y).map { abs($0.1 - $1.1) }.max() ?? 0
}

final class PromptCacheTests: XCTestCase {
    var runner: MLXRunner!

    override func setUp() async throws {
        runner = try await GPU.runner()
    }

    override func tearDown() {
        runner?.promptCache = nil  // the shared runner runs cold everywhere else (as golden run_case)
    }

    func fresh(_ maxEntries: Int = PromptCacheDefaults.maxEntries) -> PromptCacheStore {
        let s = PromptCacheStore(maxEntries: maxEntries)
        runner.promptCache = s
        return s
    }

    /// Golden decide with delta on, step by step on a fresh store: `cached_tokens`,
    /// `input_tokens`, each prefill's `(cached, computed)` and the store's
    /// entries / hits / misses / refused rewinds equal golden's exactly on every step.
    /// Single-ordering first-token steps (the synthetic edits near the end, inside the
    /// sliding window and before it; the tail change; the ticks; the cached-list
    /// fixture): probabilities within 1e-3 of golden's delta response and of this
    /// runner's own cold run (`delta_cache: false`), answers equal. Permuted,
    /// two-question and sequence steps: answers equal, distances reported, as story
    /// 2.2 reports permuted cases (a hit changes the chunk shape, and those prompts
    /// amplify bf16 noise; golden's own delta-vs-cold distance on them is printed).
    /// The store and the rewind are exact (KVRewindTests); the delta path's only
    /// freedom is the chunk shape of the rows it computes.
    func testDeltaHitMatchesColdRun() throws {
        let ref = try DeltaReference.load()
        XCTAssertEqual(ref.format, "mlx-classifiers-swift-delta-reference/1")
        var worstCold = 0.0, bitIdentical = 0, total = 0
        for sc in ref.scenarios {
            let store = fresh(sc.maxEntries)
            for (i, step) in sc.steps.enumerated() {
                let ctx = "\(sc.id)[\(i)]"
                let n0 = runner.prefillCalls.count
                let got = try decide(step.request, runner: runner)
                let calls = runner.prefillCalls[n0...].map { [$0.cached, $0.computed] }
                XCTAssertEqual(got.usage, step.response.usage, ctx)
                XCTAssertEqual(calls, step.prefillCalls, ctx)
                XCTAssertEqual([store.count, store.hits, store.misses, store.unrepresentable],
                               [step.store.entries, step.store.hits, step.store.misses, step.store.unrepresentable], ctx)
                let opts = step.request.options ?? RequestOptions()
                let single = opts.resolvedReadout == .firstToken && opts.resolvedPermutations == 1
                    && step.request.questions.count == 1
                let tol = single ? FixtureFormat.defaultProbabilityTolerance : .infinity
                let dGolden = maxProbDelta(step.response, got)
                XCTAssertLessThanOrEqual(dGolden, tol, ctx)
                XCTAssertEqual(got.answers.values.map { AnswerView($0).pick }, step.response.answers.values.map { AnswerView($0).pick }, ctx)

                var coldReq = step.request
                var o = coldReq.options ?? RequestOptions()
                o.deltaCache = false
                coldReq.options = o
                let before = (store.count, store.hits, store.misses)
                let cold = try decide(coldReq, runner: runner)
                XCTAssertTrue(before == (store.count, store.hits, store.misses), "\(ctx): delta_cache false touched the store")
                XCTAssertEqual(cold.usage, step.coldResponse.usage, ctx)
                let dCold = maxProbDelta(cold, got)
                XCTAssertLessThanOrEqual(dCold, tol, ctx)
                XCTAssertEqual(cold.answers.values.map { AnswerView($0).pick }, got.answers.values.map { AnswerView($0).pick }, ctx)
                worstCold = max(worstCold, dCold)
                total += 1
                if dCold == 0 { bitIdentical += 1 }
                print("DELTA \(ctx)\(single ? "" : " (reported)"): usage=\(got.usage.inputTokens)/\(got.usage.cachedTokens ?? -1) calls=\(calls) "
                      + "store=\(store.count)/\(store.hits)/\(store.misses)/\(store.unrepresentable) "
                      + "|dp| vs cold = \(dCold) vs golden = \(dGolden) (golden delta vs cold \(maxProbDelta(step.response, step.coldResponse)))")
            }
        }
        print("DELTA summary: \(bitIdentical)/\(total) steps bit-identical to the cold run; worst |dp| \(worstCold)")
    }

    /// The `cached-list-state` fixture through `runCase` with a store on the runner:
    /// the case runs without delta (golden `run_case`), so its cached count is still
    /// the declared prefix and the store is untouched.
    func testFixtureRunIgnoresTheStore() throws {
        let store = fresh()
        let fx = try FixtureRun.load("choice-first-token-cached-list-state")
        for _ in 0 ..< 2 {
            let n0 = runner.prefillCalls.count
            let got = try runCase(fx, runner: runner)
            XCTAssertTrue(compareCase(expected: fx.expected, result: got, caseId: fx.id).ok)
            let calls = runner.prefillCalls[n0...].map { [$0.cached, $0.computed] }
            XCTAssertEqual(calls, [[fx.expected.prefixLen, fx.expected.tokens.count - fx.expected.prefixLen]])
        }
        XCTAssertEqual(store.count, 0)
        XCTAssertEqual(store.hits + store.misses, 0)
    }

    /// Story 3.1 results table: a ~1k-token prompt with a 1-token edit near the end,
    /// full prefill (delta off) against a delta prefill from the unedited prompt's
    /// entry. Reported, not asserted (not a perf story).
    func testDeltaPrefillTiming() throws {
        let ref = try DeltaReference.load()
        let sc = try XCTUnwrap(ref.scenarios.first { $0.id == "edits-1k-8" })
        func tokens(_ r: SystemOneRequest) throws -> CompiledPrompt {
            try compileQuestion(r, name: "q", template: runner.chatTemplate, tokenize: runner.tokenize)
        }
        let base = try tokens(sc.steps[0].request), edit = try tokens(sc.steps[1].request)
        let shared = commonPrefixLen(base.tokens, edit.tokens)
        func ms(_ f: () throws -> Void) rethrows -> Double {
            let t = DispatchTime.now().uptimeNanoseconds
            try f()
            return Double(DispatchTime.now().uptimeNanoseconds - t) / 1e6
        }
        var full: [Double] = [], deltaMs: [Double] = []
        var cold: [Double] = [], hot: [Double] = []
        for rep in 0 ..< 6 {
            runner.promptCache = nil
            var r: PrefillReadout!
            let f = try ms { r = try runner.prefill(edit.tokens, prefixCache: nil, readoutPositions: [edit.readoutPosition]) }
            cold = gather(try r.at(edit.readoutPosition), Array(0 ..< 16))
            let store = fresh()
            _ = try runner.prefill(base.tokens, prefixCache: nil, readoutPositions: [base.readoutPosition])
            let d = try ms { r = try runner.prefill(edit.tokens, prefixCache: nil, readoutPositions: [edit.readoutPosition]) }
            hot = gather(try r.at(edit.readoutPosition), Array(0 ..< 16))
            XCTAssertEqual(r.cachedTokens, shared)
            XCTAssertEqual(store.hits, 1)
            if rep > 0 { full.append(f); deltaMs.append(d) }  // the first rep warms up
        }
        let lowPower = (try? shell("/usr/bin/pmset", ["-g"]))?.split(separator: "\n")
            .first { $0.contains("lowpowermode") }.map { String($0).trimmingCharacters(in: .whitespaces) } ?? "unknown"
        func med(_ x: [Double]) -> Double { x.sorted()[x.count / 2] }
        print("TIMING prompt \(edit.tokens.count) tokens, edit at \(shared), \(lowPower): full prefill median "
              + String(format: "%.1f ms (min %.1f)", med(full), full.min()!) + ", delta prefill median "
              + String(format: "%.1f ms (min %.1f)", med(deltaMs), deltaMs.min()!)
              + " computing \(edit.tokens.count - shared) rows; readout logits bit-identical: \(cold == hot)")
    }
}

func shell(_ path: String, _ args: [String]) throws -> String {
    let p = Process()
    p.executableURL = URL(fileURLWithPath: path)
    p.arguments = args
    let pipe = Pipe()
    p.standardOutput = pipe
    try p.run()
    p.waitUntilExit()
    return String(decoding: pipe.fileHandleForReading.readDataToEndOfFile(), as: UTF8.self)
}
