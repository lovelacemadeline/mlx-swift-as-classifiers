import Foundation
import XCTest

@testable import MLXClassifiers

/// The comparison semantics of golden `fixtures.py` (`run_case` + the checker):
/// raw probabilities within `defaultProbabilityTolerance` (1e-3) and the same answer.
///
/// Prompt compilation and label binding by the tokenizer are story 1.2, so these
/// tests feed `expected.tokens` and `expected.option_token_ids` (the in-context label
/// ids) straight to `scoreFirstToken`; everything from the label rules onward is real.
enum FixtureRun {
    static func load(_ id: String) throws -> GoldenFixture {
        try Fixtures.load(file: fixturesDir.appendingPathComponent(id + ".json"))
    }

    static func score(_ fx: GoldenFixture, runner: Runner) throws -> FirstTokenScore {
        let e = fx.expected
        return try scoreFirstToken(
            runner: runner, tokens: e.tokens, readoutPosition: e.readoutPosition,
            options: e.options, labels: e.labels,
            labelSequences: e.options.map { e.optionTokenIds[$0]! },
            labelStyle: fx.labelStyle, prefixLen: e.prefixLen, prefixCached: fx.prefixCached)
    }
}

// MARK: - GPU: the vendored Gemma 4 E2B on the split path

/// Needs the MLX Metal library (build with xcodebuild, see docs/runbooks/build-and-test.md)
/// and the E2B snapshot in the local HF cache. Skipped when `MLXCLASSIFIERS_SKIP_GPU=1`
/// (under xcodebuild: `TEST_RUNNER_MLXCLASSIFIERS_SKIP_GPU=1`), when the metallib is
/// absent (plain `swift test`), or when the snapshot is missing. `MLXCLASSIFIERS_MODEL_DIR`
/// overrides the snapshot directory.
final class TextFixtureTests: XCTestCase {
    nonisolated(unsafe) static var shared: MLXRunner?
    nonisolated(unsafe) static var loadError: Error?

    /// Whether MLX's Metal library sits next to the test bundle (xcodebuild puts
    /// `mlx-swift_Cmlx.bundle` there; `swift build` builds no metallib).
    static func metallibAvailable() -> Bool {
        let dir = Bundle(for: TextFixtureTests.self).bundleURL.deletingLastPathComponent()
        let fm = FileManager.default
        if fm.fileExists(atPath: dir.appendingPathComponent("mlx.metallib").path) { return true }
        let bundle = dir.appendingPathComponent("mlx-swift_Cmlx.bundle")
        guard let walk = fm.enumerator(atPath: bundle.path) else { return false }
        return walk.contains { ($0 as? String)?.hasSuffix(".metallib") == true }
    }

    func runner() throws -> MLXRunner {
        let env = ProcessInfo.processInfo.environment
        if env["MLXCLASSIFIERS_SKIP_GPU"] == "1" { throw XCTSkip("MLXCLASSIFIERS_SKIP_GPU=1") }
        guard Self.metallibAvailable() else {
            throw XCTSkip("no MLX metallib beside the test bundle; run under xcodebuild (docs/runbooks/build-and-test.md)")
        }
        if let r = Self.shared { return r }
        if let e = Self.loadError { throw e }
        let dir = env["MLXCLASSIFIERS_MODEL_DIR"].map { URL(fileURLWithPath: $0) }
            ?? MLXRunner.cachedSnapshot(revision: "42f62737af7a9fd8c1d55d79666c1a217be4e2e2")
        guard let dir else { throw XCTSkip("\(MLXRunner.defaultModel) is not in the local HF cache") }
        do {
            let t0 = Date()
            let r = try MLXRunner.load(directory: dir)
            print("MLXRunner.load: \(String(format: "%.2f", Date().timeIntervalSince(t0))) s from \(dir.path)")
            Self.shared = r
            return r
        } catch {
            Self.loadError = error
            throw error
        }
    }

    /// Runs `id` and asserts it against its fixture; returns the max |Δp|.
    @discardableResult
    func check(_ id: String) throws -> Double {
        let r = try runner()
        let fx = try FixtureRun.load(id)
        let t0 = Date()
        let s = try FixtureRun.score(fx, runner: r)
        let ms = Date().timeIntervalSince(t0) * 1e3
        let want = try XCTUnwrap(fx.expected.rawProbabilities)
        var maxDelta = 0.0
        for (i, o) in s.options.enumerated() {
            let d = abs(s.probabilities[i] - want[o]!)
            maxDelta = max(maxDelta, d)
            XCTAssertLessThanOrEqual(
                d, FixtureFormat.defaultProbabilityTolerance,
                "\(id) \(o): got \(s.probabilities[i]), want \(want[o]!)")
        }
        XCTAssertEqual(s.answer, fx.expected.answer, id)
        XCTAssertEqual(s.cachedTokens, fx.prefixCached ? fx.expected.prefixLen : 0, id)
        let logitDeltas = s.options.enumerated().map { i, o in s.logits[i] - fx.expected.rawLogits![o]! }
        print("FIXTURE \(id): max|dp| = \(maxDelta) answer=\(s.answer) logit deltas=\(logitDeltas) "
              + String(format: "%.1f ms", ms))
        return maxDelta
    }

    func testNoulFirstToken() throws { try check("noul-first-token") }

    func testChoiceFirstTokenObjectState() throws { try check("choice-first-token-object-state") }

    func testChoiceFirstTokenCachedListState() throws { try check("choice-first-token-cached-list-state") }

    func testScoreFirstTokenNoPrefix() throws { try check("score-first-token-no-prefix") }

    /// Story 2.1 AC 2: the same case twice is bit-identical.
    func testNoulFirstTokenIsBitIdenticalTwice() throws {
        let r = try runner()
        let fx = try FixtureRun.load("noul-first-token")
        let a = try FixtureRun.score(fx, runner: r)
        let b = try FixtureRun.score(fx, runner: r)
        XCTAssertEqual(a.logits, b.logits)
        XCTAssertEqual(a.probabilities, b.probabilities)
    }

    /// Golden `head_calls`: the vocab projection sees only the readout row.
    func testHeadProjectsOnlyTheReadoutRow() throws {
        let r = try runner()
        let fx = try FixtureRun.load("noul-first-token")
        let before = r.headCalls.count
        _ = try r.prefill(fx.expected.tokens, prefixCache: nil, readoutPositions: [fx.expected.readoutPosition])
        XCTAssertEqual(r.headCalls.count, before + 1)
        XCTAssertEqual(r.headCalls.last, [1, 1, 1536])
    }
}

// MARK: - no GPU: label rules and the prefill contract

/// A Runner that must never be reached (the label rules fail first).
struct UnreachableRunner: Runner {
    func tokenize(_ text: String) throws -> [Int] { XCTFail("tokenize called"); return [] }
    func cachePrefix(_ tokens: [Int]) throws -> PrefixCache { XCTFail("cachePrefix called"); return PrefixCache(tokens: tokens, state: nil) }
    func prefill(_ tokens: [Int], prefixCache: PrefixCache?, readoutPositions: [Int]?) throws -> PrefillReadout {
        XCTFail("prefill called")
        throw RunnerError.invalidArgument("unreachable")
    }
}

final class FirstTokenReadoutTests: XCTestCase {
    /// `choice-first-token-multi-token-label-error`: 'chargeback' is two tokens in
    /// context, so first-token mode refuses before any prefill, with golden's message.
    func testMultiTokenLabelError() throws {
        let fx = try FixtureRun.load("choice-first-token-multi-token-label-error")
        let want = try XCTUnwrap(fx.expected.error)
        XCTAssertThrowsError(try FixtureRun.score(fx, runner: UnreachableRunner())) { error in
            guard let e = error as? LabelTokenizationError else { return XCTFail("\(error)") }
            XCTAssertEqual(LabelTokenizationError.typeName, want.type)
            XCTAssertEqual(e.message, want.message)
        }
    }

    func testSharedFirstTokenIsAnError() {
        XCTAssertThrowsError(try bindLabels(labels: ["ab", "ac"], sequences: [[1, 2], [1, 3]])) { error in
            XCTAssertEqual(
                (error as? LabelTokenizationError)?.message,
                "labels 'ab' and 'ac' share a first token; they first differ at depth 2, which first-token mode cannot read; "
                    + LabelHint.sequence)
        }
        let tree = PrefixTree.build(labels: ["a", "b"], sequences: [[1], [1]])
        XCTAssertNil(tree.depth)
        XCTAssertEqual(tree.collisions, [["a", "b"]])
    }

    func testSoftmaxAndArgmax() {
        let p = softmax([0, Foundation.log(3)])  // shift-invariant: max is subtracted first
        XCTAssertEqual(softmax([1000, 1000]), [0.5, 0.5])
        XCTAssertEqual(p[0], 0.25, accuracy: 1e-15)
        XCTAssertEqual(p[1], 0.75, accuracy: 1e-15)
        XCTAssertEqual(argmax([0.2, 0.5, 0.5]), 1, "first maximum wins, as Python max()")
        XCTAssertEqual(softmax([0, 0], temperature: 2), [0.5, 0.5])
    }

    func testPrefillContract() throws {
        let pc = PrefixCache(tokens: [1, 2], state: nil)
        XCTAssertEqual(try checkPrefillArgs([1, 2, 3], prefixCache: pc, readoutPositions: nil).positions, [2])
        XCTAssertThrowsError(try checkPrefillArgs([1, 2], prefixCache: pc, readoutPositions: nil))
        XCTAssertThrowsError(try checkPrefillArgs([9, 2, 3], prefixCache: pc, readoutPositions: nil))
        XCTAssertThrowsError(try checkPrefillArgs([1, 2, 3], prefixCache: pc, readoutPositions: [1]))
        XCTAssertThrowsError(try checkPrefillArgs([], prefixCache: nil, readoutPositions: nil))
    }
}
