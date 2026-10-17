import Foundation
import MLX
import XCTest

@testable import MLXClassifiers

/// Story 7.1: Gemma 4 26B-A4B (MoE, bidirectional image blocks) through `MLXRunner`
/// against golden's `fixtures/26b/` (golden story 4.4), under the same per-fixture
/// gate rules as the E2B set. Model-gated: skipped without the metallib (plain
/// `swift test`), with `MLXCLASSIFIERS_SKIP_GPU=1`, or when the 26B snapshot at the
/// fixtures' revision is not in the local HF cache (`MLXCLASSIFIERS_26B_DIR`
/// overrides the snapshot directory).
enum GPU26B {
    static let repo = "mlx-community/gemma-4-26B-A4B-it-qat-4bit"
    static let dir = fixturesDir.appendingPathComponent("26b")
    nonisolated(unsafe) static var shared: MLXRunner?
    nonisolated(unsafe) static var loadSeconds: Double = 0

    static func fixtures() throws -> [GoldenFixture] { try Fixtures.load(directory: dir) }

    static func runner() async throws -> MLXRunner {
        let env = ProcessInfo.processInfo.environment
        if env["MLXCLASSIFIERS_SKIP_GPU"] == "1" { throw XCTSkip("MLXCLASSIFIERS_SKIP_GPU=1") }
        guard GPU.metallibAvailable() else { throw XCTSkip("no MLX metallib beside the test bundle; run under xcodebuild") }
        if let r = shared { return r }
        let fx = try fixtures()
        let model = fx[0].model
        let snapshot = env["MLXCLASSIFIERS_26B_DIR"].map { URL(fileURLWithPath: $0) }
            ?? MLXRunner.cachedSnapshot(repo: repo, revision: model.revision)
        guard let snapshot else { throw XCTSkip("\(repo) is not in the local HF cache") }
        let t0 = Date()
        let r = try await MLXRunner.load(directory: snapshot, modelId: repo, expectedTokenizerSha256: model.tokenizerSha256)
        loadSeconds = Date().timeIntervalSince(t0)
        print(String(format: "LOAD26B %.2f s, active %.2f GB, peak %.2f GB from %@", loadSeconds,
                     Double(Memory.activeMemory) / 1e9, Double(Memory.peakMemory) / 1e9, snapshot.path))
        r.promptCache = nil  // cold, as golden runs fixtures
        shared = r
        return r
    }
}

final class Gemma26BFixtureTests: XCTestCase {
    /// Golden story 4.4's five cases; only the near-tied gridworld action case is `answer-and-ids`.
    func testVendored26BFixturesArePinnedToThe26B() throws {
        let fx = try GPU26B.fixtures()
        XCTAssertEqual(fx.map(\.id), [
            "choice-first-token-image-gridworld-vt140", "choice-first-token-image-gridworld-you-vt140",
            "choice-sequence-multi-token-options", "noul-first-token", "noul-sequence",
        ])
        XCTAssertEqual(Set(fx.map(\.model.repo)), [GPU26B.repo])
        XCTAssertEqual(fx.filter { $0.gate != nil }.map(\.id), ["choice-first-token-image-gridworld-vt140"])
    }

    func testBoundsOffRunsMatchesGolden() {
        XCTAssertEqual(boundsOffRuns([0, 5, 10], [(3, 7)]), [0, 3, 10])
        XCTAssertEqual(boundsOffRuns([0, 3, 5, 10], [(3, 7)]), [0, 3, 7, 10])
        XCTAssertEqual(boundsOffRuns([0, 4, 10], []), [0, 4, 10])
    }

    func testImageBlockOverlay() throws {
        if ProcessInfo.processInfo.environment["MLXCLASSIFIERS_SKIP_GPU"] == "1" { throw XCTSkip("MLXCLASSIFIERS_SKIP_GPU=1") }
        guard GPU.metallibAvailable() else { throw XCTSkip("no MLX metallib beside the test bundle; run under xcodebuild") }
        // 2 cached columns + 4 segment rows; rows 1..2 are image 0
        let causal = MLXArray(Int32(2) ..< Int32(6))[0..., .newAxis] .>= MLXArray(Int32(0) ..< Int32(6))[.newAxis]
        let got = Gemma4TextModelInner.overlayImageBlocks(causal, blocks: MLXArray([-1, 0, 0, -1] as [Int32]))
        let rows = (0 ..< 4).map { got[$0].asArray(Bool.self) }
        XCTAssertEqual(rows[1], [true, true, true, true, true, false])  // image row sees the later image row
        XCTAssertEqual(rows[0], [true, true, true, false, false, false])
        XCTAssertEqual(rows[3], [true, true, true, true, true, true])
    }

    /// Every 26B fixture from its `request` through `runCase`, golden `compare` semantics
    /// under its gate (ids, image runs and answer exact; raw probabilities within 1e-3 on full).
    func test26BFixtures() async throws {
        let runner = try await GPU26B.runner()
        for fx in try GPU26B.fixtures() {
            let t0 = Date()
            let got = try runCase(fx, runner: runner)
            let ms = Date().timeIntervalSince(t0) * 1e3
            let rep = compareCase(fx, result: got)
            XCTAssertTrue(rep.tokenIdsEqual, fx.id)
            XCTAssertEqual(got.answer, fx.expected.answer, fx.id)
            XCTAssertTrue(rep.ok, "\(fx.id): \(rep.failures.joined(separator: "; "))")
            let logitDeltas = got.rawLogits.map { r in r.entries.map { $0.value - fx.expected.rawLogits![$0.key]! } } ?? []
            print("FIX26B \(fx.id) [gate \(rep.gate)]: max|dp| = \(rep.maxProbDistance.map { "\($0)" } ?? "n/a") "
                  + "answer=\(got.answer ?? "?") logit deltas=\(logitDeltas) " + String(format: "%.1f ms", ms))
        }
    }

    /// Timing: ms per decision on the text first-token case and the vt140 image case
    /// (cold prompt cache; image with the soft-token memo cleared, i.e. a new frame,
    /// and with a memo hit). Prints `TIME26B` lines; never fails on speed.
    func test26BDecisionTiming() async throws {
        let runner = try await GPU26B.runner()
        let fx = Dictionary(uniqueKeysWithValues: try GPU26B.fixtures().map { ($0.id, $0) })
        func median(_ xs: [Double]) -> Double { xs.sorted()[xs.count / 2] }
        func time(_ id: String, reps: Int = 7, clear: Bool) throws -> [Double] {
            var out: [Double] = []
            for _ in 0 ..< reps {
                if clear { runner.vision?.clear() }
                let t0 = Date()
                _ = try runCase(fx[id]!, runner: runner)
                out.append(Date().timeIntervalSince(t0) * 1e3)
            }
            return out
        }
        _ = try time("noul-first-token", reps: 2, clear: false)  // warm-up
        let text = try time("noul-first-token", clear: false)
        let imageNew = try time("choice-first-token-image-gridworld-vt140", clear: true)
        let imageHit = try time("choice-first-token-image-gridworld-vt140", clear: false)
        print(String(format: "TIME26B text first-token median %.1f ms %@", median(text), text.map { String(format: "%.0f", $0) }))
        print(String(format: "TIME26B image vt140 new frame median %.1f ms %@", median(imageNew), imageNew.map { String(format: "%.0f", $0) }))
        print(String(format: "TIME26B image vt140 memo hit median %.1f ms %@", median(imageHit), imageHit.map { String(format: "%.0f", $0) }))
        print(String(format: "MEM26B active %.2f GB, peak %.2f GB", Double(Memory.activeMemory) / 1e9, Double(Memory.peakMemory) / 1e9))
    }
}
