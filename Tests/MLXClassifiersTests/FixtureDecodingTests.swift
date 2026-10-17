import CryptoKit
import Foundation
import XCTest

@testable import MLXClassifiers

/// The repo root, found from this file's path (Tests/MLXClassifiersTests/...).
let repoRoot = URL(fileURLWithPath: #filePath)
    .deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
let fixturesDir = repoRoot.appendingPathComponent("fixtures")
let schemaDir = repoRoot.appendingPathComponent("schema")

final class FixtureDecodingTests: XCTestCase {
    func testEveryGoldenFixtureDecodesAndValidates() throws {
        let fixtures = try Fixtures.load(directory: fixturesDir)
        XCTAssertEqual(fixtures.count, 13, "golden 9b64a14 ships 13 cases")
        // Sorted by file name, as golden load_fixtures does ("-vt70.json" < ".json").
        let names = fixtures.map { $0.id + ".json" }
        XCTAssertEqual(names, names.sorted())
        for fx in fixtures {
            XCTAssertEqual(fx.model.repo, "mlx-community/gemma-4-E2B-it-qat-4bit", fx.id)
            XCTAssertEqual(fx.model.revision, "42f62737af7a9fd8c1d55d79666c1a217be4e2e2", fx.id)
            XCTAssertEqual(fx.path.prefill, "split", fx.id)
            XCTAssertEqual(fx.path.prefillStep, 2048, fx.id)
            XCTAssertNotNil(fx.theQuestion, fx.id)
        }
        let ids = Set(fixtures.map(\.id))
        for id in ["noul-first-token", "choice-first-token-image-gridworld-vt70",
                   "choice-first-token-image-gridworld-vt140", "choice-first-token-multi-token-label-error"] {
            XCTAssertTrue(ids.contains(id), id)
        }
    }

    /// Option order is semantic (it decides which option is "A."), so decoding
    /// must keep document order: option_token_ids keys equal expected.options
    /// (checked in `validate`) and the choice criteria keys equal them too.
    func testOptionOrderSurvivesDecoding() throws {
        for fx in try Fixtures.load(directory: fixturesDir) {
            switch fx.theQuestion {
            case .choice(let q):
                XCTAssertEqual(q.criteria.keys, fx.expected.options, fx.id)
            case .score(let q):
                XCTAssertEqual(fx.expected.options, q.criteria.indices.map(String.init), fx.id)
            case .noul:
                XCTAssertEqual(fx.expected.options, ["yes", "no"], fx.id)
            case nil:
                XCTFail("\(fx.id): no question")
            }
        }
    }

    func testFixtureProbabilitiesAndAnswers() throws {
        for fx in try Fixtures.load(directory: fixturesDir) {
            let e = fx.expected
            if let err = e.error {
                XCTAssertEqual(err.type, "LabelTokenizationError", fx.id)
                XCTAssertTrue(e.optionTokenIds.values.contains { $0.count > 1 }, fx.id)
                continue
            }
            let probs = try XCTUnwrap(e.rawProbabilities, fx.id)
            XCTAssertEqual(probs.values.reduce(0, +), 1, accuracy: 1e-6, fx.id)
            // answer: highest raw probability, first wins ties.
            let best = probs.entries.reduce(probs.entries[0]) { $1.value > $0.value ? $1 : $0 }
            XCTAssertEqual(e.answer, best.key, fx.id)
            if fx.readout == .firstToken {
                XCTAssertTrue(e.optionTokenIds.values.allSatisfy { $0.count == 1 }, fx.id)
            }
        }
    }

    /// Image runs: boi, `length` soft tokens, eoi; sha256 pins the PNG bytes
    /// carried inline in the request.
    func testImageRunsAndPNGHashes() throws {
        var imageCases = 0
        for fx in try Fixtures.load(directory: fixturesDir) {
            let pngs: [Data] = {
                guard case .blocks(let blocks) = fx.request.state else { return [] }
                return blocks.compactMap { if case .image(let i) = $0 { return i.pngData } else { return nil } }
            }()
            let images = fx.expected.images ?? []
            XCTAssertEqual(pngs.count, images.count, fx.id)
            if images.isEmpty { continue }
            imageCases += 1
            let t = fx.expected.tokens
            for (img, png) in zip(images, pngs) {
                XCTAssertEqual(t[img.start - 1], Gemma4ImageTokens.beginOfImage, fx.id)
                XCTAssertTrue(t[img.start..<(img.start + img.length)].allSatisfy { $0 == Gemma4ImageTokens.softToken }, fx.id)
                XCTAssertEqual(t[img.start + img.length], Gemma4ImageTokens.endOfImage, fx.id)
                let hex = SHA256.hash(data: png).map { String(format: "%02x", $0) }.joined()
                XCTAssertEqual(hex, img.sha256, fx.id)
                XCTAssertEqual(img.maxSoftTokens, fx.request.options?.visionSoftTokens?.rawValue, fx.id)
                if let budget = img.maxSoftTokens { XCTAssertLessThanOrEqual(img.length, budget, fx.id) }
            }
        }
        XCTAssertEqual(imageCases, 5)
    }

    /// Decode then encode reproduces each fixture file's JSON tree exactly,
    /// key order included.
    func testFixtureFilesRoundTripInKeyOrder() throws {
        let files = try FileManager.default.contentsOfDirectory(at: fixturesDir, includingPropertiesForKeys: nil)
            .filter { $0.pathExtension == "json" }
        XCTAssertEqual(files.count, 13)
        for f in files {
            let raw = try JSONValue.parse(try Data(contentsOf: f))
            let fx = try SystemOneJSON.decode(GoldenFixture.self, from: raw)
            XCTAssertEqual(try SystemOneJSON.encodeValue(fx), raw, f.lastPathComponent)
            let req = try SystemOneJSON.decode(SystemOneRequest.self, from: raw["request"]!)
            XCTAssertEqual(try SystemOneJSON.encodeValue(req), raw["request"], f.lastPathComponent)
            // and through bytes
            XCTAssertEqual(try SystemOneJSON.decode(GoldenFixture.self, from: try SystemOneJSON.encode(fx)), fx)
        }
    }
}
