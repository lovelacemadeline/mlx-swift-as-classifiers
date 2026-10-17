// Typed loading of the golden fixtures (fixtures/*.json, format
// "mlx-classifiers-fixture/1"), the conformance contract this port must
// reproduce (golden ADR-0002). Mirrors golden `fixtures.py`: `load_fixtures`
// (sorted by file name, format checked) and the `FixtureCase` invariants.
// The field meanings are in the golden docs/reference/fixture-format.md.

import Foundation

public enum FixtureFormat {
    public static let fixture = "mlx-classifiers-fixture/1"
    public static let results = "mlx-classifiers-results/1"
    /// Default `--prob-tol` of the golden checker.
    public static let defaultProbabilityTolerance = 1e-3
}

/// Golden `GATES` (golden story 4.3): how a fixture's result is judged. A fixture
/// without a `gate` key is `full`.
public enum FixtureGate {
    /// Ids, images, answer exact; raw probabilities within the tolerance.
    public static let full = "full"
    /// The same exact checks; max |dp| and max |dlogit| reported, never gated (near-tied cases).
    public static let answerAndIds = "answer-and-ids"
    public static let all = [full, answerAndIds]
}

/// Gemma 4 image placeholder ids as they appear in `expected.tokens`.
public enum Gemma4ImageTokens {
    public static let beginOfImage = 255_999
    public static let softToken = 258_880
    public static let endOfImage = 258_882
}

public struct FixtureError: Error, CustomStringConvertible, Sendable {
    public let file: String
    public let message: String
    public var description: String { "\(file): \(message)" }
}

/// One golden fixture: one request with exactly one question, and its expected result.
public struct GoldenFixture: Codable, Sendable, Equatable {
    public var format: String
    public var id: String
    public var description: String
    public var model: Model
    public var path: PrefillPath
    public var readout: Readout
    public var labelStyle: LabelStyle
    public var prefixCached: Bool
    /// Golden story 4.3: `"answer-and-ids"` or absent (= `"full"`); see `FixtureGate`.
    /// Kept as the raw string so an unknown value fails the case, as golden's checker does.
    public var gate: String?
    /// Name of the request's single question.
    public var question: String
    public var request: SystemOneRequest
    public var expected: Expected
    /// Package versions that wrote the file (provenance only).
    public var generator: OrderedObject<String>

    public enum CodingKeys: String, CodingKey, CaseIterable {
        case format, id, description, model, path, readout
        case labelStyle = "label_style"
        case prefixCached = "prefix_cached"
        case gate
        case question, request, expected, generator
    }

    /// The gate this fixture is checked under (`gate`, absent = `full`).
    public var resolvedGate: String { gate ?? FixtureGate.full }

    public struct Model: Codable, Sendable, Equatable {
        public var repo: String
        public var revision: String
        public var tokenizerSha256: String

        public enum CodingKeys: String, CodingKey, CaseIterable {
            case repo, revision
            case tokenizerSha256 = "tokenizer_sha256"
        }
    }

    public struct PrefillPath: Codable, Sendable, Equatable {
        /// Always "split": MLXRunner's split prefill.
        public var prefill: String
        /// Chunk size of the cache-only rows.
        public var prefillStep: Int

        public enum CodingKeys: String, CodingKey, CaseIterable {
            case prefill
            case prefillStep = "prefill_step"
        }
    }

    /// The `expected` block, which is also the shape a port's result takes.
    public struct Expected: Codable, Sendable, Equatable {
        public var tokens: [Int]
        public var prefixLen: Int
        public var readoutPosition: Int
        public var options: [String]
        public var labels: [String]
        public var images: [Image]?
        public var optionTokenIds: OrderedObject<[Int]>
        public var rawLogits: OrderedObject<Double>?
        public var rawProbabilities: OrderedObject<Double>?
        public var answer: String?
        public var error: ExpectedError?

        public enum CodingKeys: String, CodingKey, CaseIterable {
            case tokens
            case prefixLen = "prefix_len"
            case readoutPosition = "readout_position"
            case options, labels, images
            case optionTokenIds = "option_token_ids"
            case rawLogits = "raw_logits"
            case rawProbabilities = "raw_probabilities"
            case answer, error
        }
    }

    /// One image's run of soft tokens in `expected.tokens`.
    public struct Image: Codable, Sendable, Equatable {
        /// Index of the first soft token; `tokens[start - 1]` is begin-of-image.
        public var start: Int
        public var length: Int
        /// sha256 of the PNG bytes.
        public var sha256: String
        /// Present only for a non-default budget (70 or 140).
        public var maxSoftTokens: Int?

        public enum CodingKeys: String, CodingKey, CaseIterable {
            case start, length, sha256
            case maxSoftTokens = "max_soft_tokens"
        }
    }

    public struct ExpectedError: Codable, Sendable, Equatable {
        /// Compared exactly (e.g. "LabelTokenizationError").
        public var type: String
        /// Ignored by the checker.
        public var message: String
    }

    /// The single question this fixture asks.
    public var theQuestion: Question? { request.questions[question] }

    /// The `FixtureCase` invariants of golden fixtures.py, plus the shape rules
    /// of fixture-format.md that need no model to check.
    public func validate() throws {
        func fail(_ m: String) -> FixtureError { FixtureError(file: id, message: m) }
        guard format == FixtureFormat.fixture else { throw fail("format \(format), expected \(FixtureFormat.fixture)") }
        guard request.questions.count == 1 else { throw fail("a fixture request holds exactly one question") }
        guard request.questions[question] != nil else { throw fail("question '\(question)' is not in the request") }
        if prefixCached, (request.prefix ?? "").isEmpty { throw fail("prefix_cached needs a request prefix") }
        let wireStyle = request.options?.labelStyle ?? labelStyle
        guard wireStyle == labelStyle else { throw fail("request label_style disagrees with label_style") }
        guard (request.options?.resolvedReadout ?? .firstToken) == readout else { throw fail("request readout disagrees with readout") }
        let e = expected
        guard e.readoutPosition == e.tokens.count - 1 else { throw fail("readout_position is not the last prompt token") }
        guard (0...e.tokens.count).contains(e.prefixLen) else { throw fail("prefix_len out of range") }
        guard e.labels.count == e.options.count else { throw fail("labels and options differ in length") }
        guard e.optionTokenIds.keys == e.options else { throw fail("option_token_ids keys are not the options in order") }
        if e.error == nil {
            guard e.rawLogits?.keys == e.options, e.rawProbabilities?.keys == e.options, let answer = e.answer,
                  e.options.contains(answer)
            else { throw fail("a non-error fixture needs raw_logits, raw_probabilities and answer over the options") }
        } else if e.answer != nil || e.rawProbabilities != nil {
            throw fail("an error fixture carries no answer or probabilities")
        }
    }
}

public enum Fixtures {
    /// Every `*.json` fixture in `directory`, sorted by file name, decoded and
    /// validated. The id must equal the file stem.
    public static func load(directory: URL) throws -> [GoldenFixture] {
        let files = try FileManager.default.contentsOfDirectory(at: directory, includingPropertiesForKeys: nil)
            .filter { $0.pathExtension == "json" }
            .sorted { $0.lastPathComponent < $1.lastPathComponent }
        guard !files.isEmpty else { throw FixtureError(file: directory.path, message: "no fixture files") }
        return try files.map(load(file:))
    }

    public static func load(file: URL) throws -> GoldenFixture {
        let fx = try SystemOneJSON.decode(GoldenFixture.self, from: try Data(contentsOf: file))
        guard fx.id == file.deletingPathExtension().lastPathComponent else {
            throw FixtureError(file: file.lastPathComponent, message: "id \(fx.id) does not match the file name")
        }
        try fx.validate()
        return fx
    }
}
