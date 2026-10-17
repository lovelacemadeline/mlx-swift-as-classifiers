import Foundation
import XCTest

@testable import MLXClassifiers

/// Story 1.2: compile every golden fixture's request with the checkpoint's own
/// tokenizer and chat template (no weights) and reproduce `expected.tokens`,
/// `prefix_len`, `readout_position`, `options`, `labels`, `images` and
/// `option_token_ids` exactly, as golden `fixtures.compare` checks them.
final class TokenParityTests: XCTestCase {
    static let repo = "mlx-community/gemma-4-E2B-it-qat-4bit"
    static let revision = "42f62737af7a9fd8c1d55d79666c1a217be4e2e2"

    /// Loaded once for the suite (parsing tokenizer.json takes seconds).
    nonisolated(unsafe) static var shared: Task<ModelTokenizer, Error>?

    func loadTokenizer() async throws -> ModelTokenizer {
        guard let dir = ModelTokenizer.cachedSnapshot(repo: Self.repo, revision: Self.revision) else {
            throw XCTSkip("no local HF snapshot of \(Self.repo)@\(Self.revision) (tokenizer files only are needed)")
        }
        if Self.shared == nil {
            let sha = try Fixtures.load(directory: fixturesDir)[0].model.tokenizerSha256
            Self.shared = Task { try await ModelTokenizer.load(directory: dir, expectedTokenizerSha256: sha) }
        }
        return try await Self.shared!.value
    }

    /// Golden `_compare_ids`: the first divergence, then a length mismatch.
    func compareIds(_ where_: String, _ want: [Int], _ got: [Int], _ failures: inout [String]) {
        if let i = zip(want, got).enumerated().first(where: { $0.element.0 != $0.element.1 })?.offset {
            failures.append("\(where_)[\(i)]: expected token id \(want[i]), got \(got[i])")
        }
        if want.count != got.count { failures.append("\(where_): expected \(want.count) token ids, got \(got.count)") }
    }

    func result(_ fx: GoldenFixture, _ tok: ModelTokenizer) throws -> TokenResult {
        let prompt = try compileQuestion(fx.request, name: fx.question, template: tok.chatTemplate,
                                         tokenize: tok.tokenize, labelStyle: fx.labelStyle,
                                         imageTokens: tok.imageTokens)
        return try tokenResult(prompt, tokenize: tok.tokenize)
    }

    func testFixtureTokensMatch() async throws {
        let tok = try await loadTokenizer()
        let fixtures = try Fixtures.load(directory: fixturesDir)
        XCTAssertEqual(fixtures.count, 13)
        for fx in fixtures {
            XCTAssertEqual(fx.model.repo, Self.repo, fx.id)
            XCTAssertEqual(fx.model.tokenizerSha256, tok.tokenizerSha256, fx.id)
            let r = try result(fx, tok)
            let e = fx.expected
            var f: [String] = []
            compareIds("tokens", e.tokens, r.tokens, &f)
            if r.prefixLen != e.prefixLen { f.append("prefix_len: expected \(e.prefixLen), got \(r.prefixLen)") }
            if r.readoutPosition != e.readoutPosition {
                f.append("readout_position: expected \(e.readoutPosition), got \(r.readoutPosition)")
            }
            if r.images != e.images { f.append("images: expected \(String(describing: e.images)), got \(String(describing: r.images))") }
            if r.options != e.options { f.append("options: expected \(e.options), got \(r.options)") }
            if r.labels != e.labels { f.append("labels: expected \(e.labels), got \(r.labels)") }
            if r.optionTokenIds.keys != e.optionTokenIds.keys {
                f.append("option_token_ids: expected options \(e.optionTokenIds.keys), got \(r.optionTokenIds.keys)")
            }
            for entry in e.optionTokenIds.entries {
                compareIds("option_token_ids['\(entry.key)']", entry.value, r.optionTokenIds[entry.key] ?? [], &f)
            }
            if let want = e.error {
                if r.error?.type != want.type { f.append("error.type: expected \(want.type), got \(r.error?.type ?? "none")") }
            } else if let got = r.error {
                f.append("unexpected error \(got)")
            }
            XCTAssertTrue(f.isEmpty, "\(fx.id): \(f.joined(separator: "; "))")
        }
    }

    /// Image fixtures: the placeholder runs sit where `expected.images` says,
    /// bracketed by begin/end-of-image, one run per image block (the pixels are Epic 4).
    func testImageRunStructure() async throws {
        let tok = try await loadTokenizer()
        let spec = try XCTUnwrap(tok.imageTokens)
        XCTAssertEqual(spec.imageTokenId, Gemma4ImageTokens.softToken)
        XCTAssertEqual(tok.tokenize(spec.boi), [Gemma4ImageTokens.beginOfImage])
        XCTAssertEqual(tok.tokenize(spec.eoi), [Gemma4ImageTokens.endOfImage])
        var seen = 0
        for fx in try Fixtures.load(directory: fixturesDir) where fx.request.hasImages {
            seen += 1
            let r = try result(fx, tok)
            let runs = findImageRuns(r.tokens, imageTokenId: spec.imageTokenId)
            let images = try XCTUnwrap(r.images, fx.id)
            XCTAssertEqual(runs.map(\.start), images.map(\.start), fx.id)
            XCTAssertEqual(runs.map(\.length), images.map(\.length), fx.id)
            for run in runs {
                XCTAssertEqual(r.tokens[run.start - 1], Gemma4ImageTokens.beginOfImage, fx.id)
                XCTAssertEqual(r.tokens[run.start + run.length], Gemma4ImageTokens.endOfImage, fx.id)
            }
        }
        XCTAssertEqual(seen, 5, "five image fixtures at golden 9b64a14")
    }

    /// The option-text case with a two-token label fails first-token binding
    /// with LabelTokenizationError, and the per-label ids are still reported.
    func testMultiTokenOptionTextLabelFails() async throws {
        let tok = try await loadTokenizer()
        let fx = try Fixtures.load(directory: fixturesDir).first { $0.id == "choice-first-token-multi-token-label-error" }!
        XCTAssertEqual(fx.labelStyle, .optionText)
        let prompt = try compileQuestion(fx.request, name: fx.question, template: tok.chatTemplate,
                                         tokenize: tok.tokenize, labelStyle: fx.labelStyle)
        XCTAssertThrowsError(try bindOptions(prompt, tokenize: tok.tokenize)) { error in
            let e = error as? ClassifierError
            XCTAssertEqual(e?.kind, .labelTokenization)
            XCTAssertEqual(e?.type, fx.expected.error?.type)
            XCTAssertTrue(e?.message.contains("not single tokens") ?? false, "\(error)")
        }
        XCTAssertTrue(fx.expected.optionTokenIds.values.contains { $0.count == 2 })
    }

    /// The rendered Gemma 4 turn markup, checked as text against what golden's
    /// HF `apply_chat_template` renders (captured from the golden venv).
    func testChatTemplateRendersLikeHF() async throws {
        let tok = try await loadTokenizer()
        let sys = "You are the support triage router for a payments company. Answer precisely."
        let chat = try tok.chatTemplate.render(system: sys, user: "USER")
        XCTAssertEqual(chat.text, "<bos><|turn>system\n\(sys)<turn|>\n<|turn>user\nUSER<turn|>\n<|turn>model\n")
        XCTAssertEqual(chat.prefixText, "<bos><|turn>system\n\(sys)<turn|>\n")
        let bare = try tok.chatTemplate.render(system: nil, user: "USER")
        XCTAssertEqual(bare.text, "<bos><|turn>user\nUSER<turn|>\n<|turn>model\n")
        XCTAssertEqual(bare.prefixText, "")
        XCTAssertEqual(try tok.chatTemplate.generationPrompt(), "<|turn>model\n")
    }
}

/// Python-semantics helpers the prompt text depends on.
final class PythonTextTests: XCTestCase {
    func testJSONDumpsLikePython() throws {
        let v = try JSONValue.parse(#"{"b": [1, 2.5, {"z": null, "a": true}], "a": "é\n\"q\"", "B": {}, "c": [], "\u001f": 1e-7}"#)
        XCTAssertEqual(PythonJSON.dumps(v), """
        {
          "\\u001f": 1e-07,
          "B": {},
          "a": "é\\n\\"q\\"",
          "b": [
            1,
            2.5,
            {
              "a": true,
              "z": null
            }
          ],
          "c": []
        }
        """)
    }

    func testFloatRepr() {
        let cases: [(Double, String)] = [(1.0, "1.0"), (0.1, "0.1"), (1e16, "1e+16"), (1e15, "1000000000000000.0"),
                                         (0.0001, "0.0001"), (0.00001, "1e-05"), (-2.5e-10, "-2.5e-10"),
                                         (123456.789, "123456.789"), (1.5e300, "1.5e+300")]
        for (x, want) in cases { XCTAssertEqual(PythonJSON.floatRepr(x), want, "\(x)") }
    }

    func testEmittedLabel() {
        XCTAssertEqual(emittedLabel(context: "<|turn>model\n", label: "A"), "A")
        XCTAssertEqual(emittedLabel(context: "x", label: "A"), " A")
        XCTAssertEqual(emittedLabel(context: "", label: "A"), "A")
    }
}
