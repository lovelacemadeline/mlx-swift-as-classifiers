import Foundation
import XCTest

@testable import MLXClassifiers

/// The Swift types against the vendored JSON Schema files: every property and
/// enum value in the schema has a Swift counterpart and vice versa.
final class SchemaDriftTests: XCTestCase {
    func schema(_ name: String) throws -> JSONValue {
        try JSONValue.parse(try Data(contentsOf: schemaDir.appendingPathComponent(name)))
    }

    func props(_ v: JSONValue?) -> [String] { v?["properties"]?.objectValue?.keys.sorted() ?? [] }

    func keys<K: CodingKey & CaseIterable>(_ k: K.Type) -> [String] { K.allCases.map(\.stringValue).sorted() }

    func enumStrings(_ v: JSONValue?) -> [String] {
        guard case .array(let a)? = v?["enum"] else { return [] }
        return a.compactMap { if case .string(let s) = $0 { return s } else { return nil } }.sorted()
    }

    func testRequestSchemaMatchesSwiftTypes() throws {
        let s = try schema("systemone-request.schema.json")
        XCTAssertEqual(s["title"], .string("SystemOneRequest"))
        let defs = try XCTUnwrap(s["$defs"])
        XCTAssertEqual(props(s), keys(SystemOneRequest.CodingKeys.self))
        XCTAssertEqual(props(defs["ImageBlock"]), keys(ImageBlock.CodingKeys.self))
        XCTAssertEqual(props(defs["NoulQuestion"]), keys(NoulQuestion.CodingKeys.self))
        XCTAssertEqual(props(defs["ChoiceQuestion"]), keys(ChoiceQuestion.CodingKeys.self))
        XCTAssertEqual(props(defs["ScoreQuestion"]), keys(ScoreQuestion.CodingKeys.self))
        XCTAssertEqual(props(defs["RequestOptions"]), keys(RequestOptions.CodingKeys.self))
        let opts = defs["RequestOptions"]?["properties"]
        XCTAssertEqual(enumStrings(opts?["readout"]), Readout.allCases.map(\.rawValue).sorted())
        XCTAssertEqual(enumStrings(opts?["label_style"]), LabelStyle.allCases.map(\.rawValue).sorted())
        XCTAssertEqual(opts?["vision_soft_tokens"]?["enum"], .array(VisionSoftTokens.allCases.map { .int($0.rawValue) }))
        XCTAssertEqual(opts?["permutations"]?["minimum"], .int(RequestOptions.permutationsRange.lowerBound))
        XCTAssertEqual(opts?["permutations"]?["maximum"], .int(RequestOptions.permutationsRange.upperBound))
        XCTAssertEqual(defs["ImageBlock"]?["properties"]?["type"]?["const"], .string(ImageBlock.MediaType.png.rawValue))
        XCTAssertEqual(defs["Question"]?["oneOf"].map { v -> Int in if case .array(let a) = v { a.count } else { 0 } },
                       Question.Kind.allCases.count)
    }

    func testResponseSchemaMatchesSwiftTypes() throws {
        let s = try schema("systemone-response.schema.json")
        XCTAssertEqual(s["title"], .string("SystemOneResponse"))
        let defs = try XCTUnwrap(s["$defs"])
        XCTAssertEqual(props(s), keys(SystemOneResponse.CodingKeys.self))
        XCTAssertEqual(props(s["properties"]?["usage"]), keys(Usage.CodingKeys.self))
        XCTAssertEqual(props(s["properties"]?["timing_ms"]), keys(Timing.CodingKeys.self))
        XCTAssertEqual(props(defs["Debug"]), keys(Debug.CodingKeys.self))
        XCTAssertEqual(props(defs["NoulAnswer"]), keys(NoulAnswer.CodingKeys.self))
        XCTAssertEqual(props(defs["ChoiceAnswer"]), keys(ChoiceAnswer.CodingKeys.self))
        XCTAssertEqual(props(defs["ScoreAnswer"]), keys(ScoreAnswer.CodingKeys.self))
    }

    func testSchemaFilesRoundTripThroughJSONValue() throws {
        for name in ["systemone-request.schema.json", "systemone-response.schema.json"] {
            let v = try schema(name)
            XCTAssertEqual(try JSONValue.parse(v.serialized()), v, name)
        }
    }
}

final class SchemaValidationTests: XCTestCase {
    func decodeRequest(_ text: String) throws -> SystemOneRequest {
        try SystemOneJSON.decode(SystemOneRequest.self, from: Data(text.utf8))
    }

    func testAllThreeStateShapes() throws {
        let q = #""questions": {"q": {"type": "noul", "instructions": "ok?"}}"#
        XCTAssertEqual(try decodeRequest(#"{"state": "hi", \#(q)}"#).state, .text("hi"))
        XCTAssertEqual(try decodeRequest(#"{"state": {"b": 1, "a": [true, null]}, \#(q)}"#).state,
                       .object(["b": .int(1), "a": .array([.bool(true), .null])]))
        XCTAssertEqual(try decodeRequest(#"{"state": ["x", {"type": "image/png", "base64": "AA=="}], \#(q)}"#).state,
                       .blocks([.text("x"), .image(ImageBlock(base64: "AA=="))]))
    }

    func testOptionsDefaultsAndSnakeCase() throws {
        let r = try decodeRequest(#"""
            {"state": "s", "questions": {"q": {"type": "score", "instructions": "i", "criteria": ["lo", "hi"]}},
             "options": {"permutations": 3, "label_style": "option-text", "delta_cache": false,
                         "vision_soft_tokens": 140, "readout": "sequence"}}
            """#)
        let o = try XCTUnwrap(r.options)
        XCTAssertEqual(o.resolvedPermutations, 3)
        XCTAssertEqual(o.resolvedLabelStyle, .optionText)
        XCTAssertEqual(o.resolvedDeltaCache, false)
        XCTAssertEqual(o.visionSoftTokens, .t140)
        XCTAssertEqual(o.resolvedReadout, .sequence)
        XCTAssertEqual(o.resolvedCalibrated, true)
        XCTAssertEqual(o.resolvedDebug, false)
        XCTAssertEqual(RequestOptions().resolvedReadout, .firstToken)
    }

    func testSchemaConstraintsAreEnforced() {
        let noul = #"{"type": "noul", "instructions": "ok?"}"#
        let bad = [
            #"{"state": "s", "questions": {}}"#,                                              // minProperties 1
            #"{"state": "s", "questions": {"q": \#(noul)}, "extra": 1}"#,                       // additionalProperties
            #"{"state": "s", "questions": {"q": {"type": "noul", "instructions": "i", "x": 1}}}"#,
            #"{"state": "s", "questions": {"q": {"type": "choice", "instructions": "i", "criteria": {"a": null}}}}"#,
            #"{"state": "s", "questions": {"q": {"type": "score", "instructions": "i", "criteria": ["one"]}}}"#,
            #"{"state": "s", "questions": {"q": {"type": "rank", "instructions": "i"}}}"#,
            #"{"state": "s", "questions": {"q": \#(noul)}, "options": {"permutations": 9}}"#,
            #"{"state": "s", "questions": {"q": \#(noul)}, "options": {"vision_soft_tokens": 100}}"#,
            #"{"state": "s", "questions": {"q": \#(noul)}, "options": {"readout": "greedy"}}"#,
            #"{"state": ["x", {"type": "image/jpeg", "base64": "AA=="}], "questions": {"q": \#(noul)}}"#,
            #"{"state": 3, "questions": {"q": \#(noul)}}"#,
            #"{"questions": {"q": \#(noul)}}"#,
        ]
        for text in bad {
            XCTAssertThrowsError(try decodeRequest(text), text)
        }
    }

    func testResponseRoundTripAllAnswerKinds() throws {
        let text = #"""
            {"model":"mlx-community/gemma-4-E2B-it-qat-4bit","answers":{"is_payment":{"type":"noul","noul":0.91,"raw_noul":0.88},"team":{"type":"choice","choice":"billing","probabilities":{"billing":0.7,"technical":0.2,"sales":0.1},"raw_probabilities":{"billing":0.6,"technical":0.3,"sales":0.1},"confidence":0.7,"debug":{"readout_position":81,"option_token_ids":{"billing":[236776],"technical":[236799],"sales":[236780]},"raw_logits":{"billing":12.5,"technical":11.0,"sales":9.25},"temperature":1.3}},"severity":{"type":"score","score":3,"expected_score":2.6,"legend":{"0":"trivial","1":"minor","2":"major","3":"critical"},"probabilities":{"0":0.01,"1":0.04,"2":0.3,"3":0.65},"confidence":0.65}},"usage":{"input_tokens":120,"output_tokens":0,"cached_tokens":20},"timing_ms":{"prefill":41.5,"readout":0.25,"total":43.25}}
            """#
        let raw = try JSONValue.parse(text)
        let r = try SystemOneJSON.decode(SystemOneResponse.self, from: raw)
        XCTAssertEqual(r.answers.keys, ["is_payment", "team", "severity"])
        XCTAssertEqual(r.usage.outputTokens, 0)
        XCTAssertEqual(try SystemOneJSON.encodeValue(r), raw)
        XCTAssertEqual(try SystemOneJSON.encodeValue(r).serialized(), text)
        XCTAssertThrowsError(try SystemOneJSON.decode(
            SystemOneResponse.self, from: Data(#"{"model":"m","answers":{"a":{"type":"noul","noul":1.5}},"usage":{"input_tokens":1,"output_tokens":0}}"#.utf8)))
    }

    func testParserEdgeCases() throws {
        XCTAssertEqual(try JSONValue.parse(#"{"s":"é😀\n","n":-1.5e2,"i":0,"z":[]}"#),
                       .object(["s": .string("é😀\n"), "n": .double(-150), "i": .int(0), "z": .array([])]))
        for bad in ["", "{", #"{"a":1,}"#, "[1 2]", "01", #""\ud800""#, "nul", "1 2"] {
            XCTAssertThrowsError(try JSONValue.parse(bad), bad)
        }
    }
}
