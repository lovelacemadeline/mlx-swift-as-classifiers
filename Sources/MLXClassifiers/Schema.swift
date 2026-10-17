// The /v1/systemone wire types, written by hand from the vendored JSON Schema
// (schema/systemone-request.schema.json and schema/systemone-response.schema.json,
// golden commit 9b64a14; see schema/VENDORED.txt). Every property and enum in
// the schema has a field or case here; `SchemaDriftTests` fails if they part.
//
// Decode with `SystemOneJSON` to keep object key order (option order is
// semantic). Constraints the schema states (additionalProperties false,
// minProperties, minItems, minimum/maximum, const, enum) are enforced on
// decode and throw `SchemaError`.

import Foundation

public struct SchemaError: Error, CustomStringConvertible, Sendable, Equatable {
    public let path: String
    public let message: String
    public init(path: String, _ message: String) {
        self.path = path
        self.message = message
    }
    public var description: String { "\(path.isEmpty ? "$" : path): \(message)" }
}

func pathString(_ codingPath: [CodingKey]) -> String {
    codingPath.map { $0.intValue.map { "[\($0)]" } ?? ".\($0.stringValue)" }.joined()
}

extension Decoder {
    /// `additionalProperties: false`: every key must be one of `Keys`.
    func rejectUnknownKeys<Keys: CodingKey & CaseIterable>(_ keys: Keys.Type) throws {
        let allowed = Set(Keys.allCases.map(\.stringValue))
        let present = try container(keyedBy: AnyCodingKey.self).allKeys.map(\.stringValue)
        if let bad = present.first(where: { !allowed.contains($0) }) {
            throw SchemaError(path: pathString(codingPath), "unknown property '\(bad)'")
        }
    }
}

// MARK: - Request

/// `SystemOneRequest`: one state, many named questions.
public struct SystemOneRequest: Codable, Sendable, Equatable {
    /// Model id. Omit for the server default.
    public var model: String?
    /// The unstructured program state the questions are about.
    public var state: SystemOneState
    /// x-local. Stable rules/system text, KV-cached across requests.
    public var prefix: String?
    /// Question name -> question, in declared order. At least one.
    public var questions: OrderedObject<Question>
    /// x-local. Inference knobs.
    public var options: RequestOptions?

    public init(model: String? = nil, state: SystemOneState, prefix: String? = nil,
                questions: OrderedObject<Question>, options: RequestOptions? = nil) {
        self.model = model
        self.state = state
        self.prefix = prefix
        self.questions = questions
        self.options = options
    }

    /// Encoding order follows the golden request builder (state, questions,
    /// prefix, options), so fixture requests round-trip in key order.
    public enum CodingKeys: String, CodingKey, CaseIterable {
        case model, state, questions, prefix, options
    }

    public init(from decoder: Decoder) throws {
        try decoder.rejectUnknownKeys(CodingKeys.self)
        let c = try decoder.container(keyedBy: CodingKeys.self)
        model = try c.decodeIfPresent(String.self, forKey: .model)
        state = try c.decode(SystemOneState.self, forKey: .state)
        prefix = try c.decodeIfPresent(String.self, forKey: .prefix)
        questions = try c.decode(OrderedObject<Question>.self, forKey: .questions)
        guard questions.count >= 1 else {
            throw SchemaError(path: pathString(decoder.codingPath + [CodingKeys.questions]), "needs at least 1 question")
        }
        options = try c.decodeIfPresent(RequestOptions.self, forKey: .options)
    }
}

/// `state`: a string, a JSON object, or a list of blocks (strings and x-local image blocks).
public enum SystemOneState: Codable, Sendable, Equatable {
    case text(String)
    case object(OrderedObject<JSONValue>)
    case blocks([StateBlock])

    public init(from decoder: Decoder) throws {
        let c = try decoder.singleValueContainer()
        if let s = try? c.decode(String.self) {
            self = .text(s)
        } else if let blocks = try? c.decode([StateBlock].self) {
            self = .blocks(blocks)
        } else if let o = try? c.decode(OrderedObject<JSONValue>.self) {
            self = .object(o)
        } else {
            throw SchemaError(path: pathString(decoder.codingPath), "state must be a string, an object, or a list of strings and image blocks")
        }
    }

    public func encode(to encoder: Encoder) throws {
        var c = encoder.singleValueContainer()
        switch self {
        case .text(let s): try c.encode(s)
        case .object(let o): try c.encode(o)
        case .blocks(let b): try c.encode(b)
        }
    }
}

/// One entry of a list state.
public enum StateBlock: Codable, Sendable, Equatable {
    case text(String)
    case image(ImageBlock)

    public init(from decoder: Decoder) throws {
        let c = try decoder.singleValueContainer()
        if let s = try? c.decode(String.self) {
            self = .text(s)
        } else {
            self = .image(try c.decode(ImageBlock.self))
        }
    }

    public func encode(to encoder: Encoder) throws {
        var c = encoder.singleValueContainer()
        switch self {
        case .text(let s): try c.encode(s)
        case .image(let i): try c.encode(i)
        }
    }
}

/// x-local `ImageBlock`: a PNG at its list position.
public struct ImageBlock: Codable, Sendable, Equatable {
    public enum MediaType: String, Codable, Sendable, CaseIterable { case png = "image/png" }

    public let type: MediaType
    /// The PNG file, base64-encoded.
    public var base64: String

    public init(base64: String) {
        type = .png
        self.base64 = base64
    }

    public init(png: Data) { self.init(base64: png.base64EncodedString()) }

    /// The decoded PNG bytes, or nil if `base64` is not valid base64.
    public var pngData: Data? { Data(base64Encoded: base64) }

    public enum CodingKeys: String, CodingKey, CaseIterable { case type, base64 }

    public init(from decoder: Decoder) throws {
        try decoder.rejectUnknownKeys(CodingKeys.self)
        let c = try decoder.container(keyedBy: CodingKeys.self)
        type = try c.decode(MediaType.self, forKey: .type)
        base64 = try c.decode(String.self, forKey: .base64)
    }
}

// MARK: Questions

/// `Question`: one of noul, choice, score, told apart by `type`.
public enum Question: Codable, Sendable, Equatable {
    case noul(NoulQuestion)
    case choice(ChoiceQuestion)
    case score(ScoreQuestion)

    public enum Kind: String, Codable, Sendable, CaseIterable { case noul, choice, score }

    public var kind: Kind {
        switch self {
        case .noul: .noul
        case .choice: .choice
        case .score: .score
        }
    }

    public var instructions: String {
        switch self {
        case .noul(let q): q.instructions
        case .choice(let q): q.instructions
        case .score(let q): q.instructions
        }
    }

    private enum TypeKey: String, CodingKey { case type }

    public init(from decoder: Decoder) throws {
        let raw = try decoder.container(keyedBy: TypeKey.self).decode(String.self, forKey: .type)
        guard let kind = Kind(rawValue: raw) else {
            throw SchemaError(path: pathString(decoder.codingPath), "unknown question type '\(raw)'")
        }
        switch kind {
        case .noul: self = .noul(try NoulQuestion(from: decoder))
        case .choice: self = .choice(try ChoiceQuestion(from: decoder))
        case .score: self = .score(try ScoreQuestion(from: decoder))
        }
    }

    public func encode(to encoder: Encoder) throws {
        switch self {
        case .noul(let q): try q.encode(to: encoder)
        case .choice(let q): try q.encode(to: encoder)
        case .score(let q): try q.encode(to: encoder)
        }
    }
}

/// `NoulQuestion`: a yes/no question.
public struct NoulQuestion: Codable, Sendable, Equatable {
    public enum Kind: String, Codable, Sendable { case noul }
    public let type: Kind
    public var instructions: String

    public init(instructions: String) {
        type = .noul
        self.instructions = instructions
    }

    public enum CodingKeys: String, CodingKey, CaseIterable { case type, instructions }

    public init(from decoder: Decoder) throws {
        try decoder.rejectUnknownKeys(CodingKeys.self)
        let c = try decoder.container(keyedBy: CodingKeys.self)
        type = try c.decode(Kind.self, forKey: .type)
        instructions = try c.decode(String.self, forKey: .instructions)
    }
}

/// `ChoiceQuestion`: option name -> description (or null), at least two, in declared order.
public struct ChoiceQuestion: Codable, Sendable, Equatable {
    public enum Kind: String, Codable, Sendable { case choice }
    public let type: Kind
    public var instructions: String
    public var criteria: OrderedObject<String?>

    public init(instructions: String, criteria: OrderedObject<String?>) {
        type = .choice
        self.instructions = instructions
        self.criteria = criteria
    }

    public enum CodingKeys: String, CodingKey, CaseIterable { case type, instructions, criteria }

    public init(from decoder: Decoder) throws {
        try decoder.rejectUnknownKeys(CodingKeys.self)
        let c = try decoder.container(keyedBy: CodingKeys.self)
        type = try c.decode(Kind.self, forKey: .type)
        instructions = try c.decode(String.self, forKey: .instructions)
        criteria = try c.decode(OrderedObject<String?>.self, forKey: .criteria)
        guard criteria.count >= 2 else {
            throw SchemaError(path: pathString(decoder.codingPath + [CodingKeys.criteria]), "a choice needs at least 2 options")
        }
    }

    /// `encode` must write a null description as JSON null, not drop it.
    public func encode(to encoder: Encoder) throws {
        var c = encoder.container(keyedBy: CodingKeys.self)
        try c.encode(type, forKey: .type)
        try c.encode(instructions, forKey: .instructions)
        var crit = c.nestedContainer(keyedBy: AnyCodingKey.self, forKey: .criteria)
        for e in criteria.entries {
            if let d = e.value { try crit.encode(d, forKey: AnyCodingKey(e.key)) }
            else { try crit.encodeNil(forKey: AnyCodingKey(e.key)) }
        }
    }
}

/// `ScoreQuestion`: ordered levels, lowest first; the index is the score.
public struct ScoreQuestion: Codable, Sendable, Equatable {
    public enum Kind: String, Codable, Sendable { case score }
    public let type: Kind
    public var instructions: String
    public var criteria: [String]

    public init(instructions: String, criteria: [String]) {
        type = .score
        self.instructions = instructions
        self.criteria = criteria
    }

    public enum CodingKeys: String, CodingKey, CaseIterable { case type, instructions, criteria }

    public init(from decoder: Decoder) throws {
        try decoder.rejectUnknownKeys(CodingKeys.self)
        let c = try decoder.container(keyedBy: CodingKeys.self)
        type = try c.decode(Kind.self, forKey: .type)
        instructions = try c.decode(String.self, forKey: .instructions)
        criteria = try c.decode([String].self, forKey: .criteria)
        guard criteria.count >= 2 else {
            throw SchemaError(path: pathString(decoder.codingPath + [CodingKeys.criteria]), "a score needs at least 2 levels")
        }
    }
}

// MARK: Options

/// `options.readout`.
public enum Readout: String, Codable, Sendable, CaseIterable {
    /// Masked softmax over option-label tokens at the readout position.
    case firstToken = "first-token"
    /// Teacher-forced loglikelihood of each full option string (golden ADR-0003).
    case sequence
}

/// `options.label_style`.
public enum LabelStyle: String, Codable, Sendable, CaseIterable {
    /// Options shown as A., B., ...; the model answers a letter.
    case letters
    /// Options shown by name; every name must be one token in context.
    case optionText = "option-text"
}

/// `options.vision_soft_tokens`: soft-token budget per image.
public enum VisionSoftTokens: Int, Codable, Sendable, CaseIterable {
    case t70 = 70
    case t140 = 140
    case t280 = 280
}

/// x-local `RequestOptions`. Every field optional; `resolved*` apply the schema defaults.
public struct RequestOptions: Codable, Sendable, Equatable {
    /// 1...8, default 1.
    public var permutations: Int?
    /// Default true.
    public var calibrated: Bool?
    /// Default first-token.
    public var readout: Readout?
    /// Default letters.
    public var labelStyle: LabelStyle?
    /// Default true.
    public var deltaCache: Bool?
    /// Default false.
    public var debug: Bool?
    /// No schema default: omitted means the runner's budget (280 unless configured).
    public var visionSoftTokens: VisionSoftTokens?

    public init(permutations: Int? = nil, calibrated: Bool? = nil, readout: Readout? = nil,
                labelStyle: LabelStyle? = nil, deltaCache: Bool? = nil, debug: Bool? = nil,
                visionSoftTokens: VisionSoftTokens? = nil) {
        self.permutations = permutations
        self.calibrated = calibrated
        self.readout = readout
        self.labelStyle = labelStyle
        self.deltaCache = deltaCache
        self.debug = debug
        self.visionSoftTokens = visionSoftTokens
    }

    public static let permutationsRange = 1...8

    public var resolvedPermutations: Int { permutations ?? 1 }
    public var resolvedCalibrated: Bool { calibrated ?? true }
    public var resolvedReadout: Readout { readout ?? .firstToken }
    public var resolvedLabelStyle: LabelStyle { labelStyle ?? .letters }
    public var resolvedDeltaCache: Bool { deltaCache ?? true }
    public var resolvedDebug: Bool { debug ?? false }

    public enum CodingKeys: String, CodingKey, CaseIterable {
        case permutations, calibrated, readout
        case labelStyle = "label_style"
        case deltaCache = "delta_cache"
        case debug
        case visionSoftTokens = "vision_soft_tokens"
    }

    public init(from decoder: Decoder) throws {
        try decoder.rejectUnknownKeys(CodingKeys.self)
        let c = try decoder.container(keyedBy: CodingKeys.self)
        permutations = try c.decodeIfPresent(Int.self, forKey: .permutations)
        if let p = permutations, !Self.permutationsRange.contains(p) {
            throw SchemaError(path: pathString(decoder.codingPath + [CodingKeys.permutations]), "must be in 1...8, got \(p)")
        }
        calibrated = try c.decodeIfPresent(Bool.self, forKey: .calibrated)
        readout = try c.decodeIfPresent(Readout.self, forKey: .readout)
        labelStyle = try c.decodeIfPresent(LabelStyle.self, forKey: .labelStyle)
        deltaCache = try c.decodeIfPresent(Bool.self, forKey: .deltaCache)
        debug = try c.decodeIfPresent(Bool.self, forKey: .debug)
        visionSoftTokens = try c.decodeIfPresent(VisionSoftTokens.self, forKey: .visionSoftTokens)
    }
}

// MARK: - Response

/// `SystemOneResponse`.
public struct SystemOneResponse: Codable, Sendable, Equatable {
    public var model: String
    /// Question name -> answer, in request order.
    public var answers: OrderedObject<Answer>
    public var usage: Usage
    /// x-local. Wall-clock breakdown.
    public var timingMs: Timing?

    public init(model: String, answers: OrderedObject<Answer>, usage: Usage, timingMs: Timing? = nil) {
        self.model = model
        self.answers = answers
        self.usage = usage
        self.timingMs = timingMs
    }

    public enum CodingKeys: String, CodingKey, CaseIterable {
        case model, answers, usage
        case timingMs = "timing_ms"
    }

    public init(from decoder: Decoder) throws {
        try decoder.rejectUnknownKeys(CodingKeys.self)
        let c = try decoder.container(keyedBy: CodingKeys.self)
        model = try c.decode(String.self, forKey: .model)
        answers = try c.decode(OrderedObject<Answer>.self, forKey: .answers)
        usage = try c.decode(Usage.self, forKey: .usage)
        timingMs = try c.decodeIfPresent(Timing.self, forKey: .timingMs)
    }
}

public struct Usage: Codable, Sendable, Equatable {
    public var inputTokens: Int
    /// Always 0: nothing is decoded.
    public var outputTokens: Int
    /// x-local. Prefix tokens served from the KV cache.
    public var cachedTokens: Int?

    public init(inputTokens: Int, outputTokens: Int = 0, cachedTokens: Int? = nil) {
        self.inputTokens = inputTokens
        self.outputTokens = outputTokens
        self.cachedTokens = cachedTokens
    }

    public enum CodingKeys: String, CodingKey, CaseIterable {
        case inputTokens = "input_tokens"
        case outputTokens = "output_tokens"
        case cachedTokens = "cached_tokens"
    }
}

public struct Timing: Codable, Sendable, Equatable {
    public var prefill: Double?
    public var readout: Double?
    public var total: Double?

    public init(prefill: Double? = nil, readout: Double? = nil, total: Double? = nil) {
        self.prefill = prefill
        self.readout = readout
        self.total = total
    }

    public enum CodingKeys: String, CodingKey, CaseIterable { case prefill, readout, total }
}

/// `Answer`: one of noul, choice, score, told apart by `type`.
public enum Answer: Codable, Sendable, Equatable {
    case noul(NoulAnswer)
    case choice(ChoiceAnswer)
    case score(ScoreAnswer)

    private enum TypeKey: String, CodingKey { case type }

    public init(from decoder: Decoder) throws {
        let raw = try decoder.container(keyedBy: TypeKey.self).decode(String.self, forKey: .type)
        switch raw {
        case "noul": self = .noul(try NoulAnswer(from: decoder))
        case "choice": self = .choice(try ChoiceAnswer(from: decoder))
        case "score": self = .score(try ScoreAnswer(from: decoder))
        default: throw SchemaError(path: pathString(decoder.codingPath), "unknown answer type '\(raw)'")
        }
    }

    public func encode(to encoder: Encoder) throws {
        switch self {
        case .noul(let a): try a.encode(to: encoder)
        case .choice(let a): try a.encode(to: encoder)
        case .score(let a): try a.encode(to: encoder)
        }
    }
}

func checkUnit(_ x: Double?, _ key: CodingKey, _ decoder: Decoder) throws {
    if let x, !(0...1).contains(x) {
        throw SchemaError(path: pathString(decoder.codingPath + [key]), "must be in [0, 1], got \(x)")
    }
}

/// x-local `Debug`: present when options.debug is true.
public struct Debug: Codable, Sendable, Equatable {
    public var readoutPosition: Int?
    public var optionTokenIds: OrderedObject<[Int]>?
    public var rawLogits: OrderedObject<Double>?
    public var temperature: Double?
    /// Per option, max minus min calibrated probability across permutations (> 1 only).
    public var permutationSpread: OrderedObject<Double>?

    public init(readoutPosition: Int? = nil, optionTokenIds: OrderedObject<[Int]>? = nil,
                rawLogits: OrderedObject<Double>? = nil, temperature: Double? = nil,
                permutationSpread: OrderedObject<Double>? = nil) {
        self.readoutPosition = readoutPosition
        self.optionTokenIds = optionTokenIds
        self.rawLogits = rawLogits
        self.temperature = temperature
        self.permutationSpread = permutationSpread
    }

    public enum CodingKeys: String, CodingKey, CaseIterable {
        case readoutPosition = "readout_position"
        case optionTokenIds = "option_token_ids"
        case rawLogits = "raw_logits"
        case temperature
        case permutationSpread = "permutation_spread"
    }
}

public struct NoulAnswer: Codable, Sendable, Equatable {
    public enum Kind: String, Codable, Sendable { case noul }
    public let type: Kind
    /// Calibrated P(yes), in [0, 1].
    public var noul: Double
    /// x-local. Uncalibrated P(yes).
    public var rawNoul: Double?
    public var debug: Debug?

    public init(noul: Double, rawNoul: Double? = nil, debug: Debug? = nil) {
        type = .noul
        self.noul = noul
        self.rawNoul = rawNoul
        self.debug = debug
    }

    public enum CodingKeys: String, CodingKey, CaseIterable {
        case type, noul
        case rawNoul = "raw_noul"
        case debug
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        type = try c.decode(Kind.self, forKey: .type)
        noul = try c.decode(Double.self, forKey: .noul)
        rawNoul = try c.decodeIfPresent(Double.self, forKey: .rawNoul)
        debug = try c.decodeIfPresent(Debug.self, forKey: .debug)
        try checkUnit(noul, CodingKeys.noul, decoder)
        try checkUnit(rawNoul, CodingKeys.rawNoul, decoder)
    }
}

public struct ChoiceAnswer: Codable, Sendable, Equatable {
    public enum Kind: String, Codable, Sendable { case choice }
    public let type: Kind
    public var choice: String
    /// Calibrated; sums to 1 over the declared options.
    public var probabilities: OrderedObject<Double>
    /// x-local. Before temperature scaling.
    public var rawProbabilities: OrderedObject<Double>?
    /// Calibrated probability of the chosen option, in [0, 1].
    public var confidence: Double
    public var debug: Debug?

    public init(choice: String, probabilities: OrderedObject<Double>, rawProbabilities: OrderedObject<Double>? = nil,
                confidence: Double, debug: Debug? = nil) {
        type = .choice
        self.choice = choice
        self.probabilities = probabilities
        self.rawProbabilities = rawProbabilities
        self.confidence = confidence
        self.debug = debug
    }

    public enum CodingKeys: String, CodingKey, CaseIterable {
        case type, choice, probabilities
        case rawProbabilities = "raw_probabilities"
        case confidence, debug
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        type = try c.decode(Kind.self, forKey: .type)
        choice = try c.decode(String.self, forKey: .choice)
        probabilities = try c.decode(OrderedObject<Double>.self, forKey: .probabilities)
        rawProbabilities = try c.decodeIfPresent(OrderedObject<Double>.self, forKey: .rawProbabilities)
        confidence = try c.decode(Double.self, forKey: .confidence)
        debug = try c.decodeIfPresent(Debug.self, forKey: .debug)
        try checkUnit(confidence, CodingKeys.confidence, decoder)
    }
}

public struct ScoreAnswer: Codable, Sendable, Equatable {
    public enum Kind: String, Codable, Sendable { case score }
    public let type: Kind
    /// Index into legend (argmax), >= 0.
    public var score: Int
    /// x-local. Probability-weighted mean index.
    public var expectedScore: Double?
    public var legend: OrderedObject<String>
    public var probabilities: OrderedObject<Double>
    public var rawProbabilities: OrderedObject<Double>?
    public var confidence: Double
    public var debug: Debug?

    public init(score: Int, expectedScore: Double? = nil, legend: OrderedObject<String>,
                probabilities: OrderedObject<Double>, rawProbabilities: OrderedObject<Double>? = nil,
                confidence: Double, debug: Debug? = nil) {
        type = .score
        self.score = score
        self.expectedScore = expectedScore
        self.legend = legend
        self.probabilities = probabilities
        self.rawProbabilities = rawProbabilities
        self.confidence = confidence
        self.debug = debug
    }

    public enum CodingKeys: String, CodingKey, CaseIterable {
        case type, score
        case expectedScore = "expected_score"
        case legend, probabilities
        case rawProbabilities = "raw_probabilities"
        case confidence, debug
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        type = try c.decode(Kind.self, forKey: .type)
        score = try c.decode(Int.self, forKey: .score)
        guard score >= 0 else { throw SchemaError(path: pathString(decoder.codingPath + [CodingKeys.score]), "must be >= 0") }
        expectedScore = try c.decodeIfPresent(Double.self, forKey: .expectedScore)
        legend = try c.decode(OrderedObject<String>.self, forKey: .legend)
        probabilities = try c.decode(OrderedObject<Double>.self, forKey: .probabilities)
        rawProbabilities = try c.decodeIfPresent(OrderedObject<Double>.self, forKey: .rawProbabilities)
        confidence = try c.decode(Double.self, forKey: .confidence)
        debug = try c.decodeIfPresent(Debug.self, forKey: .debug)
        try checkUnit(confidence, CodingKeys.confidence, decoder)
    }
}
