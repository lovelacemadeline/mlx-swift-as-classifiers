// JSONValue — an order-preserving JSON tree, with a parser and a serialiser.
//
// Why not Foundation's JSONDecoder: key order is semantic on this wire. A
// choice question's `criteria` object declares the options in order, and that
// order decides which option is shown as "A." (golden `prompt.py`). Foundation's
// JSONDecoder/JSONSerialization go through an unordered dictionary, so
// `KeyedDecodingContainer.allKeys` comes back in hash order. `SystemOneJSON`
// (JSONCoding.swift) decodes and encodes the Codable schema types through this
// tree instead, and `OrderedObject` keeps document order end to end.

import Foundation

/// A JSON object whose keys keep their document (or insertion) order.
public struct OrderedObject<Value> {
    public struct Entry {
        public var key: String
        public var value: Value
        public init(key: String, value: Value) {
            self.key = key
            self.value = value
        }
    }

    public private(set) var entries: [Entry]

    public init() { entries = [] }

    /// Pairs in order; a repeated key keeps its first position and its last value
    /// (Python's `json.loads` keeps the last value too).
    public init(_ pairs: [(String, Value)]) {
        entries = []
        for (k, v) in pairs { self[k] = v }
    }

    public var keys: [String] { entries.map(\.key) }
    public var values: [Value] { entries.map(\.value) }
    public var count: Int { entries.count }
    public var isEmpty: Bool { entries.isEmpty }

    public subscript(key: String) -> Value? {
        get { entries.first(where: { $0.key == key })?.value }
        set {
            if let i = entries.firstIndex(where: { $0.key == key }) {
                if let newValue { entries[i].value = newValue } else { entries.remove(at: i) }
            } else if let newValue {
                entries.append(Entry(key: key, value: newValue))
            }
        }
    }

    public func mapValues<T>(_ transform: (Value) throws -> T) rethrows -> OrderedObject<T> {
        var out = OrderedObject<T>()
        for e in entries { out[e.key] = try transform(e.value) }
        return out
    }
}

extension OrderedObject: Sendable where Value: Sendable {}
extension OrderedObject.Entry: Sendable where Value: Sendable {}
extension OrderedObject.Entry: Equatable where Value: Equatable {}
extension OrderedObject: Equatable where Value: Equatable {}

extension OrderedObject: ExpressibleByDictionaryLiteral {
    public init(dictionaryLiteral elements: (String, Value)...) { self.init(elements) }
}

extension OrderedObject: Codable where Value: Codable {
    /// Keys come back in `decoder`'s `allKeys` order: document order under
    /// `SystemOneJSON`, unspecified under Foundation's JSONDecoder.
    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: AnyCodingKey.self)
        var out = OrderedObject<Value>()
        for k in c.allKeys {
            // `decode(Value.self)` with Value == Optional<T> must see JSON null.
            out.entries.append(Entry(key: k.stringValue, value: try c.decode(Value.self, forKey: k)))
        }
        self = out
    }

    public func encode(to encoder: Encoder) throws {
        var c = encoder.container(keyedBy: AnyCodingKey.self)
        for e in entries { try c.encode(e.value, forKey: AnyCodingKey(e.key)) }
    }
}

/// A coding key for arbitrary object keys.
public struct AnyCodingKey: CodingKey, Hashable, Sendable {
    public var stringValue: String
    public var intValue: Int?
    public init(_ s: String) { stringValue = s }
    public init?(stringValue: String) { self.stringValue = stringValue }
    public init?(intValue: Int) {
        stringValue = String(intValue)
        self.intValue = intValue
    }
}

/// A JSON value. Integers and non-integers are kept apart so a round trip
/// writes `20`, not `20.0`.
public enum JSONValue: Sendable, Equatable {
    case null
    case bool(Bool)
    case int(Int)
    case double(Double)
    case string(String)
    case array([JSONValue])
    case object(OrderedObject<JSONValue>)

    public subscript(key: String) -> JSONValue? {
        if case .object(let o) = self { return o[key] }
        return nil
    }

    public var objectValue: OrderedObject<JSONValue>? {
        if case .object(let o) = self { return o }
        return nil
    }
}

extension JSONValue: Codable {
    public init(from decoder: Decoder) throws {
        if let d = decoder as? JSONValueDecoderImpl {
            self = d.value
            return
        }
        // Foundation decoders: best effort (object key order is theirs).
        let c = try decoder.singleValueContainer()
        if c.decodeNil() { self = .null }
        else if let b = try? c.decode(Bool.self) { self = .bool(b) }
        else if let i = try? c.decode(Int.self) { self = .int(i) }
        else if let x = try? c.decode(Double.self) { self = .double(x) }
        else if let s = try? c.decode(String.self) { self = .string(s) }
        else if let a = try? c.decode([JSONValue].self) { self = .array(a) }
        else { self = .object(try c.decode(OrderedObject<JSONValue>.self)) }
    }

    public func encode(to encoder: Encoder) throws {
        if let e = encoder as? JSONValueEncoderImpl {
            e.node.set(self)
            return
        }
        var c = encoder.singleValueContainer()
        switch self {
        case .null: try c.encodeNil()
        case .bool(let b): try c.encode(b)
        case .int(let i): try c.encode(i)
        case .double(let x): try c.encode(x)
        case .string(let s): try c.encode(s)
        case .array(let a): try c.encode(a)
        case .object(let o): try c.encode(o)
        }
    }
}

// MARK: - Parsing

public struct JSONParseError: Error, CustomStringConvertible, Sendable {
    public let offset: Int
    public let message: String
    public var description: String { "JSON parse error at byte \(offset): \(message)" }
}

extension JSONValue {
    /// Parse UTF-8 JSON (RFC 8259), keeping object key order.
    public static func parse(_ data: Data) throws -> JSONValue {
        var p = Parser(bytes: [UInt8](data))
        p.skipWS()
        let v = try p.value(depth: 0)
        p.skipWS()
        guard p.i == p.bytes.count else { throw p.err("trailing characters") }
        return v
    }

    public static func parse(_ text: String) throws -> JSONValue { try parse(Data(text.utf8)) }

    private struct Parser {
        let bytes: [UInt8]
        var i = 0

        func err(_ m: String) -> JSONParseError { JSONParseError(offset: i, message: m) }

        mutating func skipWS() {
            while i < bytes.count, [0x20, 0x09, 0x0A, 0x0D].contains(bytes[i]) { i += 1 }
        }

        mutating func expect(_ lit: String) throws {
            for b in lit.utf8 {
                guard i < bytes.count, bytes[i] == b else { throw err("expected \(lit)") }
                i += 1
            }
        }

        mutating func value(depth: Int) throws -> JSONValue {
            guard depth < 512 else { throw err("nesting too deep") }
            guard i < bytes.count else { throw err("unexpected end") }
            switch bytes[i] {
            case UInt8(ascii: "{"): return .object(try object(depth: depth))
            case UInt8(ascii: "["): return .array(try array(depth: depth))
            case UInt8(ascii: "\""): return .string(try string())
            case UInt8(ascii: "t"): try expect("true"); return .bool(true)
            case UInt8(ascii: "f"): try expect("false"); return .bool(false)
            case UInt8(ascii: "n"): try expect("null"); return .null
            default: return try number()
            }
        }

        mutating func object(depth: Int) throws -> OrderedObject<JSONValue> {
            i += 1
            var out = OrderedObject<JSONValue>()
            skipWS()
            if i < bytes.count, bytes[i] == UInt8(ascii: "}") { i += 1; return out }
            while true {
                skipWS()
                guard i < bytes.count, bytes[i] == UInt8(ascii: "\"") else { throw err("expected object key") }
                let k = try string()
                skipWS()
                try expect(":")
                skipWS()
                out[k] = try value(depth: depth + 1)
                skipWS()
                guard i < bytes.count else { throw err("unterminated object") }
                if bytes[i] == UInt8(ascii: ",") { i += 1; continue }
                if bytes[i] == UInt8(ascii: "}") { i += 1; return out }
                throw err("expected , or }")
            }
        }

        mutating func array(depth: Int) throws -> [JSONValue] {
            i += 1
            var out: [JSONValue] = []
            skipWS()
            if i < bytes.count, bytes[i] == UInt8(ascii: "]") { i += 1; return out }
            while true {
                skipWS()
                out.append(try value(depth: depth + 1))
                skipWS()
                guard i < bytes.count else { throw err("unterminated array") }
                if bytes[i] == UInt8(ascii: ",") { i += 1; continue }
                if bytes[i] == UInt8(ascii: "]") { i += 1; return out }
                throw err("expected , or ]")
            }
        }

        mutating func hex4() throws -> UInt32 {
            guard i + 4 <= bytes.count, let v = UInt32(String(decoding: bytes[i..<i + 4], as: UTF8.self), radix: 16)
            else { throw err("bad \\u escape") }
            i += 4
            return v
        }

        mutating func string() throws -> String {
            i += 1
            var out = [UInt8]()
            while true {
                guard i < bytes.count else { throw err("unterminated string") }
                let b = bytes[i]
                if b == UInt8(ascii: "\"") { i += 1; break }
                if b < 0x20 { throw err("control character in string") }
                if b != UInt8(ascii: "\\") { out.append(b); i += 1; continue }
                i += 1
                guard i < bytes.count else { throw err("unterminated escape") }
                let e = bytes[i]
                i += 1
                switch e {
                case UInt8(ascii: "\""): out.append(0x22)
                case UInt8(ascii: "\\"): out.append(0x5C)
                case UInt8(ascii: "/"): out.append(0x2F)
                case UInt8(ascii: "b"): out.append(0x08)
                case UInt8(ascii: "f"): out.append(0x0C)
                case UInt8(ascii: "n"): out.append(0x0A)
                case UInt8(ascii: "r"): out.append(0x0D)
                case UInt8(ascii: "t"): out.append(0x09)
                case UInt8(ascii: "u"):
                    var cp = try hex4()
                    if (0xD800..<0xDC00).contains(cp) {
                        guard i + 1 < bytes.count, bytes[i] == UInt8(ascii: "\\"), bytes[i + 1] == UInt8(ascii: "u")
                        else { throw err("lone high surrogate") }
                        i += 2
                        let lo = try hex4()
                        guard (0xDC00..<0xE000).contains(lo) else { throw err("bad low surrogate") }
                        cp = 0x10000 + ((cp - 0xD800) << 10) + (lo - 0xDC00)
                    } else if (0xDC00..<0xE000).contains(cp) {
                        throw err("lone low surrogate")
                    }
                    guard let scalar = Unicode.Scalar(cp) else { throw err("bad code point") }
                    out.append(contentsOf: Array(String(Character(scalar)).utf8))
                default: throw err("bad escape")
                }
            }
            guard let s = String(bytes: out, encoding: .utf8) else { throw err("invalid UTF-8") }
            return s
        }

        mutating func number() throws -> JSONValue {
            let start = i
            var isInt = true
            if i < bytes.count, bytes[i] == UInt8(ascii: "-") { i += 1 }
            guard i < bytes.count, (0x30...0x39).contains(bytes[i]) else { throw err("unexpected character") }
            if bytes[i] == 0x30 { i += 1 } else { while i < bytes.count, (0x30...0x39).contains(bytes[i]) { i += 1 } }
            if i < bytes.count, bytes[i] == UInt8(ascii: ".") {
                isInt = false
                i += 1
                guard i < bytes.count, (0x30...0x39).contains(bytes[i]) else { throw err("bad fraction") }
                while i < bytes.count, (0x30...0x39).contains(bytes[i]) { i += 1 }
            }
            if i < bytes.count, bytes[i] == UInt8(ascii: "e") || bytes[i] == UInt8(ascii: "E") {
                isInt = false
                i += 1
                if i < bytes.count, bytes[i] == UInt8(ascii: "+") || bytes[i] == UInt8(ascii: "-") { i += 1 }
                guard i < bytes.count, (0x30...0x39).contains(bytes[i]) else { throw err("bad exponent") }
                while i < bytes.count, (0x30...0x39).contains(bytes[i]) { i += 1 }
            }
            let text = String(decoding: bytes[start..<i], as: UTF8.self)
            if isInt, let v = Int(text) { return .int(v) }
            guard let d = Double(text) else { throw err("bad number \(text)") }
            return .double(d)
        }
    }
}

// MARK: - Serialising

extension JSONValue {
    /// Compact JSON in key order. Doubles use Swift's shortest round-trip form.
    public func serialized() -> String {
        var out = ""
        write(into: &out)
        return out
    }

    private func write(into out: inout String) {
        switch self {
        case .null: out += "null"
        case .bool(let b): out += b ? "true" : "false"
        case .int(let i): out += String(i)
        case .double(let d):
            precondition(d.isFinite, "JSON cannot represent \(d)")
            out += "\(d)"
        case .string(let s): JSONValue.writeString(s, into: &out)
        case .array(let a):
            out += "["
            for (n, v) in a.enumerated() {
                if n > 0 { out += "," }
                v.write(into: &out)
            }
            out += "]"
        case .object(let o):
            out += "{"
            for (n, e) in o.entries.enumerated() {
                if n > 0 { out += "," }
                JSONValue.writeString(e.key, into: &out)
                out += ":"
                e.value.write(into: &out)
            }
            out += "}"
        }
    }

    private static func writeString(_ s: String, into out: inout String) {
        out += "\""
        for u in s.unicodeScalars {
            switch u {
            case "\"": out += "\\\""
            case "\\": out += "\\\\"
            case "\n": out += "\\n"
            case "\r": out += "\\r"
            case "\t": out += "\\t"
            case _ where u.value < 0x20:
                out += String(format: "\\u%04x", u.value)
            default: out.unicodeScalars.append(u)
            }
        }
        out += "\""
    }
}
