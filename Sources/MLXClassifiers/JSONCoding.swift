// Codable over JSONValue: a Decoder whose keyed containers list `allKeys` in
// document order, and an Encoder that writes keys in encoding order. Together
// they make the schema types round-trip byte-for-byte in key order.

import Foundation

/// Order-preserving JSON coding for the wire and fixture types.
public enum SystemOneJSON {
    public static func decode<T: Decodable>(_ type: T.Type, from data: Data) throws -> T {
        try decode(type, from: try JSONValue.parse(data))
    }

    public static func decode<T: Decodable>(_ type: T.Type, from value: JSONValue) throws -> T {
        try type.init(from: JSONValueDecoderImpl(value: value, codingPath: []))
    }

    public static func encodeValue<T: Encodable>(_ value: T) throws -> JSONValue {
        let root = EncodeNode()
        try value.encode(to: JSONValueEncoderImpl(node: root, codingPath: []))
        return root.value
    }

    public static func encode<T: Encodable>(_ value: T) throws -> Data {
        Data(try encodeValue(value).serialized().utf8)
    }
}

// MARK: - Decoder

struct JSONValueDecoderImpl: Decoder {
    let value: JSONValue
    let codingPath: [CodingKey]
    var userInfo: [CodingUserInfoKey: Any] { [:] }

    func mismatch(_ expected: Any.Type) -> DecodingError {
        .typeMismatch(expected, .init(codingPath: codingPath, debugDescription: "expected \(expected), found \(value.kindName)"))
    }

    func container<Key: CodingKey>(keyedBy type: Key.Type) throws -> KeyedDecodingContainer<Key> {
        guard case .object(let o) = value else { throw mismatch([String: Any].self) }
        return KeyedDecodingContainer(Keyed<Key>(object: o, codingPath: codingPath))
    }

    func unkeyedContainer() throws -> UnkeyedDecodingContainer {
        guard case .array(let a) = value else { throw mismatch([Any].self) }
        return Unkeyed(array: a, codingPath: codingPath)
    }

    func singleValueContainer() throws -> SingleValueDecodingContainer { Single(value: value, codingPath: codingPath) }

    struct Keyed<Key: CodingKey>: KeyedDecodingContainerProtocol {
        let object: OrderedObject<JSONValue>
        let codingPath: [CodingKey]
        var allKeys: [Key] { object.keys.compactMap { Key(stringValue: $0) } }
        func contains(_ key: Key) -> Bool { object[key.stringValue] != nil }

        func child(_ key: Key) throws -> JSONValueDecoderImpl {
            guard let v = object[key.stringValue] else {
                throw DecodingError.keyNotFound(key, .init(codingPath: codingPath, debugDescription: "missing key \(key.stringValue)"))
            }
            return JSONValueDecoderImpl(value: v, codingPath: codingPath + [key])
        }

        func decodeNil(forKey key: Key) throws -> Bool { try child(key).value == .null }
        func decode<T: Decodable>(_ type: T.Type, forKey key: Key) throws -> T { try type.init(from: child(key)) }
        func decode(_ type: Bool.Type, forKey key: Key) throws -> Bool { try type.init(from: child(key)) }
        func decode(_ type: String.Type, forKey key: Key) throws -> String { try type.init(from: child(key)) }
        func decode(_ type: Double.Type, forKey key: Key) throws -> Double { try type.init(from: child(key)) }
        func decode(_ type: Float.Type, forKey key: Key) throws -> Float { try type.init(from: child(key)) }
        func decode(_ type: Int.Type, forKey key: Key) throws -> Int { try type.init(from: child(key)) }
        func decode(_ type: Int8.Type, forKey key: Key) throws -> Int8 { try type.init(from: child(key)) }
        func decode(_ type: Int16.Type, forKey key: Key) throws -> Int16 { try type.init(from: child(key)) }
        func decode(_ type: Int32.Type, forKey key: Key) throws -> Int32 { try type.init(from: child(key)) }
        func decode(_ type: Int64.Type, forKey key: Key) throws -> Int64 { try type.init(from: child(key)) }
        func decode(_ type: UInt.Type, forKey key: Key) throws -> UInt { try type.init(from: child(key)) }
        func decode(_ type: UInt8.Type, forKey key: Key) throws -> UInt8 { try type.init(from: child(key)) }
        func decode(_ type: UInt16.Type, forKey key: Key) throws -> UInt16 { try type.init(from: child(key)) }
        func decode(_ type: UInt32.Type, forKey key: Key) throws -> UInt32 { try type.init(from: child(key)) }
        func decode(_ type: UInt64.Type, forKey key: Key) throws -> UInt64 { try type.init(from: child(key)) }

        func nestedContainer<NestedKey: CodingKey>(keyedBy type: NestedKey.Type, forKey key: Key) throws -> KeyedDecodingContainer<NestedKey> {
            try child(key).container(keyedBy: type)
        }
        func nestedUnkeyedContainer(forKey key: Key) throws -> UnkeyedDecodingContainer { try child(key).unkeyedContainer() }
        func superDecoder() throws -> Decoder { JSONValueDecoderImpl(value: .object(object), codingPath: codingPath) }
        func superDecoder(forKey key: Key) throws -> Decoder { try child(key) }
    }

    struct Unkeyed: UnkeyedDecodingContainer {
        let array: [JSONValue]
        let codingPath: [CodingKey]
        var count: Int? { array.count }
        var isAtEnd: Bool { currentIndex >= array.count }
        var currentIndex = 0

        init(array: [JSONValue], codingPath: [CodingKey]) {
            self.array = array
            self.codingPath = codingPath
        }

        mutating func next() throws -> JSONValueDecoderImpl {
            guard !isAtEnd else {
                throw DecodingError.valueNotFound(Any.self, .init(codingPath: codingPath, debugDescription: "array exhausted"))
            }
            defer { currentIndex += 1 }
            return JSONValueDecoderImpl(value: array[currentIndex], codingPath: codingPath + [AnyCodingKey(intValue: currentIndex)!])
        }

        mutating func decodeNil() throws -> Bool {
            guard !isAtEnd, array[currentIndex] == .null else { return false }
            currentIndex += 1
            return true
        }
        mutating func decode<T: Decodable>(_ type: T.Type) throws -> T { try type.init(from: next()) }
        mutating func decode(_ type: Bool.Type) throws -> Bool { try type.init(from: next()) }
        mutating func decode(_ type: String.Type) throws -> String { try type.init(from: next()) }
        mutating func decode(_ type: Double.Type) throws -> Double { try type.init(from: next()) }
        mutating func decode(_ type: Float.Type) throws -> Float { try type.init(from: next()) }
        mutating func decode(_ type: Int.Type) throws -> Int { try type.init(from: next()) }
        mutating func decode(_ type: Int8.Type) throws -> Int8 { try type.init(from: next()) }
        mutating func decode(_ type: Int16.Type) throws -> Int16 { try type.init(from: next()) }
        mutating func decode(_ type: Int32.Type) throws -> Int32 { try type.init(from: next()) }
        mutating func decode(_ type: Int64.Type) throws -> Int64 { try type.init(from: next()) }
        mutating func decode(_ type: UInt.Type) throws -> UInt { try type.init(from: next()) }
        mutating func decode(_ type: UInt8.Type) throws -> UInt8 { try type.init(from: next()) }
        mutating func decode(_ type: UInt16.Type) throws -> UInt16 { try type.init(from: next()) }
        mutating func decode(_ type: UInt32.Type) throws -> UInt32 { try type.init(from: next()) }
        mutating func decode(_ type: UInt64.Type) throws -> UInt64 { try type.init(from: next()) }
        mutating func nestedContainer<NestedKey: CodingKey>(keyedBy type: NestedKey.Type) throws -> KeyedDecodingContainer<NestedKey> {
            try next().container(keyedBy: type)
        }
        mutating func nestedUnkeyedContainer() throws -> UnkeyedDecodingContainer { try next().unkeyedContainer() }
        mutating func superDecoder() throws -> Decoder { try next() }
    }

    struct Single: SingleValueDecodingContainer {
        let value: JSONValue
        let codingPath: [CodingKey]

        func mismatch(_ t: Any.Type) -> DecodingError {
            .typeMismatch(t, .init(codingPath: codingPath, debugDescription: "expected \(t), found \(value.kindName)"))
        }

        func integer<I: FixedWidthInteger>(_ t: I.Type) throws -> I {
            switch value {
            case .int(let i):
                guard let v = I(exactly: i) else { throw mismatch(t) }
                return v
            case .double(let d):
                guard let v = I(exactly: d) else { throw mismatch(t) }
                return v
            default: throw mismatch(t)
            }
        }

        func decodeNil() -> Bool { value == .null }
        func decode(_ type: Bool.Type) throws -> Bool {
            guard case .bool(let b) = value else { throw mismatch(type) }
            return b
        }
        func decode(_ type: String.Type) throws -> String {
            guard case .string(let s) = value else { throw mismatch(type) }
            return s
        }
        func decode(_ type: Double.Type) throws -> Double {
            switch value {
            case .double(let d): return d
            case .int(let i): return Double(i)
            default: throw mismatch(type)
            }
        }
        func decode(_ type: Float.Type) throws -> Float { Float(try decode(Double.self)) }
        func decode(_ type: Int.Type) throws -> Int { try integer(type) }
        func decode(_ type: Int8.Type) throws -> Int8 { try integer(type) }
        func decode(_ type: Int16.Type) throws -> Int16 { try integer(type) }
        func decode(_ type: Int32.Type) throws -> Int32 { try integer(type) }
        func decode(_ type: Int64.Type) throws -> Int64 { try integer(type) }
        func decode(_ type: UInt.Type) throws -> UInt { try integer(type) }
        func decode(_ type: UInt8.Type) throws -> UInt8 { try integer(type) }
        func decode(_ type: UInt16.Type) throws -> UInt16 { try integer(type) }
        func decode(_ type: UInt32.Type) throws -> UInt32 { try integer(type) }
        func decode(_ type: UInt64.Type) throws -> UInt64 { try integer(type) }
        func decode<T: Decodable>(_ type: T.Type) throws -> T {
            try type.init(from: JSONValueDecoderImpl(value: value, codingPath: codingPath))
        }
    }
}

extension JSONValue {
    var kindName: String {
        switch self {
        case .null: "null"
        case .bool: "bool"
        case .int: "integer"
        case .double: "number"
        case .string: "string"
        case .array: "array"
        case .object: "object"
        }
    }
}

// MARK: - Encoder

/// A mutable tree node the encoder fills in; `value` freezes it.
final class EncodeNode {
    enum Content {
        case unset
        case leaf(JSONValue)
        case object([(String, EncodeNode)])
        case array([EncodeNode])
    }

    var content: Content = .unset

    func set(_ v: JSONValue) { content = .leaf(v) }

    func asObject() {
        if case .object = content { return }
        content = .object([])
    }

    func asArray() {
        if case .array = content { return }
        content = .array([])
    }

    func child(forKey key: String) -> EncodeNode {
        asObject()
        guard case .object(var pairs) = content else { fatalError("unreachable") }
        if let existing = pairs.first(where: { $0.0 == key }) { return existing.1 }
        let n = EncodeNode()
        pairs.append((key, n))
        content = .object(pairs)
        return n
    }

    func appendChild() -> EncodeNode {
        asArray()
        guard case .array(var items) = content else { fatalError("unreachable") }
        let n = EncodeNode()
        items.append(n)
        content = .array(items)
        return n
    }

    var count: Int {
        switch content {
        case .array(let a): a.count
        case .object(let o): o.count
        default: 0
        }
    }

    var value: JSONValue {
        switch content {
        case .unset: .object(OrderedObject())
        case .leaf(let v): v
        case .object(let pairs): .object(OrderedObject(pairs.map { ($0.0, $0.1.value) }))
        case .array(let items): .array(items.map(\.value))
        }
    }
}

struct JSONValueEncoderImpl: Encoder {
    let node: EncodeNode
    let codingPath: [CodingKey]
    var userInfo: [CodingUserInfoKey: Any] { [:] }

    func container<Key: CodingKey>(keyedBy type: Key.Type) -> KeyedEncodingContainer<Key> {
        node.asObject()
        return KeyedEncodingContainer(Keyed<Key>(node: node, codingPath: codingPath))
    }

    func unkeyedContainer() -> UnkeyedEncodingContainer {
        node.asArray()
        return Unkeyed(node: node, codingPath: codingPath)
    }

    func singleValueContainer() -> SingleValueEncodingContainer { Single(node: node, codingPath: codingPath) }

    static func leaf<T: Encodable>(_ v: T, into node: EncodeNode, path: [CodingKey]) throws {
        switch v {
        case let b as Bool: node.set(.bool(b))
        case let s as String: node.set(.string(s))
        case let d as Double: node.set(.double(d))
        case let f as Float: node.set(.double(Double(f)))
        case let i as Int: node.set(.int(i))
        case let i as Int8: node.set(.int(Int(i)))
        case let i as Int16: node.set(.int(Int(i)))
        case let i as Int32: node.set(.int(Int(i)))
        case let i as Int64: node.set(.int(Int(i)))
        case let i as UInt: node.set(.int(Int(i)))
        case let i as UInt8: node.set(.int(Int(i)))
        case let i as UInt16: node.set(.int(Int(i)))
        case let i as UInt32: node.set(.int(Int(i)))
        case let i as UInt64: node.set(.int(Int(i)))
        default: try v.encode(to: JSONValueEncoderImpl(node: node, codingPath: path))
        }
    }

    struct Keyed<Key: CodingKey>: KeyedEncodingContainerProtocol {
        let node: EncodeNode
        let codingPath: [CodingKey]
        mutating func encodeNil(forKey key: Key) throws { node.child(forKey: key.stringValue).set(.null) }
        mutating func encode<T: Encodable>(_ value: T, forKey key: Key) throws {
            try JSONValueEncoderImpl.leaf(value, into: node.child(forKey: key.stringValue), path: codingPath + [key])
        }
        mutating func nestedContainer<NestedKey: CodingKey>(keyedBy keyType: NestedKey.Type, forKey key: Key) -> KeyedEncodingContainer<NestedKey> {
            let c = node.child(forKey: key.stringValue)
            c.asObject()
            return KeyedEncodingContainer(Keyed<NestedKey>(node: c, codingPath: codingPath + [key]))
        }
        mutating func nestedUnkeyedContainer(forKey key: Key) -> UnkeyedEncodingContainer {
            let c = node.child(forKey: key.stringValue)
            c.asArray()
            return Unkeyed(node: c, codingPath: codingPath + [key])
        }
        mutating func superEncoder() -> Encoder {
            JSONValueEncoderImpl(node: node.child(forKey: "super"), codingPath: codingPath)
        }
        mutating func superEncoder(forKey key: Key) -> Encoder {
            JSONValueEncoderImpl(node: node.child(forKey: key.stringValue), codingPath: codingPath + [key])
        }
    }

    struct Unkeyed: UnkeyedEncodingContainer {
        let node: EncodeNode
        let codingPath: [CodingKey]
        var count: Int { node.count }
        mutating func encodeNil() throws { node.appendChild().set(.null) }
        mutating func encode<T: Encodable>(_ value: T) throws {
            let path = codingPath + [AnyCodingKey(intValue: count)!]
            try JSONValueEncoderImpl.leaf(value, into: node.appendChild(), path: path)
        }
        mutating func nestedContainer<NestedKey: CodingKey>(keyedBy keyType: NestedKey.Type) -> KeyedEncodingContainer<NestedKey> {
            let c = node.appendChild()
            c.asObject()
            return KeyedEncodingContainer(Keyed<NestedKey>(node: c, codingPath: codingPath))
        }
        mutating func nestedUnkeyedContainer() -> UnkeyedEncodingContainer {
            let c = node.appendChild()
            c.asArray()
            return Unkeyed(node: c, codingPath: codingPath)
        }
        mutating func superEncoder() -> Encoder { JSONValueEncoderImpl(node: node.appendChild(), codingPath: codingPath) }
    }

    struct Single: SingleValueEncodingContainer {
        let node: EncodeNode
        let codingPath: [CodingKey]
        mutating func encodeNil() throws { node.set(.null) }
        mutating func encode<T: Encodable>(_ value: T) throws {
            try JSONValueEncoderImpl.leaf(value, into: node, path: codingPath)
        }
    }
}
