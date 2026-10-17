// Python string semantics the golden prompt code relies on. Swift's String
// compares by grapheme cluster under canonical equivalence; Python compares code
// points. Prompt compilation must use Python's rules, so these work on scalars.

import Foundation

extension String {
    /// Python `str.startswith`: a code-point prefix (UTF-8 byte prefix).
    func pyStartsWith(_ prefix: String) -> Bool {
        let a = Array(utf8), p = Array(prefix.utf8)
        return a.count >= p.count && Array(a[0..<p.count]) == p
    }

    /// Python `str.count(sub)` (non-overlapping), over UTF-8 bytes.
    func pyCount(_ sub: String) -> Int {
        let a = Array(utf8), s = Array(sub.utf8)
        guard !s.isEmpty, a.count >= s.count else { return 0 }
        var n = 0, i = 0
        while i <= a.count - s.count {
            if Array(a[i..<i + s.count]) == s { n += 1; i += s.count } else { i += 1 }
        }
        return n
    }

    /// Python `str.replace(old, new)` over code points.
    func pyReplacing(_ old: String, with new: String) -> String {
        let a = Array(utf8), o = Array(old.utf8), n = Array(new.utf8)
        guard !o.isEmpty else { return self }
        var out: [UInt8] = []
        out.reserveCapacity(a.count)
        var i = 0
        while i < a.count {
            if i + o.count <= a.count, Array(a[i..<i + o.count]) == o {
                out += n
                i += o.count
            } else {
                out.append(a[i])
                i += 1
            }
        }
        return String(decoding: out, as: UTF8.self)
    }

    /// Python `str.isspace()` of the last code point (false for "").
    var pyEndsWithSpace: Bool {
        guard let last = unicodeScalars.last else { return false }
        // Python's isspace: Unicode White_Space plus the C0 separators 0x1C-0x1F.
        return last.properties.isWhitespace || (0x1C...0x1F).contains(last.value)
    }
}

/// Python `json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False)`.
enum PythonJSON {
    static func dumps(_ value: JSONValue) -> String {
        var out = ""
        write(value, indent: 0, into: &out)
        return out
    }

    private static func write(_ v: JSONValue, indent: Int, into out: inout String) {
        switch v {
        case .null: out += "null"
        case .bool(let b): out += b ? "true" : "false"
        case .int(let i): out += String(i)
        case .double(let d): out += floatRepr(d)
        case .string(let s): writeString(s, into: &out)
        case .array(let a):
            if a.isEmpty { out += "[]"; return }
            let pad = String(repeating: " ", count: indent + 2)
            out += "[\n"
            for (k, x) in a.enumerated() {
                out += pad
                write(x, indent: indent + 2, into: &out)
                if k < a.count - 1 { out += "," }
                out += "\n"
            }
            out += String(repeating: " ", count: indent) + "]"
        case .object(let o):
            if o.isEmpty { out += "{}"; return }
            let pad = String(repeating: " ", count: indent + 2)
            // sort_keys: Python sorts str keys by code point.
            let entries = o.entries.sorted { $0.key.unicodeScalars.map(\.value).lexicographicallyPrecedes($1.key.unicodeScalars.map(\.value)) }
            out += "{\n"
            for (k, e) in entries.enumerated() {
                out += pad
                writeString(e.key, into: &out)
                out += ": "
                write(e.value, indent: indent + 2, into: &out)
                if k < entries.count - 1 { out += "," }
                out += "\n"
            }
            out += String(repeating: " ", count: indent) + "}"
        }
    }

    /// `json.encoder.py_encode_basestring` (ensure_ascii=False).
    private static func writeString(_ s: String, into out: inout String) {
        out += "\""
        for u in s.unicodeScalars {
            switch u {
            case "\"": out += "\\\""
            case "\\": out += "\\\\"
            case "\n": out += "\\n"
            case "\r": out += "\\r"
            case "\t": out += "\\t"
            case "\u{08}": out += "\\b"
            case "\u{0C}": out += "\\f"
            case _ where u.value < 0x20: out += String(format: "\\u%04x", u.value)
            default: out.unicodeScalars.append(u)
            }
        }
        out += "\""
    }

    /// Python `float.__repr__` (what json uses): the shortest round-trip digits,
    /// fixed notation for decimal exponents -4...15, else `d.ddde+XX`.
    static func floatRepr(_ d: Double) -> String {
        if d.isNaN { return "NaN" }
        if d.isInfinite { return d < 0 ? "-Infinity" : "Infinity" }
        if d == 0 { return d.sign == .minus ? "-0.0" : "0.0" }
        // Swift's description is also shortest round-trip; take its digits and exponent.
        var s = "\(abs(d))"
        var exp10 = 0
        if let e = s.firstIndex(where: { $0 == "e" || $0 == "E" }) {
            exp10 = Int(s[s.index(after: e)...])!
            s = String(s[..<e])
        }
        let parts = s.split(separator: ".", omittingEmptySubsequences: false)
        let intPart = String(parts[0]), frac = parts.count > 1 ? String(parts[1]) : ""
        var digits = intPart + frac
        var point = intPart.count + exp10  // decimal point position within digits
        let lead = digits.prefix { $0 == "0" }.count
        digits.removeFirst(lead)
        point -= lead
        while digits.hasSuffix("0") && digits.count > 1 { digits.removeLast() }
        let x = point - 1  // scientific exponent
        let sign = d < 0 ? "-" : ""
        if x < -4 || x >= 16 {
            let mant = digits.count > 1 ? "\(digits.first!).\(digits.dropFirst())" : "\(digits)"
            let es = String(format: "%02d", abs(x))
            return "\(sign)\(mant)e\(x < 0 ? "-" : "+")\(es)"
        }
        if point <= 0 { return "\(sign)0.\(String(repeating: "0", count: -point))\(digits)" }
        if point >= digits.count { return "\(sign)\(digits)\(String(repeating: "0", count: point - digits.count)).0" }
        let i = digits.index(digits.startIndex, offsetBy: point)
        return "\(sign)\(digits[..<i]).\(digits[i...])"
    }
}
