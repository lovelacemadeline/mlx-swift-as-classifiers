// CPython's `random.Random`, only as far as golden `permutation_orders` needs it:
// seeding from a str (version 2: the bytes plus their SHA-512, as one big-endian
// integer, fed to MT19937 `init_by_array`) and `shuffle` (Python >= 3.11:
// `_randbelow_with_getrandbits`). Same seed, same permutation as Python.

import CryptoKit
import Foundation

struct PythonRandom {
    private var mt = [UInt32](repeating: 0, count: 624)
    private var index = 625

    init(seed: String) {
        let bytes = Array(seed.utf8)
        let digest = Array(SHA512.hash(data: bytes))
        // int.from_bytes(bytes + digest, "big") as 32-bit words, least significant first.
        var bigEndian = bytes + digest
        while let first = bigEndian.first, first == 0 { bigEndian.removeFirst() }
        var key: [UInt32] = []
        var i = bigEndian.count
        while i > 0 {
            let lo = max(0, i - 4)
            var w: UInt32 = 0
            for b in bigEndian[lo ..< i] { w = (w << 8) | UInt32(b) }
            key.append(w)
            i = lo
        }
        if key.isEmpty { key = [0] }
        initByArray(key)
    }

    private mutating func initGenrand(_ s: UInt32) {
        mt[0] = s
        for i in 1 ..< 624 {
            mt[i] = 1_812_433_253 &* (mt[i - 1] ^ (mt[i - 1] >> 30)) &+ UInt32(i)
        }
        index = 624
    }

    private mutating func initByArray(_ key: [UInt32]) {
        initGenrand(19_650_218)
        var i = 1, j = 0
        for _ in 0 ..< max(624, key.count) {
            mt[i] = (mt[i] ^ ((mt[i - 1] ^ (mt[i - 1] >> 30)) &* 1_664_525)) &+ key[j] &+ UInt32(j)
            i += 1; j += 1
            if i >= 624 { mt[0] = mt[623]; i = 1 }
            if j >= key.count { j = 0 }
        }
        for _ in 0 ..< 623 {
            mt[i] = (mt[i] ^ ((mt[i - 1] ^ (mt[i - 1] >> 30)) &* 1_566_083_941)) &- UInt32(i)
            i += 1
            if i >= 624 { mt[0] = mt[623]; i = 1 }
        }
        mt[0] = 0x8000_0000
    }

    mutating func nextUInt32() -> UInt32 {
        if index >= 624 {
            for k in 0 ..< 624 {
                let y = (mt[k] & 0x8000_0000) | (mt[(k + 1) % 624] & 0x7FFF_FFFF)
                mt[k] = mt[(k + 397) % 624] ^ (y >> 1) ^ ((y & 1) == 0 ? 0 : 0x9908_B0DF)
            }
            index = 0
        }
        var y = mt[index]
        index += 1
        y ^= y >> 11
        y ^= (y << 7) & 0x9D2C_5680
        y ^= (y << 15) & 0xEFC6_0000
        y ^= y >> 18
        return y
    }

    /// `getrandbits(k)` for 0 < k <= 32.
    mutating func getrandbits(_ k: Int) -> UInt32 {
        precondition(0 < k && k <= 32)
        return nextUInt32() >> UInt32(32 - k)
    }

    /// `_randbelow_with_getrandbits(n)`, n > 0.
    mutating func randbelow(_ n: Int) -> Int {
        let k = Int.bitWidth - n.leadingZeroBitCount
        var r = Int(getrandbits(k))
        while r >= n { r = Int(getrandbits(k)) }
        return r
    }

    /// `random.shuffle(x)`.
    mutating func shuffle<T>(_ x: inout [T]) {
        guard x.count > 1 else { return }
        for i in stride(from: x.count - 1, through: 1, by: -1) {
            x.swapAt(i, randbelow(i + 1))
        }
    }
}
