// Image preprocessing for the Gemma 4 vision tower, matching golden
// `vision/processor.py` (vendored there from mlx-vlm 0.7.8) bit for bit:
// PNG -> RGB uint8 (PIL `Image.open(...).convert("RGB")`), then, when the
// aspect-preserving size differs, PIL `Image.resize((w, h), Image.BICUBIC)`,
// then float32 `* (1/255)`, channel-first `[1, 3, H, W]`.
//
// Neither CoreImage nor CoreGraphics is used: both resample in float (and may
// colour-manage), while PIL resamples uint8 in fixed point. `resizedUInt8` is a
// port of Pillow's `ImagingResample` (libImaging/Resample.c, Pillow 12.3.0 in the
// golden venv): a separable two-pass filter, horizontal first, each pass clamped
// to uint8; the bicubic kernel with a = -0.5 and support 2 (scaled by the
// downscale factor); per output pixel the weights are normalised in double and
// converted to 22-bit fixed point (`normalize_coeffs_8bpc`), accumulated in Int32
// from a half-unit rounding offset and clipped with `clip8`.
// The PNG decoder handles what the PNG spec calls 8-bit non-interlaced images
// (grey, RGB, palette, grey+alpha, RGBA); alpha is dropped as PIL's RGB convert does.

import Compression
import Foundation
import MLX

/// An 8-bit image, row-major HWC.
public struct RGBImage: Sendable, Equatable {
    public let width: Int
    public let height: Int
    public let channels: Int
    public var pixels: [UInt8]
}

// MARK: - PNG decode

/// Golden `Image.open(png).convert("RGB")` as uint8 HWC, for 8-bit non-interlaced PNGs.
public func decodePNG(_ data: Data) throws -> RGBImage {
    let b = [UInt8](data)
    func fail(_ m: String) -> ClassifierError { ClassifierError(.prompt, "image block PNG: \(m)") }
    guard b.count > 33, Array(b[0 ..< 8]) == [0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A] else { throw fail("bad signature") }
    func be32(_ i: Int) -> Int { b[i ..< i + 4].reduce(0) { $0 << 8 | Int($1) } }
    var i = 8
    var width = 0, height = 0, depth = 0, colour = 0, interlace = 0
    var palette: [UInt8] = []
    var idat = Data()
    while i + 8 <= b.count {
        let n = be32(i)
        let kind = String(decoding: b[i + 4 ..< i + 8], as: UTF8.self)
        let body = i + 8
        guard body + n + 4 <= b.count else { throw fail("truncated \(kind) chunk") }
        switch kind {
        case "IHDR":
            (width, height, depth, colour, interlace) = (be32(body), be32(body + 4), Int(b[body + 8]), Int(b[body + 9]), Int(b[body + 12]))
        case "PLTE": palette = Array(b[body ..< body + n])
        case "IDAT": idat.append(contentsOf: b[body ..< body + n])
        default: break
        }
        if kind == "IEND" { break }
        i = body + n + 4
    }
    guard width > 0, height > 0 else { throw fail("no IHDR") }
    guard depth == 8, interlace == 0 else { throw fail("only 8-bit non-interlaced PNGs are supported (depth \(depth), interlace \(interlace))") }
    let samples: Int
    switch colour {
    case 0: samples = 1
    case 2: samples = 3
    case 3: samples = 1
    case 4: samples = 2
    case 6: samples = 4
    default: throw fail("colour type \(colour)")
    }
    let stride = width * samples
    let raw = try inflateZlib(idat, expected: height * (stride + 1))
    guard raw.count == height * (stride + 1) else { throw fail("IDAT holds \(raw.count) bytes, expected \(height * (stride + 1))") }
    // unfilter (PNG spec 9.2), bpp = samples bytes
    var cur = [UInt8](repeating: 0, count: stride), prev = [UInt8](repeating: 0, count: stride)
    var out = [UInt8](repeating: 0, count: width * height * 3)
    for y in 0 ..< height {
        let row = y * (stride + 1)
        let f = raw[row]
        for x in 0 ..< stride {
            let v = raw[row + 1 + x]
            let a = x >= samples ? Int(cur[x - samples]) : 0
            let up = Int(prev[x])
            let c = x >= samples ? Int(prev[x - samples]) : 0
            let pred: Int
            switch f {
            case 0: pred = 0
            case 1: pred = a
            case 2: pred = up
            case 3: pred = (a + up) / 2
            case 4:
                let p = a + up - c, pa = abs(p - a), pb = abs(p - up), pc = abs(p - c)
                pred = pa <= pb && pa <= pc ? a : (pb <= pc ? up : c)
            default: throw fail("filter type \(f)")
            }
            cur[x] = UInt8(truncatingIfNeeded: Int(v) + pred)
        }
        for x in 0 ..< width {
            let o = (y * width + x) * 3
            switch colour {
            case 0, 4:
                let g = cur[x * samples]
                (out[o], out[o + 1], out[o + 2]) = (g, g, g)
            case 3:
                let k = Int(cur[x]) * 3
                guard k + 2 < palette.count else { throw fail("palette index out of range") }
                (out[o], out[o + 1], out[o + 2]) = (palette[k], palette[k + 1], palette[k + 2])
            default:
                (out[o], out[o + 1], out[o + 2]) = (cur[x * samples], cur[x * samples + 1], cur[x * samples + 2])
            }
        }
        swap(&cur, &prev)
    }
    return RGBImage(width: width, height: height, channels: 3, pixels: out)
}

/// zlib stream (RFC 1950) to bytes: Apple's `COMPRESSION_ZLIB` is raw DEFLATE, so the
/// 2-byte header is skipped (the Adler-32 trailer is not checked).
func inflateZlib(_ data: Data, expected: Int) throws -> [UInt8] {
    guard data.count > 6 else { throw ClassifierError(.prompt, "image block PNG: empty IDAT") }
    let src = [UInt8](data.dropFirst(2))
    var dst = [UInt8](repeating: 0, count: expected + 1)
    let n = compression_decode_buffer(&dst, dst.count, src, src.count, nil, COMPRESSION_ZLIB)
    return Array(dst[..<n])
}

// MARK: - PIL bicubic on uint8

let pilPrecisionBits = 32 - 8 - 2

/// Pillow `bicubic_filter` (a = -0.5).
func pilBicubic(_ x: Double) -> Double {
    let a = -0.5
    let x = abs(x)
    if x < 1.0 { return ((a + 2.0) * x - (a + 3.0)) * x * x + 1 }
    if x < 2.0 { return (((x - 5) * x + 8) * x - 4) * a }
    return 0.0
}

/// Pillow `precompute_coeffs` + `normalize_coeffs_8bpc` over the full box: per output
/// index, the first input index and the fixed-point weights.
func pilCoefficients(inSize: Int, outSize: Int) -> [(min: Int, k: [Int32])] {
    let scale = Double(inSize) / Double(outSize)
    let filterScale = max(scale, 1.0)
    let support = 2.0 * filterScale
    return (0 ..< outSize).map { xx in
        let center = (Double(xx) + 0.5) * scale
        let ss = 1.0 / filterScale
        let xmin = max(Int(center - support + 0.5), 0)  // C (int) truncation; negatives are clamped
        let xmax = min(Int(center + support + 0.5), inSize) - xmin
        var k = (0 ..< xmax).map { pilBicubic((Double($0 + xmin) - center + 0.5) * ss) }
        var ww = 0.0
        for v in k { ww += v }
        if ww != 0 { k = k.map { $0 / ww } }
        let one = Double(1 << pilPrecisionBits)
        return (xmin, k.map { $0 < 0 ? Int32(-0.5 + $0 * one) : Int32(0.5 + $0 * one) })
    }
}

@inline(__always) func pilClip8(_ v: Int32) -> UInt8 {
    if v >= Int32((1 << pilPrecisionBits) << 8) { return 255 }
    if v <= 0 { return 0 }
    return UInt8(v >> Int32(pilPrecisionBits))
}

/// One resampling pass (Pillow `ImagingResampleHorizontal_8bpc` / `..Vertical_8bpc`):
/// `count` output samples along an axis of `inSize`, each the fixed-point dot product
/// of `coeffs` with input samples `stride` apart.
func pilPass(_ src: UnsafeBufferPointer<UInt8>, _ dst: UnsafeMutableBufferPointer<UInt8>,
             coeffs: [(min: Int, k: [Int32])], lines: Int, lineStrideIn: Int, lineStrideOut: Int,
             sampleStrideIn: Int, sampleStrideOut: Int, channels: Int) {
    let half = Int32(1 << (pilPrecisionBits - 1))
    for (o, co) in coeffs.enumerated() {
        co.k.withUnsafeBufferPointer { k in
            for line in 0 ..< lines {
                let inBase = line * lineStrideIn + co.min * sampleStrideIn
                let outBase = line * lineStrideOut + o * sampleStrideOut
                for ch in 0 ..< channels {
                    var ss = half
                    var at = inBase + ch
                    for w in k {
                        ss = ss &+ Int32(src[at]) &* w
                        at += sampleStrideIn
                    }
                    dst[outBase + ch] = pilClip8(ss)
                }
            }
        }
    }
}

/// Pillow `Image.resize((width, height), Image.BICUBIC)` on an 8-bit image: the
/// horizontal pass first, then the vertical one, each clamped to uint8.
public func pilResizeBicubic(_ img: RGBImage, width tw: Int, height th: Int) -> RGBImage {
    let c = img.channels
    var src = img
    if tw != img.width {
        let coeffs = pilCoefficients(inSize: img.width, outSize: tw)
        var out = [UInt8](repeating: 0, count: tw * img.height * c)
        img.pixels.withUnsafeBufferPointer { p in
            out.withUnsafeMutableBufferPointer { q in
                pilPass(p, q, coeffs: coeffs, lines: img.height, lineStrideIn: img.width * c, lineStrideOut: tw * c,
                        sampleStrideIn: c, sampleStrideOut: c, channels: c)
            }
        }
        src = RGBImage(width: tw, height: img.height, channels: c, pixels: out)
    }
    if th != src.height {
        let coeffs = pilCoefficients(inSize: src.height, outSize: th)
        var out = [UInt8](repeating: 0, count: tw * th * c)
        src.pixels.withUnsafeBufferPointer { p in
            out.withUnsafeMutableBufferPointer { q in
                // lines are columns (x), samples run down the rows
                pilPass(p, q, coeffs: coeffs, lines: tw, lineStrideIn: c, lineStrideOut: c,
                        sampleStrideIn: tw * c, sampleStrideOut: tw * c, channels: c)
            }
        }
        src = RGBImage(width: tw, height: th, channels: c, pixels: out)
    }
    return src
}

/// Golden `processor.resized_uint8`: the PNG decoded to RGB and bicubic-resized (PIL)
/// to the aspect-preserving size for `maxSoftTokens`; H and W multiples of
/// `patchSize * poolingKernelSize`.
public func resizedUInt8(_ png: Data, patchSize: Int = 16, maxSoftTokens: Int = 280,
                         poolingKernelSize: Int = 3) throws -> RGBImage {
    let img = try decodePNG(png)
    let (th, tw) = try resizedSize(height: img.height, width: img.width, patchSize: patchSize,
                                   maxPatches: maxSoftTokens * poolingKernelSize * poolingKernelSize,
                                   pooling: poolingKernelSize)
    if th == img.height, tw == img.width { return img }
    return pilResizeBicubic(img, width: tw, height: th)
}

/// Golden `processor.preprocess_png`: `[1, 3, H, W]` float32 in [0, 1] (uint8 as
/// float32 times float32(1/255), as numpy computes it).
public func pixelValues(_ img: RGBImage) -> MLXArray {
    let r = Float(1.0 / 255.0)
    let hwc = MLXArray(img.pixels.map { Float($0) * r }, [img.height, img.width, img.channels])
    return hwc.transposed(2, 0, 1)[.newAxis]
}
