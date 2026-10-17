// The text side of image blocks, ported from golden `images.py`: the placeholder
// run `boi + image * N + eoi` a block expands to, N from Gemma 4's
// aspect-preserving resize arithmetic, and the runs' positions in the token ids.
// The pixels (decode, resize, the tower) are Vision/ and Models/Gemma4Vision.swift
// (story 4.1); this file needs only the PNG IHDR size.

import CryptoKit
import Foundation

public enum SoftTokenBudget {
    /// Gemma 4's processor default (`max_soft_tokens`).
    public static let defaultBudget = 280
}

/// Golden `png_size`: `(width, height)` from the IHDR chunk.
public func pngSize(_ data: Data) throws -> (width: Int, height: Int) {
    let b = [UInt8](data.prefix(24))
    let signature: [UInt8] = [0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A]
    guard b.count >= 24, Array(b[0..<8]) == signature, Array(b[12..<16]) == Array("IHDR".utf8) else {
        throw ClassifierError(.prompt, "image block is not a PNG (bad signature or missing IHDR)")
    }
    func be32(_ i: Int) -> Int { b[i..<i + 4].reduce(0) { $0 << 8 | Int($1) } }
    let width = be32(16), height = be32(20)
    guard width > 0, height > 0 else { throw ClassifierError(.prompt, "image block is an empty PNG") }
    return (width, height)
}

/// Golden `decode_png_base64`: the PNG bytes of an image block, validated as a PNG.
public func decodePNGBase64(_ text: String) throws -> Data {
    guard let data = Data(base64Encoded: text) else {
        throw ClassifierError(.prompt, "image block base64 does not decode")
    }
    _ = try pngSize(data)
    return data
}

/// Golden `resized_size` (vendored there from mlx-vlm 0.7.8
/// `get_aspect_ratio_preserving_size`): the `(height, width)` Gemma 4 resizes to.
public func resizedSize(height: Int, width: Int, patchSize: Int = 16, maxPatches: Int = 2520,
                        pooling: Int = 3) throws -> (height: Int, width: Int) {
    let targetPx = Double(maxPatches * patchSize * patchSize)
    let factor = (targetPx / Double(height * width)).squareRoot()
    let sideMult = pooling * patchSize
    var targetHeight = Int((factor * Double(height) / Double(sideMult)).rounded(.down)) * sideMult
    var targetWidth = Int((factor * Double(width) / Double(sideMult)).rounded(.down)) * sideMult
    if targetHeight == 0, targetWidth == 0 { throw ClassifierError(.prompt, "image resizes to 0 x 0") }
    let maxSideLength = (maxPatches / (pooling * pooling)) * sideMult
    if targetHeight == 0 {
        targetHeight = sideMult
        targetWidth = min(Int((Double(width) / Double(height)).rounded(.down)) * sideMult, maxSideLength)
    } else if targetWidth == 0 {
        targetWidth = sideMult
        targetHeight = min(Int((Double(height) / Double(width)).rounded(.down)) * sideMult, maxSideLength)
    }
    return (targetHeight, targetWidth)
}

/// Golden `soft_token_count`.
public func softTokenCount(height: Int, width: Int, patchSize: Int = 16, maxSoftTokens: Int = 280,
                           pooling: Int = 3) throws -> Int {
    let (h, w) = try resizedSize(height: height, width: width, patchSize: patchSize,
                                 maxPatches: maxSoftTokens * pooling * pooling, pooling: pooling)
    return (h / patchSize) * (w / patchSize) / (pooling * pooling)
}

/// Golden `ImageTokens`: a model's image placeholder convention.
public struct ImageTokens: Sendable, Equatable {
    public var boi: String
    public var image: String
    public var eoi: String
    public var imageTokenId: Int
    public var patchSize: Int = 16
    public var poolingKernelSize: Int = 3
    public var maxSoftTokens: Int = SoftTokenBudget.defaultBudget

    public init(boi: String, image: String, eoi: String, imageTokenId: Int, patchSize: Int = 16,
                poolingKernelSize: Int = 3, maxSoftTokens: Int = SoftTokenBudget.defaultBudget) {
        self.boi = boi
        self.image = image
        self.eoi = eoi
        self.imageTokenId = imageTokenId
        self.patchSize = patchSize
        self.poolingKernelSize = poolingKernelSize
        self.maxSoftTokens = maxSoftTokens
    }

    public func softTokens(_ png: Data) throws -> Int {
        let (width, height) = try pngSize(png)
        return try softTokenCount(height: height, width: width, patchSize: patchSize,
                                  maxSoftTokens: maxSoftTokens, pooling: poolingKernelSize)
    }

    /// The text the template's image marker expands to for this image.
    public func placeholder(_ png: Data) throws -> String {
        boi + String(repeating: image, count: try softTokens(png)) + eoi
    }

    /// Golden `image_tokens_from_config`: from a snapshot's `config.json`,
    /// `tokenizer_config.json` and `processor_config.json` (JSON only, no
    /// weights). Nil when the checkpoint has no vision tower.
    public static func fromConfig(modelDirectory d: URL) throws -> ImageTokens? {
        func json(_ name: String) throws -> JSONValue { try JSONValue.parse(Data(contentsOf: d.appendingPathComponent(name))) }
        func int(_ v: JSONValue?) -> Int? {
            switch v {
            case .int(let i): i
            case .double(let x): Int(x)
            default: nil
            }
        }
        func str(_ v: JSONValue?) -> String? { if case .string(let s) = v { s } else { nil } }
        let config = try json("config.json")
        guard let vision = config["vision_config"], vision != .null, vision.objectValue?.isEmpty == false,
              let imageTokenId = int(config["image_token_id"])
        else { return nil }
        let tok = try json("tokenizer_config.json")
        let procURL = d.appendingPathComponent("processor_config.json")
        let proc: JSONValue = FileManager.default.fileExists(atPath: procURL.path)
            ? (try json("processor_config.json")["image_processor"] ?? .object(OrderedObject())) : .object(OrderedObject())
        guard let boi = str(tok["boi_token"]), let image = str(tok["image_token"]), let eoi = str(tok["eoi_token"]) else {
            throw ClassifierError(.prompt, "tokenizer_config.json lacks boi_token, image_token or eoi_token")
        }
        return ImageTokens(
            boi: boi, image: image, eoi: eoi, imageTokenId: imageTokenId,
            patchSize: int(proc["patch_size"]) ?? int(vision["patch_size"]) ?? 16,
            poolingKernelSize: int(proc["pooling_kernel_size"]) ?? int(vision["pooling_kernel_size"]) ?? 3,
            maxSoftTokens: int(proc["max_soft_tokens"]) ?? int(config["vision_soft_tokens_per_image"]) ?? 280)
    }
}

/// Golden `ImageSlot`: one image's run of soft tokens in the prompt. The PNG bytes
/// ride along in `data` (not part of equality, as golden's `compare=False`).
public struct ImageSlot: Sendable, Equatable {
    public var start: Int
    public var length: Int
    /// sha256 of the PNG bytes, lowercase hex.
    public var sha256: String
    /// The budget the run was compiled at; nil means the runner's own.
    public var maxSoftTokens: Int?
    public var data: Data

    public init(start: Int, length: Int, png: Data, maxSoftTokens: Int? = nil) {
        self.start = start
        self.length = length
        sha256 = sha256Hex(png)
        self.maxSoftTokens = maxSoftTokens
        data = png
    }

    public static func == (a: ImageSlot, b: ImageSlot) -> Bool {
        a.start == b.start && a.length == b.length && a.sha256 == b.sha256 && a.maxSoftTokens == b.maxSoftTokens
    }

    public var end: Int { start + length }

    /// The fixture/wire record (golden `describe`): `max_soft_tokens` only when not 280.
    public var described: GoldenFixture.Image {
        GoldenFixture.Image(start: start, length: length, sha256: sha256,
                            maxSoftTokens: maxSoftTokens.flatMap { $0 == SoftTokenBudget.defaultBudget ? nil : $0 })
    }
}

func sha256Hex(_ data: Data) -> String {
    SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
}

/// Golden `image_key_id`: the negative stand-in id for an image's soft tokens in a
/// delta-cache key; with a budget it also depends on it (the same PNG at 140 and 280
/// is different soft tokens and must not share a cached prefix).
public func imageKeyId(sha256: String, maxSoftTokens: Int?) -> Int {
    let digest = maxSoftTokens.map { sha256Hex(Data("\(sha256):\($0)".utf8)) } ?? sha256
    return -1 - Int(digest.prefix(15), radix: 16)!
}

/// Golden `keyed_tokens`: `tokens` with each image run's ids replaced by `imageKeyId`.
public func keyedTokens(_ tokens: [Int], _ images: [ImageSlot]) throws -> [Int] {
    var key = tokens
    for s in images {
        guard s.end <= key.count else {
            throw RunnerError.invalidArgument("image run [\(s.start), \(s.end)) beyond \(key.count) tokens")
        }
        let id = imageKeyId(sha256: s.sha256, maxSoftTokens: s.maxSoftTokens)
        for i in s.start ..< s.end { key[i] = id }
    }
    return key
}

/// Golden `find_image_runs`: `(start, length)` of each maximal run of `imageTokenId`.
public func findImageRuns(_ tokens: [Int], imageTokenId: Int) -> [(start: Int, length: Int)] {
    var runs: [(start: Int, length: Int)] = []
    var start: Int?
    for (i, t) in tokens.enumerated() {
        if t == imageTokenId {
            if start == nil { start = i }
        } else if let s = start {
            runs.append((s, i - s))
            start = nil
        }
    }
    if let s = start { runs.append((s, tokens.count - s)) }
    return runs
}
