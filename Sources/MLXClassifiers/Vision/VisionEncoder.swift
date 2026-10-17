// The image encode path, golden `vision/tower.py::VisionTower.encode_png`: PNG bytes
// to `[N, text hidden]` soft tokens in the text embedding space (bf16), memoised by
// (PNG sha256, budget), at most `cacheSize` entries, LRU. The tower and `embed_vision`
// are the vendored `Gemma4VisionModel` / `Gemma4MultimodalEmbedder`, loaded with the
// rest of the checkpoint (`Gemma4Model.sanitize` keeps their weights).

import Foundation
import MLX

public final class VisionEncoder: @unchecked Sendable {
    let model: Gemma4Model
    public let patchSize: Int
    public let poolingKernelSize: Int
    public let cacheSize: Int
    private var order: [String] = []
    private var features: [String: MLXArray] = [:]
    private let lock = NSLock()
    /// Encodes actually run (memo misses) and their wall-clock milliseconds, for tests.
    public private(set) var encodes = 0
    public private(set) var encodeMs: [Double] = []

    init(model: Gemma4Model, cacheSize: Int = 16) {
        self.model = model
        patchSize = model.visionConfig?.patchSize ?? 16
        poolingKernelSize = model.visionConfig?.poolingKernelSize ?? 3
        self.cacheSize = cacheSize
    }

    /// Golden `VisionTower.pixel_values`.
    public func pixelValues(_ png: Data, maxSoftTokens: Int) throws -> MLXArray {
        MLXClassifiers.pixelValues(try resizedUInt8(png, patchSize: patchSize, maxSoftTokens: maxSoftTokens,
                                                    poolingKernelSize: poolingKernelSize))
    }

    /// `vision_tower(pixels)` alone, before `embed_vision` (for parity tests).
    public func tower(_ pixels: MLXArray) -> MLXArray { model.visionTower!(pixels) }

    /// Golden `encode_png`.
    public func encodePNG(_ png: Data, sha256: String? = nil, maxSoftTokens: Int) throws -> MLXArray {
        let key = "\(sha256 ?? sha256Hex(png)):\(maxSoftTokens)"
        if let hit = lock.withLock({ () -> MLXArray? in
            guard let f = features[key] else { return nil }
            order.removeAll { $0 == key }
            order.append(key)
            return f
        }) { return hit }
        let t0 = DispatchTime.now().uptimeNanoseconds
        let feats = model.encodeImage(try pixelValues(png, maxSoftTokens: maxSoftTokens))![0]
        eval(feats)
        let ms = Double(DispatchTime.now().uptimeNanoseconds - t0) / 1e6
        lock.withLock {
            encodes += 1
            encodeMs.append(ms)
            features[key] = feats
            order.append(key)
            while order.count > cacheSize { features.removeValue(forKey: order.removeFirst()) }
        }
        return feats
    }

    /// Seed the memo with given soft tokens (diagnostics: run the text side on another encoder's output).
    func inject(sha256: String, maxSoftTokens: Int, features: MLXArray) {
        lock.withLock {
            let key = "\(sha256):\(maxSoftTokens)"
            self.features[key] = features
            order.append(key)
        }
    }

    public func clear() {
        lock.withLock {
            features.removeAll()
            order.removeAll()
        }
    }
}
