// Story 5.1 on-device probe: do the golden INT8 / MoE Metal kernels compile and run
// on the A19? `int8-kernels.json` is written by scripts/capture_int8_kernels.py from
// the golden repo on an M5: each kernel's exact source and header (int8/wiring.py,
// int8/experts.py, moe.py), one small call's launch geometry and input bytes, and
// the M5 output bytes. Each kernel is rebuilt with `MLXFast.metalKernel`, called with
// the same inputs, and compared with the M5 output (bit-identical, else max |diff|).
// Compile errors are caught via `withError` and logged verbatim.

import Foundation
import MLX

enum KernelProbe {
    static func dtype(_ s: String) -> DType? { DType.allCases.first { "\($0)" == s } }

    static func array(_ spec: [String: Any]) throws -> MLXArray {
        guard let shape = spec["shape"] as? [Int], let ds = spec["dtype"] as? String, let dt = dtype(ds),
              let b64 = spec["b64"] as? String, let data = Data(base64Encoded: b64)
        else { throw NSError(domain: "KernelProbe", code: 1, userInfo: [NSLocalizedDescriptionKey: "bad array spec"]) }
        return MLXArray(data, [data.count], dtype: .uint8).view(dtype: dt).reshaped(shape)
    }

    static func run(log: (String) -> Void) {
        guard let url = Bundle.main.url(forResource: "int8-kernels", withExtension: "json"),
              let data = try? Data(contentsOf: url),
              let root = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
              let kernels = root["kernels"] as? [[String: Any]]
        else { log("KERNEL ERROR int8-kernels.json missing or unreadable"); return }
        log("KERNEL captured on mlx \(root["mlx"] ?? "?") \(root["device"] ?? "?"); \(kernels.count) kernels")
        var ok = 0
        for k in kernels {
            let name = k["name"] as? String ?? "?"
            log("KERNEL \(name) start")
            do {
                let inputs = try (k["inputs"] as? [[String: Any]] ?? []).map(array)
                let want = try (k["outputs"] as? [[String: Any]] ?? []).map(array)
                let g = k["grid"] as? [Int] ?? [1, 1, 1], t = k["threadgroup"] as? [Int] ?? [1, 1, 1]
                let template: [(String, any KernelTemplateArg)] = (k["template"] as? [[Any]] ?? []).compactMap { p in
                    guard let key = p.first as? String, p.count == 2 else { return nil }
                    if let s = p[1] as? String, let d = dtype(s) { return (key, d) }
                    if let n = p[1] as? Int { return (key, n) }
                    if let b = p[1] as? Bool { return (key, b) }
                    return nil
                }
                let kernel = MLXFast.metalKernel(
                    name: name, inputNames: k["input_names"] as? [String] ?? [],
                    outputNames: k["output_names"] as? [String] ?? [],
                    source: k["source"] as? String ?? "", header: k["header"] as? String ?? "",
                    ensureRowContiguous: k["ensure_row_contiguous"] as? Bool ?? true)
                let t0 = Date()
                let got = try withError {
                    let outs = kernel(inputs, template: template.isEmpty ? nil : template,
                                      grid: (g[0], g[1], g[2]), threadGroup: (t[0], t[1], t[2]),
                                      outputShapes: want.map(\.shape), outputDTypes: want.map(\.dtype))
                    eval(outs)
                    return outs
                }
                let ms = Date().timeIntervalSince(t0) * 1e3
                var parts: [String] = []
                for (i, (a, b)) in zip(got, want).enumerated() {
                    let same = a.view(dtype: .uint8).asData() == b.view(dtype: .uint8).asData()
                    let diff = (a.asType(.float32) - b.asType(.float32)).abs().max().item(Float.self)
                    parts.append("out\(i) \(same ? "bit-identical" : "max|diff|=\(diff)")")
                }
                log("KERNEL \(name) OK compile+run " + String(format: "%.1f ms (first call, incl. JIT)", ms)
                    + " vs M5: " + parts.joined(separator: ", "))
                ok += 1
            } catch {
                log("KERNEL \(name) FAILED: \(error)")
            }
        }
        log("KERNEL summary \(ok)/\(kernels.count) compiled and ran")
    }
}
