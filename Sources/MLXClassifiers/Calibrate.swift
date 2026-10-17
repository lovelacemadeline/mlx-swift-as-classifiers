// Masked softmax and single-scalar temperature calibration, ported verbatim from
// golden `calibrate.py` (pure arithmetic on a handful of floats; deterministic).
// `math.fsum` is stood in for by a Neumaier-compensated sum.

import Foundation

/// The logits of `tokenIds` from a vocab row (the option mask).
public func gather(_ row: LogitRow, _ tokenIds: [Int]) -> [Double] {
    tokenIds.map { row[$0] }
}

/// Numerically stable softmax of `logits / temperature` (golden `softmax`).
public func softmax(_ logits: [Double], temperature: Double = 1.0) -> [Double] {
    precondition(temperature > 0, "temperature must be positive")
    let scaled = logits.map { $0 / temperature }
    let m = scaled.max()!
    let exps = scaled.map { Foundation.exp($0 - m) }
    let total = exactSum(exps)
    return exps.map { $0 / total }
}

/// Index of the first maximum (Python `max(range(n), key=...)`).
public func argmax(_ xs: [Double]) -> Int {
    var best = 0
    for i in xs.indices where xs[i] > xs[best] { best = i }
    return best
}

/// Neumaier-compensated sum, standing in for Python's `math.fsum` on a handful of terms.
func exactSum(_ xs: [Double]) -> Double {
    var sum = 0.0, c = 0.0
    for x in xs {
        let t = sum + x
        c += abs(sum) >= abs(x) ? (sum - t) + x : (x - t) + sum
        sum = t
    }
    return sum + c
}

/// `log softmax(row)[tokenId]` over the full vocab (sequence mode; golden `token_logprob`).
public func tokenLogprob(_ row: LogitRow, _ tokenId: Int) -> Double {
    row[tokenId] - row.logsumexp()
}

func logSoftmaxAt(_ logits: [Double], _ index: Int, _ temperature: Double) -> Double {
    let scaled = logits.map { $0 / temperature }
    let m = scaled.max()!
    return scaled[index] - m - Foundation.log(exactSum(scaled.map { Foundation.exp($0 - m) }))
}

public enum CalibrationError: Error, Equatable {
    case invalidArgument(String)
}

/// Mean negative log-likelihood of `labels` under `softmax(rows / temperature)` (golden `nll`).
public func nll(_ logitRows: [[Double]], _ labels: [Int], temperature: Double = 1.0) throws -> Double {
    guard logitRows.count == labels.count, !labels.isEmpty else {
        throw CalibrationError.invalidArgument("need one label per logit row, and at least one row")
    }
    return exactSum(zip(logitRows, labels).map { -logSoftmaxAt($0, $1, temperature) }) / Double(labels.count)
}

/// Top-label ECE with `nBins` equal-width confidence bins (golden `expected_calibration_error`):
/// the sample-weighted mean of |accuracy - mean confidence| over the bins.
public func expectedCalibrationError(_ probRows: [[Double]], _ labels: [Int], nBins: Int = 15) throws -> Double {
    guard probRows.count == labels.count, !labels.isEmpty else {
        throw CalibrationError.invalidArgument("need one label per probability row, and at least one row")
    }
    var bins = [[(Double, Bool)]](repeating: [], count: nBins)
    for (probs, y) in zip(probRows, labels) {
        let conf = probs.max()!
        let pred = probs.firstIndex(of: conf)!
        let b = min(Int(conf * Double(nBins)), nBins - 1)
        bins[b].append((conf, pred == y))
    }
    var ece = 0.0
    for members in bins where !members.isEmpty {
        let acc = Double(members.filter(\.1).count) / Double(members.count)
        let conf = exactSum(members.map(\.0)) / Double(members.count)
        ece += Double(members.count) / Double(labels.count) * abs(acc - conf)
    }
    return ece
}

/// Golden `CalibrationReport`.
public struct CalibrationReport: Sendable, Equatable {
    public let temperature: Double
    public let n: Int
    public let nllBefore: Double
    public let nllAfter: Double
    public let eceBefore: Double
    public let eceAfter: Double
}

/// Golden `TemperatureScaler`: fit one temperature T by minimising NLL on labelled
/// decisions. NLL is convex in 1/T, so a golden-section search over log T finds
/// the global optimum deterministically. T > 1 softens an over-confident model.
public struct TemperatureScaler: Sendable {
    public var temperature: Double
    public var bounds: (Double, Double)
    public private(set) var report: CalibrationReport?

    public init(temperature: Double = 1.0, bounds: (Double, Double) = (0.05, 20.0)) {
        self.temperature = temperature
        self.bounds = bounds
    }

    @discardableResult
    public mutating func fit(_ logitRows: [[Double]], _ labels: [Int], nBins: Int = 15,
                             iterations: Int = 100) throws -> CalibrationReport {
        let lo = Foundation.log(bounds.0), hi = Foundation.log(bounds.1)
        let g = (5.0.squareRoot() - 1) / 2
        func f(_ u: Double) throws -> Double { try nll(logitRows, labels, temperature: Foundation.exp(u)) }
        var a = lo, b = hi
        var c = b - g * (b - a), d = a + g * (b - a)
        var fc = try f(c), fd = try f(d)
        for _ in 0 ..< iterations {
            if fc <= fd {
                b = d; d = c; fd = fc
                c = b - g * (b - a)
                fc = try f(c)
            } else {
                a = c; c = d; fc = fd
                d = a + g * (b - a)
                fd = try f(d)
            }
        }
        temperature = Foundation.exp((a + b) / 2)
        let t = temperature
        let before = logitRows.map { softmax($0) }
        let after = logitRows.map { softmax($0, temperature: t) }
        let r = CalibrationReport(
            temperature: t, n: labels.count,
            nllBefore: try nll(logitRows, labels, temperature: 1.0),
            nllAfter: try nll(logitRows, labels, temperature: t),
            eceBefore: try expectedCalibrationError(before, labels, nBins: nBins),
            eceAfter: try expectedCalibrationError(after, labels, nBins: nBins))
        report = r
        return r
    }

    public func apply(_ logits: [Double]) -> [Double] { softmax(logits, temperature: temperature) }
}
