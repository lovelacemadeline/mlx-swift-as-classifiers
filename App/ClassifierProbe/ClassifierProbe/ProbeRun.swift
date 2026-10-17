// Story 6.1: load Gemma 4 E2B from the app's Documents directory and run the
// vendored golden fixtures through the unchanged `runCase` / `compareCase`, logging
// memory at each stage. Every line goes to os.Logger (subsystem
// com.maddie.classifierprobe, streamed by scripts/deploy.sh) and to
// Documents/probe-log.txt, which `scripts/deploy.sh --pull` copies back.
//
// Modes (launch arguments, `scripts/deploy.sh --run [--text-only]`):
//   default        text + vision tower, all 13 fixtures (8 text, 5 image)
//   --text-only    vision tower not built (its weights dropped like audio's), 8 text fixtures

import Foundation
import MLX
import MLXClassifiers
import os

let probeLog = Logger(subsystem: "com.maddie.classifierprobe", category: "probe")

/// The model directory inside the app container: Documents/<this name>/ (config.json, model.safetensors, ...).
let modelDirName = "gemma-4-E2B-it-qat-4bit"

struct FixtureLine: Identifiable, Sendable {
    let id: String
    let kind: String
    let ok: Bool
    let maxDp: Double?
    let ms: Double
    let answer: String
    let failures: [String]
}

@MainActor
final class ProbeState: ObservableObject {
    @Published var stage = "starting"
    @Published var memory: [String] = []
    @Published var fixtures: [FixtureLine] = []
    @Published var notes: [String] = []
}

final class ProbeLogFile: @unchecked Sendable {
    private let handle: FileHandle?
    init(url: URL) {
        FileManager.default.createFile(atPath: url.path, contents: nil)
        handle = try? FileHandle(forWritingTo: url)
    }
    func write(_ line: String) {
        handle?.write((line + "\n").data(using: .utf8)!)
        try? handle?.synchronize()
    }
}

struct ProbeRun: Sendable {
    let state: ProbeState
    let documents: URL
    let file: ProbeLogFile

    init(state: ProbeState) {
        self.state = state
        documents = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
        file = ProbeLogFile(url: documents.appendingPathComponent("probe-log.txt"))
    }

    func log(_ s: String) {
        probeLog.notice("CLASSIFIER_PROBE \(s, privacy: .public)")
        file.write("\(Date().timeIntervalSince1970) \(s)")
    }

    func mem(_ stage: String) -> MemorySample {
        let m = MemorySample.now()
        log("MEM \(stage) \(m)")
        Task { @MainActor in state.memory.append("\(stage): \(m)") }
        return m
    }

    func setStage(_ s: String) {
        log("STAGE \(s)")
        Task { @MainActor in state.stage = s }
    }

    func note(_ s: String) {
        log(s)
        Task { @MainActor in state.notes.append(s) }
    }

    func run(arguments: [String]) async {
        let textOnly = arguments.contains("--text-only")
        let build = Bundle.main.infoDictionary?["CFBundleVersion"] as? String ?? "?"
        log("LAUNCHED build=\(build) mode=\(textOnly ? "text-only" : "full") "
            + "physical=\(gib(ProcessInfo.processInfo.physicalMemory))")
        _ = mem("launch")
        let dir = documents.appendingPathComponent(modelDirName)
        guard FileManager.default.fileExists(atPath: dir.appendingPathComponent("config.json").path) else {
            note("MODEL_MISSING \(dir.path) — run scripts/deploy.sh --push-model")
            setStage("model missing")
            return
        }
        guard let fixturesDir = Bundle.main.url(forResource: "fixtures", withExtension: nil) else {
            note("FIXTURES_MISSING in the app bundle")
            setStage("fixtures missing")
            return
        }

        do {
            let all = try Fixtures.load(directory: fixturesDir)
            let text = all.filter { !$0.request.hasImages }
            let images = textOnly ? [] : all.filter { $0.request.hasImages }
            let sha = all[0].model.tokenizerSha256

            setStage(textOnly ? "loading E2B (text only)" : "loading E2B (text + vision)")
            _ = mem("before-load")
            let t0 = Date()
            let runner = try await MLXRunner.load(directory: dir, expectedTokenizerSha256: sha,
                                                  includeVision: !textOnly)
            runner.promptCache = nil
            let loadS = Date().timeIntervalSince(t0)
            log(String(format: "LOAD %.2f s vision=%@", loadS, runner.vision == nil ? "no" : "yes"))
            _ = mem("after-load")

            for fx in text + images {
                setStage("fixture \(fx.id)")
                let t = Date()
                let got = try runCase(fx, runner: runner)
                let ms = Date().timeIntervalSince(t) * 1e3
                let rep = compareCase(expected: fx.expected, result: got, caseId: fx.id)
                let kind = fx.request.hasImages ? "image/\(fx.readout.rawValue)" : "text/\(fx.readout.rawValue)"
                let line = FixtureLine(id: fx.id, kind: kind, ok: rep.ok, maxDp: rep.maxProbDistance, ms: ms,
                                       answer: got.answer ?? got.error?.type ?? "?", failures: rep.failures)
                log("FIXTURE \(fx.id) kind=\(kind) \(rep.ok ? "PASS" : "FAIL") "
                    + "max|dp|=\(rep.maxProbDistance.map { String(format: "%.3g", $0) } ?? "n/a") "
                    + String(format: "ms=%.1f", ms) + " answer=\(line.answer)"
                    + (rep.ok ? "" : " failures=\(rep.failures.joined(separator: "; "))"))
                _ = mem("after-\(fx.id)")
                Task { @MainActor in state.fixtures.append(line) }
            }
            // Second pass, warm: the first pass includes kernel JIT and first-touch costs.
            for fx in text + images {
                let t = Date()
                _ = try runCase(fx, runner: runner)
                log("WARM \(fx.id) " + String(format: "ms=%.1f", Date().timeIntervalSince(t) * 1e3))
            }
            _ = mem("end")
            setStage("done")
            log("DONE")
        } catch {
            note("ERROR \(error)")
            setStage("error")
            _ = mem("error")
        }
    }
}
