// ClassifierProbe — story 6.1's on-device probe: MLXClassifiers on an iPhone 17.
// One screen mirroring the log lines; the run starts on launch.

import SwiftUI

@main
struct ClassifierProbeApp: App {
    @StateObject private var state = ProbeState()

    var body: some Scene {
        WindowGroup {
            ProbeView(state: state)
                .task {
                    let run = ProbeRun(state: state)
                    let args = ProcessInfo.processInfo.arguments
                    await Task.detached(priority: .userInitiated) { await run.run(arguments: args) }.value
                }
        }
    }
}

struct ProbeView: View {
    @ObservedObject var state: ProbeState

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 10) {
                Text("ClassifierProbe").font(.title.bold())
                Text(state.stage).font(.headline)
                ForEach(state.fixtures) { f in
                    HStack {
                        Text(f.ok ? "PASS" : "FAIL").foregroundStyle(f.ok ? .green : .red).bold()
                        VStack(alignment: .leading) {
                            Text(f.id).font(.caption.bold())
                            Text("\(f.kind)  max|dp| \(f.maxDp.map { String(format: "%.2g", $0) } ?? "n/a")  "
                                 + String(format: "%.0f ms", f.ms) + "  \(f.answer)")
                                .font(.caption2.monospaced())
                        }
                    }
                }
                ForEach(Array(state.memory.enumerated()), id: \.offset) { _, m in
                    Text(m).font(.caption2.monospaced()).foregroundStyle(.secondary)
                }
                ForEach(Array(state.notes.enumerated()), id: \.offset) { _, n in
                    Text(n).font(.caption2.monospaced())
                }
            }
            .padding()
        }
    }
}
