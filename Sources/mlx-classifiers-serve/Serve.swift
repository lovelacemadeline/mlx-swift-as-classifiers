// mlx-classifiers-serve (story 7.2): a localhost HTTP shim serving golden
// `server.py`'s `POST /v1/systemone` over one loaded `MLXRunner`.
//
// The route only (no chat completions). Request and response bodies are the
// wire schema (`SystemOneRequest` / `SystemOneResponse` through `SystemOneJSON`,
// key order kept). Golden's statuses: 200 with the response; 422
// `{"detail": "request body is not valid JSON"}`; 422 `{"detail": "request does
// not match the SystemOneRequest schema", "errors": [{loc, msg, type}]}` (the
// field paths are best effort, Swift's DecodingError is not pydantic); 422
// `{"detail": <message>}` on a ClassifierError; anything else 404. Headers:
// `x-prefix-cache-tokens` (the first prefill's cached tokens, the delta store's
// hit) and `x-prefix-cache: none` (no declared-prefix LRU here: the runner's
// delta store serves repeated prefixes). Decisions run one at a time on one
// serial queue: one model, one GPU. HTTP/1.1 with keep-alive and Content-Length
// bodies only (what httpx sends); Network.framework, no package dependency.
//
// Needs the MLX metallib beside the binary: build with xcodebuild
// (docs/runbooks/build-and-test.md), e.g.
//   xcodebuild build -scheme mlx-classifiers-serve -configuration Release \
//     -destination 'platform=macOS' -derivedDataPath .build/xcode \
//     -skipMacroValidation -skipPackagePluginValidation
//   .build/xcode/Build/Products/Release/mlx-classifiers-serve \
//     --model mlx-community/gemma-4-26B-A4B-it-qat-4bit --port 8081 --vision-tokens 140

import Foundation
import MLXClassifiers
import Network

struct Options {
    var model = MLXRunner.defaultModel
    var modelDir: String?
    var port: UInt16 = 8081
    var visionTokens: Int?

    static func parse(_ args: [String]) -> Options {
        var o = Options()
        var i = 1
        func value() -> String {
            i += 1
            guard i < args.count else { fail("missing value for \(args[i - 1])") }
            return args[i]
        }
        while i < args.count {
            switch args[i] {
            case "--model": o.model = value()
            case "--model-dir": o.modelDir = value()
            case "--port": o.port = UInt16(value()) ?? { fail("bad --port") }()
            case "--vision-tokens":
                let v = Int(value())
                guard let v, [70, 140, 280].contains(v) else { fail("--vision-tokens must be 70, 140 or 280") }
                o.visionTokens = v
            case "-h", "--help":
                print("usage: mlx-classifiers-serve [--model REPO] [--model-dir DIR] [--port PORT] [--vision-tokens {70,140,280}]")
                exit(0)
            default: fail("unknown argument \(args[i])")
            }
            i += 1
        }
        return o
    }
}

func fail(_ msg: String) -> Never {
    FileHandle.standardError.write(Data("mlx-classifiers-serve: \(msg)\n".utf8))
    exit(2)
}

struct HTTPReply {
    var status: Int
    var body: Data
    var headers: [(String, String)] = []

    static func json(_ status: Int, _ obj: [String: Any]) -> HTTPReply {
        HTTPReply(status: status, body: (try? JSONSerialization.data(withJSONObject: obj, options: [.sortedKeys])) ?? Data())
    }

    func bytes(keepAlive: Bool) -> Data {
        let reason = [200: "OK", 404: "Not Found", 405: "Method Not Allowed", 422: "Unprocessable Entity",
                      500: "Internal Server Error"][status] ?? "Status"
        var head = "HTTP/1.1 \(status) \(reason)\r\ncontent-type: application/json\r\ncontent-length: \(body.count)\r\n"
        for (k, v) in headers { head += "\(k): \(v)\r\n" }
        head += keepAlive ? "connection: keep-alive\r\n\r\n" : "connection: close\r\n\r\n"
        return Data(head.utf8) + body
    }
}

/// The one route over one runner; `handle` is only ever called on `queue`.
final class SystemOneService: @unchecked Sendable {
    let runner: MLXRunner
    let queue = DispatchQueue(label: "mlx-classifiers-model")

    init(runner: MLXRunner) { self.runner = runner }

    static func errorPaths(_ error: Error) -> [[String: Any]] {
        func loc(_ path: [CodingKey], _ key: CodingKey? = nil) -> String {
            (path + (key.map { [$0] } ?? [])).map { $0.intValue.map(String.init) ?? $0.stringValue }.joined(separator: ".")
        }
        switch error as? DecodingError {
        case .keyNotFound(let key, let ctx)?:
            return [["loc": loc(ctx.codingPath, key), "msg": "Field required", "type": "missing"]]
        case .typeMismatch(_, let ctx)?, .valueNotFound(_, let ctx)?, .dataCorrupted(let ctx)?:
            return [["loc": loc(ctx.codingPath), "msg": ctx.debugDescription, "type": "value_error"]]
        default:
            return [["loc": "", "msg": "\(error)", "type": "value_error"]]
        }
    }

    func handle(method: String, path: String, body: Data) -> HTTPReply {
        guard path == "/v1/systemone" else { return .json(404, ["status_code": 404, "detail": "Not Found"]) }
        guard method == "POST" else { return .json(405, ["status_code": 405, "detail": "Method Not Allowed"]) }
        let raw: JSONValue
        do { raw = try JSONValue.parse(body) } catch {
            return .json(422, ["detail": "request body is not valid JSON"])
        }
        let req: SystemOneRequest
        do { req = try SystemOneJSON.decode(SystemOneRequest.self, from: raw) } catch {
            return .json(422, ["detail": "request does not match the SystemOneRequest schema",
                               "errors": Self.errorPaths(error)])
        }
        let before = runner.prefillCalls.count
        do {
            let resp = try decide(req, runner: runner)
            let hit = runner.prefillCalls.count > before ? runner.prefillCalls[before].cached : 0
            runner.clearCallRecords()  // a server never reads older calls (golden `del calls[:]`)
            return HTTPReply(status: 200, body: try SystemOneJSON.encode(resp),
                             headers: [("x-prefix-cache", "none"), ("x-prefix-cache-tokens", "\(hit)")])
        } catch let e as ClassifierError {
            runner.clearCallRecords()
            return .json(422, ["detail": e.message])
        } catch {
            runner.clearCallRecords()
            return .json(500, ["detail": "\(error)"])
        }
    }
}

/// One HTTP/1.1 connection: requests parsed off the byte stream in order, replies written in order.
final class Connection: @unchecked Sendable {
    let conn: NWConnection
    let service: SystemOneService
    var buffer = Data()

    init(_ conn: NWConnection, service: SystemOneService) {
        self.conn = conn
        self.service = service
    }

    func start() {
        conn.start(queue: service.queue)
        receive()
    }

    func receive() {
        conn.receive(minimumIncompleteLength: 1, maximumLength: 1 << 20) { [self] data, _, done, error in
            if let data { buffer.append(data) }
            if error != nil { conn.cancel(); return }
            if !drain() { return }
            if done { conn.cancel(); return }
            receive()
        }
    }

    /// Serve every complete request in `buffer`; false when the connection was closed.
    func drain() -> Bool {
        let sep = Data("\r\n\r\n".utf8)
        while let end = buffer.range(of: sep) {
            let head = String(decoding: buffer[buffer.startIndex ..< end.lowerBound], as: UTF8.self)
            let lines = head.components(separatedBy: "\r\n")
            let parts = lines[0].split(separator: " ")
            guard parts.count >= 2 else { conn.cancel(); return false }
            var length = 0, keepAlive = !lines[0].hasSuffix("HTTP/1.0")
            for line in lines.dropFirst() {
                guard let colon = line.firstIndex(of: ":") else { continue }
                let name = line[..<colon].lowercased()
                let value = line[line.index(after: colon)...].trimmingCharacters(in: .whitespaces)
                if name == "content-length" { length = Int(value) ?? 0 }
                if name == "connection" { keepAlive = value.lowercased() != "close" }
            }
            let bodyStart = end.upperBound
            guard buffer.count - (bodyStart - buffer.startIndex) >= length else { return true }  // wait for the body
            let body = buffer[bodyStart ..< bodyStart + length]
            buffer = Data(buffer[(bodyStart + length)...])
            let target = String(parts[1]).split(separator: "?").first.map(String.init) ?? "/"
            let reply = service.handle(method: String(parts[0]), path: target, body: Data(body))
            conn.send(content: reply.bytes(keepAlive: keepAlive), completion: .contentProcessed { _ in })
            if !keepAlive {
                conn.send(content: nil, isComplete: true, completion: .contentProcessed { [conn] _ in conn.cancel() })
                return false
            }
        }
        return true
    }
}

@main
struct Serve {
    static func main() async throws {
        setvbuf(stdout, nil, _IOLBF, 0)  // line-buffered even when redirected to a log
        let opts = Options.parse(CommandLine.arguments)
        let dir = opts.modelDir.map { URL(fileURLWithPath: $0) } ?? MLXRunner.cachedSnapshot(repo: opts.model)
        guard let dir else { fail("\(opts.model) is not in the local HF cache (pass --model-dir)") }
        let t0 = Date()
        let runner = try await MLXRunner.load(directory: dir, modelId: opts.model)
        if let vt = opts.visionTokens { runner.imageTokens?.maxSoftTokens = vt }
        print(String(format: "loaded %@ in %.2f s from %@%@", opts.model, Date().timeIntervalSince(t0), dir.path,
                     opts.visionTokens.map { " (vision tokens \($0))" } ?? ""))
        let service = SystemOneService(runner: runner)
        let params = NWParameters.tcp
        params.requiredLocalEndpoint = NWEndpoint.hostPort(host: "127.0.0.1", port: NWEndpoint.Port(rawValue: opts.port)!)
        params.allowLocalEndpointReuse = true
        let listener = try NWListener(using: params)
        listener.newConnectionHandler = { Connection($0, service: service).start() }
        listener.stateUpdateHandler = { state in
            switch state {
            case .ready: print("serving POST /v1/systemone on http://127.0.0.1:\(opts.port)")
            case .failed(let e): fail("listener failed: \(e)")
            default: break
            }
        }
        listener.start(queue: service.queue)
        while true { try await Task.sleep(nanoseconds: 3_600_000_000_000) }
    }
}
