// A checkpoint's text front end: the swift-transformers tokenizer, its chat
// template and its image placeholder convention, loaded from a local Hugging
// Face snapshot directory (no weights, no network). `tokenize` is golden
// `MLXRunner.tokenize`: `tokenizer.encode(text, add_special_tokens=False)`.

import CryptoKit
import Foundation
import Tokenizers

public struct ModelTokenizer: @unchecked Sendable {
    public let directory: URL
    public let tokenizer: any Tokenizer
    public let chatTemplate: JinjaChatTemplate
    public let imageTokens: ImageTokens?
    /// sha256 of `tokenizer.json`, lowercase hex (golden `model.tokenizer_sha256`).
    public let tokenizerSha256: String

    public func tokenize(_ text: String) -> [Int] {
        tokenizer.encode(text: text, addSpecialTokens: false)
    }

    /// Load from a snapshot directory. With `expectedTokenizerSha256`, the
    /// `tokenizer.json` digest must match it (a fixture's `model.tokenizer_sha256`).
    public static func load(directory: URL, expectedTokenizerSha256: String? = nil) async throws -> ModelTokenizer {
        let data = try Data(contentsOf: directory.appendingPathComponent("tokenizer.json"))
        let sha = SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
        if let want = expectedTokenizerSha256, want != sha {
            throw ClassifierError(.prompt, "tokenizer.json sha256 \(sha) does not match the expected \(want)")
        }
        let tokenizer = try await AutoTokenizer.from(modelFolder: directory)
        return ModelTokenizer(directory: directory, tokenizer: tokenizer,
                              chatTemplate: try JinjaChatTemplate.fromModelDirectory(directory),
                              imageTokens: try ImageTokens.fromConfig(modelDirectory: directory),
                              tokenizerSha256: sha)
    }

    /// The local Hugging Face cache snapshot of `repo` at `revision`
    /// (`$HF_HUB_CACHE`, else `$HF_HOME/hub`, else `~/.cache/huggingface/hub`), or
    /// nil if it holds no `tokenizer.json`.
    public static func cachedSnapshot(repo: String, revision: String) -> URL? {
        let env = ProcessInfo.processInfo.environment
        let hub: URL
        if let c = env["HF_HUB_CACHE"] {
            hub = URL(fileURLWithPath: c)
        } else if let h = env["HF_HOME"] {
            hub = URL(fileURLWithPath: h).appendingPathComponent("hub")
        } else {
            hub = homeDirectory().appendingPathComponent(".cache/huggingface/hub")
        }
        let dir = hub.appendingPathComponent("models--" + repo.replacingOccurrences(of: "/", with: "--"))
            .appendingPathComponent("snapshots").appendingPathComponent(revision)
        return FileManager.default.fileExists(atPath: dir.appendingPathComponent("tokenizer.json").path) ? dir : nil
    }
}
