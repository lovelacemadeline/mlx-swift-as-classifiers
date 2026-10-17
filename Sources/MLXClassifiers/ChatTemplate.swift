// Chat templates, ported from golden `template.py`: render a (system, user) pair
// with the assistant turn opened, and give the cacheable prefix (the system
// turn) as text. Decisions always render with thinking off.
//
// `JinjaChatTemplate` is golden's `HFChatTemplate`: it renders the checkpoint's
// own Jinja template the way Hugging Face `apply_chat_template` does
// (trim_blocks and lstrip_blocks on, `messages`, `add_generation_prompt`,
// `enable_thinking = false` and the special-token strings in the context), with
// swift-jinja, the engine swift-transformers' `applyChatTemplate` uses. That
// method returns token ids only, and compilation needs the text (the prefix
// split, the image markers), so the template is driven directly.

import Foundation
import Jinja

/// `text` is the whole prompt ending in the open assistant turn; `prefixText` is
/// its cacheable head (empty when there is no system turn).
public struct RenderedChat: Sendable, Equatable {
    public let prefixText: String
    public let text: String

    public init(prefixText: String, text: String) throws {
        guard text.pyStartsWith(prefixText) else {
            throw ClassifierError(.prompt, "rendered prompt does not start with its prefix text")
        }
        self.prefixText = prefixText
        self.text = text
    }
}

public protocol ChatTemplate: Sendable {
    func render(system: String?, user: String) throws -> RenderedChat
}

public struct ChatMessage: Sendable, Equatable {
    public var role: String
    public var content: String
    public init(role: String, content: String) {
        self.role = role
        self.content = content
    }
}

/// Golden `HFChatTemplate` over swift-jinja.
public struct JinjaChatTemplate: ChatTemplate, @unchecked Sendable {
    public let source: String
    private let template: Template
    /// Special-token strings HF puts in the template context (`bos_token`, ...).
    public let specialTokens: [String: String]
    /// Extra context, always including `enable_thinking: false`.
    public let templateContext: [String: Value]

    /// The attributes `PreTrainedTokenizerBase.special_tokens_map` exposes to templates.
    public static let specialTokenAttributes = [
        "bos_token", "eos_token", "unk_token", "sep_token", "pad_token", "cls_token", "mask_token",
    ]

    public init(source: String, specialTokens: [String: String], context: [String: Value] = [:]) throws {
        self.source = source
        template = try Template(source, with: .init(lstripBlocks: true, trimBlocks: true))
        self.specialTokens = specialTokens
        templateContext = ["enable_thinking": .boolean(false)].merging(context) { _, new in new }
    }

    /// The template of a Hugging Face snapshot directory: `chat_template.jinja` if
    /// present (as transformers prefers), else `tokenizer_config.json`'s
    /// `chat_template`; special tokens from `tokenizer_config.json`.
    public static func fromModelDirectory(_ d: URL) throws -> JinjaChatTemplate {
        let config = try JSONValue.parse(Data(contentsOf: d.appendingPathComponent("tokenizer_config.json")))
        let jinjaURL = d.appendingPathComponent("chat_template.jinja")
        let source: String
        if FileManager.default.fileExists(atPath: jinjaURL.path) {
            source = try String(contentsOf: jinjaURL, encoding: .utf8)
        } else if case .string(let s) = config["chat_template"] {
            source = s
        } else {
            throw ClassifierError(.prompt, "no chat template in \(d.path)")
        }
        var special: [String: String] = [:]
        for key in specialTokenAttributes {
            switch config[key] {
            case .string(let s): special[key] = s
            case .object(let o): if case .string(let s) = o["content"] { special[key] = s }
            default: break
            }
        }
        return try JinjaChatTemplate(source: source, specialTokens: special)
    }

    public func apply(_ messages: [ChatMessage], addGenerationPrompt: Bool) throws -> String {
        var context = templateContext
        context["messages"] = .array(messages.map {
            .object(["role": .string($0.role), "content": .string($0.content)])
        })
        context["add_generation_prompt"] = .boolean(addGenerationPrompt)
        for (k, v) in specialTokens { context[k] = .string(v) }
        return try template.render(context)
    }

    /// Golden `HFChatTemplate.render`.
    public func render(system: String?, user: String) throws -> RenderedChat {
        let sys = system.map { [ChatMessage(role: "system", content: $0)] } ?? []
        let text = try apply(sys + [ChatMessage(role: "user", content: user)], addGenerationPrompt: true)
        let prefix = system != nil ? try apply(sys, addGenerationPrompt: false) : ""
        guard text.pyStartsWith(prefix) else {
            throw ClassifierError(.prompt, "chat template renders the system turn differently in context")
        }
        return try RenderedChat(prefixText: prefix, text: text)
    }

    /// Golden `HFChatTemplate.generation_prompt`: the assistant-start marker.
    public func generationPrompt() throws -> String {
        let msgs = [ChatMessage(role: "user", content: "x")]
        let closed = try apply(msgs, addGenerationPrompt: false), opened = try apply(msgs, addGenerationPrompt: true)
        guard opened.pyStartsWith(closed) else {
            throw ClassifierError(.prompt, "generation prompt is not a suffix of the closed conversation")
        }
        return String(decoding: Array(opened.utf8).dropFirst(closed.utf8.count), as: UTF8.self)
    }
}
