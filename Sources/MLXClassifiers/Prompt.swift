// Compile a request into one token sequence per question, with a known readout
// position. A faithful port of golden `prompt.py` (separate mode):
//
// - system turn = `request.prefix`, the cacheable head;
// - user turn = the rendered `state` plus one question with its labelled options
//   (letters `A.`, `B.`, ... by default; with `label_style = "option-text"` or in
//   sequence mode, each option by its own name);
// - the assistant turn is opened, so the next token is the answer label; the
//   readout position is the last token of the sequence.
//
// Image blocks render as `[i] ` plus the placeholder run `boi + image * N + eoi`
// (`ImageTokens`), substituted for a private-use sentinel after the chat
// template renders; each run is recorded as an `ImageSlot`.

import Foundation

public typealias Tokenize = (String) -> [Int]

public enum PromptLabels {
    public static let letters: [String] = (UInt8(ascii: "A")...UInt8(ascii: "Z")).map { String(UnicodeScalar($0)) }
    /// Internal option names of a noul question; the answer reports P(yes).
    public static let noulOptions = ["yes", "no"]
    /// The fixed label pair the model is asked to emit.
    public static let noulLabels = ["Yes", "No"]
}

/// Stand-in for image `k` while the chat template renders (private-use code points).
func imageSentinel(_ k: Int) -> String { "\u{E000}\(k)\u{E001}" }

/// Golden `question_options`: criteria keys for choice, level indices for score,
/// `["yes", "no"]` for noul.
public func questionOptions(_ question: Question) -> [String] {
    switch question {
    case .noul: PromptLabels.noulOptions
    case .choice(let q): q.criteria.keys
    case .score(let q): q.criteria.indices.map(String.init)
    }
}

/// Golden `names_options`: options shown and answered by their own text.
public func namesOptions(readout: Readout, labelStyle: LabelStyle) -> Bool {
    readout == .sequence || labelStyle == .optionText
}

/// Golden `option_labels`.
public func optionLabels(_ question: Question, readout: Readout = .firstToken,
                         labelStyle: LabelStyle = .letters) throws -> [String] {
    switch question {
    case .noul: return PromptLabels.noulLabels
    case .choice(let q) where namesOptions(readout: readout, labelStyle: labelStyle): return q.criteria.keys
    case .score(let q) where namesOptions(readout: readout, labelStyle: labelStyle): return q.criteria
    default:
        let n = questionOptions(question).count
        guard n <= PromptLabels.letters.count else {
            throw ClassifierError(.tooManyOptions,
                "\(n) options exceeds the \(PromptLabels.letters.count) single-letter labels; "
                + "split the question or use independent per-option scoring (a future story)")
        }
        return Array(PromptLabels.letters.prefix(n))
    }
}

/// Golden `render_state`: a string as-is; an object as sorted-key pretty JSON; a
/// list as numbered blocks. An image block renders as `image(k)`, by default
/// `<image k>`.
public func renderState(_ state: SystemOneState, image: ((Int) -> String)? = nil) -> String {
    switch state {
    case .text(let s): return s
    case .object(let o): return PythonJSON.dumps(.object(o))
    case .blocks(let blocks):
        let image = image ?? { "<image \($0)>" }
        var parts: [String] = [], k = 0
        for (i, block) in blocks.enumerated() {
            switch block {
            case .image:
                parts.append("[\(i + 1)] \(image(k))")
                k += 1
            case .text(let s):
                parts.append("[\(i + 1)] \(s)")
            }
        }
        return parts.joined(separator: "\n\n")
    }
}

/// Golden `render_user_turn`.
public func renderUserTurn(_ state: SystemOneState, _ question: Question, labels: [String]? = nil,
                           order: [Int]? = nil, readout: Readout = .firstToken, labelStyle: LabelStyle = .letters,
                           image: ((Int) -> String)? = nil) throws -> String {
    let labels = try labels ?? optionLabels(question, readout: readout, labelStyle: labelStyle)
    var lines = ["State:", renderState(state, image: image), "", "Question: \(question.instructions)", ""]
    let entries: [String], noun: String
    switch question {
    case .noul:
        lines.append("Answer \(labels[0]) or \(labels[1]) only.")
        return lines.joined(separator: "\n")
    case .choice(let q):
        entries = q.criteria.entries.map { e in
            if let desc = e.value, !desc.isEmpty { "\(e.key): \(desc)" } else { e.key }
        }
        lines.append("Options:")
        noun = "option"
    case .score(let q):
        entries = q.criteria
        lines.append("Scale (lowest to highest):")
        noun = "level"
    }
    let order = order ?? Array(entries.indices)
    guard order.sorted() == Array(entries.indices) else {
        throw ClassifierError(.prompt, "order \(order) is not a permutation of \(entries.count) options")
    }
    if namesOptions(readout: readout, labelStyle: labelStyle) {
        lines += order.map { "- \(entries[$0])" }
        lines += ["", "Answer with one \(noun) exactly as written."]
        return lines.joined(separator: "\n")
    }
    lines += order.enumerated().map { k, i in "\(labels[k]). \(entries[i])" }
    lines += ["", "Answer with the letter of one \(noun) only."]
    return lines.joined(separator: "\n")
}

/// Golden `CompiledPrompt`: one question's prompt. `tokens[..<prefixLen]` is the
/// cacheable system turn; `readoutPosition` indexes the token whose logits pick
/// the label. `labels[k]` is the label shown for `options[order[k]]`.
public struct CompiledPrompt: Sendable, Equatable {
    public var name: String
    public var question: Question
    public var text: String
    public var tokens: [Int]
    public var prefixLen: Int
    public var readoutPosition: Int
    public var options: [String]
    public var labels: [String]
    public var order: [Int]
    public var readout: Readout
    public var labelStyle: LabelStyle
    public var images: [ImageSlot]

    public var prefixTokens: [Int] { Array(tokens.prefix(prefixLen)) }

    public func label(for optionIndex: Int) -> String { labels[order.firstIndex(of: optionIndex)!] }

    /// Labels in option order (the fixture's `expected.labels`).
    public var optionOrderLabels: [String] { options.indices.map(label(for:)) }
}

extension SystemOneRequest {
    /// True when the state list holds at least one image block.
    public var hasImages: Bool {
        if case .blocks(let b) = state {
            return b.contains { if case .image = $0 { true } else { false } }
        }
        return false
    }
}

/// Golden `compile_question`: render and tokenize one question of `request`.
/// `readout` and `labelStyle` default to `request.options`. A state with image
/// blocks needs `imageTokens`.
public func compileQuestion(_ request: SystemOneRequest, name: String, template: ChatTemplate,
                            tokenize: Tokenize, order: [Int]? = nil, readout: Readout? = nil,
                            labelStyle: LabelStyle? = nil, imageTokens: ImageTokens? = nil) throws -> CompiledPrompt {
    guard let question = request.questions[name] else {
        throw ClassifierError(.prompt, "no question named \(name)")
    }
    let readout = readout ?? request.options?.resolvedReadout ?? .firstToken
    let labelStyle = labelStyle ?? request.options?.resolvedLabelStyle ?? .letters
    let options = questionOptions(question)
    var labels = try optionLabels(question, readout: readout, labelStyle: labelStyle)
    var order = order
    if case .noul = question { order = [0, 1] }  // the Yes/No pair is fixed, never permuted
    let ord = order ?? Array(options.indices)
    if case .noul = question {} else if namesOptions(readout: readout, labelStyle: labelStyle) {
        labels = ord.map { labels[$0] }  // labels are the texts: they follow the order
    }
    var pngs: [Data] = []
    var image: ((Int) -> String)?
    var spec = imageTokens
    if request.hasImages {
        guard var s = spec else {
            throw ClassifierError(.prompt, "the state holds image blocks but this runner has no vision path")
        }
        if let budget = request.options?.visionSoftTokens?.rawValue, budget != s.maxSoftTokens {
            s.maxSoftTokens = budget
        }
        spec = s
        if case .blocks(let blocks) = request.state {
            for case .image(let b) in blocks { pngs.append(try decodePNGBase64(b.base64)) }
        }
        image = imageSentinel
    }
    let user = try renderUserTurn(request.state, question, labels: labels, order: ord, readout: readout,
                                  labelStyle: labelStyle, image: image)
    let chat = try template.render(system: request.prefix, user: user)
    var text = chat.text
    if let spec, !pngs.isEmpty {
        for (k, png) in pngs.enumerated() {
            let marker = imageSentinel(k)
            guard text.pyCount(marker) == 1, chat.prefixText.pyCount(marker) == 0 else {
                throw ClassifierError(.prompt, "the chat template did not keep the image position")
            }
            text = text.pyReplacing(marker, with: try spec.placeholder(png))
        }
    }
    let tokens = tokenize(text)
    var prefixLen = 0
    if !chat.prefixText.isEmpty {
        let prefixIds = tokenize(chat.prefixText)
        guard Array(tokens.prefix(prefixIds.count)) == prefixIds else {
            throw ClassifierError(.prompt,
                "the prefix does not tokenize to a token-aligned head of the prompt; it cannot be KV-cached")
        }
        prefixLen = prefixIds.count
    }
    guard tokens.count > prefixLen else { throw ClassifierError(.prompt, "prompt has no tokens after the prefix") }
    let images = pngs.isEmpty ? [] : try imageSlots(tokens, pngs, spec!)
    return CompiledPrompt(name: name, question: question, text: text, tokens: tokens, prefixLen: prefixLen,
                          readoutPosition: tokens.count - 1, options: options, labels: labels, order: ord,
                          readout: readout, labelStyle: labelStyle, images: images)
}

/// Golden `_image_slots`: one slot per image; the runs must match the images one to one.
func imageSlots(_ tokens: [Int], _ pngs: [Data], _ spec: ImageTokens) throws -> [ImageSlot] {
    let runs = findImageRuns(tokens, imageTokenId: spec.imageTokenId)
    guard runs.count == pngs.count else {
        throw ClassifierError(.prompt,
            "\(runs.count) image-token runs in the prompt for \(pngs.count) image blocks "
            + "(does the text itself contain the image placeholder?)")
    }
    return try zip(runs, pngs).map { run, png in
        let want = try spec.softTokens(png)
        guard run.length == want else {
            throw ClassifierError(.prompt, "image run at \(run.start) has \(run.length) tokens, expected \(want)")
        }
        return ImageSlot(start: run.start, length: run.length, png: png, maxSoftTokens: spec.maxSoftTokens)
    }
}

/// Golden `compile_request`: one `CompiledPrompt` per question, in request order.
public func compileRequest(_ request: SystemOneRequest, template: ChatTemplate, tokenize: Tokenize,
                           readout: Readout? = nil, imageTokens: ImageTokens? = nil) throws -> [CompiledPrompt] {
    try request.questions.keys.map {
        try compileQuestion(request, name: $0, template: template, tokenize: tokenize, readout: readout,
                            imageTokens: imageTokens)
    }
}
