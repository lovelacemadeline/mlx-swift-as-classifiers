// Request-compile errors, ported from golden `errors.py`. Golden raises a class
// hierarchy (ClassifierError > PromptError, OptionBindingError >
// TooManyOptionsError, LabelTokenizationError); the class name is what a
// fixture's `expected.error.type` records, so `Kind.rawValue` is that name.

public struct ClassifierError: Error, CustomStringConvertible, Sendable, Equatable {
    public enum Kind: String, Sendable, CaseIterable {
        /// The chat template or tokenizer cannot give a stable prefix/readout split.
        case prompt = "PromptError"
        /// Options cannot be bound to labels.
        case optionBinding = "OptionBindingError"
        /// More options than single-letter labels (A..Z).
        case tooManyOptions = "TooManyOptionsError"
        /// A label is not exactly one token at the readout position (golden ADR-0003).
        case labelTokenization = "LabelTokenizationError"
    }

    public let kind: Kind
    public let message: String

    public init(_ kind: Kind, _ message: String) {
        self.kind = kind
        self.message = message
    }

    /// The golden error class name, e.g. "LabelTokenizationError".
    public var type: String { kind.rawValue }
    public var description: String { "\(kind.rawValue): \(message)" }

    /// True for the option-binding family (golden `OptionBindingError` subclasses).
    public var isOptionBinding: Bool { kind != .prompt }
}
