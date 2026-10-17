// swift-tools-version: 6.0
// MLXClassifiers — Swift port of the golden Python System One classifier
// (github.com/lovelacemadeline/mlx-lm-as-classifiers). Definition of done:
// reproduce the vendored golden fixtures (fixtures/, see fixtures/VENDORED.txt).
//
// Dependency pins (verified to resolve and build on 2026-10-17, see
// docs/research/swift-port-recon-2026-10.md):
//   mlx-swift 0.32.3          — MLX core, MLXNN, MLXFast (custom Metal kernels)
//   mlx-swift-lm 3.32.3       — MLXLMCommon (KVCache, model protocols); the LLM/VLM
//                               libraries moved here out of mlx-swift-examples
//   swift-transformers 1.3.5  — Tokenizers (tokenizer.json loading)
import PackageDescription

let package = Package(
    name: "MLXClassifiers",
    platforms: [
        .macOS(.v14),
        .iOS(.v17),
    ],
    products: [
        .library(name: "MLXClassifiers", targets: ["MLXClassifiers"])
    ],
    dependencies: [
        .package(url: "https://github.com/ml-explore/mlx-swift", exact: "0.32.3"),
        .package(url: "https://github.com/ml-explore/mlx-swift-lm", exact: "3.32.3"),
        .package(url: "https://github.com/huggingface/swift-transformers", exact: "1.3.5"),
    ],
    targets: [
        .target(
            name: "MLXClassifiers",
            dependencies: [
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
                .product(name: "MLXLMCommon", package: "mlx-swift-lm"),
                .product(name: "Tokenizers", package: "swift-transformers"),
            ],
            // Models/ holds the vendored Gemma 4 text model (ADR-0004); its upstream licence ships beside it.
            exclude: ["Models/LICENSE-mlx-swift-lm"]
        ),
        .testTarget(
            name: "MLXClassifiersTests",
            dependencies: ["MLXClassifiers"]
        ),
    ]
)
