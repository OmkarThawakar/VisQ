import Testing
import UIKit
@testable import VisualSeek

// MARK: - QwenVLDescriptionGenerator Tests

@MainActor
struct QwenVLDescriptionGeneratorTests {

    private let generator = QwenVLDescriptionGenerator()

    /// Creates a synthetic 768-D normalized embedding from a seed value.
    private func syntheticEmbedding(seed: Float) -> [Float] {
        var values = (0..<768).map { i in
            sin(Float(i) * seed * 0.1) * cos(Float(i + 1) * seed * 0.05)
        }
        // L2-normalize
        let norm = sqrt(values.reduce(0) { $0 + $1 * $1 })
        if norm > 0 { values = values.map { $0 / norm } }
        return values
    }

    /// Creates a simple solid-color test image.
    private func testImage(color: UIColor = .systemBlue) -> UIImage {
        let renderer = UIGraphicsImageRenderer(size: CGSize(width: 224, height: 224))
        return renderer.image { context in
            color.setFill()
            context.fill(CGRect(x: 0, y: 0, width: 224, height: 224))
        }
    }

    // MARK: - Core output quality tests

    @Test func descriptionIsNonEmptyForValidEmbedding() {
        let description = generator.generate(
            embedding: syntheticEmbedding(seed: 1.0),
            image: testImage()
        )
        #expect(!description.isEmpty)
    }

    @Test func descriptionMeetsMinimumLength() {
        let description = generator.generate(
            embedding: syntheticEmbedding(seed: 2.5),
            image: testImage(color: .systemGreen)
        )
        // A Qwen-VL description should be substantially longer than a keyword list
        #expect(description.count >= 80, "Description too short: \(description.count) chars")
    }

    @Test func descriptionContainsMultipleSentences() {
        let description = generator.generate(
            embedding: syntheticEmbedding(seed: 3.7),
            image: testImage(color: .systemOrange)
        )
        let sentenceCount = description.components(separatedBy: ".").filter { !$0.trimmingCharacters(in: .whitespaces).isEmpty }.count
        #expect(sentenceCount >= 3, "Expected ≥ 3 sentences, got \(sentenceCount)")
    }

    @Test func descriptionContainsSemanticKeywords() {
        let description = generator.generate(
            embedding: syntheticEmbedding(seed: 5.1),
            image: testImage()
        ).lowercased()

        // Should contain at least one scene-type term
        let sceneKeywords = ["outdoor", "indoor", "landscape", "scene", "urban", "environment", "setting", "room"]
        let hasScene = sceneKeywords.contains { description.contains($0) }
        #expect(hasScene, "Missing scene-type keyword in: \(description)")

        // Should contain at least one color-related term
        let colorKeywords = ["tone", "hue", "color", "warm", "cool", "palette", "bright", "vivid", "saturat"]
        let hasColor = colorKeywords.contains { description.contains($0) }
        #expect(hasColor, "Missing color keyword in: \(description)")
    }

    // MARK: - Fallback behavior

    @Test func emptyEmbeddingStillProducesDescription() {
        let image = testImage(color: .systemYellow)
        // The generator now uses Vision analysis regardless of whether an embedding
        // is provided, so an empty embedding must still produce a valid description.
        let description = generator.generate(embedding: [], image: image)
        #expect(!description.isEmpty)
        #expect(description.count >= 80, "Description too short for empty embedding: \(description.count) chars")
    }

    // MARK: - Determinism

    @Test func descriptionIsDeterministicForSameEmbedding() {
        let embedding = syntheticEmbedding(seed: 9.9)
        let image = testImage()
        let d1 = generator.generate(embedding: embedding, image: image)
        let d2 = generator.generate(embedding: embedding, image: image)
        #expect(d1 == d2, "Description should be deterministic for the same embedding")
    }

    /// Verifies descriptions differ for visually different images.
    /// (Descriptions are now image-content-driven, not embedding-driven.)
    @Test func descriptionsDifferForDifferentImages() {
        let d1 = generator.generate(embedding: [], image: testImage(color: .systemBlue))
        let d2 = generator.generate(embedding: [], image: testImage(color: .systemRed))
        // Even solid-color images differ in color analysis, so descriptions must differ
        #expect(d1 != d2, "Visually distinct images should produce different descriptions")
    }
}
