import Foundation
import CoreML
import UIKit

/// High-level interface for providing visual and text embeddings.
/// Abstracts whether we are using QwenVisionEncoder or a Text fusion model.
protocol EmbeddingProvider {
    func embedImage(_ image: UIImage) async throws -> [Float]
    func embedComposed(referenceImage: UIImage, text: String) async throws -> [Float]
}

class EmbeddingModel: EmbeddingProvider {
    private let visionEncoder: QwenVisionEncoderProtocol
    
    init(visionEncoder: QwenVisionEncoderProtocol) {
        self.visionEncoder = visionEncoder
    }
    
    func embedImage(_ image: UIImage) async throws -> [Float] {
        return try await visionEncoder.encode(image: image)
    }
    
    func embedComposed(referenceImage: UIImage, text: String) async throws -> [Float] {
        return try await visionEncoder.encode(image: referenceImage, textGuidance: text)
    }
}
