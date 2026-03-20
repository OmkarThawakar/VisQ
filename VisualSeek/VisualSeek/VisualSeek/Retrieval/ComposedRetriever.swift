import Foundation
import UIKit

class ComposedRetriever {
    private let store: EmbeddingStore
    private let vectorIndex: VectorIndex
    private let visionEncoder: QwenVisionEncoderProtocol
    private let languageModel: QwenLanguageModelProtocol
    
    init(store: EmbeddingStore, vectorIndex: VectorIndex, visionEncoder: QwenVisionEncoderProtocol, languageModel: QwenLanguageModelProtocol) {
        self.store = store
        self.vectorIndex = vectorIndex
        self.visionEncoder = visionEncoder
        self.languageModel = languageModel
    }
    
    /// Composed Image Retrieval: Reference Image + Text Edit -> Target Image
    func search(referenceImage: UIImage, editText: String, topK: Int = 20) async throws -> [RetrievalResult] {
        
        let reasoningPrompt = """
        You are analyzing a visual modification request.
        Request edit: "\(editText)"
        Generate a structured analysis of what the TARGET image should contain:
        - Object states: What physical states change?
        - Action phases: What temporal progression is implied?
        - Scene changes: What environment modifications?
        - Camera/framing: Any cinematographic changes?
        - Tempo/mood: Pacing or emotional changes?
        """
        
        // 1. After-Effect Reasoning
        let reasoningTrace = try await languageModel.generate(prompt: reasoningPrompt, image: referenceImage, maxTokens: 256)
        
        // 2. Hypothetical Target Description
        let hypotheticalDescription = try await languageModel.generateDescription(reasoning: reasoningTrace, editText: editText)
        
        // 3. Multimodal Query embedding (Image + Text Guidance)
        let queryEmbedding = try await visionEncoder.encode(image: referenceImage, textGuidance: hypotheticalDescription)
        
        // 4. Vector Search
        let rawResults = vectorIndex.search(query: queryEmbedding, topK: topK)
        let descriptionsByID = try Dictionary(
            uniqueKeysWithValues: store.fetchAllEmbeddings().map { record in
                (record.assetLocalIdentifier, record.imageDescription)
            }
        )
        
        return rawResults.map { result in
            let description = descriptionsByID[result.id] ?? hypotheticalDescription
            return RetrievalResult(
                id: result.id,
                title: MatchExplanationFactory.excerpt(from: description) ?? "Expected: \(reasoningTrace.states)",
                score: result.score,
                matchExplanation: MatchExplanationFactory.composedMatch(
                    editText: editText,
                    reasoning: reasoningTrace,
                    documentText: description
                )
            )
        }
    }
}
