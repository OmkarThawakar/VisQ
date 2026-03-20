import Foundation
import os

protocol VectorIndexProtocol {
    func add(embeddings: [[Float]], ids: [String])
    func search(query: [Float], topK: Int) -> [(id: String, score: Float)]
    func buildIndex(from store: EmbeddingStore) throws
}

/// A simple vector index using Accelerate for brute-force matching. 
/// Useful for small to medium-sized databases (< 50,000 to 100,000 photos).
class AccelerateVectorIndex: VectorIndexProtocol {
    private var embeddings: [[Float]] = []
    private var ids: [String] = []
    
    init() {}
    
    func add(embeddings newEmbeddings: [[Float]], ids newIds: [String]) {
        self.embeddings.append(contentsOf: newEmbeddings)
        self.ids.append(contentsOf: newIds)
    }
    
    func buildIndex(from store: EmbeddingStore) throws {
        let records = try store.fetchAllEmbeddings()
        self.embeddings = records.compactMap { record in
            let floatArray = record.embedding.withUnsafeBytes { bufferPointer in
                Array(bufferPointer.bindMemory(to: Float.self))
            }
            return floatArray
        }
        self.ids = records.map { $0.assetLocalIdentifier }
    }
    
    func search(query: [Float], topK: Int) -> [(id: String, score: Float)] {
        let results = CosineSimilarity.topKResults(query: query, corpus: embeddings, k: topK)
        return results.map { (ids[$0.index], $0.score) }
    }
}

/// A robust proxy VectorIndex that initializes AccelerateIndex or FAISSIndex (hypothetical) depending on photo count.
class VectorIndex {
    private let FAISS_THRESHOLD = 100_000
    private var index: VectorIndexProtocol
    
    init(embeddingCount: Int) {
        if embeddingCount > FAISS_THRESHOLD {
            // Placeholder for real FAISS wrapper
            // index = FAISSIndex(dimension: 1536, metric: .innerProduct)
            AppLog.retrieval.warning("FAISS threshold exceeded; falling back to Accelerate index")
            self.index = AccelerateVectorIndex()
        } else {
            self.index = AccelerateVectorIndex()
        }
    }
    
    func rebuild(from store: EmbeddingStore) throws {
        try index.buildIndex(from: store)
    }
    
    func search(query: [Float], topK: Int) -> [(id: String, score: Float)] {
        return index.search(query: query, topK: topK)
    }
}
