import Foundation
import Accelerate

enum CosineSimilarity {
    
    /// Computes the top K most similar vectors to the given query using Accelerate (vDSP).
    /// Assumes embeddings are already L2-normalized, so dot product is equivalent to cosine similarity.
    ///
    /// - Parameters:
    ///   - query: The query embedding (1D array of Floats).
    ///   - corpus: An array of embeddings (each a 1D array of Floats) to search against.
    ///   - k: The number of top results to return.
    /// - Returns: An array of tuples containing the index of the best matches and their similarity scores.
    static func topKResults(
        query: [Float],
        corpus: [[Float]],
        k: Int
    ) -> [(index: Int, score: Float)] {
        
        let dim = query.count
        var scores = [Float](repeating: 0, count: corpus.count)
        
        // Ensure that the dimension of all corpus embeddings match the query
        for (i, embedding) in corpus.enumerated() {
            guard embedding.count == dim else {
                continue
            }
            var dot: Float = 0
            // Core vDSP dot product for fast execution
            vDSP_dotpr(query, 1, embedding, 1, &dot, vDSP_Length(dim))
            scores[i] = dot
        }
        
        // Find the top K scores using a partial sort algorithm.
        // For Swift simplicity we use a full sort prefixed by K. 
        // In a highly optimized scenario, a min-heap or true partial sort could be used.
        return scores.enumerated()
            .sorted { $0.element > $1.element }
            .prefix(k)
            .map { (index: $0.offset, score: $0.element) }
    }
}
