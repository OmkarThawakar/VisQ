import Foundation
import Accelerate
import CoreML

/// Importance-Weighted Pooling Head
/// Ported from CoVR-R's generate_embeddings.py
/// This takes the output of a Vision Transformer and pools token embeddings
/// based on attention priorities.
struct ImportanceWeightedPooling {
    
    /// Pools token embeddings based on attention weights.
    ///
    /// - Parameters:
    ///   - tokenEmbeddings: A flat array representing (N tokens x D dimension)
    ///   - attentionWeights: A flat array representing (N tokens) attention logits
    ///   - dimension: D, the size of each embedding (e.g. 1536)
    /// - Returns: A pooled, L2-normalized embedding of size D.
    static func pool(tokenEmbeddings: [Float], attentionWeights: [Float], dimension: Int) -> [Float] {
        let numTokens = attentionWeights.count
        guard numTokens * dimension == tokenEmbeddings.count, numTokens > 0 else {
            return [Float](repeating: 0, count: dimension)
        }
        
        // Compute softmax over attentionWeights to get actual importance probabilities
        var maxWeight: Float = -Float.greatestFiniteMagnitude
        vDSP_maxv(attentionWeights, 1, &maxWeight, vDSP_Length(numTokens))
        
        var subtracted = [Float](repeating: 0, count: numTokens)
        var negativeMax = -maxWeight
        vDSP_vsadd(attentionWeights, 1, &negativeMax, &subtracted, 1, vDSP_Length(numTokens))
        
        var expWeights = [Float](repeating: 0, count: numTokens)
        var count = Int32(numTokens)
        vvexpf(&expWeights, subtracted, &count)
        
        var sumExp: Float = 0
        vDSP_sve(expWeights, 1, &sumExp, vDSP_Length(numTokens))
        
        var probabilities = [Float](repeating: 0, count: numTokens)
        var divisor = sumExp
        vDSP_vsdiv(expWeights, 1, &divisor, &probabilities, 1, vDSP_Length(numTokens))
        
        // Compute weighted sum
        var pooledResult = [Float](repeating: 0, count: dimension)
        for i in 0..<numTokens {
            let prob = probabilities[i]
            let offset = i * dimension
            let tokenArr = Array(tokenEmbeddings[offset..<offset+dimension])
            
            // multiply token[i] by probability
            let probArray = [Float](repeating: prob, count: dimension)
            var scaledToken = [Float](repeating: 0, count: dimension)
            vDSP_vmul(tokenArr, 1, probArray, 1, &scaledToken, 1, vDSP_Length(dimension))
            
            // add to pooledResult
            vDSP_vadd(pooledResult, 1, scaledToken, 1, &pooledResult, 1, vDSP_Length(dimension))
        }
        
        // L2 Normalize
        var sumSquares: Float = 0
        vDSP_svesq(pooledResult, 1, &sumSquares, vDSP_Length(dimension))
        let norm = sqrt(sumSquares)
        
        if norm > 0 {
            var inverseNorm = 1.0 / norm
            vDSP_vsmul(pooledResult, 1, &inverseNorm, &pooledResult, 1, vDSP_Length(dimension))
        }
        
        return pooledResult
    }
}
