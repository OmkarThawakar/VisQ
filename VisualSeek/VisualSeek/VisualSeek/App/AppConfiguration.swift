import Foundation
import CoreML

struct AppConfiguration {
    // Model settings
    // Prefer GPU/CPU by default because the bundled Qwen2-VL models are large
    // enough to fail ANE compilation on some devices.
    var computeUnits: MLComputeUnits = .cpuAndGPU
    // Keep fewer preprocessed images resident at once to reduce launch/indexing
    // memory pressure on-device.
    var batchSize: Int = 4
    // The current app runtime persists 768-D retrieval embeddings even though
    // the underlying vision patch representation uses 1536 values per patch.
    var embeddingDimension: Int = 768
    
    // Indexing settings
    var indexNewPhotosAutomatically: Bool = true
    var backgroundIndexingEnabled: Bool = true       // Uses BGTaskScheduler
    var indexOnWifiOnly: Bool = false
    var indexOnChargingOnly: Bool = true
    
    // Retrieval settings
    var defaultTopK: Int = 20
    var minimumSimilarityScore: Float = 0.3
    var useReasoningForComposedRetrieval: Bool = true
    var reasoningMaxTokens: Int = 256
    
    // Storage settings
    var embeddingStoragePath: URL {
        URL.documentsDirectory.appendingPathComponent("embeddings.sqlite")
    }
    var maxStorageGB: Double = 2.0
}
