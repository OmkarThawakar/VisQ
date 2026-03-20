import Foundation
import UIKit
import CoreML

class BatchProcessor {
    let batchSize: Int
    let visionEncoder: QwenVisionEncoderProtocol
    
    init(batchSize: Int = 8, visionEncoder: QwenVisionEncoderProtocol) {
        self.batchSize = batchSize
        self.visionEncoder = visionEncoder
    }
    
    func processBatch(_ images: [UIImage]) async throws -> [[Float]] {
        return try await visionEncoder.encodeBatch(images: images)
    }
}
