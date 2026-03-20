import Foundation
import AVFoundation
import UIKit

class VideoFrameExtractor {
    
    /// Extracts a fixed number of frames from a local video URL.
    func extractFrames(from url: URL, frameCount: Int = 4, targetSize: CGSize = CGSize(width: 448, height: 448)) async throws -> [UIImage] {
        let asset = AVAsset(url: url)
        return try await extractFrames(from: asset, frameCount: frameCount, targetSize: targetSize)
    }

    /// Extracts a fixed number of frames from any AVAsset-backed video.
    func extractFrames(from asset: AVAsset, frameCount: Int = 4, targetSize: CGSize = CGSize(width: 448, height: 448)) async throws -> [UIImage] {
        guard try await asset.load(.isPlayable) else {
            throw NSError(domain: "VideoFrameExtractor", code: -1, userInfo: [NSLocalizedDescriptionKey: "Video is not playable"])
        }
        
        let duration = try await asset.load(.duration)
        let durationSeconds = CMTimeGetSeconds(duration)
        
        var times: [NSValue] = []
        let interval = durationSeconds / Double(frameCount + 1)
        for i in 1...frameCount {
            let time = CMTime(seconds: interval * Double(i), preferredTimescale: 600)
            times.append(NSValue(time: time))
        }
        
        let generator = AVAssetImageGenerator(asset: asset)
        generator.appliesPreferredTrackTransform = true
        generator.maximumSize = targetSize
        generator.requestedTimeToleranceBefore = .zero
        generator.requestedTimeToleranceAfter = .zero
        let expectedCount = times.count
        
        return try await withCheckedThrowingContinuation { continuation in
            var images: [UIImage] = []
            var errors: [Error] = []
            var completedCount = 0
            
            generator.generateCGImagesAsynchronously(forTimes: times) { requestedTime, image, actualTime, result, error in
                DispatchQueue.main.async {
                    completedCount += 1
                    if let image = image {
                        images.append(UIImage(cgImage: image))
                    } else if let error = error {
                        errors.append(error)
                    }
                    
                    if completedCount == expectedCount {
                        if !images.isEmpty {
                            // Sort images if needed, but AVAssetImageGenerator gives them in order hopefully
                            continuation.resume(returning: images)
                        } else {
                            continuation.resume(throwing: errors.first ?? NSError(domain: "VideoFrameExtractor", code: -1, userInfo: [NSLocalizedDescriptionKey: "Failed to extract any frames"]))
                        }
                    }
                }
            }
        }
    }
}
