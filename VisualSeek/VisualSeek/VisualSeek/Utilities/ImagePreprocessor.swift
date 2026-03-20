import Foundation
import UIKit
import CoreML

struct ImagePreprocessor {
    
    /// Resizes an image to the expected coreML resolution, usually 448x448, and extracts pixel buffers.
    static func preprocessForQwen(_ image: UIImage, targetResolution: CGSize = CGSize(width: 448, height: 448)) throws -> UIImage {
        // Redraw image to specific resolution to be ready for ANE inference
        UIGraphicsBeginImageContextWithOptions(targetResolution, false, 1.0)
        image.draw(in: CGRect(origin: .zero, size: targetResolution))
        guard let resized = UIGraphicsGetImageFromCurrentImageContext() else {
            UIGraphicsEndImageContext()
            throw NSError(domain: "ImagePreprocessor", code: -1, userInfo: [NSLocalizedDescriptionKey: "Failed to resize image"])
        }
        UIGraphicsEndImageContext()
        return resized
    }

    static func makeVisionPixelValues(_ image: UIImage, targetResolution: CGSize = CGSize(width: 448, height: 448)) throws -> [Float] {
        let resized = try preprocessForQwen(image, targetResolution: targetResolution)
        guard let cgImage = resized.cgImage else {
            throw NSError(domain: "ImagePreprocessor", code: -2, userInfo: [NSLocalizedDescriptionKey: "Missing CGImage"])
        }

        let width = Int(targetResolution.width)
        let height = Int(targetResolution.height)
        let bytesPerPixel = 4
        let bytesPerRow = width * bytesPerPixel
        var buffer = [UInt8](repeating: 0, count: height * bytesPerRow)

        let colorSpace = CGColorSpaceCreateDeviceRGB()
        guard let context = CGContext(
            data: &buffer,
            width: width,
            height: height,
            bitsPerComponent: 8,
            bytesPerRow: bytesPerRow,
            space: colorSpace,
            bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue
        ) else {
            throw NSError(domain: "ImagePreprocessor", code: -3, userInfo: [NSLocalizedDescriptionKey: "Failed to create bitmap context"])
        }

        context.draw(cgImage, in: CGRect(origin: .zero, size: targetResolution))

        let patchSize = 16
        let temporalPatchSize = 2
        let patchesPerSide = width / patchSize
        var values: [Float] = []
        values.reserveCapacity(patchesPerSide * patchesPerSide * 3 * temporalPatchSize * patchSize * patchSize)

        for patchY in 0..<patchesPerSide {
            for patchX in 0..<patchesPerSide {
                for channel in 0..<3 {
                    for temporalIndex in 0..<temporalPatchSize {
                        _ = temporalIndex
                        for y in 0..<patchSize {
                            for x in 0..<patchSize {
                                let pixelX = patchX * patchSize + x
                                let pixelY = patchY * patchSize + y
                                let offset = pixelY * bytesPerRow + pixelX * bytesPerPixel
                                let component = Float(buffer[offset + channel]) / 255.0
                                let normalized = (component - 0.5) / 0.5
                                values.append(normalized)
                            }
                        }
                    }
                }
            }
        }

        return values
    }
}
