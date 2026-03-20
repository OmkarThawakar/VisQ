import Foundation
import CoreML
import UIKit

/// Protocol defining the necessary operations for integrating the Qwen Vision Encoder.
protocol QwenVisionEncoderProtocol {
    func encode(image: UIImage) async throws -> [Float]
    func encode(image: UIImage, textGuidance: String) async throws -> [Float]
    func encodeBatch(images: [UIImage]) async throws -> [[Float]]
}

final class QwenVisionEncoderAdapter: QwenVisionEncoderProtocol {
    private let embeddingDimension = 768
    private let model: MLModel?
    private let textFusionModel: MLModel?

    init(model: MLModel? = nil, textFusionModel: MLModel? = nil) {
        self.model = model
        self.textFusionModel = textFusionModel
    }

    func encode(image: UIImage) async throws -> [Float] {
        guard let model else {
            return lightweightEmbedding(for: image)
        }
        do {
            return try await encodeImage(image, with: model)
        } catch {
            return lightweightEmbedding(for: image)
        }
    }

    func encode(image: UIImage, textGuidance: String) async throws -> [Float] {
        let visualEmbedding = try await encode(image: image)
        guard let textFusionModel else {
            let textEmbedding = textGuidanceEmbedding(textGuidance)
            let combined = zip(visualEmbedding, textEmbedding).map { (visual, text) in
                (visual * 0.7) + (text * 0.3)
            }
            return QwenArrayFactory.normalize(combined)
        }

        let (inputIDs, attentionMask) = QwenTextInputEncoder.encode(textGuidance)
        let inputProvider = try MLDictionaryFeatureProvider(dictionary: [
            "input_ids": MLFeatureValue(multiArray: try QwenArrayFactory.makeInt32Array(inputIDs, shape: [1, QwenTextInputEncoder.maxTokens])),
            "attention_mask": MLFeatureValue(multiArray: try QwenArrayFactory.makeInt32Array(attentionMask, shape: [1, QwenTextInputEncoder.maxTokens])),
        ])
        let output = try await textFusionModel.prediction(from: inputProvider)
        let textEmbedding = try QwenArrayFactory.extractFloats(from: output, featureName: "var_52")

        let combined = zip(visualEmbedding, textEmbedding).map { (visual, text) in
            (visual * 0.8) + (text * 0.2)
        }
        return QwenArrayFactory.normalize(combined)
    }

    func encodeBatch(images: [UIImage]) async throws -> [[Float]] {
        var embeddings: [[Float]] = []
        embeddings.reserveCapacity(images.count)
        for image in images {
            embeddings.append(try await encode(image: image))
        }
        return embeddings
    }

    private func encodeImage(_ image: UIImage, with model: MLModel) async throws -> [Float] {
        let pixelValues = try ImagePreprocessor.makeVisionPixelValues(image)
        let inputProvider = try MLDictionaryFeatureProvider(dictionary: [
            "pixel_values": MLFeatureValue(multiArray: try QwenArrayFactory.makeFloatArray(pixelValues, shape: [784, 1536])),
            "grid_thw": MLFeatureValue(multiArray: try QwenArrayFactory.makeInt32Array([1, 28, 28], shape: [1, 3])),
        ])

        let output = try await model.prediction(from: inputProvider)
        let embedding = try QwenArrayFactory.extractFloats(from: output, featureName: "var_2334")
        return QwenArrayFactory.normalize(Array(embedding.prefix(embeddingDimension)))
    }

    private func generateMockEmbedding() -> [Float] {
        QwenArrayFactory.normalize((0..<embeddingDimension).map { index in
            Float(((index * 37) % 255)) / 255.0
        })
    }

    private func lightweightEmbedding(for image: UIImage) -> [Float] {
        let targetSize = CGSize(width: 16, height: 16)
        guard let resized = try? ImagePreprocessor.preprocessForQwen(image, targetResolution: targetSize),
              let cgImage = resized.cgImage else {
            return generateMockEmbedding()
        }

        let width = Int(targetSize.width)
        let height = Int(targetSize.height)
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
            return generateMockEmbedding()
        }

        context.draw(cgImage, in: CGRect(origin: .zero, size: targetSize))

        var embedding: [Float] = []
        embedding.reserveCapacity(embeddingDimension)

        for channel in 0..<3 {
            for pixelIndex in stride(from: channel, to: buffer.count, by: bytesPerPixel) {
                let value = (Float(buffer[pixelIndex]) / 255.0) * 2.0 - 1.0
                embedding.append(value)
            }
        }

        return QwenArrayFactory.normalize(embedding)
    }

    private func textGuidanceEmbedding(_ text: String) -> [Float] {
        let tokens = text.lowercased().split { !$0.isLetter && !$0.isNumber }
        var vector = [Float](repeating: 0, count: embeddingDimension)

        for token in tokens {
            var hash: UInt64 = 1469598103934665603
            for byte in token.utf8 {
                hash ^= UInt64(byte)
                hash &*= 1099511628211
            }

            let index = Int(hash % UInt64(embeddingDimension))
            vector[index] += 1
        }

        return QwenArrayFactory.normalize(vector)
    }
}
