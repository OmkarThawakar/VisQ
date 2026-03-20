import Foundation
import CoreML
import UIKit

/// Represents the generated reasoning trace from CoVR-R style prompts
struct ReasoningTrace: Codable {
    var states: String
    var actions: String
    var scene: String
    var camera: String
    var tempo: String
}

protocol QwenLanguageModelProtocol {
    func generate(prompt: String, image: UIImage, maxTokens: Int) async throws -> ReasoningTrace
    func generateDescription(reasoning: ReasoningTrace, editText: String) async throws -> String
}

class QwenLanguageModel: QwenLanguageModelProtocol {
    private let model: MLModel?
    private let descriptors = [
        "bright", "moody", "calm", "dynamic", "warm", "cool", "soft", "dramatic",
        "wide", "close", "minimal", "detailed", "quiet", "vivid", "shadowy", "cinematic"
    ]

    init(model: MLModel? = nil) {
        self.model = model
    }
    
    func generate(prompt: String, image: UIImage, maxTokens: Int) async throws -> ReasoningTrace {
        guard let model else {
            return ReasoningTrace(
                states: "transition from bright to dark",
                actions: "person moving off-screen",
                scene: "lighting changes to sunset",
                camera: "static shot",
                tempo: "slow down"
            )
        }

        let embedding = try await encodePrompt(prompt, with: model)
        let brightness = image.averageBrightness

        return ReasoningTrace(
            states: descriptor(from: embedding, offset: 0, fallback: brightness > 0.55 ? "brightened scene" : "darker scene"),
            actions: descriptor(from: embedding, offset: 1, fallback: "gentle motion") + " progression",
            scene: descriptor(from: embedding, offset: 2, fallback: "updated environment"),
            camera: descriptor(from: embedding, offset: 3, fallback: "stable framing"),
            tempo: descriptor(from: embedding, offset: 4, fallback: brightness > 0.55 ? "energetic" : "measured")
        )
    }
    
    func generateDescription(reasoning: ReasoningTrace, editText: String) async throws -> String {
        guard let model else {
            return "Hypothetical modified scene: \(editText). The scene has \(reasoning.scene) with \(reasoning.states)."
        }

        let embedding = try await encodePrompt(editText, with: model)
        let style = descriptor(from: embedding, offset: 5, fallback: "cohesive")
        return "Hypothetical modified scene: \(editText). The scene has \(reasoning.scene) with \(reasoning.states), \(reasoning.camera), and a \(style) feel."
    }

    private func encodePrompt(_ prompt: String, with model: MLModel) async throws -> [Float] {
        let (inputIDs, attentionMask) = QwenTextInputEncoder.encode(prompt)
        let inputProvider = try MLDictionaryFeatureProvider(dictionary: [
            "input_ids": MLFeatureValue(multiArray: try QwenArrayFactory.makeInt32Array(inputIDs, shape: [1, QwenTextInputEncoder.maxTokens])),
            "attention_mask": MLFeatureValue(multiArray: try QwenArrayFactory.makeInt32Array(attentionMask, shape: [1, QwenTextInputEncoder.maxTokens])),
        ])

        let output = try await model.prediction(from: inputProvider)
        return try QwenArrayFactory.extractFloats(from: output, featureName: "var_52")
    }

    private func descriptor(from embedding: [Float], offset: Int, fallback: String) -> String {
        guard !embedding.isEmpty else { return fallback }
        let index = Int(abs(embedding[offset % embedding.count]) * 10_000) % descriptors.count
        return descriptors[index]
    }
}

private extension UIImage {
    var averageBrightness: CGFloat {
        guard let cgImage = cgImage else { return 0.5 }
        let width = 16
        let height = 16
        let bytesPerPixel = 4
        let bytesPerRow = width * bytesPerPixel
        var buffer = [UInt8](repeating: 0, count: width * height * bytesPerPixel)
        let colorSpace = CGColorSpaceCreateDeviceRGB()
        guard let context = CGContext(
            data: &buffer,
            width: width,
            height: height,
            bitsPerComponent: 8,
            bytesPerRow: bytesPerRow,
            space: colorSpace,
            bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue
        ) else { return 0.5 }
        context.draw(cgImage, in: CGRect(x: 0, y: 0, width: width, height: height))
        let total = stride(from: 0, to: buffer.count, by: bytesPerPixel).reduce(Float(0)) { partial, index in
            partial + Float(buffer[index]) + Float(buffer[index + 1]) + Float(buffer[index + 2])
        }
        return CGFloat(total / Float(width * height * 3 * 255))
    }
}
