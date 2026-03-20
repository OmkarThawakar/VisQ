import Testing
import UIKit
@testable import VisualSeek

@MainActor
struct VisualSeekTests {

    @Test func textEncoderProducesFixedLengthIDsAndMask() async throws {
        let encoded = QwenTextInputEncoder.encode("Golden retriever running in autumn leaves")

        #expect(encoded.ids.count == QwenTextInputEncoder.maxTokens)
        #expect(encoded.mask.count == QwenTextInputEncoder.maxTokens)
        #expect(encoded.ids.first == QwenTextInputEncoder.bosToken)
        #expect(encoded.mask.contains(1))
    }

    @Test func modelArrayFactoryNormalizesEmbeddings() async throws {
        let normalized = QwenArrayFactory.normalize([3, 4])
        #expect(abs(normalized[0] - 0.6) < 0.0001)
        #expect(abs(normalized[1] - 0.8) < 0.0001)
    }

    @Test func imagePreprocessorProducesExpectedPatchVectorLength() async throws {
        let renderer = UIGraphicsImageRenderer(size: CGSize(width: 448, height: 448))
        let image = renderer.image { context in
            UIColor.orange.setFill()
            context.fill(CGRect(x: 0, y: 0, width: 448, height: 448))
        }

        let pixels = try ImagePreprocessor.makeVisionPixelValues(image)
        #expect(pixels.count == 784 * 1536)
    }

}
