import Foundation
import CoreML
import UIKit

// MARK: - Qwen2VLInferencePipeline
//
// On-device autoregressive description generator using the three CoreML models
// produced by scripts/convert_qwen2vl_to_coreml.py (Qwen3-VL-2B-Instruct):
//
//   Qwen2VLVisionEncoder.mlpackage  — ViT + spatial merger + deepstack features
//   Qwen2VLPrefill.mlpackage        — Full-prompt prefill, returns KV cache
//   Qwen2VLDecodeStep.mlpackage     — Single-token decode step
//
// All constants must match qwen2vl_config.json printed by the conversion script.

final class Qwen2VLInferencePipeline {

    // MARK: - Constants (must match qwen2vl_config.json)
    private enum C {
        // Vision
        static let numVisualTokens  = 196       // 28×28 / 2² spatial merge
        static let patchFlatSize    = 1536      // 3 × temporal(2) × patch(16) × patch(16)
        static let patchesPerSide   = 28        // 448 / patch_size(16)
        static let patchSize        = 16
        static let temporalPatchSize = 2

        // Language model
        static let hiddenSize   = 2048
        static let numLayers    = 28
        static let numKVHeads   = 8
        static let headDim      = 128
        static let maxNewTokens = 200
        static let imageStartIndex = 15        // position of first <|image_pad|>
        static let promptLength    = 285       // total prompt tokens

        // Special token IDs (Qwen3-VL vocabulary)
        static let imStartID:     Int32 = 151_644
        static let imEndID:       Int32 = 151_645
        static let visionStartID: Int32 = 151_652
        static let visionEndID:   Int32 = 151_653
        static let imagePadID:    Int32 = 151_655
        static let newlineID:     Int32 = 198
    }

    // MARK: - Models

    private let visionEncoder: MLModel
    private let prefillModel:  MLModel
    private let decodeModel:   MLModel

    // MARK: - Init

    init(visionEncoder: MLModel, prefillModel: MLModel, decodeModel: MLModel) {
        self.visionEncoder = visionEncoder
        self.prefillModel  = prefillModel
        self.decodeModel   = decodeModel
    }

    // MARK: - Public API

    /// Generates a rich natural-language description of `image`.
    /// `progressHandler` is called on a background thread.
    func generateDescription(
        for image: UIImage,
        progressHandler: ((String) -> Void)? = nil
    ) async throws -> String {
        try Task.checkCancellation()
        progressHandler?("Preprocessing image…")

        // 1 — Image → pixel patches [N, 1536] and grid [1, 3]
        let (pixelValues, gridTHW) = try Qwen3VLImagePreprocessor.preprocess(
            image,
            patchesPerSide: C.patchesPerSide,
            patchSize: C.patchSize,
            temporalPatchSize: C.temporalPatchSize
        )

        try Task.checkCancellation()
        progressHandler?("Running vision encoder…")

        // 2 — Vision encoder → pooler_output [196, 2048] + deepstack_0/1/2 [196, 2048]
        let (imageFeatures, ds0, ds1, ds2) = try await runVisionEncoder(
            pixelValues: pixelValues, gridTHW: gridTHW
        )

        try Task.checkCancellation()
        progressHandler?("Building prompt…")

        // 3 — Build prompt token IDs (system + vision + user + assistant header)
        let promptTokenIDs = buildPromptTokenIDs()

        try Task.checkCancellation()
        progressHandler?("Running language model prefill…")

        // 4 — Prefill → first logits + KV cache
        let visualPosMask = try buildVisualPosMask(promptLen: promptTokenIDs.count)
        let (firstLogits, kvKeys, kvVals) = try await runPrefill(
            promptTokenIDs: promptTokenIDs,
            imageFeatures:  imageFeatures,
            ds0: ds0, ds1: ds1, ds2: ds2,
            visualPosMask: visualPosMask
        )

        try Task.checkCancellation()

        // 5 — Greedy decode loop
        var generatedIDs: [Int32] = []
        var currentLogits = firstLogits
        var currentKVKeys = kvKeys
        var currentKVVals = kvVals

        for step in 0..<C.maxNewTokens {
            try Task.checkCancellation()

            let nextID = argmax(currentLogits)
            if nextID == C.imEndID { break }
            generatedIDs.append(nextID)

            if step % 25 == 0 {
                progressHandler?("Generating… (\(generatedIDs.count) tokens)")
            }

            let (nextLogits, nextKeys, nextVals) = try await runDecodeStep(
                tokenID: nextID, kvKeys: currentKVKeys, kvVals: currentKVVals
            )
            currentLogits  = nextLogits
            currentKVKeys  = nextKeys
            currentKVVals  = nextVals
        }

        // 6 — Token IDs → text
        let text = Qwen2VLTokenDecoder.decode(generatedIDs)
        return text.trimmingCharacters(in: .whitespacesAndNewlines)
    }

    // MARK: - Vision Encoder

    private func runVisionEncoder(
        pixelValues: MLMultiArray,
        gridTHW:     MLMultiArray
    ) async throws -> (imageFeatures: MLMultiArray,
                       ds0: MLMultiArray,
                       ds1: MLMultiArray,
                       ds2: MLMultiArray) {
        let input = try MLDictionaryFeatureProvider(dictionary: [
            "pixel_values": MLFeatureValue(multiArray: pixelValues),
            "grid_thw":     MLFeatureValue(multiArray: gridTHW),
        ])
        let output = try await visionEncoder.prediction(from: input)

        guard let feat = output.featureValue(for: "pooler_output")?.multiArrayValue,
              let ds0  = output.featureValue(for: "deepstack_0")?.multiArrayValue,
              let ds1  = output.featureValue(for: "deepstack_1")?.multiArrayValue,
              let ds2  = output.featureValue(for: "deepstack_2")?.multiArrayValue else {
            throw Qwen2VLError.missingOutput("vision encoder outputs")
        }
        return (feat, ds0, ds1, ds2)
    }

    // MARK: - Prefill

    private func runPrefill(
        promptTokenIDs: [Int32],
        imageFeatures:  MLMultiArray,
        ds0: MLMultiArray,
        ds1: MLMultiArray,
        ds2: MLMultiArray,
        visualPosMask:  MLMultiArray
    ) async throws -> (logits: MLMultiArray,
                       kvKeys: MLMultiArray,
                       kvVals: MLMultiArray) {
        let T = promptTokenIDs.count
        let inputIDs = try MLMultiArray(shape: [1, NSNumber(value: T)], dataType: .int32)
        for (i, id) in promptTokenIDs.enumerated() {
            inputIDs[i] = NSNumber(value: id)
        }

        let input = try MLDictionaryFeatureProvider(dictionary: [
            "input_ids":      MLFeatureValue(multiArray: inputIDs),
            "image_features": MLFeatureValue(multiArray: imageFeatures),
            "deepstack_0":    MLFeatureValue(multiArray: ds0),
            "deepstack_1":    MLFeatureValue(multiArray: ds1),
            "deepstack_2":    MLFeatureValue(multiArray: ds2),
            // visual_pos_mask was converted from bool → fp32 by coremltools
            "visual_pos_mask": MLFeatureValue(multiArray: visualPosMask),
        ])
        let output = try await prefillModel.prediction(from: input)

        guard let logits  = output.featureValue(for: "logits")?.multiArrayValue,
              let kvKeys  = output.featureValue(for: "kv_keys")?.multiArrayValue,
              let kvVals  = output.featureValue(for: "kv_vals")?.multiArrayValue else {
            throw Qwen2VLError.missingOutput("prefill outputs")
        }
        return (logits, kvKeys, kvVals)
    }

    // MARK: - Single Decode Step

    private func runDecodeStep(
        tokenID: Int32,
        kvKeys: MLMultiArray,
        kvVals: MLMultiArray
    ) async throws -> (logits: MLMultiArray,
                       newKVKeys: MLMultiArray,
                       newKVVals: MLMultiArray) {
        let tokenArray = try MLMultiArray(shape: [1, 1], dataType: .int32)
        tokenArray[0] = NSNumber(value: tokenID)

        let input = try MLDictionaryFeatureProvider(dictionary: [
            "token_id": MLFeatureValue(multiArray: tokenArray),
            "kv_keys":  MLFeatureValue(multiArray: kvKeys),
            "kv_vals":  MLFeatureValue(multiArray: kvVals),
        ])
        let output = try await decodeModel.prediction(from: input)

        guard let logits    = output.featureValue(for: "logits")?.multiArrayValue,
              let newKVKeys = output.featureValue(for: "new_kv_keys")?.multiArrayValue,
              let newKVVals = output.featureValue(for: "new_kv_vals")?.multiArrayValue else {
            throw Qwen2VLError.missingOutput("decode outputs")
        }
        return (logits, newKVKeys, newKVVals)
    }

    // MARK: - Visual Position Mask

    /// Builds the float32 mask (converted from bool by coremltools) where True (1.0)
    /// marks the image-pad token positions [imageStartIndex ..< imageStartIndex + numVisualTokens].
    private func buildVisualPosMask(promptLen: Int) throws -> MLMultiArray {
        let mask: MLMultiArray
        do {
            mask = try MLMultiArray(shape: [1, NSNumber(value: promptLen)], dataType: .float32)
        } catch {
            throw Qwen2VLError.allocationFailed("visual_pos_mask")
        }
        let s = C.imageStartIndex
        let e = s + C.numVisualTokens
        for i in 0..<promptLen {
            mask[i] = NSNumber(value: (i >= s && i < e) ? Float(1.0) : Float(0.0))
        }
        return mask
    }

    // MARK: - Greedy argmax

    private func argmax(_ logits: MLMultiArray) -> Int32 {
        let count = logits.count
        var bestIdx = 0
        var bestVal = Float(truncating: logits[0])
        for i in 1..<count {
            let v = Float(truncating: logits[i])
            if v > bestVal { bestVal = v; bestIdx = i }
        }
        return Int32(bestIdx)
    }

    // MARK: - Prompt template

    private func buildPromptTokenIDs() -> [Int32] {
        guard let tokenizer = QwenBPETokenizer.shared else {
            return buildHardcodedPromptTokenIDs()
        }

        let systemText = "You are a helpful assistant."
        // Must match USER_PROMPT in scripts/convert_qwen2vl_to_coreml.py exactly
        // (same string → same 67 tokens → same total 285 tokens the prefill model expects)
        let userText   = "Describe this image in comprehensive detail for fine-grained image retrieval. " +
            "Include: all visible people (count, appearance, clothing, activity, expression), " +
            "all objects and their positions, scene type (indoor/outdoor), setting/venue, " +
            "background elements, dominant colors, lighting conditions, time of day, " +
            "composition, and any visible text."

        var ids: [Int32] = []
        ids.append(C.imStartID)
        ids.append(contentsOf: tokenizer.encode("system"))
        ids.append(C.newlineID)
        ids.append(contentsOf: tokenizer.encode(systemText))
        ids.append(C.imEndID)
        ids.append(C.newlineID)

        ids.append(C.imStartID)
        ids.append(contentsOf: tokenizer.encode("user"))
        ids.append(C.newlineID)
        ids.append(C.visionStartID)
        ids.append(contentsOf: [Int32](repeating: C.imagePadID, count: C.numVisualTokens))
        ids.append(C.visionEndID)
        ids.append(C.newlineID)
        ids.append(contentsOf: tokenizer.encode(userText))
        ids.append(C.imEndID)
        ids.append(C.newlineID)

        ids.append(C.imStartID)
        ids.append(contentsOf: tokenizer.encode("assistant"))
        ids.append(C.newlineID)
        return ids
    }

    /// Fallback with pre-computed token IDs when the BPE tokenizer is unavailable.
    /// These IDs correspond exactly to the 285-token prompt used during conversion.
    /// Generated by running build_prompt_token_ids() with the Qwen3-VL-2B-Instruct tokenizer.
    private func buildHardcodedPromptTokenIDs() -> [Int32] {
        // Prefix: <|im_start|>(151644) system(8948) \n(198)
        //         You(2610) are(525) a(264) helpful(10950) assistant(17847) .(13)
        //         <|im_end|>(151645) \n(198)
        //         <|im_start|>(151644) user(872) \n(198) <|vision_start|>(151652)
        var ids: [Int32] = [
            151644, 8948, 198,
            2610, 525, 264, 10950, 17847, 13,
            151645, 198,
            151644, 872, 198,
            151652
        ]
        // 196 × <|image_pad|>
        ids.append(contentsOf: [Int32](repeating: 151655, count: C.numVisualTokens))
        // Suffix: <|vision_end|>(151653) \n(198)
        // "Describe this image in comprehensive detail ..." (67 tokens)
        // <|im_end|>(151645) \n(198) <|im_start|>(151644) assistant(77091) \n(198)
        ids.append(contentsOf: [
            151653, 198,
            74785, 419, 2168, 304, 15817, 7716, 369, 6915, 24321, 2627, 2168, 56370, 13,
            29734, 25, 678, 9434, 1251, 320, 1830, 11, 11094, 11, 17438, 11, 5702, 11, 7493, 701,
            678, 6171, 323, 862, 9892, 11, 6109, 943, 320, 484, 10692, 48316, 10787, 701,
            6243, 14, 7140, 11, 4004, 5424, 11, 24456, 7987, 11, 17716, 4682, 11,
            882, 315, 1899, 11, 18037, 11, 323, 894, 9434, 1467, 13,
            151645, 198, 151644, 77091, 198
        ])
        return ids
    }
}

// MARK: - Qwen3VLImagePreprocessor

enum Qwen3VLImagePreprocessor {
    /// Converts a UIImage into the flat patch tensor expected by Qwen3-VL's ViT.
    ///
    /// Qwen3-VL patch format (per patch):
    ///   channels(3) × temporal_patch_size(2) × patch_h(16) × patch_w(16) = 1536 values
    ///
    /// Returns:
    ///   pixel_values  MLMultiArray  [N_patches, 1536]  float32
    ///   grid_thw      MLMultiArray  [1, 3]             int32
    static func preprocess(
        _ image: UIImage,
        patchesPerSide: Int,
        patchSize: Int,
        temporalPatchSize: Int
    ) throws -> (pixelValues: MLMultiArray, gridTHW: MLMultiArray) {
        let targetSide = patchesPerSide * patchSize          // 28 × 16 = 448
        let targetSize = CGSize(width: targetSide, height: targetSide)

        guard let cgImage = image.cgImage else {
            throw Qwen2VLError.imageConversionFailed
        }
        let bytesPerPixel = 4
        let rowBytes = targetSide * bytesPerPixel
        var rgba = [UInt8](repeating: 0, count: targetSide * rowBytes)
        let space = CGColorSpaceCreateDeviceRGB()
        guard let ctx = CGContext(
            data: &rgba,
            width: targetSide, height: targetSide,
            bitsPerComponent: 8, bytesPerRow: rowBytes,
            space: space,
            bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue
        ) else { throw Qwen2VLError.imageConversionFailed }
        ctx.draw(cgImage, in: CGRect(origin: .zero, size: targetSize))

        // ImageNet mean/std normalisation
        let mean: (Float, Float, Float) = (0.48145466, 0.4578275,  0.40821073)
        let std:  (Float, Float, Float) = (0.26862954, 0.26130258, 0.27577711)

        let N            = patchesPerSide * patchesPerSide
        let patchFlatSize = 3 * temporalPatchSize * patchSize * patchSize

        let pixelValues = try MLMultiArray(
            shape: [NSNumber(value: N), NSNumber(value: patchFlatSize)],
            dataType: .float32
        )

        // Fill: [patch_idx, channel * temporal_patch_size * patch_h * patch_w]
        var patchIndex = 0
        for py in 0..<patchesPerSide {
            for px in 0..<patchesPerSide {
                var flatIdx = patchIndex * patchFlatSize
                for ch in 0..<3 {
                    let m = ch == 0 ? mean.0 : (ch == 1 ? mean.1 : mean.2)
                    let s = ch == 0 ? std.0  : (ch == 1 ? std.1  : std.2)
                    for _ in 0..<temporalPatchSize {   // duplicate frame
                        for ky in 0..<patchSize {
                            for kx in 0..<patchSize {
                                let pixY   = py * patchSize + ky
                                let pixX   = px * patchSize + kx
                                let offset = (pixY * targetSide + pixX) * bytesPerPixel + ch
                                let raw    = Float(rgba[offset]) / 255.0
                                pixelValues[flatIdx] = NSNumber(value: (raw - m) / s)
                                flatIdx += 1
                            }
                        }
                    }
                }
                patchIndex += 1
            }
        }

        let gridTHW = try MLMultiArray(shape: [1, 3], dataType: .int32)
        gridTHW[0] = 1
        gridTHW[1] = NSNumber(value: patchesPerSide)
        gridTHW[2] = NSNumber(value: patchesPerSide)

        return (pixelValues, gridTHW)
    }
}

// MARK: - Qwen2VLTokenDecoder

enum Qwen2VLTokenDecoder {
    private static let decodeTable: [Int32: String] = {
        guard let tok = QwenBPETokenizer.shared else { return [:] }
        return tok.reverseVocabulary()
    }()

    private static let byteDecoder: [Character: UInt8] = {
        var bs = Array(UInt8(33)...UInt8(126))
        bs += Array(UInt8(161)...UInt8(172))
        bs += Array(UInt8(174)...UInt8(255))
        var cs = bs.map(Int.init)
        var next = 0
        for byte in UInt8.min...UInt8.max where !bs.contains(byte) {
            bs.append(byte); cs.append(256 + next); next += 1
        }
        var result: [Character: UInt8] = [:]
        for (byte, scalar) in zip(bs, cs) {
            if let uc = Unicode.Scalar(scalar) {
                result[Character(uc)] = byte
            }
        }
        return result
    }()

    static func decode(_ tokenIDs: [Int32]) -> String {
        var bytes: [UInt8] = []
        for id in tokenIDs {
            guard let tokenStr = decodeTable[id] else { continue }
            for ch in tokenStr {
                if let byte = byteDecoder[ch] { bytes.append(byte) }
            }
        }
        return String(bytes: bytes, encoding: .utf8) ?? ""
    }
}

// MARK: - Errors

enum Qwen2VLError: Error, LocalizedError {
    case missingOutput(String)
    case imageConversionFailed
    case modelNotLoaded
    case allocationFailed(String)

    var errorDescription: String? {
        switch self {
        case .missingOutput(let name):  return "Qwen2VL: missing model output '\(name)'"
        case .imageConversionFailed:    return "Qwen2VL: failed to convert image to patch tensor"
        case .modelNotLoaded:           return "Qwen2VL: one or more CoreML models are not loaded"
        case .allocationFailed(let name): return "Qwen2VL: failed to allocate '\(name)'"
        }
    }
}
