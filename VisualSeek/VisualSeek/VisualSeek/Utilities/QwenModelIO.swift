import Foundation
import CoreML

enum QwenModelIOError: Error {
    case outputMissing(String)
    case unsupportedArrayType
}

enum QwenTextInputEncoder {
    static let maxTokens = 32
    static let vocabSize: Int32 = 151_936
    static let bosToken: Int32 = 151_644
    static let eosToken: Int32 = 151_645
    static let padToken: Int32 = 151_643

    static func encode(_ text: String, maxTokens: Int = maxTokens) -> (ids: [Int32], mask: [Int32]) {
        var ids: [Int32] = [bosToken]
        if let tokenizer = QwenBPETokenizer.shared {
            ids.append(contentsOf: tokenizer.encode(text))
        } else {
            let normalized = text.lowercased()
            let pieces = normalized.split { !$0.isLetter && !$0.isNumber }
            for piece in pieces.prefix(max(0, maxTokens - 2)) {
                ids.append(stableTokenID(for: String(piece)))
            }
        }
        ids.append(eosToken)

        if ids.count > maxTokens {
            ids = Array(ids.prefix(maxTokens))
            ids[maxTokens - 1] = eosToken
        }

        let activeCount = ids.count
        if ids.count < maxTokens {
            ids.append(contentsOf: Array(repeating: padToken, count: maxTokens - ids.count))
        }

        let mask = (0..<maxTokens).map { $0 < activeCount ? Int32(1) : Int32(0) }
        return (ids, mask)
    }

    private static func stableTokenID(for token: String) -> Int32 {
        var hash: UInt64 = 1469598103934665603
        for byte in token.utf8 {
            hash ^= UInt64(byte)
            hash &*= 1099511628211
        }
        let usableRange = UInt64(vocabSize - 2048)
        return Int32(hash % usableRange) + 2048
    }
}

enum QwenArrayFactory {
    static func makeFloatArray(_ values: [Float], shape: [Int]) throws -> MLMultiArray {
        let array = try MLMultiArray(shape: shape as [NSNumber], dataType: .float32)
        for (index, value) in values.enumerated() {
            array[index] = NSNumber(value: value)
        }
        return array
    }

    static func makeInt32Array(_ values: [Int32], shape: [Int]) throws -> MLMultiArray {
        let array = try MLMultiArray(shape: shape as [NSNumber], dataType: .int32)
        for (index, value) in values.enumerated() {
            array[index] = NSNumber(value: value)
        }
        return array
    }

    static func extractFloats(from featureProvider: MLFeatureProvider, featureName: String) throws -> [Float] {
        guard let array = featureProvider.featureValue(for: featureName)?.multiArrayValue else {
            throw QwenModelIOError.outputMissing(featureName)
        }
        return try extractFloats(from: array)
    }

    static func extractFloats(from array: MLMultiArray) throws -> [Float] {
        switch array.dataType {
        case .float16:
            let pointer = UnsafeMutablePointer<Float16>(OpaquePointer(array.dataPointer))
            return (0..<array.count).map { Float(pointer[$0]) }
        case .float32:
            let pointer = UnsafeMutablePointer<Float32>(OpaquePointer(array.dataPointer))
            return (0..<array.count).map { Float(pointer[$0]) }
        case .double:
            let pointer = UnsafeMutablePointer<Double>(OpaquePointer(array.dataPointer))
            return (0..<array.count).map { Float(pointer[$0]) }
        default:
            throw QwenModelIOError.unsupportedArrayType
        }
    }

    static func normalize(_ values: [Float]) -> [Float] {
        let norm = sqrt(values.reduce(0) { $0 + $1 * $1 })
        guard norm > 0 else { return values }
        return values.map { $0 / norm }
    }
}
