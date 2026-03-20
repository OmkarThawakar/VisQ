import Foundation
import os

private final class QwenTokenizerBundleMarker {}

final class QwenBPETokenizer {
    static let shared = QwenBPETokenizer()

    private let encoder: [String: Int32]
    private let bpeRanks: [String: Int]
    private let regex: NSRegularExpression
    private let byteEncoder: [UInt8: String]
    private var cache: [String: [String]] = [:]
    private let lock = NSLock()

    private init?() {
        let bundle = Bundle.main.url(forResource: "vocab", withExtension: "json", subdirectory: "Tokenizer") != nil
            ? Bundle.main
            : Bundle(for: QwenTokenizerBundleMarker.self)

        guard
            let vocabURL = bundle.url(forResource: "vocab", withExtension: "json", subdirectory: "Tokenizer"),
            let mergesURL = bundle.url(forResource: "merges", withExtension: "txt", subdirectory: "Tokenizer"),
            let vocabData = try? Data(contentsOf: vocabURL),
            let vocabObject = try? JSONSerialization.jsonObject(with: vocabData) as? [String: Int]
        else {
            return nil
        }

        var convertedEncoder: [String: Int32] = [:]
        convertedEncoder.reserveCapacity(vocabObject.count)
        for (key, value) in vocabObject {
            convertedEncoder[key] = Int32(value)
        }
        encoder = convertedEncoder

        let mergesLines = (try? String(contentsOf: mergesURL, encoding: .utf8).components(separatedBy: .newlines)) ?? []
        var ranks: [String: Int] = [:]
        for (index, line) in mergesLines.enumerated() where !line.isEmpty && !line.hasPrefix("#") {
            ranks[line] = index
        }
        bpeRanks = ranks

        guard let regex = try? NSRegularExpression(
            pattern: "'s|'t|'re|'ve|'m|'ll|'d| ?\\p{L}+| ?\\p{N}+| ?[^\\s\\p{L}\\p{N}]+|\\s+(?!\\S)|\\s+",
            options: []
        ) else {
            AppLog.model.error("Failed to initialize Qwen tokenizer regex")
            return nil
        }
        self.regex = regex
        byteEncoder = QwenBPETokenizer.makeByteEncoder()
    }

    /// Returns a token-ID → BPE-token-string reverse mapping, used by Qwen2VLTokenDecoder
    /// to convert generated token IDs back into UTF-8 text.
    func reverseVocabulary() -> [Int32: String] {
        var rev = [Int32: String](minimumCapacity: encoder.count)
        for (token, id) in encoder { rev[id] = token }
        return rev
    }

    func encode(_ text: String) -> [Int32] {
        let normalized = text.precomposedStringWithCompatibilityMapping
        let range = NSRange(normalized.startIndex..<normalized.endIndex, in: normalized)
        let matches = regex.matches(in: normalized, options: [], range: range)

        var output: [Int32] = []
        output.reserveCapacity(matches.count * 2)

        for match in matches {
            guard let matchRange = Range(match.range, in: normalized) else { continue }
            let token = String(normalized[matchRange])
            let transformed = token.utf8.map { byteEncoder[$0] ?? "" }.joined()
            for piece in bpe(transformed) {
                if let value = encoder[piece] {
                    output.append(value)
                }
            }
        }

        return output
    }

    private func bpe(_ token: String) -> [String] {
        lock.lock()
        if let cached = cache[token] {
            lock.unlock()
            return cached
        }
        lock.unlock()

        var word = token.map(String.init)
        guard word.count > 1 else {
            lock.lock()
            cache[token] = word
            lock.unlock()
            return word
        }

        while true {
            let pairs = adjacentPairs(in: word)
            guard !pairs.isEmpty else { break }

            var bestPair: (left: String, right: String)?
            var bestRank = Int.max

            for pair in pairs {
                let key = "\(pair.left) \(pair.right)"
                if let rank = bpeRanks[key], rank < bestRank {
                    bestRank = rank
                    bestPair = pair
                }
            }

            guard let bestPair else { break }

            var merged: [String] = []
            var index = 0
            while index < word.count {
                if index < word.count - 1, word[index] == bestPair.left, word[index + 1] == bestPair.right {
                    merged.append(bestPair.left + bestPair.right)
                    index += 2
                } else {
                    merged.append(word[index])
                    index += 1
                }
            }

            word = merged
            if word.count == 1 { break }
        }

        lock.lock()
        cache[token] = word
        lock.unlock()
        return word
    }

    private func adjacentPairs(in word: [String]) -> [(left: String, right: String)] {
        guard word.count > 1 else { return [] }
        return (0..<(word.count - 1)).map { (word[$0], word[$0 + 1]) }
    }

    private static func makeByteEncoder() -> [UInt8: String] {
        var bs = Array(UInt8(33)...UInt8(126))
        bs += Array(UInt8(161)...UInt8(172))
        bs += Array(UInt8(174)...UInt8(255))

        var cs = bs.map(Int.init)
        var next = 0
        for byte in UInt8.min...UInt8.max where !bs.contains(byte) {
            bs.append(byte)
            cs.append(256 + next)
            next += 1
        }

        var table: [UInt8: String] = [:]
        for (byte, scalarValue) in zip(bs, cs) {
            if let scalar = UnicodeScalar(scalarValue) {
                table[byte] = String(scalar)
            }
        }
        return table
    }
}
