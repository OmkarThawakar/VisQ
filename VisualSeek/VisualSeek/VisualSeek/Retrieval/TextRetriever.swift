import Foundation

struct RetrievalResult: Identifiable {
    let id: String
    let title: String?
    let score: Float
    let matchExplanation: MatchExplanation?
}

struct MatchExplanation: Codable, Hashable {
    let summary: String
    let featureChips: [String]
    let reasonChips: [String]
    let detail: String
}

enum MatchExplanationFactory {
    private static let stopWords: Set<String> = [
        "a", "an", "and", "are", "as", "at", "be", "for", "from", "has", "in", "into",
        "is", "it", "its", "of", "on", "or", "that", "the", "this", "to", "was", "with",
        "your", "their", "there", "here", "then", "than", "over", "under", "near",
        "show", "shows", "photo", "image", "video", "clip", "scene", "frame"
    ]
    private static let lightingTerms: Set<String> = [
        "backlit", "bright", "dark", "dusk", "glow", "golden", "light", "lighting",
        "lit", "moody", "neon", "night", "shadow", "shadowy", "silhouette", "sunrise",
        "sunset", "warm"
    ]
    private static let colorTerms: Set<String> = [
        "amber", "blue", "color", "colour", "cool", "gold", "green", "orange",
        "pastel", "pink", "purple", "red", "teal", "vivid", "yellow"
    ]
    private static let sceneTerms: Set<String> = [
        "beach", "building", "car", "city", "coast", "desert", "forest", "indoor",
        "interior", "kitchen", "lake", "mountain", "ocean", "outdoor", "park", "rain",
        "road", "room", "sea", "sky", "snow", "street", "sunset", "water"
    ]
    private static let poseTerms: Set<String> = [
        "arm", "child", "dance", "face", "figure", "hand", "jump", "man", "motion",
        "move", "people", "person", "portrait", "pose", "profile", "run", "sit",
        "silhouette", "stand", "walk", "woman"
    ]
    private static let moodTerms: Set<String> = [
        "calm", "cinematic", "dramatic", "dreamy", "energetic", "gentle", "mood",
        "moody", "quiet", "romantic", "serene", "soft"
    ]

    static func textMatch(query: String, documentText: String) -> MatchExplanation {
        let featureChips = overlappingDisplayChips(queryText: query, documentText: documentText)
        let reasonChips = reasonChips(context: [query, documentText].joined(separator: " "), usesReferencePose: false)
        let summary = summaryText(featureChips: featureChips, reasonChips: reasonChips, usesReferenceLanguage: false)
        let detail = detailText(featureChips: featureChips, reasonChips: reasonChips, usesReferenceLanguage: false)

        return MatchExplanation(
            summary: summary,
            featureChips: featureChips,
            reasonChips: reasonChips,
            detail: detail
        )
    }

    static func composedMatch(editText: String, reasoning: ReasoningTrace, documentText: String) -> MatchExplanation {
        let targetDescription = [
            editText,
            reasoning.states,
            reasoning.actions,
            reasoning.scene,
            reasoning.camera,
            reasoning.tempo
        ].joined(separator: " ")

        let featureChips = overlappingDisplayChips(queryText: targetDescription, documentText: documentText)
        let reasonChips = reasonChips(context: targetDescription + " " + documentText, usesReferencePose: true)
        let summary = summaryText(featureChips: featureChips, reasonChips: reasonChips, usesReferenceLanguage: true)
        let detail = detailText(featureChips: featureChips, reasonChips: reasonChips, usesReferenceLanguage: true)

        return MatchExplanation(
            summary: summary,
            featureChips: featureChips,
            reasonChips: reasonChips,
            detail: detail
        )
    }

    static func excerpt(from description: String) -> String? {
        let trimmed = description.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return nil }

        if let sentenceEnd = trimmed.firstIndex(of: ".") {
            return String(trimmed[..<sentenceEnd])
        }

        return String(trimmed.prefix(96))
    }

    private static func overlappingDisplayChips(queryText: String, documentText: String, limit: Int = 4) -> [String] {
        let queryTokens = displayTokens(from: queryText)
        let documentTokens = Set(displayTokens(from: documentText).map(normalizeToken))
        let documentNormalizedText = normalizedText(from: documentText)

        var chips: [String] = []
        let queryBigrams = zip(queryTokens, queryTokens.dropFirst()).map { [$0.0, $0.1] }
        for bigram in queryBigrams {
            let normalizedBigram = bigram.map(normalizeToken).joined(separator: " ")
            guard documentNormalizedText.contains(normalizedBigram) else { continue }
            appendChip(formatChip(bigram.joined(separator: " ")), to: &chips, limit: limit)
        }

        for token in queryTokens where documentTokens.contains(normalizeToken(token)) {
            appendChip(formatChip(token), to: &chips, limit: limit)
        }

        if chips.isEmpty {
            for token in queryTokens.prefix(limit) {
                appendChip(formatChip(token), to: &chips, limit: limit)
            }
        }

        return chips
    }

    private static func reasonChips(context: String, usesReferencePose: Bool) -> [String] {
        let tokens = Set(displayTokens(from: context).map(normalizeToken))
        var reasons: [String] = []

        if !tokens.intersection(poseTerms).isEmpty {
            reasons.append(usesReferencePose ? "reference pose" : "pose")
        }
        if !tokens.intersection(lightingTerms.union(colorTerms)).isEmpty {
            reasons.append("color")
        }
        if !tokens.intersection(sceneTerms).isEmpty {
            reasons.append("scene")
        }
        if !tokens.intersection(moodTerms).isEmpty {
            reasons.append("mood")
        }

        if reasons.isEmpty {
            reasons.append(usesReferencePose ? "reference scene" : "visual overlap")
        }

        return reasons
    }

    private static func summaryText(featureChips: [String], reasonChips: [String], usesReferenceLanguage: Bool) -> String {
        let reasons = naturalLanguageList(reasonChips)
        let features = naturalLanguageList(featureChips)

        if !reasonChips.isEmpty, !featureChips.isEmpty {
            let prefix = usesReferenceLanguage ? "Matched because of \(reasons)" : "Matched because of \(reasons)"
            return "\(prefix), with overlap on \(features)."
        }

        if !reasonChips.isEmpty {
            return "Matched because of \(reasons)."
        }

        return "Matched on \(features)."
    }

    private static func detailText(featureChips: [String], reasonChips: [String], usesReferenceLanguage: Bool) -> String {
        let features = naturalLanguageList(featureChips)
        let reasons = naturalLanguageList(reasonChips)

        if !featureChips.isEmpty, !reasonChips.isEmpty {
            let lead = usesReferenceLanguage
                ? "The retrieved media stayed close to the reference-guided target on \(features)."
                : "The indexed description overlaps with your request on \(features)."
            return "\(lead) It also aligned on \(reasons)."
        }

        if !featureChips.isEmpty {
            return usesReferenceLanguage
                ? "The retrieved media stayed close to the reference-guided target on \(features)."
                : "The indexed description overlaps with your request on \(features)."
        }

        return usesReferenceLanguage
            ? "The result aligned with the reference-guided target description."
            : "The result aligned with the strongest terms in your request."
    }

    private static func displayTokens(from text: String) -> [String] {
        text
            .lowercased()
            .replacingOccurrences(of: "[^a-z0-9\\s]", with: " ", options: .regularExpression)
            .split(whereSeparator: \.isWhitespace)
            .map(String.init)
            .filter { token in
                token.count > 2 && !stopWords.contains(token)
            }
    }

    private static func normalizedText(from text: String) -> String {
        displayTokens(from: text)
            .map(normalizeToken)
            .joined(separator: " ")
    }

    private static func normalizeToken(_ token: String) -> String {
        if token.count > 5, token.hasSuffix("ing") {
            return String(token.dropLast(3))
        }

        if token.count > 4, token.hasSuffix("ed") {
            return String(token.dropLast(2))
        }

        if token.count > 4, token.hasSuffix("ies") {
            return String(token.dropLast(3)) + "y"
        }

        if token.count > 4,
           token.hasSuffix("ches") || token.hasSuffix("shes") ||
           token.hasSuffix("sses") || token.hasSuffix("xes") ||
           token.hasSuffix("zes") {
            return String(token.dropLast(2))
        }

        if token.count > 3, token.hasSuffix("s"), !token.hasSuffix("ss") {
            return String(token.dropLast())
        }

        return token
    }

    private static func formatChip(_ chip: String) -> String {
        chip
            .split(whereSeparator: \.isWhitespace)
            .prefix(3)
            .joined(separator: " ")
    }

    private static func appendChip(_ chip: String, to chips: inout [String], limit: Int) {
        guard !chip.isEmpty, chips.count < limit else { return }
        guard !chips.contains(chip) else { return }
        chips.append(chip)
    }

    private static func naturalLanguageList(_ items: [String]) -> String {
        switch items.count {
        case 0:
            return "the strongest overlap"
        case 1:
            return items[0]
        case 2:
            return "\(items[0]) and \(items[1])"
        default:
            let head = items.dropLast().joined(separator: ", ")
            return "\(head), and \(items[items.count - 1])"
        }
    }
}

final class TextRetriever {
    private let store: EmbeddingStore
    private let stopWords: Set<String> = [
        "a", "an", "and", "are", "as", "at", "be", "for", "from", "has", "in", "into",
        "is", "it", "its", "of", "on", "or", "that", "the", "this", "to", "was", "with"
    ]

    init(store: EmbeddingStore) {
        self.store = store
    }

    func search(query: String, topK: Int = 20) async throws -> [RetrievalResult] {
        let queryFeatures = features(for: query)
        guard !queryFeatures.termFrequencies.isEmpty else { return [] }

        let records = try store.fetchAllEmbeddings()
        let documents = records.map { record in
            SearchDocument(record: record, features: features(for: record.imageDescription))
        }

        let documentFrequency = buildDocumentFrequency(from: documents)
        let documentCount = max(documents.count, 1)

        let scoredResults = documents.compactMap { document -> RetrievalResult? in
            let score = score(query: queryFeatures, document: document.features, documentFrequency: documentFrequency, documentCount: documentCount)
            guard score > 0.08 else { return nil }

            return RetrievalResult(
                id: document.record.assetLocalIdentifier,
                title: MatchExplanationFactory.excerpt(from: document.record.imageDescription),
                score: score,
                matchExplanation: MatchExplanationFactory.textMatch(
                    query: query,
                    documentText: document.record.imageDescription
                )
            )
        }

        return scoredResults
            .sorted { lhs, rhs in
                if lhs.score == rhs.score {
                    return lhs.id < rhs.id
                }
                return lhs.score > rhs.score
            }
            .prefix(topK)
            .map { $0 }
    }

    private func score(
        query: TextFeatures,
        document: TextFeatures,
        documentFrequency: [String: Int],
        documentCount: Int
    ) -> Float {
        let cosine = cosineSimilarity(query: query, document: document, documentFrequency: documentFrequency, documentCount: documentCount)
        let overlapCount = query.uniqueTerms.intersection(document.uniqueTerms).count
        let coverage = Float(overlapCount) / Float(max(query.uniqueTerms.count, 1))
        let phraseBoost: Float = document.normalizedText.contains(query.normalizedText) ? 0.18 : 0
        let bigramOverlap = jaccardSimilarity(query.bigrams, document.bigrams)

        return min(1.0, cosine * 0.62 + coverage * 0.23 + bigramOverlap * 0.15 + phraseBoost)
    }

    private func cosineSimilarity(
        query: TextFeatures,
        document: TextFeatures,
        documentFrequency: [String: Int],
        documentCount: Int
    ) -> Float {
        var numerator: Float = 0
        var queryMagnitude: Float = 0
        var documentMagnitude: Float = 0

        let queryWeights = tfidfWeights(for: query.termFrequencies, documentFrequency: documentFrequency, documentCount: documentCount)
        let documentWeights = tfidfWeights(for: document.termFrequencies, documentFrequency: documentFrequency, documentCount: documentCount)

        for (term, queryWeight) in queryWeights {
            queryMagnitude += queryWeight * queryWeight
            if let documentWeight = documentWeights[term] {
                numerator += queryWeight * documentWeight
            }
        }

        for documentWeight in documentWeights.values {
            documentMagnitude += documentWeight * documentWeight
        }

        guard numerator > 0, queryMagnitude > 0, documentMagnitude > 0 else { return 0 }
        return numerator / (sqrt(queryMagnitude) * sqrt(documentMagnitude))
    }

    private func tfidfWeights(
        for termFrequencies: [String: Int],
        documentFrequency: [String: Int],
        documentCount: Int
    ) -> [String: Float] {
        var weights: [String: Float] = [:]

        for (term, frequency) in termFrequencies {
            let tf = 1 + logf(Float(frequency))
            let df = Float(documentFrequency[term] ?? 0)
            let idf = logf((Float(documentCount) + 1) / (df + 1)) + 1
            weights[term] = tf * idf
        }

        return weights
    }

    private func buildDocumentFrequency(from documents: [SearchDocument]) -> [String: Int] {
        var frequency: [String: Int] = [:]
        for document in documents {
            for term in document.features.uniqueTerms {
                frequency[term, default: 0] += 1
            }
        }
        return frequency
    }

    private func features(for text: String) -> TextFeatures {
        let normalizedText = normalizedText(from: text)
        let tokens = normalizedText
            .split(separator: " ")
            .map(String.init)
            .filter { !stopWords.contains($0) }

        var termFrequencies: [String: Int] = [:]
        for token in tokens {
            termFrequencies[token, default: 0] += 1
        }

        let bigrams = Set(zip(tokens, tokens.dropFirst()).map { "\($0.0) \($0.1)" })

        return TextFeatures(
            normalizedText: normalizedText,
            termFrequencies: termFrequencies,
            uniqueTerms: Set(tokens),
            bigrams: bigrams
        )
    }

    private func normalizedText(from text: String) -> String {
        text
            .lowercased()
            .replacingOccurrences(of: "[^a-z0-9\\s]", with: " ", options: .regularExpression)
            .split(whereSeparator: \.isWhitespace)
            .map { normalizeToken(String($0)) }
            .filter { !$0.isEmpty }
            .joined(separator: " ")
    }

    private func normalizeToken(_ token: String) -> String {
        if token.count > 5, token.hasSuffix("ing") {
            return String(token.dropLast(3))
        }

        if token.count > 4, token.hasSuffix("ed") {
            return String(token.dropLast(2))
        }

        if token.count > 3, token.hasSuffix("es") {
            return String(token.dropLast(2))
        }

        if token.count > 3, token.hasSuffix("s") {
            return String(token.dropLast())
        }

        return token
    }

    private func jaccardSimilarity(_ lhs: Set<String>, _ rhs: Set<String>) -> Float {
        guard !lhs.isEmpty, !rhs.isEmpty else { return 0 }
        let intersection = lhs.intersection(rhs).count
        let union = lhs.union(rhs).count
        return Float(intersection) / Float(max(union, 1))
    }
}

private struct SearchDocument {
    let record: EmbeddingRecord
    let features: TextFeatures
}

private struct TextFeatures {
    let normalizedText: String
    let termFrequencies: [String: Int]
    let uniqueTerms: Set<String>
    let bigrams: Set<String>
}
