import Foundation
import Testing
import UIKit
@testable import VisualSeek

@MainActor
struct ProductionReadinessTests {

    @Test func tokenizerLoadsAndEncodesTextWithoutCrashing() {
        let tokenizer = QwenBPETokenizer.shared
        #expect(tokenizer != nil)
        #expect(!(tokenizer?.encode("A bright beach portrait with ocean in the background").isEmpty ?? true))
    }

    @Test func backgroundTaskIdentifiersUseProductionBundlePrefix() {
        #expect(BackgroundTaskIdentifier.indexing == "omkar.VisualSeek.indexing")
        #expect(BackgroundTaskIdentifier.processing == "omkar.VisualSeek.processing")
    }

    @Test func embeddingStoreRoundTripsAndUpdatesDescriptions() async throws {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString, isDirectory: true)
        let storeURL = root.appendingPathComponent("embeddings.sqlite")
        let store = EmbeddingStore(storageURL: storeURL)

        let embedding = (0..<1536).map { i in Float(i % 13) / 13.0 }

        try await store.saveEmbedding(
            assetId: "asset-1",
            assetType: 1,
            creationDate: Date(timeIntervalSince1970: 1_700_000_000),
            embedding: embedding,
            imageDescription: "Original description"
        )

        #expect(try store.countIndexedPhotos() == 1)

        var records = try store.fetchAllEmbeddings()
        #expect(records.count == 1)
        #expect(records[0].assetLocalIdentifier == "asset-1")
        #expect(records[0].imageDescription == "Original description")
        #expect(records[0].embedding.count == embedding.count * MemoryLayout<Float>.size)

        try store.updateDescription(assetId: "asset-1", description: "Updated description")
        records = try store.fetchAllEmbeddings()
        #expect(records[0].imageDescription == "Updated description")

        try store.clearAllEmbeddings()
        #expect(try store.countIndexedPhotos() == 0)
    }
}
