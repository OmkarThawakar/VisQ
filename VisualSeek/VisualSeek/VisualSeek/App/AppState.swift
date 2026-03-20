import Foundation
import Combine
import SwiftUI
import Photos
import UIKit

@MainActor
class AppState: ObservableObject {
    @Published var configuration = AppConfiguration()
    @Published var isIndexing = false
    @Published var indexedPhotoCount = 0
    @Published var totalPhotoCount = 0
    @Published var selectedPhotoCount = 0
    @Published var selectedIndexedCount = 0
    @Published var selectedAssetIdentifiers: [String] = []
    @Published var indexingProgress: Float = 0.0
    @Published var isModelLoading = false
    @Published var modelStatusMessage = "Loading models..."
    @Published var lastErrorMessage: String?
    @Published var diagnosticsReport: String?
    @Published var isRunningDiagnostics = false
    @Published var lastIndexingDuration: TimeInterval?
    @Published var lastTextSearchDuration: TimeInterval?
    @Published var lastComposedSearchDuration: TimeInterval?
    @Published var indexingStatusMessage = "Select photos or videos to begin indexing."
    @Published var usingBundledModels = false
    @Published var indexedRecords: [EmbeddingRecord] = []
    @Published var isGeneratingDescriptions = false
    @Published var descriptionGenerationMessage = ""

    let scanner = PhotoLibraryScanner()
    let embeddingStore: EmbeddingStore
    let vectorIndex: VectorIndex
    let modelLoader: ModelLoader

    private(set) var visionEncoder: QwenVisionEncoderProtocol?
    private(set) var languageModel: QwenLanguageModel?
    private(set) var batchProcessor: BatchProcessor?
    private(set) var indexingPipeline: IndexingPipeline?
    private(set) var textRetriever: TextRetriever?
    private(set) var composedRetriever: ComposedRetriever?
    private(set) var qwen2VLPipeline: Qwen2VLInferencePipeline?
    private let descriptionGenerator = QwenVLDescriptionGenerator()
    private var indexingBaseSelectedIndexedCount = 0
    private var qwen2VLPipelineLoadTask: Task<Qwen2VLInferencePipeline?, Never>?

    init() {
        self.embeddingStore = EmbeddingStore(storageURL: AppConfiguration().embeddingStoragePath)
        let indexedCount = (try? embeddingStore.countIndexedPhotos()) ?? 0
        self.vectorIndex = VectorIndex(embeddingCount: indexedCount)
        self.modelLoader = ModelLoader(config: AppConfiguration())
        self.indexedPhotoCount = indexedCount

        Task {
            // Request photo-library authorization proactively so the system accounts
            // daemon settles before any indexing starts. Calling it here (at launch,
            // while the app is foregrounded) prevents the spurious
            // "com.apple.accounts Code=7" error that appears when authorization is
            // requested lazily during indexing.
            await prefetchPhotoAuthorization()
            await refreshLibraryStats()
            await refreshIndexedRecords()
            await loadRuntime()
        }
    }

    func loadRuntime() async {
        isModelLoading = true
        modelStatusMessage = "Loading runtime…"
        qwen2VLPipeline = nil
        qwen2VLPipelineLoadTask = nil
        descriptionGenerator.qwen2VLPipeline = nil
        descriptionGenerator.qwen2VLPipelineProvider = { [weak self] in
            await self?.loadQwen2VLPipelineIfNeeded()
        }

        let loadedModels = await modelLoader.loadRuntimeModels()
        let visionEncoder = loadedModels.visionEncoder
        let languageModel = loadedModels.languageModel
        let batchProcessor = BatchProcessor(batchSize: configuration.batchSize, visionEncoder: visionEncoder)
        // Pass visionEncoder into IndexingPipeline so it can generate rich descriptions
        let indexingPipeline = IndexingPipeline(
            scanner: scanner,
            store: embeddingStore,
            batchProcessor: batchProcessor,
            vectorIndex: vectorIndex,
            visionEncoder: visionEncoder
        )
        let textRetriever = TextRetriever(store: embeddingStore)
        let composedRetriever = ComposedRetriever(
            store: embeddingStore,
            vectorIndex: vectorIndex,
            visionEncoder: visionEncoder,
            languageModel: languageModel
        )

        self.visionEncoder = visionEncoder
        self.languageModel = languageModel
        self.batchProcessor = batchProcessor
        self.indexingPipeline = indexingPipeline
        self.textRetriever = textRetriever
        self.composedRetriever = composedRetriever
        bindIndexingPipeline(indexingPipeline)
        usingBundledModels = loadedModels.usingBundledModels

        if let firstRecord = try? embeddingStore.fetchAllEmbeddings().first,
           firstRecord.embedding.count != 768 * MemoryLayout<Float>.size {
            try? embeddingStore.clearAllEmbeddings()
            indexingStatusMessage = "Cleared incompatible cached embeddings. Please re-index your selected media."
        }

        try? vectorIndex.rebuild(from: embeddingStore)
        indexedPhotoCount = (try? embeddingStore.countIndexedPhotos()) ?? 0
        await refreshIndexedRecords()
        let genStatus = " + on-demand Qwen3-VL descriptions"
        modelStatusMessage = loadedModels.usingBundledModels
            ? "Qwen3-VL-2B Core ML runtime active\(genStatus)"
            : "Lightweight on-device runtime active (Qwen3-VL-2B Core ML unavailable)\(genStatus)"

        isModelLoading = false
    }

    private func loadQwen2VLPipelineIfNeeded() async -> Qwen2VLInferencePipeline? {
        if let qwen2VLPipeline {
            return qwen2VLPipeline
        }

        if let qwen2VLPipelineLoadTask {
            return await qwen2VLPipelineLoadTask.value
        }

        let task = Task { [modelLoader] in
            await modelLoader.loadQwen2VLPipeline()
        }
        qwen2VLPipelineLoadTask = task

        let pipeline = await task.value
        qwen2VLPipeline = pipeline
        descriptionGenerator.qwen2VLPipeline = pipeline
        qwen2VLPipelineLoadTask = nil
        return pipeline
    }

    func refreshLibraryStats() async {
        let fetchResult = PHAsset.fetchAssets(with: PHFetchOptions())
        totalPhotoCount = fetchResult.count
        indexedPhotoCount = (try? embeddingStore.countIndexedPhotos()) ?? 0
    }

    func refreshIndexedRecords() async {
        indexedRecords = (try? embeddingStore.fetchAllEmbeddings()) ?? []
    }

    func runTextSearch(_ query: String, topK: Int = 20) async throws -> [RetrievalResult] {
        guard let textRetriever else {
            throw NSError(domain: "AppState", code: -10, userInfo: [NSLocalizedDescriptionKey: "Text retriever is not ready yet"])
        }
        try vectorIndex.rebuild(from: embeddingStore)
        lastTextSearchDuration = nil
        let start = Date()
        let results = try await textRetriever.search(query: query, topK: topK)
        lastTextSearchDuration = Date().timeIntervalSince(start)
        return results
    }

    func runComposedSearch(referenceImage: UIImage, editText: String, topK: Int = 20) async throws -> (ReasoningTrace, [RetrievalResult]) {
        guard let composedRetriever, let languageModel else {
            throw NSError(domain: "AppState", code: -11, userInfo: [NSLocalizedDescriptionKey: "Composed retriever is not ready yet"])
        }

        let reasoningPrompt = """
        You are analyzing a visual modification request.
        Request edit: "\(editText)"
        Generate a structured analysis of what the TARGET image should contain:
        - Object states: What physical states change?
        - Action phases: What temporal progression is implied?
        - Scene changes: What environment modifications?
        - Camera/framing: Any cinematographic changes?
        - Tempo/mood: Pacing or emotional changes?
        """

        lastComposedSearchDuration = nil
        let start = Date()
        let reasoning = try await languageModel.generate(prompt: reasoningPrompt, image: referenceImage, maxTokens: configuration.reasoningMaxTokens)
        try vectorIndex.rebuild(from: embeddingStore)
        let results = try await composedRetriever.search(referenceImage: referenceImage, editText: editText, topK: topK)
        lastComposedSearchDuration = Date().timeIntervalSince(start)
        return (reasoning, results)
    }

    func startIndexing() async {
        guard let indexingPipeline else {
            lastErrorMessage = "Indexing pipeline is not ready yet."
            return
        }
        guard !selectedAssetIdentifiers.isEmpty else {
            lastErrorMessage = "Select photos or videos before starting indexing."
            return
        }

        isIndexing = true
        lastErrorMessage = nil
        lastIndexingDuration = nil
        indexingStatusMessage = "Starting indexing..."
        indexingBaseSelectedIndexedCount = selectedIndexedCount
        let start = Date()
        await indexingPipeline.startIndexing(selectedAssetIDs: selectedAssetIdentifiers)
        lastIndexingDuration = Date().timeIntervalSince(start)
        indexingProgress = indexingPipeline.progress
        isIndexing = indexingPipeline.isIndexing
        indexedPhotoCount = (try? embeddingStore.countIndexedPhotos()) ?? indexedPhotoCount
        await refreshIndexedRecords()
        await refreshSelectionStats()
        await refreshLibraryStats()
    }

    func pauseIndexing() {
        indexingPipeline?.pauseIndexing()
        isIndexing = false
        indexingStatusMessage = "Indexing paused."
    }

    func clearEmbeddings() {
        do {
            try embeddingStore.clearAllEmbeddings()
            indexedPhotoCount = 0
            indexedRecords = []
            try? vectorIndex.rebuild(from: embeddingStore)
            Task { await refreshSelectionStats() }
            diagnosticsReport = "Embedding store cleared."
        } catch {
            lastErrorMessage = "Failed to clear embeddings: \(error.localizedDescription)"
        }
    }

    /// Regenerates rich VL descriptions for any already-indexed photos that
    /// only have a short rule-based caption (< 80 chars). Does NOT re-embed,
    /// so it is fast even for large libraries.
    func regenerateDescriptions() async {
        guard !isGeneratingDescriptions, !isIndexing else { return }
        isGeneratingDescriptions = true
        descriptionGenerationMessage = "Finding indexed assets to re-describe…"
        lastErrorMessage = nil

        do {
            let staleRecords = try embeddingStore.fetchRecordsWithEmptyDescriptions(minLength: 80)
            guard !staleRecords.isEmpty else {
                descriptionGenerationMessage = "All descriptions are already up to date."
                isGeneratingDescriptions = false
                return
            }

            descriptionGenerationMessage = "Regenerating descriptions for \(staleRecords.count) assets…"
            let targetSize = CGSize(width: 448, height: 448)
            var updated = 0

            for record in staleRecords {
                guard isGeneratingDescriptions else { break } // allow cancellation

                // Decode the stored embedding from raw bytes
                let storedFloats: [Float] = record.embedding.withUnsafeBytes { buffer in
                    Array(buffer.bindMemory(to: Float.self))
                }

                // Fetch thumbnail for OCR augmentation
                let fetchResult = PHAsset.fetchAssets(
                    withLocalIdentifiers: [record.assetLocalIdentifier],
                    options: nil
                )
                guard let asset = fetchResult.firstObject else { continue }

                let image = await withCheckedContinuation { continuation in
                    let opts = PHImageRequestOptions()
                    opts.isNetworkAccessAllowed = true
                    opts.deliveryMode = .fastFormat
                    PHImageManager.default().requestImage(
                        for: asset,
                        targetSize: targetSize,
                        contentMode: .aspectFill,
                        options: opts
                    ) { img, _ in
                        continuation.resume(returning: img ?? UIImage())
                    }
                }

                // Generate description — reuse stored embedding (no re-inference)
                let description: String
                if !storedFloats.isEmpty {
                    description = descriptionGenerator.generate(
                        embedding: storedFloats,
                        image: image
                    )
                } else {
                    description = await descriptionGenerator.generate(
                        image: image,
                        using: visionEncoder
                    )
                }

                try embeddingStore.updateDescription(
                    assetId: record.assetLocalIdentifier,
                    description: description
                )
                updated += 1
                descriptionGenerationMessage = "Regenerated \(updated) of \(staleRecords.count) descriptions…"
            }

            descriptionGenerationMessage = updated == 0
                ? "No descriptions updated."
                : "Updated descriptions for \(updated) assets."
            await refreshIndexedRecords()

        } catch {
            lastErrorMessage = "Description regeneration failed: \(error.localizedDescription)"
            descriptionGenerationMessage = ""
        }

        isGeneratingDescriptions = false
    }

    /// Cancels an in-progress description regeneration job.
    func cancelDescriptionRegeneration() {
        isGeneratingDescriptions = false
        descriptionGenerationMessage = "Description regeneration cancelled."
    }

    func updateSelectedAssets(_ assetIDs: [String]) async {
        selectedAssetIdentifiers = Array(NSOrderedSet(array: assetIDs)) as? [String] ?? []
        lastErrorMessage = nil
        indexingProgress = 0.0
        lastIndexingDuration = nil
        indexingStatusMessage = selectedAssetIdentifiers.isEmpty ? "Select photos or videos to begin indexing." : "Selected \(selectedAssetIdentifiers.count) assets."
        await refreshSelectionStats()
    }

    func clearSelectedAssets() async {
        selectedAssetIdentifiers = []
        selectedPhotoCount = 0
        selectedIndexedCount = 0
        indexingProgress = 0.0
        lastIndexingDuration = nil
        indexingStatusMessage = "Selection cleared."
    }

    func refreshSelectionStats() async {
        selectedPhotoCount = selectedAssetIdentifiers.count

        guard !selectedAssetIdentifiers.isEmpty else {
            selectedIndexedCount = 0
            return
        }

        let indexedIDs = Set((try? embeddingStore.fetchAllEmbeddings().map { $0.assetLocalIdentifier }) ?? [])
        selectedIndexedCount = selectedAssetIdentifiers.reduce(into: 0) { count, assetID in
            if indexedIDs.contains(assetID) {
                count += 1
            }
        }

        if selectedPhotoCount > 0 {
            indexingProgress = Float(selectedIndexedCount) / Float(selectedPhotoCount)
        } else {
            indexingProgress = 0.0
        }
    }

    func formatDuration(_ duration: TimeInterval?) -> String? {
        guard let duration else { return nil }
        if duration >= 1 {
            return String(format: "%.2f s", duration)
        }
        return String(format: "%.0f ms", duration * 1000)
    }

    private func bindIndexingPipeline(_ pipeline: IndexingPipeline) {
        pipeline.onProgressUpdate = { [weak self] processed, total, message in
            guard let self else { return }
            self.indexingStatusMessage = message
            self.isIndexing = pipeline.isIndexing
            self.indexingProgress = total > 0 ? Float(processed) / Float(total) : 0.0
            self.selectedIndexedCount = min(self.selectedPhotoCount, self.indexingBaseSelectedIndexedCount + processed)
            self.indexedPhotoCount = (try? self.embeddingStore.countIndexedPhotos()) ?? self.indexedPhotoCount
        }

        pipeline.onError = { [weak self] message in
            guard let self else { return }
            self.lastErrorMessage = message
            self.indexingStatusMessage = message
        }

        pipeline.onCompletion = { [weak self] processed, total in
            guard let self else { return }
            self.indexingStatusMessage = total == 0 ? "Nothing new to index." : "Finished creating \(processed) embeddings."
            self.indexingProgress = total > 0 ? 1.0 : self.indexingProgress
            self.indexedPhotoCount = (try? self.embeddingStore.countIndexedPhotos()) ?? self.indexedPhotoCount
        }
    }

    func runDiagnostics() async {
        isRunningDiagnostics = true
        defer { isRunningDiagnostics = false }

        var lines: [String] = []
        lines.append("Models: \(modelStatusMessage)")
        lines.append("Indexed embeddings: \((try? embeddingStore.countIndexedPhotos()) ?? 0)")

        let samplePrompt = "sunset at the beach"
        let encoded = QwenTextInputEncoder.encode(samplePrompt)
        let activeTokens = zip(encoded.ids, encoded.mask).filter { $0.1 == 1 }.map(\.0)
        lines.append("Tokenizer sample: \(samplePrompt)")
        lines.append("Token count: \(activeTokens.count)")
        let tokenPreview = activeTokens.prefix(8).map(String.init).joined(separator: ", ")
        lines.append("First tokens: \(tokenPreview)")

        if let visionEncoder {
            do {
                let testImage = makeDiagnosticsImage()
                let embedding = try await visionEncoder.encode(image: testImage)
                let preview = embedding.prefix(6).map { String(format: "%.4f", $0) }.joined(separator: ", ")
                lines.append("Vision embedding preview: [\(preview)]")
            } catch {
                lines.append("Vision embedding error: \(error.localizedDescription)")
            }
        }

        if let languageModel {
            do {
                let reasoning = try await languageModel.generate(
                    prompt: "Describe a calm bright outdoor scene",
                    image: makeDiagnosticsImage(),
                    maxTokens: configuration.reasoningMaxTokens
                )
                lines.append("Reasoning sample: \(reasoning.states) / \(reasoning.scene) / \(reasoning.camera)")
            } catch {
                lines.append("Language model error: \(error.localizedDescription)")
            }
        }

        diagnosticsReport = lines.joined(separator: "\n")
    }

    /// Requests photo-library authorization at launch so the system accounts daemon
    /// is ready before any indexing starts. If authorization has already been determined
    /// this is a no-op (Photos returns the cached status instantly without UI).
    private func prefetchPhotoAuthorization() async {
        let current = PHPhotoLibrary.authorizationStatus(for: .readWrite)
        guard current == .notDetermined else { return }
        _ = await PHPhotoLibrary.requestAuthorization(for: .readWrite)
    }

    private func makeDiagnosticsImage() -> UIImage {
        let renderer = UIGraphicsImageRenderer(size: CGSize(width: 448, height: 448))
        return renderer.image { context in
            UIColor.systemBlue.setFill()
            context.fill(CGRect(x: 0, y: 0, width: 448, height: 448))
            UIColor.systemYellow.setFill()
            context.fill(CGRect(x: 40, y: 40, width: 160, height: 160))
            UIColor.systemPink.setFill()
            context.fill(CGRect(x: 220, y: 220, width: 180, height: 120))
        }
    }
}
