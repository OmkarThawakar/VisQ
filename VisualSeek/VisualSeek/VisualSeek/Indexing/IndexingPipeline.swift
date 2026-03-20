import Foundation
import Photos
import UIKit
import Combine

class IndexingPipeline: ObservableObject {
    private let scanner: PhotoLibraryScanner
    private let store: EmbeddingStore
    private let batchProcessor: BatchProcessor
    private let vectorIndex: VectorIndex
    // Vision encoder reference so we can pass embeddings to the description generator
    private let visionEncoder: QwenVisionEncoderProtocol
    private let descriptionGenerator = QwenVLDescriptionGenerator()
    private let videoFrameExtractor = VideoFrameExtractor()
    
    @Published var progress: Float = 0.0
    @Published var isIndexing = false
    @Published var remainingCount = 0
    var onProgressUpdate: (@MainActor (_ processed: Int, _ total: Int, _ message: String) -> Void)?
    var onError: (@MainActor (_ message: String) -> Void)?
    var onCompletion: (@MainActor (_ processed: Int, _ total: Int) -> Void)?
    
    init(
        scanner: PhotoLibraryScanner,
        store: EmbeddingStore,
        batchProcessor: BatchProcessor,
        vectorIndex: VectorIndex,
        visionEncoder: QwenVisionEncoderProtocol
    ) {
        self.scanner = scanner
        self.store = store
        self.batchProcessor = batchProcessor
        self.vectorIndex = vectorIndex
        self.visionEncoder = visionEncoder
    }
    
    @MainActor
    func startIndexing(selectedAssetIDs: [String]) async {
        isIndexing = true
        progress = 0.0
        remainingCount = 0
        
        do {
            let status = await scanner.requestAccess()
            guard status == .authorized || status == .limited else {
                onError?("Photo library access denied.")
                isIndexing = false
                return
            }
            
            let unindexedAssets = try scanner.fetchUnindexedAssets(selectedAssetIDs: selectedAssetIDs, store: store)
            remainingCount = unindexedAssets.count
            let total = remainingCount
            
            guard total > 0 else {
                onProgressUpdate?(0, 0, "Selected photos are already indexed.")
                isIndexing = false
                return
            }

            onProgressUpdate?(0, total, "Preparing \(total) selected assets for embedding.")
            
            var batchImages: [UIImage] = []
            var batchAssets: [PHAsset] = []
            let targetSize = CGSize(width: 448, height: 448)
            var processed = 0
            
            for asset in unindexedAssets {
                guard isIndexing else { break }

                switch asset.mediaType {
                case .image:
                    do {
                        onProgressUpdate?(processed, total, "Encoding photo \(processed + 1) of \(total)...")
                        let image = try await scanner.fetchImage(for: asset, targetSize: targetSize)
                        let processedImage = try ImagePreprocessor.preprocessForQwen(image, targetResolution: targetSize)
                        
                        batchImages.append(processedImage)
                        batchAssets.append(asset)
                        
                        if batchImages.count >= batchProcessor.batchSize {
                            try await flushBatch(images: batchImages, assets: batchAssets)
                            processed += batchImages.count
                            remainingCount -= batchImages.count
                            progress = Float(processed) / Float(total)
                            onProgressUpdate?(processed, total, "Saved \(processed) of \(total) embeddings.")
                            
                            batchImages.removeAll(keepingCapacity: true)
                            batchAssets.removeAll(keepingCapacity: true)
                        }
                    } catch {
                        onError?("Failed to process a selected photo: \(error.localizedDescription)")
                    }
                case .video:
                    do {
                        if !batchImages.isEmpty {
                            try await flushBatch(images: batchImages, assets: batchAssets)
                            processed += batchImages.count
                            remainingCount -= batchImages.count
                            progress = Float(processed) / Float(total)
                            onProgressUpdate?(processed, total, "Saved \(processed) of \(total) embeddings.")
                            batchImages.removeAll(keepingCapacity: true)
                            batchAssets.removeAll(keepingCapacity: true)
                        }

                        onProgressUpdate?(processed, total, "Encoding video \(processed + 1) of \(total)...")
                        try await indexVideo(asset: asset, targetSize: targetSize)
                        processed += 1
                        remainingCount -= 1
                        progress = Float(processed) / Float(total)
                        onProgressUpdate?(processed, total, "Saved \(processed) of \(total) embeddings.")
                    } catch {
                        onError?("Failed to process a selected video: \(error.localizedDescription)")
                    }
                default:
                    continue
                }
            }
            
            // Flush remaining
            if !batchImages.isEmpty && isIndexing {
                try await flushBatch(images: batchImages, assets: batchAssets)
                processed += batchImages.count
                remainingCount -= batchImages.count
                progress = 1.0
                onProgressUpdate?(processed, total, "Saved \(processed) of \(total) embeddings.")
            }
            
            // Rebuild the in-memory vector index after insertion
            try vectorIndex.rebuild(from: store)
            onCompletion?(processed, total)
            
        } catch {
            onError?("Indexing error: \(error.localizedDescription)")
        }
        
        isIndexing = false
    }
    
    @MainActor
    func pauseIndexing() {
        isIndexing = false
    }
    
    // MARK: - Private

    /// Encodes a batch of images, generates VL descriptions from the resulting
    /// embeddings, then persists both embedding + description to the store.
    private func flushBatch(images: [UIImage], assets: [PHAsset]) async throws {
        // Compute visual embeddings — one call per image (already batched above)
        var embeddings: [[Float]] = []
        for image in images {
            let embedding = (try? await visionEncoder.encode(image: image)) ?? []
            embeddings.append(embedding)
        }

        guard embeddings.count == assets.count else { return }

        for (i, embedding) in embeddings.enumerated() {
            let asset = assets[i]
            let image = images[i]
            let assetType = asset.mediaType == .image ? 0 : 1

            // Generate a rich multi-sentence description from the visual embedding.
            // Because we already have the embedding in hand, this is free —
            // no additional model inference is required.
            let description = embedding.isEmpty
                ? ImageSemanticAnalyzer.description(for: image)   // fallback
                : descriptionGenerator.generate(embedding: embedding, image: image)

            try await store.saveEmbedding(
                assetId: asset.localIdentifier,
                assetType: assetType,
                creationDate: asset.creationDate,
                embedding: embedding,
                imageDescription: description
            )
        }
    }

    private func indexVideo(asset: PHAsset, targetSize: CGSize) async throws {
        let avAsset = try await scanner.fetchAVAsset(for: asset)
        let rawFrames = try await videoFrameExtractor.extractFrames(from: avAsset, frameCount: 4, targetSize: targetSize)
        let frames = try rawFrames.map { try ImagePreprocessor.preprocessForQwen($0, targetResolution: targetSize) }
        let embeddings = try await batchProcessor.processBatch(frames)
        let pooledEmbedding = pooledVideoEmbedding(from: embeddings)
        let description = makeVideoDescription(from: frames, pooledEmbedding: pooledEmbedding)

        try await store.saveEmbedding(
            assetId: asset.localIdentifier,
            assetType: PHAssetMediaType.video.rawValue,
            creationDate: asset.creationDate,
            embedding: pooledEmbedding,
            imageDescription: description
        )
    }

    private func pooledVideoEmbedding(from frameEmbeddings: [[Float]]) -> [Float] {
        guard let dimension = frameEmbeddings.first?.count, dimension > 0 else { return [] }

        var pooled = [Float](repeating: 0, count: dimension)
        var validCount: Float = 0

        for embedding in frameEmbeddings where embedding.count == dimension {
            for index in embedding.indices {
                pooled[index] += embedding[index]
            }
            validCount += 1
        }

        guard validCount > 0 else { return [] }
        pooled = pooled.map { $0 / validCount }
        return QwenArrayFactory.normalize(pooled)
    }

    private func makeVideoDescription(from frames: [UIImage], pooledEmbedding: [Float]) -> String {
        guard !frames.isEmpty else {
            return "This indexed asset is a video clip."
        }

        let representativeFrame = frames[frames.count / 2]
        let baseDescription = pooledEmbedding.isEmpty
            ? ImageSemanticAnalyzer.description(for: representativeFrame)
            : descriptionGenerator.generate(embedding: pooledEmbedding, image: representativeFrame)

        var summaries: [String] = []
        if let firstFrame = frames.first {
            summaries.append("Opening frame: \(ImageSemanticAnalyzer.description(for: firstFrame))")
        }
        if frames.count > 2 {
            let middleFrame = frames[frames.count / 2]
            summaries.append("Middle frame: \(ImageSemanticAnalyzer.description(for: middleFrame))")
        }
        if let lastFrame = frames.last, frames.count > 1 {
            summaries.append("Closing frame: \(ImageSemanticAnalyzer.description(for: lastFrame))")
        }

        return ([
            "This indexed asset is a video clip.",
            "Representative scene: \(baseDescription)"
        ] + summaries).joined(separator: " ")
    }
}
