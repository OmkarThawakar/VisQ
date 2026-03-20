import Foundation
import Photos
import UIKit
import AVFoundation

/// Scans the Photo Library for new assets to index.
class PhotoLibraryScanner {
    
    init() {}
    
    /// Requests access to the user's photo library.
    func requestAccess() async -> PHAuthorizationStatus {
        return await PHPhotoLibrary.requestAuthorization(for: .readWrite)
    }
    
    /// Fetches selected image and video assets that have not been indexed yet.
    func fetchUnindexedAssets(selectedAssetIDs: [String], store: EmbeddingStore) throws -> [PHAsset] {
        let normalizedIDs = Array(NSOrderedSet(array: selectedAssetIDs)) as? [String] ?? []
        guard !normalizedIDs.isEmpty else { return [] }

        var unindexedAssets: [PHAsset] = []
        let indexedIds = Set(try store.fetchAllEmbeddings().map { $0.assetLocalIdentifier })
        let fetchResult = PHAsset.fetchAssets(withLocalIdentifiers: normalizedIDs, options: nil)

        fetchResult.enumerateObjects { asset, _, _ in
            guard asset.mediaType == .image || asset.mediaType == .video else { return }
            if !indexedIds.contains(asset.localIdentifier) {
                unindexedAssets.append(asset)
            }
        }

        unindexedAssets.sort {
            ($0.creationDate ?? .distantPast) > ($1.creationDate ?? .distantPast)
        }

        return unindexedAssets
    }
    
    /// Fetches the full-resolution UIImage for a given PHAsset (or a smaller sized one suitable for ANE).
    func fetchImage(for asset: PHAsset, targetSize: CGSize) async throws -> UIImage {
        return try await withCheckedThrowingContinuation { continuation in
            let options = PHImageRequestOptions()
            options.isNetworkAccessAllowed = true
            options.deliveryMode = .highQualityFormat
            options.isSynchronous = false
            
            PHImageManager.default().requestImage(
                for: asset,
                targetSize: targetSize,
                contentMode: .aspectFit,
                options: options) { image, info in
                    
                    if let error = info?[PHImageErrorKey] as? Error {
                        continuation.resume(throwing: error)
                        return
                    }
                    if let image = image {
                        continuation.resume(returning: image)
                    } else {
                        continuation.resume(throwing: NSError(domain: "PhotoLibraryScanner", code: -1, userInfo: [NSLocalizedDescriptionKey: "Image is nil"]))
                    }
            }
        }
    }

    func fetchAVAsset(for asset: PHAsset) async throws -> AVAsset {
        try await withCheckedThrowingContinuation { continuation in
            let options = PHVideoRequestOptions()
            options.deliveryMode = .highQualityFormat
            options.isNetworkAccessAllowed = true

            PHImageManager.default().requestAVAsset(forVideo: asset, options: options) { avAsset, _, info in
                if let error = info?[PHImageErrorKey] as? Error {
                    continuation.resume(throwing: error)
                    return
                }

                if let avAsset {
                    continuation.resume(returning: avAsset)
                } else {
                    continuation.resume(
                        throwing: NSError(
                            domain: "PhotoLibraryScanner",
                            code: -2,
                            userInfo: [NSLocalizedDescriptionKey: "Video asset is unavailable"]
                        )
                    )
                }
            }
        }
    }
}
