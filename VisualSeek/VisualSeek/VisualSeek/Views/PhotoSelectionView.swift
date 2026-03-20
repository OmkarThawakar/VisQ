import SwiftUI
import Photos

struct PhotoSelectionView: View {
    @Environment(\.dismiss) private var dismiss

    let initiallySelectedIDs: Set<String>
    let onDone: ([String]) -> Void

    @State private var assets: [PHAsset] = []
    @State private var selectedIDs: Set<String>
    @State private var authorizationStatus: PHAuthorizationStatus = .notDetermined

    private let columns = [
        GridItem(.adaptive(minimum: 120, maximum: 180), spacing: 14)
    ]

    init(initiallySelectedIDs: Set<String>, onDone: @escaping ([String]) -> Void) {
        self.initiallySelectedIDs = initiallySelectedIDs
        self.onDone = onDone
        _selectedIDs = State(initialValue: initiallySelectedIDs)
    }

    var body: some View {
        NavigationStack {
            ZStack {
                VSAppBackground()

                ScrollView {
                    VStack(alignment: .leading, spacing: 20) {
                        VSHeroHeader(
                            eyebrow: "Working Set",
                            title: "Choose Media",
                            subtitle: "Only the selected photos and videos will be turned into embeddings. Curate a tight set or sweep the whole archive.",
                            trailingBadge: "\(selectedIDs.count) selected"
                        )

                        if authorizationStatus == .authorized || authorizationStatus == .limited {
                            LazyVGrid(columns: columns, spacing: 14) {
                                ForEach(assets, id: \.localIdentifier) { asset in
                                    Button {
                                        toggleSelection(for: asset.localIdentifier)
                                    } label: {
                                        SelectionThumbnailCard(
                                            asset: asset,
                                            isSelected: selectedIDs.contains(asset.localIdentifier)
                                        )
                                    }
                                    .buttonStyle(.plain)
                                }
                            }
                        } else {
                            VSEmptyState(
                                title: "Library Access Is Needed",
                                subtitle: "Allow photo-library access to choose which photos and videos should be indexed.",
                                systemImage: "photo.stack"
                            )
                        }
                    }
                    .padding(.horizontal, 20)
                    .padding(.top, 16)
                    .padding(.bottom, 24)
                }
                .scrollIndicators(.hidden)
            }
            .toolbar(.hidden, for: .navigationBar)
            .safeAreaInset(edge: .bottom, spacing: 0) {
                HStack(spacing: 12) {
                    Button("Cancel") {
                        dismiss()
                    }
                    .buttonStyle(VSSecondaryButtonStyle())

                    Button("Use Selection") {
                        let orderedSelection = assets
                            .filter { selectedIDs.contains($0.localIdentifier) }
                            .map(\.localIdentifier)
                        onDone(orderedSelection)
                        dismiss()
                    }
                    .buttonStyle(VSPrimaryButtonStyle())
                    .disabled(selectedIDs.isEmpty)
                }
                .padding(.horizontal, 20)
                .padding(.top, 10)
                .padding(.bottom, 12)
            }
            .task {
                await requestAccessAndLoadAssets()
            }
        }
    }

    private func toggleSelection(for assetID: String) {
        if selectedIDs.contains(assetID) {
            selectedIDs.remove(assetID)
        } else {
            selectedIDs.insert(assetID)
        }
    }

    private func requestAccessAndLoadAssets() async {
        authorizationStatus = await PHPhotoLibrary.requestAuthorization(for: .readWrite)
        guard authorizationStatus == .authorized || authorizationStatus == .limited else {
            assets = []
            return
        }

        let fetchOptions = PHFetchOptions()
        fetchOptions.sortDescriptors = [NSSortDescriptor(key: "creationDate", ascending: false)]
        fetchOptions.predicate = NSPredicate(
            format: "mediaType == %d OR mediaType == %d",
            PHAssetMediaType.image.rawValue,
            PHAssetMediaType.video.rawValue
        )

        let fetchResult = PHAsset.fetchAssets(with: fetchOptions)
        var loadedAssets: [PHAsset] = []
        fetchResult.enumerateObjects { asset, _, _ in
            loadedAssets.append(asset)
        }
        assets = loadedAssets
    }
}

private struct SelectionThumbnailCard: View {
    let asset: PHAsset
    let isSelected: Bool

    var body: some View {
        ZStack(alignment: .topTrailing) {
            AssetThumbnailView(asset: asset)
                .frame(height: 138)
                .clipShape(RoundedRectangle(cornerRadius: 24, style: .continuous))
                .overlay {
                    RoundedRectangle(cornerRadius: 24, style: .continuous)
                        .stroke(isSelected ? VSTheme.teal : Color.white.opacity(0.28), lineWidth: isSelected ? 3 : 1)
                }

            Image(systemName: isSelected ? "checkmark.circle.fill" : "circle")
                .font(.system(size: 24, weight: .bold))
                .foregroundStyle(isSelected ? VSTheme.teal : Color.white.opacity(0.88))
                .padding(10)

            if asset.mediaType == .video {
                Image(systemName: "video.fill")
                    .font(.system(size: 12, weight: .bold))
                    .foregroundStyle(.white)
                    .padding(.horizontal, 8)
                    .padding(.vertical, 6)
                    .background(Color.black.opacity(0.55), in: Capsule())
                    .padding(.leading, 10)
                    .padding(.top, 10)
                    .frame(maxWidth: .infinity, alignment: .topLeading)
            }
        }
        .background(VSTheme.card, in: RoundedRectangle(cornerRadius: 24, style: .continuous))
    }
}
