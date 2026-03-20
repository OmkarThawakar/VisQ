import SwiftUI
import Photos

struct LibraryView: View {
    @EnvironmentObject var appState: AppState
    @State private var assets: [PHAsset] = []
    @State private var authorizationStatus: PHAuthorizationStatus = .notDetermined

    private let columns = [
        GridItem(.adaptive(minimum: 150, maximum: 220), spacing: 14)
    ]

    private var indexedIDs: Set<String> {
        Set(appState.indexedRecords.map(\.assetLocalIdentifier))
    }

    var body: some View {
        NavigationStack {
            ScrollView {
                VStack(alignment: .leading, spacing: 20) {
                    VSHeroHeader(
                        eyebrow: "Visual Archive",
                        title: "Photo Library",
                        subtitle: "Browse recent images, inspect full-frame detail, and keep an eye on what has already been indexed.",
                        trailingBadge: authorizationBadge
                    )

                    metricStrip

                    content
                }
                .padding(.horizontal, 20)
                .padding(.top, 16)
                .padding(.bottom, 24)
            }
            .scrollIndicators(.hidden)
            .toolbar(.hidden, for: .navigationBar)
            .task {
                await appState.refreshLibraryStats()
                loadAssets()
            }
        }
    }

    @ViewBuilder
    private var content: some View {
        switch authorizationStatus {
        case .authorized, .limited:
            if assets.isEmpty {
                VSEmptyState(
                    title: "Nothing To Show Yet",
                    subtitle: "Your photo library is available, but no recent images were returned.",
                    systemImage: "photo.on.rectangle.angled"
                )
            } else {
                photoGrid
            }
        case .denied, .restricted:
            VSEmptyState(
                title: "Photo Access Is Off",
                subtitle: "Open iOS Settings and allow photo-library access for VisQ to browse and inspect your images.",
                systemImage: "lock.shield"
            )
        default:
            VSSectionCard {
                HStack(spacing: 12) {
                    ProgressView()
                    Text("Requesting photo-library access…")
                        .font(.system(size: 15, weight: .medium, design: .rounded))
                        .foregroundStyle(VSTheme.mutedInk)
                }
            }
        }
    }

    private var metricStrip: some View {
        LazyVGrid(columns: [GridItem(.flexible()), GridItem(.flexible())], spacing: 14) {
            VSMetricTile(label: "Visible", value: "\(assets.count)", accent: VSTheme.apricot)
            VSMetricTile(label: "Indexed", value: "\(indexedIDs.count)", accent: VSTheme.teal)
        }
    }

    private var photoGrid: some View {
        VStack(alignment: .leading, spacing: 14) {
            VSSectionHeading(
                title: "Recent Frames",
                subtitle: "Tap any tile to open a clean, immersive detail view."
            )

            LazyVGrid(columns: columns, spacing: 14) {
                ForEach(assets, id: \.localIdentifier) { asset in
                    NavigationLink(destination: PhotoDetailView(asset: asset)) {
                        LibraryThumbnailCard(
                            asset: asset,
                            isIndexed: indexedIDs.contains(asset.localIdentifier)
                        )
                    }
                    .buttonStyle(.plain)
                }
            }
        }
    }

    private var authorizationBadge: String {
        switch authorizationStatus {
        case .authorized:
            return "Full Access"
        case .limited:
            return "Limited"
        case .denied, .restricted:
            return "Locked"
        default:
            return "Checking"
        }
    }

    private func loadAssets() {
        let status = PHPhotoLibrary.authorizationStatus(for: .readWrite)
        authorizationStatus = status

        if status == .notDetermined {
            PHPhotoLibrary.requestAuthorization(for: .readWrite) { newStatus in
                DispatchQueue.main.async {
                    authorizationStatus = newStatus
                    if newStatus == .authorized || newStatus == .limited {
                        fetchAssets()
                    }
                }
            }
        } else if status == .authorized || status == .limited {
            fetchAssets()
        }
    }

    private func fetchAssets() {
        let options = PHFetchOptions()
        options.sortDescriptors = [NSSortDescriptor(key: "creationDate", ascending: false)]
        options.fetchLimit = 300

        let result = PHAsset.fetchAssets(with: .image, options: options)
        var fetched: [PHAsset] = []
        fetched.reserveCapacity(result.count)
        result.enumerateObjects { asset, _, _ in
            fetched.append(asset)
        }

        DispatchQueue.main.async {
            assets = fetched
        }
    }
}

private struct LibraryThumbnailCard: View {
    let asset: PHAsset
    let isIndexed: Bool

    var body: some View {
        ZStack(alignment: .bottomLeading) {
            AssetThumbnailView(asset: asset)
                .aspectRatio(0.85, contentMode: .fill)
                .clipShape(RoundedRectangle(cornerRadius: 28, style: .continuous))

            LinearGradient(
                colors: [Color.clear, Color.black.opacity(0.72)],
                startPoint: .center,
                endPoint: .bottom
            )
            .clipShape(RoundedRectangle(cornerRadius: 28, style: .continuous))

            VStack(alignment: .leading, spacing: 8) {
                if isIndexed {
                    VSPill(text: "Indexed", tint: VSTheme.teal)
                }

                Text(asset.creationDate?.formatted(date: .abbreviated, time: .omitted) ?? "Recent Photo")
                    .font(.system(size: 15, weight: .bold, design: .rounded))
                    .foregroundStyle(.white)
                    .lineLimit(1)
            }
            .padding(14)
        }
        .background(VSTheme.card, in: RoundedRectangle(cornerRadius: 28, style: .continuous))
        .overlay {
            RoundedRectangle(cornerRadius: 28, style: .continuous)
                .stroke(VSTheme.line, lineWidth: 1)
        }
        .shadow(color: VSTheme.shadow, radius: 16, y: 8)
    }
}

struct AssetThumbnailView: View {
    let asset: PHAsset
    @State private var image: UIImage?
    @State private var requestID: PHImageRequestID?

    var body: some View {
        GeometryReader { proxy in
            ZStack {
                RoundedRectangle(cornerRadius: 24, style: .continuous)
                    .fill(VSTheme.cardStrong)

                if let image {
                    Image(uiImage: image)
                        .resizable()
                        .aspectRatio(contentMode: .fill)
                        .frame(width: proxy.size.width, height: proxy.size.height)
                        .clipped()
                        .transition(.opacity.animation(.easeIn(duration: 0.18)))
                } else {
                    Image(systemName: "photo")
                        .font(.system(size: 26, weight: .medium))
                        .foregroundStyle(VSTheme.ink.opacity(0.64))
                }
            }
            .contentShape(Rectangle())
            .onAppear { fetchImage(size: proxy.size) }
            .onDisappear { cancelRequest() }
        }
    }

    private func fetchImage(size: CGSize) {
        guard image == nil else { return }
        let scale = UIScreen.main.scale
        let targetSize = CGSize(width: size.width * scale, height: size.height * scale)

        let options = PHImageRequestOptions()
        options.isNetworkAccessAllowed = true
        options.deliveryMode = .opportunistic
        options.resizeMode = .fast

        requestID = PHImageManager.default().requestImage(
            for: asset,
            targetSize: targetSize,
            contentMode: .aspectFill,
            options: options
        ) { img, _ in
            if let img {
                image = img
            }
        }
    }

    private func cancelRequest() {
        if let requestID {
            PHImageManager.default().cancelImageRequest(requestID)
        }
    }
}
