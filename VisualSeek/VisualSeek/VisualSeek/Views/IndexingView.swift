import SwiftUI
import Photos

struct IndexingView: View {
    @EnvironmentObject var appState: AppState
    @State private var showPhotoPicker = false
    @State private var showIndexedContent = false

    private let metricColumns = [
        GridItem(.flexible(), spacing: 14),
        GridItem(.flexible(), spacing: 14)
    ]

    var body: some View {
        NavigationStack {
            ScrollView {
                VStack(alignment: .leading, spacing: 20) {
                    VSHeroHeader(
                        eyebrow: "Embedding Studio",
                        title: "Indexing Dashboard",
                        subtitle: "Choose a working set, create embeddings, refresh descriptions, and audit what is already stored on-device.",
                        trailingBadge: appState.isIndexing ? "Running" : "Ready"
                    )

                    progressCard
                    libraryMetrics
                    actionCard
                    statusCard

                    if let message = appState.lastErrorMessage, !message.isEmpty {
                        VSStatusBanner(icon: "exclamationmark.triangle.fill", text: message, tint: VSTheme.danger)
                    }
                }
                .padding(.horizontal, 20)
                .padding(.top, 16)
                .padding(.bottom, 24)
            }
            .scrollIndicators(.hidden)
            .toolbar(.hidden, for: .navigationBar)
            .task {
                await appState.refreshLibraryStats()
                await appState.refreshSelectionStats()
            }
            .fullScreenCover(isPresented: $showPhotoPicker) {
                PhotoSelectionView(initiallySelectedIDs: Set(appState.selectedAssetIdentifiers)) { assetIDs in
                    Task { await appState.updateSelectedAssets(assetIDs) }
                }
            }
            .sheet(isPresented: $showIndexedContent) {
                IndexedContentView()
                    .environmentObject(appState)
            }
        }
    }

    private var progressCard: some View {
        VSSectionCard {
            VStack(alignment: .leading, spacing: 18) {
                HStack(alignment: .top) {
                    VSSectionHeading(
                        title: "Current Progress",
                        subtitle: appState.indexingStatusMessage
                    )
                    Spacer()
                    VSPill(
                        text: "\(Int(appState.indexingProgress * 100))%",
                        tint: appState.isIndexing ? VSTheme.apricot : VSTheme.teal
                    )
                }

                GeometryReader { proxy in
                    ZStack(alignment: .leading) {
                        RoundedRectangle(cornerRadius: 999, style: .continuous)
                            .fill(Color.white.opacity(0.45))
                        RoundedRectangle(cornerRadius: 999, style: .continuous)
                            .fill(
                                LinearGradient(
                                    colors: [VSTheme.apricot, VSTheme.ember],
                                    startPoint: .leading,
                                    endPoint: .trailing
                                )
                            )
                            .frame(width: max(proxy.size.width * CGFloat(appState.indexingProgress), 10))
                    }
                }
                .frame(height: 16)

                HStack(spacing: 12) {
                    Button(appState.selectedPhotoCount == 0 ? "Select Media" : "Update Selection") {
                        showPhotoPicker = true
                    }
                    .buttonStyle(VSPrimaryButtonStyle())

                    Button("Show Indexed Content") {
                        showIndexedContent = true
                    }
                    .buttonStyle(VSSecondaryButtonStyle())
                    .disabled(appState.indexedPhotoCount == 0)
                }
            }
        }
    }

    private var libraryMetrics: some View {
        VStack(alignment: .leading, spacing: 14) {
            VSSectionHeading(
                title: "Selection Snapshot",
                subtitle: "Track total library volume against the current indexing set."
            )

            LazyVGrid(columns: metricColumns, spacing: 14) {
                VSMetricTile(label: "Library", value: "\(appState.totalPhotoCount)", accent: VSTheme.apricot)
                VSMetricTile(label: "Selected", value: "\(appState.selectedPhotoCount)", accent: VSTheme.teal)
                VSMetricTile(label: "Embedded", value: "\(appState.selectedIndexedCount)", accent: VSTheme.moss)
                VSMetricTile(label: "Remaining", value: "\(max(0, appState.selectedPhotoCount - appState.selectedIndexedCount))", accent: VSTheme.ember)
            }
        }
    }

    private var actionCard: some View {
        VSSectionCard {
            VStack(alignment: .leading, spacing: 14) {
                VSSectionHeading(
                    title: "Operations",
                    subtitle: "All current indexing actions remain here, just in a denser layout."
                )

                Button("Create Embeddings") {
                    Task { await appState.startIndexing() }
                }
                .buttonStyle(VSPrimaryButtonStyle(tint: VSTheme.teal))
                .disabled(appState.isIndexing || appState.isModelLoading || appState.selectedPhotoCount == 0)

                HStack(spacing: 12) {
                    Button("Pause Indexing") {
                        appState.pauseIndexing()
                    }
                    .buttonStyle(VSSecondaryButtonStyle(tint: VSTheme.apricot))
                    .disabled(!appState.isIndexing)

                    Button("Regenerate Descriptions") {
                        Task { await appState.regenerateDescriptions() }
                    }
                    .buttonStyle(VSSecondaryButtonStyle(tint: VSTheme.teal))
                    .disabled(appState.isIndexing || appState.isGeneratingDescriptions || appState.indexedPhotoCount == 0)
                }

                if appState.isGeneratingDescriptions {
                    Button("Cancel Regeneration") {
                        appState.cancelDescriptionRegeneration()
                    }
                    .buttonStyle(VSDestructiveButtonStyle())
                }

                Button("Clear Selection") {
                    Task { await appState.clearSelectedAssets() }
                }
                .buttonStyle(VSDestructiveButtonStyle())
                .disabled(appState.selectedPhotoCount == 0 || appState.isIndexing)
            }
        }
    }

    private var statusCard: some View {
        VSSectionCard {
            VStack(alignment: .leading, spacing: 14) {
                VSSectionHeading(
                    title: "Runtime Status",
                    subtitle: "Diagnostics for the active runtime, persistence layer, and description refreshes."
                )

                IndexingDetailRow(label: "Runtime", value: appState.usingBundledModels ? "Core ML" : "Lightweight")
                IndexingDetailRow(label: "Embeddings", value: "\(appState.indexedPhotoCount)")
                IndexingDetailRow(label: "Database", value: appState.configuration.embeddingStoragePath.lastPathComponent)

                if let indexingTime = appState.formatDuration(appState.lastIndexingDuration) {
                    IndexingDetailRow(label: "Last Run", value: indexingTime)
                }

                if appState.isGeneratingDescriptions {
                    VSStatusBanner(
                        icon: "text.badge.checkmark",
                        text: appState.descriptionGenerationMessage,
                        tint: VSTheme.teal
                    )
                } else if !appState.descriptionGenerationMessage.isEmpty {
                    VSStatusBanner(
                        icon: "text.bubble",
                        text: appState.descriptionGenerationMessage,
                        tint: VSTheme.apricot
                    )
                }
            }
        }
    }
}

private struct IndexingDetailRow: View {
    let label: String
    let value: String

    var body: some View {
        HStack {
            Text(label)
                .font(.system(size: 14, weight: .bold, design: .rounded))
                .foregroundStyle(VSTheme.mutedInk)
            Spacer()
            Text(value)
                .font(.system(size: 14, weight: .bold, design: .rounded))
                .foregroundStyle(VSTheme.ink)
        }
        .padding(.vertical, 2)
    }
}

struct IndexedContentView: View {
    @Environment(\.dismiss) private var dismiss
    @EnvironmentObject private var appState: AppState

    var body: some View {
        NavigationStack {
            ZStack {
                LinearGradient(
                    colors: [
                        Color.black,
                        Color(red: 0.10, green: 0.11, blue: 0.13),
                        Color(red: 0.15, green: 0.15, blue: 0.18)
                    ],
                    startPoint: .top,
                    endPoint: .bottom
                )
                .ignoresSafeArea()

                ScrollView {
                    VStack(alignment: .leading, spacing: 20) {
                        HStack {
                            VSHeroHeader(
                                eyebrow: "Stored Archive",
                                title: "Indexed Content",
                                subtitle: "Review the image records and generated descriptions currently persisted on-device.",
                                trailingBadge: "\(appState.indexedRecords.count) saved"
                            )
                            Spacer()
                        }

                        if appState.indexedRecords.isEmpty {
                            darkEmptyState
                        } else {
                            LazyVStack(spacing: 14) {
                                ForEach(appState.indexedRecords, id: \.assetLocalIdentifier) { record in
                                    NavigationLink {
                                        IndexedRecordDetailView(record: record)
                                    } label: {
                                        IndexedRecordRow(record: record)
                                    }
                                    .buttonStyle(.plain)
                                }
                            }
                        }
                    }
                    .padding(.horizontal, 20)
                    .padding(.top, 16)
                    .padding(.bottom, 24)
                }
            }
            .scrollIndicators(.hidden)
            .toolbar(.hidden, for: .navigationBar)
            .toolbarColorScheme(.dark, for: .navigationBar)
            .safeAreaInset(edge: .bottom, spacing: 0) {
                HStack(spacing: 12) {
                    Button("Close") {
                        dismiss()
                    }
                    .buttonStyle(VSSecondaryButtonStyle(tint: .white))

                    Button("Refresh") {
                        Task { await appState.refreshIndexedRecords() }
                    }
                    .buttonStyle(VSPrimaryButtonStyle(tint: VSTheme.teal))
                }
                .padding(.horizontal, 20)
                .padding(.top, 10)
                .padding(.bottom, 12)
            }
            .task {
                await appState.refreshIndexedRecords()
            }
        }
    }

    private var darkEmptyState: some View {
        VStack(spacing: 16) {
            Image(systemName: "photo.badge.exclamationmark")
                .font(.system(size: 30, weight: .medium))
                .foregroundStyle(.white)
                .frame(width: 72, height: 72)
                .background(Color.white.opacity(0.12), in: RoundedRectangle(cornerRadius: 24, style: .continuous))

            VStack(spacing: 8) {
                Text("No Indexed Content")
                    .font(.system(size: 22, weight: .bold, design: .rounded))
                    .foregroundStyle(.white)

                Text("Create embeddings first, then your stored image records will appear here.")
                    .font(.system(size: 14, weight: .medium, design: .rounded))
                    .foregroundStyle(Color.white.opacity(0.72))
                    .multilineTextAlignment(.center)
            }
        }
        .frame(maxWidth: .infinity)
        .padding(.vertical, 24)
        .padding(.horizontal, 18)
        .background(Color.white.opacity(0.08), in: RoundedRectangle(cornerRadius: 28, style: .continuous))
        .overlay {
            RoundedRectangle(cornerRadius: 28, style: .continuous)
                .stroke(Color.white.opacity(0.10), lineWidth: 1)
        }
    }
}

private struct IndexedRecordRow: View {
    let record: EmbeddingRecord

    private var descriptionSummary: String {
        let trimmed = record.imageDescription.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return "No description saved" }
        return String(trimmed.prefix(110))
    }

    private var embeddingDimension: Int {
        record.embedding.count / MemoryLayout<Float>.size
    }

    var body: some View {
        HStack(alignment: .top, spacing: 14) {
            IndexedAssetThumbnailView(assetLocalIdentifier: record.assetLocalIdentifier)
                .frame(width: 88, height: 88)
                .clipShape(RoundedRectangle(cornerRadius: 24, style: .continuous))

            VStack(alignment: .leading, spacing: 8) {
                Text(descriptionSummary)
                    .font(.system(size: 16, weight: .bold, design: .rounded))
                    .foregroundStyle(.white)
                    .lineLimit(3)

                Text(record.assetLocalIdentifier)
                    .font(.caption.monospaced())
                    .foregroundStyle(Color.white.opacity(0.74))
                    .lineLimit(2)
                    .truncationMode(.middle)
                    .fixedSize(horizontal: false, vertical: true)

                Text("\(assetTypeLabel(record.assetType)) • \(embeddingDimension)-D • \(record.indexedAt.formatted(date: .abbreviated, time: .shortened))")
                    .font(.system(size: 13, weight: .medium, design: .rounded))
                    .foregroundStyle(Color.white.opacity(0.68))
                    .fixedSize(horizontal: false, vertical: true)
            }

            Spacer(minLength: 0)
        }
        .padding(16)
        .background(Color.white.opacity(0.08), in: RoundedRectangle(cornerRadius: 28, style: .continuous))
        .overlay {
            RoundedRectangle(cornerRadius: 28, style: .continuous)
                .stroke(Color.white.opacity(0.10), lineWidth: 1)
        }
    }
}

private struct IndexedRecordDetailView: View {
    let record: EmbeddingRecord

    private var descriptionText: String {
        let trimmed = record.imageDescription.trimmingCharacters(in: .whitespacesAndNewlines)
        return trimmed.isEmpty ? "No description was stored for this image." : trimmed
    }

    private var embeddingDimension: Int {
        record.embedding.count / MemoryLayout<Float>.size
    }

    var body: some View {
        ZStack {
            LinearGradient(
                colors: [
                    Color.black,
                    Color(red: 0.10, green: 0.11, blue: 0.13),
                    Color(red: 0.15, green: 0.15, blue: 0.18)
                ],
                startPoint: .top,
                endPoint: .bottom
            )
            .ignoresSafeArea()

            ScrollView {
                VStack(alignment: .leading, spacing: 20) {
                    IndexedAssetThumbnailView(assetLocalIdentifier: record.assetLocalIdentifier, contentMode: .fit)
                        .frame(maxWidth: .infinity)
                        .frame(height: 300)
                        .clipShape(RoundedRectangle(cornerRadius: 34, style: .continuous))

                    darkSectionCard(
                        title: "Record Metadata",
                        subtitle: "Persistence details for this indexed asset."
                    ) {
                        detailRow("Asset ID", value: record.assetLocalIdentifier, monospaced: true)
                        detailRow("Type", value: assetTypeLabel(record.assetType))
                        detailRow("Embedding Size", value: "\(embeddingDimension) floats")
                        detailRow("Embedding Version", value: "\(record.embeddingVersion)")
                        detailRow("Indexed At", value: record.indexedAt.formatted(date: .complete, time: .standard))
                        detailRow("Created At", value: record.creationDate?.formatted(date: .complete, time: .shortened) ?? "Unknown")
                    }

                    darkSectionCard(
                        title: "Stored Description",
                        subtitle: "The retrieval-oriented description saved alongside the embedding."
                    ) {
                        Text(descriptionText)
                            .font(.system(size: 15, weight: .medium, design: .rounded))
                            .foregroundStyle(Color.white.opacity(0.92))
                            .textSelection(.enabled)
                    }
                }
                .padding(.horizontal, 20)
                .padding(.vertical, 16)
            }
        }
        .scrollIndicators(.hidden)
        .navigationBarTitleDisplayMode(.inline)
        .toolbarBackground(.hidden, for: .navigationBar)
        .toolbarColorScheme(.dark, for: .navigationBar)
    }

    private func detailRow(_ label: String, value: String, monospaced: Bool = false) -> some View {
        VStack(alignment: .leading, spacing: 4) {
            Text(label)
                .font(.system(size: 12, weight: .bold, design: .rounded))
                .foregroundStyle(Color.white.opacity(0.64))
            Text(value)
                .font(monospaced ? .caption.monospaced() : .system(size: 15, weight: .medium, design: .rounded))
                .foregroundStyle(Color.white.opacity(0.92))
                .frame(maxWidth: .infinity, alignment: .leading)
                .fixedSize(horizontal: false, vertical: true)
                .textSelection(.enabled)
        }
    }

    private func darkSectionCard<Content: View>(
        title: String,
        subtitle: String,
        @ViewBuilder content: () -> Content
    ) -> some View {
        VStack(alignment: .leading, spacing: 12) {
            VStack(alignment: .leading, spacing: 4) {
                Text(title)
                    .font(.system(size: 20, weight: .bold, design: .rounded))
                    .foregroundStyle(.white)

                Text(subtitle)
                    .font(.system(size: 13, weight: .medium, design: .rounded))
                    .foregroundStyle(Color.white.opacity(0.68))
            }

            content()
        }
        .padding(18)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(Color.white.opacity(0.08), in: RoundedRectangle(cornerRadius: 28, style: .continuous))
        .overlay {
            RoundedRectangle(cornerRadius: 28, style: .continuous)
                .stroke(Color.white.opacity(0.10), lineWidth: 1)
        }
    }
}

private struct IndexedAssetThumbnailView: View {
    let assetLocalIdentifier: String
    var contentMode: ContentMode = .fill

    @State private var image: UIImage?
    @State private var didAttemptLoad = false

    var body: some View {
        ZStack {
            RoundedRectangle(cornerRadius: 24, style: .continuous)
                .fill(Color.white.opacity(0.12))

            if let image {
                Group {
                    if contentMode == .fill {
                        Image(uiImage: image)
                            .resizable()
                            .scaledToFill()
                    } else {
                        Image(uiImage: image)
                            .resizable()
                            .scaledToFit()
                            .padding(10)
                    }
                }
            } else {
                Image(systemName: didAttemptLoad ? "photo.badge.exclamationmark" : "photo")
                    .font(.system(size: 22, weight: .medium))
                    .foregroundStyle(Color.white.opacity(0.72))
            }
        }
        .clipped()
        .task {
            guard image == nil else { return }
            await loadThumbnail()
        }
    }

    private func loadThumbnail() async {
        let fetchResult = PHAsset.fetchAssets(withLocalIdentifiers: [assetLocalIdentifier], options: nil)
        guard let asset = fetchResult.firstObject else {
            didAttemptLoad = true
            return
        }

        let targetSize = CGSize(width: 400, height: 400)
        let options = PHImageRequestOptions()
        options.isNetworkAccessAllowed = true
        options.deliveryMode = .highQualityFormat

        await withCheckedContinuation { continuation in
            PHImageManager.default().requestImage(
                for: asset,
                targetSize: targetSize,
                contentMode: .aspectFill,
                options: options
            ) { loadedImage, _ in
                image = loadedImage
                didAttemptLoad = true
                continuation.resume()
            }
        }
    }
}

private func assetTypeLabel(_ assetType: Int) -> String {
    switch assetType {
    case PHAssetMediaType.image.rawValue:
        return "Image"
    case PHAssetMediaType.video.rawValue:
        return "Video"
    case PHAssetMediaType.audio.rawValue:
        return "Audio"
    default:
        return "Unknown"
    }
}
