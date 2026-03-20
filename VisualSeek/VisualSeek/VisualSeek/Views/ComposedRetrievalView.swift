import SwiftUI
import PhotosUI
import Photos
import AVFoundation
import CoreTransferable
import UniformTypeIdentifiers

struct ComposedRetrievalView: View {
    @EnvironmentObject var appState: AppState
    @State private var referenceImage: UIImage?
    @State private var selectedPhotoItem: PhotosPickerItem?
    @State private var isVideoReference = false
    @State private var editText = ""
    @State private var isSearching = false
    @State private var isLoadingReference = false
    @State private var reasoningTrace: String?
    @State private var results: [RetrievalResult] = []
    @FocusState private var isEditFieldFocused: Bool

    private let videoFrameExtractor = VideoFrameExtractor()

    var body: some View {
        NavigationStack {
            ScrollView {
                VStack(alignment: .leading, spacing: 20) {
                    VSHeroHeader(
                        eyebrow: "Guided Remix",
                        title: "Composed Retrieval",
                        subtitle: "Start with a reference image or video, describe the transformation you want, and search for nearby results that already look that way.",
                        trailingBadge: appState.configuration.useReasoningForComposedRetrieval ? "Reasoning On" : "Reasoning Off"
                    )

                    referenceSection
                    promptSection
                    actionSection

                    if let reasoningTrace {
                        VSSectionCard {
                            VStack(alignment: .leading, spacing: 12) {
                                VSSectionHeading(
                                    title: "Reasoning Trace",
                                    subtitle: "How the retrieval system interpreted your edit instruction."
                                )

                                Text(reasoningTrace)
                                    .font(.system(size: 14, weight: .medium, design: .rounded))
                                    .foregroundStyle(VSTheme.mutedInk)
                                    .textSelection(.enabled)
                            }
                        }
                    }

                    resultSection

                    if let error = appState.lastErrorMessage, !error.isEmpty {
                        VSStatusBanner(icon: "exclamationmark.triangle.fill", text: error, tint: VSTheme.danger)
                    }
                }
                .padding(.horizontal, 20)
                .padding(.top, 16)
                .padding(.bottom, 24)
            }
            .scrollIndicators(.hidden)
            .scrollDismissesKeyboard(.interactively)
            .toolbar(.hidden, for: .navigationBar)
            .toolbar {
                ToolbarItemGroup(placement: .keyboard) {
                    Spacer()
                    Button("Done") {
                        isEditFieldFocused = false
                    }
                }
            }
        }
        .onChange(of: selectedPhotoItem) { _, newValue in
            loadReferenceImage(from: newValue)
        }
    }

    private var referenceSection: some View {
        VSSectionCard {
            VStack(alignment: .leading, spacing: 16) {
                VSSectionHeading(
                    title: "Reference Frame",
                    subtitle: "Pick the source image or video you want to reinterpret."
                )

                PhotosPicker(selection: $selectedPhotoItem, matching: .any(of: [.images, .videos])) {
                    ZStack {
                        RoundedRectangle(cornerRadius: 28, style: .continuous)
                            .fill(VSTheme.cardStrong)
                            .frame(height: 240)

                        if let referenceImage {
                            Image(uiImage: referenceImage)
                                .resizable()
                                .scaledToFill()
                                .frame(maxWidth: .infinity)
                                .frame(height: 240)
                                .clipShape(RoundedRectangle(cornerRadius: 28, style: .continuous))

                            if isVideoReference {
                                Circle()
                                    .fill(.ultraThinMaterial)
                                    .frame(width: 58, height: 58)
                                    .overlay {
                                        Image(systemName: "play.fill")
                                            .font(.system(size: 22, weight: .bold))
                                            .foregroundStyle(.white)
                                            .padding(.leading, 3)
                                    }
                                    .shadow(color: .black.opacity(0.18), radius: 14, y: 6)
                            }
                        } else {
                            VStack(spacing: 14) {
                                Image(systemName: "film.stack")
                                    .font(.system(size: 30, weight: .medium))
                                    .foregroundStyle(VSTheme.apricot)

                                Text("Choose Reference Media")
                                    .font(.system(size: 18, weight: .bold, design: .rounded))
                                    .foregroundStyle(VSTheme.ink)

                                Text("Tap to open your library and load a single image or video for composed search.")
                                    .font(.system(size: 14, weight: .medium, design: .rounded))
                                    .foregroundStyle(VSTheme.mutedInk)
                                    .multilineTextAlignment(.center)
                            }
                            .frame(maxWidth: .infinity)
                            .padding(.horizontal, 24)
                            .padding(.top, 24)
                        }
                    }
                    .frame(maxWidth: .infinity)
                    .frame(height: 240)
                    .clipShape(RoundedRectangle(cornerRadius: 28, style: .continuous))
                    .overlay(alignment: .topTrailing) {
                        VSPill(
                            text: isLoadingReference ? "Loading…" : (referenceImage == nil ? "Awaiting Media" : (isVideoReference ? "Video Loaded" : "Image Loaded")),
                            tint: (referenceImage == nil && !isLoadingReference) ? VSTheme.apricot : VSTheme.teal
                        )
                        .padding(16)
                    }
                }
                .buttonStyle(.plain)
            }
        }
    }

    private var promptSection: some View {
        VSSectionCard {
            VStack(alignment: .leading, spacing: 16) {
                VSSectionHeading(
                    title: "Edit Direction",
                    subtitle: "Tell the system what changes from the reference media to the target result."
                )

                VSGlassField {
                    TextField(
                        "Turn this daytime street clip into a rainy neon night scene",
                        text: $editText,
                        axis: .vertical
                    )
                    .lineLimit(4...8)
                    .focused($isEditFieldFocused)
                    .textInputAutocapitalization(.sentences)
                }
            }
        }
    }

    private var actionSection: some View {
        VSSectionCard {
            VStack(alignment: .leading, spacing: 14) {
                VSSectionHeading(
                    title: "Run Search",
                    subtitle: "Visual reasoning will infer the target scene before matching."
                )

                if let retrievalTime = appState.formatDuration(appState.lastComposedSearchDuration) {
                    VSPill(text: "Last run \(retrievalTime)", tint: VSTheme.teal)
                }

                Button(isSearching ? "Searching…" : "Find Matching Media", action: performComposedSearch)
                    .buttonStyle(VSPrimaryButtonStyle(tint: VSTheme.teal))
                    .disabled(referenceImage == nil || editText.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty || isSearching)

                if isSearching {
                    VSStatusBanner(
                        icon: "wand.and.stars",
                        text: "Analyzing the edit instruction, generating reasoning, and retrieving nearby results…",
                        tint: VSTheme.teal
                    )
                }
            }
        }
    }

    @ViewBuilder
    private var resultSection: some View {
        if results.isEmpty && !isSearching {
            VSEmptyState(
                title: "Waiting For A Remix",
                subtitle: "Load a reference image or video and describe the transformation to see composed matches.",
                systemImage: "wand.and.stars"
            )
        } else if !results.isEmpty {
            VStack(alignment: .leading, spacing: 14) {
                VSSectionHeading(
                    title: "Remix Matches",
                    subtitle: "These photos and videos most closely align with your described transformation."
                )

                ResultsView(results: results)
            }
        }
    }

    private func loadReferenceImage(from item: PhotosPickerItem?) {
        guard let item else { return }

        Task {
            do {
                await MainActor.run {
                    isLoadingReference = true
                    results = []
                    reasoningTrace = nil
                    appState.lastErrorMessage = nil
                }

                if item.supportedContentTypes.contains(where: { $0.conforms(to: .movie) || $0.conforms(to: .video) }),
                   let pickedVideo = try await item.loadTransferable(type: PickedVideo.self) {
                    let avAsset = AVAsset(url: pickedVideo.url)
                    let image = try await makeRepresentativeFrame(from: avAsset)
                    await MainActor.run {
                        referenceImage = image
                        isVideoReference = true
                        isLoadingReference = false
                    }
                    return
                }

                if let itemIdentifier = item.itemIdentifier,
                   let asset = PHAsset.fetchAssets(withLocalIdentifiers: [itemIdentifier], options: nil).firstObject {
                    let image = try await loadReferenceImage(for: asset)
                    await MainActor.run {
                        referenceImage = image
                        isVideoReference = asset.mediaType == .video
                        isLoadingReference = false
                    }
                    return
                }

                if let data = try await item.loadTransferable(type: Data.self),
                   let image = UIImage(data: data) {
                    await MainActor.run {
                        referenceImage = image
                        isVideoReference = false
                        isLoadingReference = false
                    }
                    return
                }

                throw NSError(
                    domain: "ComposedRetrievalView",
                    code: -1,
                    userInfo: [NSLocalizedDescriptionKey: "Unable to load the selected media."]
                )
            } catch {
                await MainActor.run {
                    appState.lastErrorMessage = error.localizedDescription
                    isLoadingReference = false
                }
            }
        }
    }

    private func loadReferenceImage(for asset: PHAsset) async throws -> UIImage {
        let targetSize = CGSize(width: 448, height: 448)

        if asset.mediaType == .video {
            let avAsset: AVAsset = try await withCheckedThrowingContinuation { continuation in
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
                                domain: "ComposedRetrievalView",
                                code: -2,
                                userInfo: [NSLocalizedDescriptionKey: "Video asset could not be loaded."]
                            )
                        )
                    }
                }
            }

            return try await makeRepresentativeFrame(from: avAsset)
        }

        return try await withCheckedThrowingContinuation { continuation in
            let options = PHImageRequestOptions()
            options.isNetworkAccessAllowed = true
            options.deliveryMode = .highQualityFormat

            PHImageManager.default().requestImage(
                for: asset,
                targetSize: targetSize,
                contentMode: .aspectFit,
                options: options
            ) { image, info in
                if let error = info?[PHImageErrorKey] as? Error {
                    continuation.resume(throwing: error)
                    return
                }

                if let image {
                    continuation.resume(returning: image)
                } else {
                    continuation.resume(
                        throwing: NSError(
                            domain: "ComposedRetrievalView",
                            code: -4,
                            userInfo: [NSLocalizedDescriptionKey: "Reference image could not be loaded."]
                        )
                    )
                }
            }
        }
    }

    private func makeRepresentativeFrame(from avAsset: AVAsset) async throws -> UIImage {
        let targetSize = CGSize(width: 448, height: 448)
        let frames = try await videoFrameExtractor.extractFrames(from: avAsset, frameCount: 5, targetSize: targetSize)
        guard let representativeFrame = frames[safe: frames.count / 2] ?? frames.first else {
            throw NSError(
                domain: "ComposedRetrievalView",
                code: -3,
                userInfo: [NSLocalizedDescriptionKey: "No preview frame could be extracted from the selected video."]
            )
        }
        return representativeFrame
    }

    private func performComposedSearch() {
        guard let referenceImage else { return }
        let trimmed = editText.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return }

        isEditFieldFocused = false
        isSearching = true
        reasoningTrace = nil
        results = []
        appState.lastErrorMessage = nil

        Task {
            do {
                let (reasoning, searchResults) = try await appState.runComposedSearch(referenceImage: referenceImage, editText: trimmed, topK: 12)
                await MainActor.run {
                    reasoningTrace = """
                    States: \(reasoning.states)
                    Actions: \(reasoning.actions)
                    Scene: \(reasoning.scene)
                    Camera: \(reasoning.camera)
                    Tempo: \(reasoning.tempo)
                    """
                    results = searchResults
                    isSearching = false
                }
            } catch {
                await MainActor.run {
                    appState.lastErrorMessage = error.localizedDescription
                    isSearching = false
                }
            }
        }
    }
}

private struct PickedVideo: Transferable {
    let url: URL

    static var transferRepresentation: some TransferRepresentation {
        FileRepresentation(contentType: .movie) { video in
            SentTransferredFile(video.url)
        } importing: { received in
            let copyURL = URL.temporaryDirectory
                .appendingPathComponent(UUID().uuidString)
                .appendingPathExtension(received.file.pathExtension.isEmpty ? "mov" : received.file.pathExtension)
            try FileManager.default.copyItem(at: received.file, to: copyURL)
            return Self(url: copyURL)
        }
    }
}

private extension Array {
    subscript(safe index: Int) -> Element? {
        guard indices.contains(index) else { return nil }
        return self[index]
    }
}
