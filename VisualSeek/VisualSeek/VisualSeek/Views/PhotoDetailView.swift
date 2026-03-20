import SwiftUI
import Photos
import AVKit

struct PhotoDetailView: View {
    let asset: PHAsset
    var retrievalResult: RetrievalResult? = nil
    @State private var image: UIImage?
    @State private var player: AVPlayer?
    @State private var isLoading = true

    var body: some View {
        ZStack {
            Color.black.ignoresSafeArea()

            if asset.mediaType == .video, let player {
                VideoPlayer(player: player)
                    .ignoresSafeArea()
                    .onAppear {
                        player.play()
                    }
            } else if let image {
                Image(uiImage: image)
                    .resizable()
                    .scaledToFit()
                    .frame(maxWidth: .infinity, maxHeight: .infinity)
                    .ignoresSafeArea()
                    .transition(.opacity.animation(.easeIn(duration: 0.2)))
            } else if isLoading {
                ProgressView()
                    .tint(.white)
                    .scaleEffect(1.3)
            }
        }
        .navigationBarTitleDisplayMode(.inline)
        .toolbarColorScheme(.dark, for: .navigationBar)
        .toolbarBackground(.hidden, for: .navigationBar)
        .safeAreaInset(edge: .bottom, spacing: 0) {
            detailOverlay
        }
        .toolbar {
            ToolbarItem(placement: .topBarTrailing) {
                Button {
                    shareAsset()
                } label: {
                    Image(systemName: "square.and.arrow.up")
                }
                .disabled(image == nil)
            }
        }
        .onAppear(perform: loadAssetContent)
        .onDisappear {
            player?.pause()
        }
    }

    private var detailOverlay: some View {
        VStack(spacing: 0) {
            LinearGradient(
                colors: [Color.clear, Color.black.opacity(0.5)],
                startPoint: .top,
                endPoint: .bottom
            )
            .frame(height: 60)

            VStack(spacing: 0) {
                HStack(alignment: .center, spacing: 16) {
                    VStack(alignment: .leading, spacing: 6) {
                        Text(asset.creationDate?.formatted(date: .abbreviated, time: .omitted) ?? "Photo Detail")
                            .font(.system(size: 20, weight: .bold, design: .serif))
                            .foregroundStyle(.white)

                        Text(detailCaption)
                            .font(.system(size: 13, weight: .medium, design: .rounded))
                            .foregroundStyle(.white.opacity(0.72))
                    }

                    Spacer()

                    Button {
                        shareAsset()
                    } label: {
                        Image(systemName: "square.and.arrow.up")
                            .font(.system(size: 16, weight: .bold))
                            .foregroundStyle(.white)
                            .frame(width: 46, height: 46)
                            .background(Color.white.opacity(0.14), in: RoundedRectangle(cornerRadius: 16, style: .continuous))
                    }
                    .disabled(image == nil)
                }
                .padding(.horizontal, 20)
                .padding(.top, 16)
                .padding(.bottom, retrievalResult?.matchExplanation == nil ? 18 : 12)

                if let explanation = retrievalResult?.matchExplanation {
                    VStack(alignment: .leading, spacing: 12) {
                        Text("Why This Matched")
                            .font(.system(size: 15, weight: .bold, design: .rounded))
                            .foregroundStyle(.white)

                        Text(explanation.summary)
                            .font(.system(size: 13, weight: .medium, design: .rounded))
                            .foregroundStyle(.white.opacity(0.82))

                        VSMatchChipRow(chips: explanation.featureChips, tint: .white)

                        if !explanation.reasonChips.isEmpty {
                            Text("Matched because of \(joinedReasonText(explanation.reasonChips)).")
                                .font(.system(size: 12, weight: .medium, design: .rounded))
                                .foregroundStyle(.white.opacity(0.72))
                        }

                        Text(explanation.detail)
                            .font(.system(size: 12, weight: .medium, design: .rounded))
                            .foregroundStyle(.white.opacity(0.68))
                    }
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .padding(.horizontal, 20)
                    .padding(.bottom, 18)
                }
            }
            .background(.ultraThinMaterial)
        }
    }

    private var detailCaption: String {
        let kind = asset.mediaType == .video ? "Video frame preview" : "Image preview"
        return "\(kind) • \(asset.pixelWidth) × \(asset.pixelHeight)"
    }

    private func loadAssetContent() {
        if asset.mediaType == .video {
            fetchVideoPreview()
            fetchPlayerItem()
        } else {
            fetchHighQualityImage()
        }
    }

    private func fetchHighQualityImage() {
        let manager = PHImageManager.default()
        let options = PHImageRequestOptions()
        options.isNetworkAccessAllowed = true
        options.deliveryMode = .opportunistic
        options.resizeMode = .none

        let screenScale = UIScreen.main.scale
        let screenBounds = UIScreen.main.bounds
        let targetSize = CGSize(
            width: screenBounds.width * screenScale,
            height: screenBounds.height * screenScale
        )

        manager.requestImage(
            for: asset,
            targetSize: targetSize,
            contentMode: .aspectFit,
            options: options
        ) { image, info in
            guard let image else { return }
            let isDegraded = (info?[PHImageResultIsDegradedKey] as? Bool) ?? false
            self.image = image
            if !isDegraded {
                isLoading = false
            }
        }
    }

    private func fetchVideoPreview() {
        let manager = PHImageManager.default()
        let options = PHImageRequestOptions()
        options.isNetworkAccessAllowed = true
        options.deliveryMode = .highQualityFormat

        let targetSize = CGSize(width: 1200, height: 1200)
        manager.requestImage(
            for: asset,
            targetSize: targetSize,
            contentMode: .aspectFit,
            options: options
        ) { image, _ in
            if let image {
                self.image = image
            }
        }
    }

    private func fetchPlayerItem() {
        let manager = PHImageManager.default()
        let options = PHVideoRequestOptions()
        options.deliveryMode = .highQualityFormat
        options.isNetworkAccessAllowed = true

        manager.requestPlayerItem(forVideo: asset, options: options) { item, _ in
            guard let item else { return }
            DispatchQueue.main.async {
                player = AVPlayer(playerItem: item)
                isLoading = false
            }
        }
    }

    private func shareAsset() {
        guard let image else { return }
        let activityController = UIActivityViewController(activityItems: [image], applicationActivities: nil)
        if let scene = UIApplication.shared.connectedScenes.first as? UIWindowScene,
           let root = scene.windows.first?.rootViewController {
            root.present(activityController, animated: true)
        }
    }

    private func joinedReasonText(_ reasons: [String]) -> String {
        switch reasons.count {
        case 0:
            return "visual overlap"
        case 1:
            return reasons[0]
        case 2:
            return "\(reasons[0]) and \(reasons[1])"
        default:
            let head = reasons.dropLast().joined(separator: ", ")
            return "\(head), and \(reasons[reasons.count - 1])"
        }
    }
}
