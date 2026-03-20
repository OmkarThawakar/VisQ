import SwiftUI
import Photos

struct ResultsView: View {
    let results: [RetrievalResult]

    private let columns = [
        GridItem(.adaptive(minimum: 170, maximum: 260), spacing: 14, alignment: .top)
    ]

    var body: some View {
        LazyVGrid(columns: columns, spacing: 14) {
            ForEach(results) { result in
                AssetResultCard(result: result, scoreLabel: scoreLabel(for: result.score))
            }
        }
    }

    private func scoreLabel(for score: Float) -> String {
        "\(Int(score * 100))% match"
    }
}

private struct AssetResultCard: View {
    let result: RetrievalResult
    let scoreLabel: String

    @State private var asset: PHAsset?

    var body: some View {
        Group {
            if let asset {
                NavigationLink(destination: PhotoDetailView(asset: asset, retrievalResult: result)) {
                    cardBody(asset: asset)
                }
                .buttonStyle(.plain)
            } else {
                cardBody(asset: nil)
                    .task {
                        loadAsset()
                    }
            }
        }
    }

    @ViewBuilder
    private func cardBody(asset: PHAsset?) -> some View {
        VStack(alignment: .leading, spacing: 12) {
            ZStack(alignment: .topLeading) {
                ResultThumbnailView(assetID: result.id)
                    .aspectRatio(0.88, contentMode: .fit)
                    .clipShape(RoundedRectangle(cornerRadius: 28, style: .continuous))

                VSPill(text: scoreLabel, tint: VSTheme.apricot)
                    .padding(12)

                if asset?.mediaType == .video {
                    VSPill(text: "Video", tint: VSTheme.teal)
                        .padding(.horizontal, 12)
                        .padding(.top, 52)
                }
            }

            VStack(alignment: .leading, spacing: 6) {
                Text(result.title ?? "Indexed Media")
                    .font(.system(size: 16, weight: .bold, design: .rounded))
                    .foregroundStyle(VSTheme.ink)
                    .lineLimit(2)

                Text(asset?.mediaType == .video ? "Matching video clip" : "Visual similarity match")
                    .font(.system(size: 13, weight: .medium, design: .rounded))
                    .foregroundStyle(VSTheme.ink.opacity(0.76))

                if let explanation = result.matchExplanation {
                    VSMatchChipRow(chips: explanation.featureChips, tint: VSTheme.teal)

                    Text(explanation.summary)
                        .font(.system(size: 12, weight: .medium, design: .rounded))
                        .foregroundStyle(VSTheme.mutedInk)
                        .lineLimit(2)
                }

                Text(result.id)
                    .font(.caption.monospaced())
                    .foregroundStyle(VSTheme.ink.opacity(0.68))
                    .lineLimit(1)
            }
        }
        .padding(12)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(VSTheme.card, in: RoundedRectangle(cornerRadius: 30, style: .continuous))
        .overlay {
            RoundedRectangle(cornerRadius: 30, style: .continuous)
                .stroke(VSTheme.line, lineWidth: 1)
        }
        .shadow(color: VSTheme.shadow, radius: 14, y: 8)
    }

    private func loadAsset() {
        let fetchResult = PHAsset.fetchAssets(withLocalIdentifiers: [result.id], options: nil)
        asset = fetchResult.firstObject
    }
}

struct ResultThumbnailView: View {
    let assetID: String

    @State private var image: UIImage?
    @State private var didRequestImage = false

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
                } else {
                    Image(systemName: "photo")
                        .font(.system(size: 24, weight: .medium))
                        .foregroundStyle(VSTheme.ink.opacity(0.68))
                }
            }
            .task {
                guard !didRequestImage else { return }
                didRequestImage = true
                fetchThumbnail(size: proxy.size)
            }
        }
    }

    private func fetchThumbnail(size: CGSize) {
        let result = PHAsset.fetchAssets(withLocalIdentifiers: [assetID], options: nil)
        guard let asset = result.firstObject else { return }

        let options = PHImageRequestOptions()
        options.isNetworkAccessAllowed = true
        options.deliveryMode = .opportunistic
        options.resizeMode = .fast

        let scale = UIScreen.main.scale
        let targetSize = CGSize(width: max(size.width, 1) * scale, height: max(size.height, 1) * scale)

        PHImageManager.default().requestImage(
            for: asset,
            targetSize: targetSize,
            contentMode: .aspectFill,
            options: options
        ) { image, _ in
            self.image = image
        }
    }
}
