import SwiftUI

struct SettingsView: View {
    @EnvironmentObject var appState: AppState

    var body: some View {
        NavigationStack {
            ScrollView {
                VStack(alignment: .leading, spacing: 20) {
                    VSHeroHeader(
                        eyebrow: "Studio Controls",
                        title: "Settings",
                        subtitle: "Tune indexing behavior, retrieval defaults, storage visibility, and runtime diagnostics from one place.",
                        trailingBadge: appState.usingBundledModels ? "Core ML" : "Lightweight"
                    )

                    modelCard
                    indexingCard
                    retrievalCard
                    storageCard
                    diagnosticsCard
                    attributionFooter

                    if let lastErrorMessage = appState.lastErrorMessage, !lastErrorMessage.isEmpty {
                        VSStatusBanner(icon: "exclamationmark.triangle.fill", text: lastErrorMessage, tint: VSTheme.danger)
                    }
                }
                .padding(.horizontal, 20)
                .padding(.top, 16)
                .padding(.bottom, 24)
            }
            .scrollIndicators(.hidden)
            .toolbar(.hidden, for: .navigationBar)
        }
    }

    private var modelCard: some View {
        VSSectionCard {
            VStack(alignment: .leading, spacing: 14) {
                VSSectionHeading(
                    title: "Runtime",
                    subtitle: "What inference path and footprint the app is using right now."
                )

                settingsRow("Inference Mode", value: appState.usingBundledModels ? "Qwen3-VL-2B Core ML" : "Lightweight On-Device")
                settingsRow("Batch Size", value: "\(appState.configuration.batchSize)")
                settingsRow("Embedding Dimension", value: "\(appState.configuration.embeddingDimension)")
                settingsRow("Model Status", value: appState.modelStatusMessage)
            }
        }
    }

    private var indexingCard: some View {
        VSSectionCard {
            VStack(alignment: .leading, spacing: 14) {
                VSSectionHeading(
                    title: "Indexing Behavior",
                    subtitle: "Choose how aggressively the library is processed in the background."
                )

                Toggle("Background Indexing", isOn: $appState.configuration.backgroundIndexingEnabled)
                Toggle("Index on Wi-Fi Only", isOn: $appState.configuration.indexOnWifiOnly)
                Toggle("Index when Charging Only", isOn: $appState.configuration.indexOnChargingOnly)
            }
            .toggleStyle(SwitchToggleStyle(tint: VSTheme.apricot))
        }
    }

    private var retrievalCard: some View {
        VSSectionCard {
            VStack(alignment: .leading, spacing: 14) {
                VSSectionHeading(
                    title: "Retrieval",
                    subtitle: "Control the number of results returned and whether composed reasoning is active."
                )

                VStack(alignment: .leading, spacing: 8) {
                    HStack {
                        Text("Default Top K")
                            .font(.system(size: 15, weight: .bold, design: .rounded))
                            .foregroundStyle(VSTheme.ink)
                        Spacer()
                        Text("\(appState.configuration.defaultTopK)")
                            .font(.system(size: 15, weight: .bold, design: .rounded))
                            .foregroundStyle(VSTheme.apricot)
                    }

                    Stepper("", value: $appState.configuration.defaultTopK, in: 5...100, step: 5)
                        .labelsHidden()
                }

                Toggle("Use Reasoning For Composed Retrieval", isOn: $appState.configuration.useReasoningForComposedRetrieval)
            }
            .toggleStyle(SwitchToggleStyle(tint: VSTheme.teal))
        }
    }

    private var storageCard: some View {
        VSSectionCard {
            VStack(alignment: .leading, spacing: 14) {
                VSSectionHeading(
                    title: "Storage",
                    subtitle: "Inspect the persistence layer and clear everything when you need a clean slate."
                )

                settingsRow("Database", value: appState.configuration.embeddingStoragePath.lastPathComponent)
                settingsRow("Saved Embeddings", value: "\(appState.indexedPhotoCount)")

                Button("Clear All Embeddings") {
                    appState.clearEmbeddings()
                }
                .buttonStyle(VSDestructiveButtonStyle())
            }
        }
    }

    private var diagnosticsCard: some View {
        VSSectionCard {
            VStack(alignment: .leading, spacing: 14) {
                VSSectionHeading(
                    title: "Diagnostics",
                    subtitle: "Run a self-check and inspect tokenizer or runtime outputs."
                )

                Button {
                    Task { await appState.runDiagnostics() }
                } label: {
                    HStack {
                        if appState.isRunningDiagnostics {
                            ProgressView()
                                .tint(.white)
                        } else {
                            Image(systemName: "stethoscope")
                        }
                        Text(appState.isRunningDiagnostics ? "Running Self-Check…" : "Run Model Self-Check")
                    }
                }
                .buttonStyle(VSPrimaryButtonStyle(tint: VSTheme.teal))

                if let diagnosticsReport = appState.diagnosticsReport {
                    ScrollView(.horizontal, showsIndicators: false) {
                        Text(diagnosticsReport)
                            .font(.caption.monospaced())
                            .foregroundStyle(VSTheme.ink)
                            .padding(14)
                    }
                    .background(VSTheme.field, in: RoundedRectangle(cornerRadius: 20, style: .continuous))
                    .overlay {
                        RoundedRectangle(cornerRadius: 20, style: .continuous)
                            .stroke(VSTheme.line, lineWidth: 1)
                    }
                }
            }
        }
    }

    private var attributionFooter: some View {
        VStack(alignment: .leading, spacing: 6) {
            Text("Designed and Built by Omkar Thawakar")
                .font(.system(size: 15, weight: .semibold, design: .rounded))
                .foregroundStyle(VSTheme.ink)

            Text("PhD Student at MBZUAI")
                .font(.system(size: 13, weight: .medium, design: .rounded))
                .foregroundStyle(VSTheme.mutedInk)
        }
        .padding(.horizontal, 4)
        .padding(.top, 4)
        .frame(maxWidth: .infinity, alignment: .leading)
    }

    private func settingsRow(_ label: String, value: String) -> some View {
        HStack(alignment: .top, spacing: 12) {
            Text(label)
                .font(.system(size: 15, weight: .bold, design: .rounded))
                .foregroundStyle(VSTheme.mutedInk)
            Spacer()
            Text(value)
                .font(.system(size: 15, weight: .bold, design: .rounded))
                .foregroundStyle(VSTheme.ink)
                .multilineTextAlignment(.trailing)
        }
    }
}
