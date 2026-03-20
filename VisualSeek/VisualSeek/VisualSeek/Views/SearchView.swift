import SwiftUI

struct SearchView: View {
    @EnvironmentObject var appState: AppState
    @State private var searchText = ""
    @State private var isSearching = false
    @State private var results: [RetrievalResult] = []
    @FocusState private var isSearchFieldFocused: Bool

    var body: some View {
        NavigationStack {
            ScrollView {
                VStack(alignment: .leading, spacing: 20) {
                    VSHeroHeader(
                        eyebrow: "Semantic Retrieval",
                        title: "Search With Language",
                        subtitle: "Describe a scene, object, or mood and let the indexed library surface the closest visual matches.",
                        trailingBadge: appState.usingBundledModels ? "Core ML" : "Lightweight"
                    )

                    searchComposer

                    if appState.isModelLoading {
                        VSStatusBanner(
                            icon: "cpu",
                            text: appState.modelStatusMessage,
                            tint: VSTheme.teal
                        )
                    }

                    if isSearching {
                        VSStatusBanner(
                            icon: "sparkle.magnifyingglass",
                            text: "Searching your indexed photos and ranking the strongest matches…",
                            tint: VSTheme.apricot
                        )
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
                        isSearchFieldFocused = false
                    }
                }
            }
        }
    }

    private var searchComposer: some View {
        VSSectionCard {
            VStack(alignment: .leading, spacing: 16) {
                VSSectionHeading(
                    title: "Describe The Photo/Video",
                    subtitle: "Natural language works best: try people, objects, location, light, color, action, or motion."
                )

                VSGlassField {
                    HStack(spacing: 12) {
                        Image(systemName: "magnifyingglass")
                            .foregroundStyle(VSTheme.ink.opacity(0.64))

                        TextField("Sunset beach with silhouettes and warm sky", text: $searchText)
                            .textInputAutocapitalization(.never)
                            .autocorrectionDisabled()
                            .submitLabel(.search)
                            .focused($isSearchFieldFocused)
                            .onSubmit(performSearch)
                    }
                }

                HStack(spacing: 12) {
                    Button("Search Library", action: performSearch)
                        .buttonStyle(VSPrimaryButtonStyle())
                        .disabled(searchText.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty || isSearching)

                    if !searchText.isEmpty {
                        Button("Clear") {
                            searchText = ""
                            results = []
                            appState.lastErrorMessage = nil
                        }
                        .buttonStyle(VSSecondaryButtonStyle())
                        .disabled(isSearching)
                    }
                }
            }
        }
    }

    @ViewBuilder
    private var resultSection: some View {
        if results.isEmpty && !isSearching {
            VSEmptyState(
                title: "No Query Yet",
                subtitle: "Run a search to see your matches appear here as visual cards.",
                systemImage: "magnifyingglass.circle"
            )
        } else if !results.isEmpty {
            VStack(alignment: .leading, spacing: 14) {
                HStack {
                    VSSectionHeading(
                        title: "Matches",
                        subtitle: "Ranked by similarity across your indexed collection."
                    )
                    Spacer()
                    if let retrievalTime = appState.formatDuration(appState.lastTextSearchDuration) {
                        VSPill(text: retrievalTime, tint: VSTheme.teal)
                    }
                }

                ResultsView(results: results)
            }
        }
    }

    private func performSearch() {
        let trimmed = searchText.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return }

        isSearchFieldFocused = false
        isSearching = true
        appState.lastErrorMessage = nil
        results = []

        Task {
            do {
                let searchResults = try await appState.runTextSearch(trimmed, topK: 20)
                await MainActor.run {
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
