import SwiftUI
import CoreML

@main
struct VisualSeekApp: App {
    @StateObject private var appState = AppState()

    init() {
        guard !RuntimeEnvironment.isRunningTests else { return }
        ProductionMetricsMonitor.shared.start()
        BackgroundIndexer.shared.registerBackgroundTasks()
    }

    var body: some Scene {
        WindowGroup {
            ContentView()
                .environmentObject(appState)
        }
    }
}
