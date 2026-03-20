import Foundation
import MetricKit
import os

enum AppLog {
    private static let subsystem = Bundle.main.bundleIdentifier ?? "omkar.VisualSeek"

    static let app = Logger(subsystem: subsystem, category: "app")
    static let model = Logger(subsystem: subsystem, category: "model")
    static let storage = Logger(subsystem: subsystem, category: "storage")
    static let retrieval = Logger(subsystem: subsystem, category: "retrieval")
    static let background = Logger(subsystem: subsystem, category: "background")
    static let metrics = Logger(subsystem: subsystem, category: "metrics")
}

enum BackgroundTaskIdentifier {
    static let indexing = "omkar.VisualSeek.indexing"
    static let processing = "omkar.VisualSeek.processing"
}

enum RuntimeEnvironment {
    static var isRunningTests: Bool {
        ProcessInfo.processInfo.environment["XCTestConfigurationFilePath"] != nil
    }
}

final class ProductionMetricsMonitor: NSObject, MXMetricManagerSubscriber {
    static let shared = ProductionMetricsMonitor()

    private var isStarted = false

    private override init() {}

    func start() {
        guard !isStarted else { return }
        MXMetricManager.shared.add(self)
        isStarted = true
        AppLog.metrics.info("MetricKit monitoring started")
    }

    func didReceive(_ payloads: [MXMetricPayload]) {
        AppLog.metrics.notice("Received \(payloads.count, privacy: .public) metric payload(s)")
    }

    func didReceive(_ payloads: [MXDiagnosticPayload]) {
        let crashCount = payloads.reduce(0) { $0 + ($1.crashDiagnostics?.count ?? 0) }
        let hangCount = payloads.reduce(0) { $0 + ($1.hangDiagnostics?.count ?? 0) }
        let cpuExceptionCount = payloads.reduce(0) { $0 + ($1.cpuExceptionDiagnostics?.count ?? 0) }
        let diskWriteCount = payloads.reduce(0) { $0 + ($1.diskWriteExceptionDiagnostics?.count ?? 0) }

        AppLog.metrics.error(
            """
            Received diagnostic payloads=\(payloads.count, privacy: .public) \
            crashes=\(crashCount, privacy: .public) \
            hangs=\(hangCount, privacy: .public) \
            cpuExceptions=\(cpuExceptionCount, privacy: .public) \
            diskWriteExceptions=\(diskWriteCount, privacy: .public)
            """
        )
    }
}
