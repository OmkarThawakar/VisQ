import Foundation
import BackgroundTasks
import os

final class BackgroundIndexer {
    static let shared = BackgroundIndexer()
    
    private let processingIdentifier = BackgroundTaskIdentifier.processing
    
    private init() {}
    
    func registerBackgroundTasks() {
        BGTaskScheduler.shared.register(forTaskWithIdentifier: processingIdentifier, using: nil) { task in
            guard let processingTask = task as? BGProcessingTask else {
                AppLog.background.error("Received unexpected background task type for identifier \(self.processingIdentifier, privacy: .public)")
                task.setTaskCompleted(success: false)
                return
            }
            self.handleAppRefresh(task: processingTask)
        }
    }
    
    func scheduleBackgroundIndexing(requiresNetwork: Bool = false, requiresCharging: Bool = true) {
        let request = BGProcessingTaskRequest(identifier: processingIdentifier)
        request.requiresNetworkConnectivity = requiresNetwork
        request.requiresExternalPower = requiresCharging
        
        do {
            try BGTaskScheduler.shared.submit(request)
        } catch {
            AppLog.background.error("Could not schedule background indexing: \(error.localizedDescription, privacy: .public)")
        }
    }
    
    private func handleAppRefresh(task: BGProcessingTask) {
        scheduleBackgroundIndexing() // Reschedule for next time
        
        let queue = OperationQueue()
        queue.maxConcurrentOperationCount = 1
        
        let operation = BlockOperation {
            // Wait for semaphore or use async/await bridge to run indexer pipeline
            let semaphore = DispatchSemaphore(value: 0)
            
            Task {
                await MainActor.run {
                    AppLog.background.info("Background indexing started")
                }
                // A real implementation would invoke IndexingPipeline.startIndexing() here
                // Note: Models and pipelines need access from background
                try? await Task.sleep(nanoseconds: 5_000_000_000) // Mock 5 seconds of work
                await MainActor.run {
                    AppLog.background.info("Background indexing completed")
                }
                semaphore.signal()
            }
            
            semaphore.wait()
        }
        
        task.expirationHandler = {
            queue.cancelAllOperations()
        }
        
        operation.completionBlock = {
            task.setTaskCompleted(success: !operation.isCancelled)
        }
        
        queue.addOperation(operation)
    }
}
