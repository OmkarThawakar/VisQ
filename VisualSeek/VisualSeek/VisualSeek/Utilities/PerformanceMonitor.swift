import Foundation
import Combine

class PerformanceMonitor: ObservableObject {
    @Published var currentANEThroughput: Double = 0.0 // images/sec
    @Published var currentQueryLatencyMs: Double = 0.0
    
    // In reality this would integrate with os_signpost or Instruments
    // to periodically update metrics in the UI.
    
    func logIndexingBatch(batchSize: Int, durationInSeconds: Double) {
        let throughput = Double(batchSize) / durationInSeconds
        DispatchQueue.main.async {
            self.currentANEThroughput = throughput
        }
    }
    
    func logRetrievalQuery(durationInSeconds: Double) {
        let latencyMs = durationInSeconds * 1000.0
        DispatchQueue.main.async {
            self.currentQueryLatencyMs = latencyMs
        }
    }
}
