import Foundation
import CoreML
import os

enum ModelLoadingError: Error {
    case modelNotFound
    case compilationFailed
    case instantiationFailed
}

struct LoadedModelBundle {
    let visionEncoder: QwenVisionEncoderProtocol
    let languageModel: QwenLanguageModel
    let usingBundledModels: Bool
}

class ModelLoader {
    private enum RuntimeModel {
        static let visionEncoder = "QwenVisionEncoder"
        static let textFusion = "QwenTextFusion"
        static let qwen3DisplayName = "Qwen3-VL-2B"
    }

    private let config: AppConfiguration
    
    init(config: AppConfiguration) {
        self.config = config
    }
    
    func createMLModelConfiguration(computeUnits: MLComputeUnits? = nil) -> MLModelConfiguration {
        let configuration = MLModelConfiguration()
        configuration.computeUnits = computeUnits ?? config.computeUnits
        return configuration
    }

    private func preferredComputeUnits() -> [MLComputeUnits] {
        switch config.computeUnits {
        case .all:
            // Skip ANE-first loading for these oversized models to avoid
            // MILCompilerForANE / ANECCompile() failures on-device.
            return [.cpuAndGPU, .cpuOnly]
        case .cpuAndNeuralEngine:
            return [.cpuAndNeuralEngine, .cpuAndGPU, .cpuOnly]
        case .cpuAndGPU:
            return [.cpuAndGPU, .cpuOnly]
        case .cpuOnly:
            return [.cpuOnly]
        @unknown default:
            return [.cpuAndGPU, .cpuOnly]
        }
    }

    private func compileBundledModel(named name: String) async throws -> URL {
        if let compiledURL = Bundle.main.url(forResource: name, withExtension: "mlmodelc") {
            return compiledURL
        }

        if let packageURL = Bundle.main.url(forResource: name, withExtension: "mlpackage") {
            return try await MLModel.compileModel(at: packageURL)
        }

        throw ModelLoadingError.modelNotFound
    }

    private func loadBundledModel(named name: String) async throws -> MLModel {
        let compiledURL = try await compileBundledModel(named: name)
        var lastError: Error?

        for computeUnits in preferredComputeUnits() {
            do {
                return try MLModel(
                    contentsOf: compiledURL,
                    configuration: createMLModelConfiguration(computeUnits: computeUnits)
                )
            } catch {
                lastError = error
                AppLog.model.error("Failed to load \(name, privacy: .public) with compute units \(String(describing: computeUnits), privacy: .public): \(error.localizedDescription, privacy: .public)")
            }
        }

        if let lastError {
            throw lastError
        }

        throw ModelLoadingError.modelNotFound
    }
    
    func loadVisionEncoder() async throws -> QwenVisionEncoderProtocol {
        do {
            let model = try await loadBundledModel(named: RuntimeModel.visionEncoder)
            let textFusionModel = try? await loadBundledModel(named: RuntimeModel.textFusion)
            AppLog.model.info("Loaded \(RuntimeModel.qwen3DisplayName, privacy: .public) vision encoder bundle")
            return QwenVisionEncoderAdapter(model: model, textFusionModel: textFusionModel)
        } catch {
            AppLog.model.error("Falling back to mock QwenVisionEncoder: \(error.localizedDescription, privacy: .public)")
            return QwenVisionEncoderAdapter()
        }
    }
    
    func loadLanguageModel() async throws -> QwenLanguageModel {
        do {
            let model = try await loadBundledModel(named: RuntimeModel.textFusion)
            AppLog.model.info("Loaded \(RuntimeModel.qwen3DisplayName, privacy: .public) text fusion bundle")
            return QwenLanguageModel(model: model)
        } catch {
            AppLog.model.error("Falling back to mock QwenLanguageModel: \(error.localizedDescription, privacy: .public)")
            return QwenLanguageModel()
        }
    }
    
    func loadTextFusionModel() async throws -> MLModel {
        return try await loadBundledModel(named: RuntimeModel.textFusion)
    }

    func loadRuntimeModels() async -> LoadedModelBundle {
        do {
            async let visionModel = loadBundledModel(named: RuntimeModel.visionEncoder)
            async let textFusionModel = loadBundledModel(named: RuntimeModel.textFusion)
            let (resolvedVisionModel, resolvedTextFusionModel) = try await (visionModel, textFusionModel)

            AppLog.model.info("Loaded \(RuntimeModel.qwen3DisplayName, privacy: .public) runtime bundle for startup")
            return LoadedModelBundle(
                visionEncoder: QwenVisionEncoderAdapter(
                    model: resolvedVisionModel,
                    textFusionModel: resolvedTextFusionModel
                ),
                languageModel: QwenLanguageModel(model: resolvedTextFusionModel),
                usingBundledModels: true
            )
        } catch {
            AppLog.model.error("Startup runtime fell back to lightweight mode because \(RuntimeModel.qwen3DisplayName, privacy: .public) Core ML models failed to load: \(error.localizedDescription, privacy: .public)")
            return LoadedModelBundle(
                visionEncoder: QwenVisionEncoderAdapter(),
                languageModel: QwenLanguageModel(),
                usingBundledModels: false
            )
        }
    }

    /// Loads the three Qwen2-VL generation models and returns a ready-to-use pipeline.
    /// Returns nil silently when the models haven't been converted yet (expected pre-conversion
    /// state). Only logs on genuinely unexpected errors such as corrupt model files.
    func loadQwen2VLPipeline() async -> Qwen2VLInferencePipeline? {
        guard #available(iOS 18.0, *) else {
            AppLog.model.notice("Qwen2VL generation requires iOS 18 or newer; using Vision fallback")
            return nil
        }

        do {
            let visionEncoder = try await loadBundledModel(named: "Qwen2VLVisionEncoder")
            let prefillModel  = try await loadBundledModel(named: "Qwen2VLPrefill")
            let decodeModel   = try await loadBundledModel(named: "Qwen2VLDecodeStep")
            AppLog.model.info("Qwen2VL pipeline loaded successfully")
            return Qwen2VLInferencePipeline(
                visionEncoder: visionEncoder,
                prefillModel:  prefillModel,
                decodeModel:   decodeModel
            )
        } catch ModelLoadingError.modelNotFound {
            // Models haven't been converted yet — this is the expected state before
            // running scripts/convert_qwen2vl_to_coreml.py. Fall back to Vision silently.
            return nil
        } catch {
            // Unexpected error (corrupt weights, ABI mismatch, …) — worth logging.
            AppLog.model.error("Qwen2VL pipeline error: \(error.localizedDescription, privacy: .public)")
            return nil
        }
    }
}
