import Foundation
import UIKit
import Vision
import os

// MARK: - QwenVLDescriptionGenerator
//
// Produces rich, multi-sentence natural-language descriptions optimised for
// fine-grained image retrieval. Six Vision requests run in a single handler
// pass to cover every dimension needed for accurate search:
//
//   • People       — VNDetectHumanRectanglesRequest (catches from-behind/profile)
//                    + VNDetectFaceRectanglesRequest (camera-facing count)
//   • Activity     — VNDetectHumanBodyPoseRequest (standing/sitting/arms raised/in-motion)
//   • Animals      — VNRecognizeAnimalsRequest (species-level)
//   • Objects/scene— VNClassifyImageRequest (top 20, threshold 0.03)
//   • Text         — VNRecognizeTextRequest (.accurate)
//   • Color/light  — pixel sampling (64×64) for palette, sky, time-of-day

final class QwenVLDescriptionGenerator {

    // MARK: - Qwen2VL pipeline (set by AppState after model load)

    /// When set, all description requests are routed through the real Qwen2-VL
    /// language model. The Vision-framework path is used as a fallback when the
    /// pipeline is nil or throws.
    var qwen2VLPipeline: Qwen2VLInferencePipeline?
    var qwen2VLPipelineProvider: (() async -> Qwen2VLInferencePipeline?)?

    // MARK: - Public API

    func generate(embedding: [Float], image: UIImage) -> String {
        // Synchronous path — Qwen2VL requires async; use Vision fallback here.
        buildDescription(for: image)
    }

    func generate(image: UIImage, using encoder: QwenVisionEncoderProtocol?) async -> String {
        // Try Qwen2-VL first; fall back to Vision framework if unavailable or on error.
        if let pipeline = await resolvePipeline() {
            do {
                return try await pipeline.generateDescription(for: image)
            } catch is CancellationError {
                return buildDescription(for: image)   // cancelled — use fast fallback
            } catch {
                AppLog.model.error("Qwen2VL generation failed; using Vision fallback: \(error.localizedDescription, privacy: .public)")
            }
        }
        return buildDescription(for: image)
    }

    private func resolvePipeline() async -> Qwen2VLInferencePipeline? {
        if let qwen2VLPipeline {
            return qwen2VLPipeline
        }

        guard let qwen2VLPipelineProvider else {
            return nil
        }

        let pipeline = await qwen2VLPipelineProvider()
        qwen2VLPipeline = pipeline
        return pipeline
    }

    // MARK: - Core Builder

    private func buildDescription(for image: UIImage) -> String {
        guard let cgImage = image.cgImage else {
            return ImageSemanticAnalyzer.description(for: image)
        }

        // ── Six Vision requests in one handler pass ──────────────────────────
        let classifyReq = VNClassifyImageRequest()

        let textReq = VNRecognizeTextRequest()
        textReq.recognitionLevel = .accurate
        textReq.usesLanguageCorrection = true

        // faceReq  → how many people are FACING the camera
        let faceReq  = VNDetectFaceRectanglesRequest()
        // humanReq → TOTAL human body count (face OR back OR profile OR silhouette)
        let humanReq = VNDetectHumanRectanglesRequest()
        let animalReq = VNRecognizeAnimalsRequest()
        // poseReq  → joint positions for activity inference
        let poseReq  = VNDetectHumanBodyPoseRequest()

        let handler = VNImageRequestHandler(cgImage: cgImage, options: [:])
        try? handler.perform([classifyReq, textReq, faceReq, humanReq, animalReq, poseReq])

        // ── Collect results ──────────────────────────────────────────────────
        // Lower threshold (0.03) and keep top 20 so that objects in the mid/background
        // that scored ~0.04-0.06 are not silently dropped.
        let classifications = (classifyReq.results ?? [])
            .filter { $0.confidence > 0.03 }
            .sorted { $0.confidence > $1.confidence }
            .prefix(20)

        let faces  = faceReq.results  ?? []
        let humans = humanReq.results ?? []
        // Use the larger count — human body detector catches people not facing camera
        let personCount  = max(faces.count, humans.count)
        let facingCamera = faces.count

        let poseObs      = poseReq.results ?? []
        let poseActivity = inferPoseActivities(from: poseObs)

        let animals: [String] = (animalReq.results ?? [])
            .flatMap { $0.labels.filter { $0.confidence > 0.4 }.map { $0.identifier.lowercased() } }
            .uniqued()

        let ocrLines = (textReq.results ?? [])
            .compactMap { $0.topCandidates(1).first?.string }
            .map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
            .filter { $0.count > 2 }
            .uniqued()
            .prefix(5)

        let labels = Array(classifications)
            .map { readableLabel($0.identifier) }
            .filter { !blocklist.contains($0) && !$0.isEmpty }

        let color = colorAnalysis(for: cgImage)

        // ── Assemble description ─────────────────────────────────────────────
        var sentences: [String] = []

        // 1. Primary subjects — people (from body detector), animals, top objects
        sentences.append(
            primarySubjectSentence(labels: labels, personCount: personCount,
                                   animals: animals, image: image)
        )

        // 2. People detail — ALWAYS added when at least one person is detected.
        //    This is the sentence that was missing for the "person on beach" case.
        if personCount > 0 {
            sentences.append(
                peopleDetailSentence(
                    personCount: personCount,
                    facingCamera: facingCamera,
                    humans: humans,
                    faces: faces,
                    poseActivity: poseActivity,
                    labels: labels
                )
            )
        }

        // 3. Scene / environment
        if let scene = sceneSentence(labels: labels, color: color) {
            sentences.append(scene)
        }

        // 4. Notable objects and foreground details
        if let fg = objectsSentence(labels: labels, personCount: personCount) {
            sentences.append(fg)
        }

        // 5. Background / environmental elements
        if let bg = backgroundSentence(labels: labels, color: color) {
            sentences.append(bg)
        }

        // 6. Color palette
        sentences.append(color.atmosphereSentence)

        // 7. Lighting / time of day
        sentences.append(color.lightingSentence)

        // 8. Composition and framing
        sentences.append(
            compositionSentence(image: image, personCount: personCount,
                                faces: faces, humans: humans, labels: labels)
        )

        // 9. Visible text
        if !ocrLines.isEmpty {
            let joined = ocrLines.joined(separator: "; ")
            sentences.append("Text visible in the image: \"\(joined)\".")
        }

        return sentences.joined(separator: " ")
    }

    // MARK: - Sentence: Primary Subjects

    private func primarySubjectSentence(
        labels: [String],
        personCount: Int,
        animals: [String],
        image: UIImage
    ) -> String {
        var tokens: [String] = []

        // People — personCount comes from the human-body detector so it catches
        // people facing away, in silhouette, at a distance, etc.
        if personCount == 1 {
            // Try to find a more specific age/gender label from VNClassify
            var specificLabel = "a person"
            for term in personSpecificTerms where labels.contains(where: { $0.contains(term.keyword) }) {
                specificLabel = term.label
                break
            }
            tokens.append(specificLabel)
        } else if personCount == 2 {
            tokens.append("two people")
        } else if personCount == 3 {
            tokens.append("three people")
        } else if personCount >= 4 {
            tokens.append("a group of \(personCount) people")
        }

        // Animals
        for animal in animals.prefix(2) { tokens.append("a \(animal)") }

        // Top non-person, non-animal objects from classification
        let topObjects = labels.filter { label in
            let isPersonLabel = personCount > 0 && personTerms.contains(where: { label.contains($0) })
            let isAnimalLabel = Set(animals).contains(where: { label.contains($0) })
            return !isPersonLabel && !isAnimalLabel
        }.prefix(5)

        for obj in topObjects where !tokens.contains(obj) { tokens.append(obj) }

        guard !tokens.isEmpty else { return fallbackSubjectSentence(for: image) }
        return "This photo shows \(tokens.joined(separator: ", "))."
    }

    // MARK: - Sentence: People Detail

    /// Provides camera orientation, spatial position, and activity for every
    /// detected person. This is the key sentence for gallery photos.
    private func peopleDetailSentence(
        personCount: Int,
        facingCamera: Int,
        humans: [VNHumanObservation],
        faces: [VNFaceObservation],
        poseActivity: [String],
        labels: [String]
    ) -> String {
        var parts: [String] = []

        // ── Camera orientation ──
        if facingCamera == personCount && personCount > 0 {
            parts.append(personCount == 1 ? "facing the camera" : "all facing the camera")
        } else if facingCamera == 0 {
            parts.append(personCount == 1
                ? "facing away or in profile (back/side to camera)"
                : "not facing the camera directly")
        } else {
            parts.append("\(facingCamera) of \(personCount) facing the camera")
        }

        // ── Spatial position (from largest human bounding box area) ──
        let largestHuman = humans.max(by: {
            $0.boundingBox.width * $0.boundingBox.height < $1.boundingBox.width * $1.boundingBox.height
        })
        if let box = largestHuman?.boundingBox {
            let area = box.width * box.height
            if area > 0.25 {
                parts.append("prominently close up and filling most of the frame")
            } else if area > 0.10 {
                parts.append("clearly visible in the foreground")
            } else if area > 0.03 {
                parts.append("visible in the mid-ground")
            } else {
                parts.append("appearing small in the distance or background")
            }

            // Horizontal position
            let cx = box.midX
            if cx < 0.35      { parts.append("positioned toward the left side") }
            else if cx > 0.65 { parts.append("positioned toward the right side") }
            else              { parts.append("roughly centred horizontally") }
        }

        // ── Activity: body pose takes priority, classification labels as fallback ──
        if !poseActivity.isEmpty {
            parts.append(contentsOf: poseActivity.prefix(2))
        } else {
            let activityLabels = labels
                .filter { l in classificationActivityKeywords.contains { l.contains($0) } }
                .prefix(2)
            parts.append(contentsOf: activityLabels)
        }

        let subject = personCount == 1 ? "The person is" : "The people are"
        return "\(subject) \(parts.joined(separator: ", "))."
    }

    // MARK: - Sentence: Scene Context

    private func sceneSentence(labels: [String], color: ColorInfo) -> String? {
        let indoorKeywords = [
            "kitchen", "living room", "bedroom", "bathroom", "office", "restaurant",
            "café", "cafe", "shop", "mall", "gym", "classroom", "library", "hospital",
            "corridor", "hallway", "interior", "room", "bar", "pub", "studio",
            "apartment", "home", "house", "hall", "lounge", "canteen", "store", "garage"
        ]
        let outdoorKeywords = [
            "beach", "mountain", "forest", "park", "garden", "street", "highway",
            "field", "desert", "lake", "river", "ocean", "valley", "cliff",
            "countryside", "cityscape", "skyline", "rooftop", "playground",
            "coast", "shore", "sea", "meadow", "path", "trail", "pier", "dock",
            "yard", "patio", "deck", "terrace", "poolside", "waterfront"
        ]
        let eventKeywords = [
            "wedding", "birthday", "party", "celebration", "graduation", "concert",
            "festival", "ceremony", "sport", "game", "match", "race", "competition",
            "meeting", "conference", "market", "fair", "parade", "event", "show",
            "performance", "gathering", "reunion", "holiday", "vacation", "trip"
        ]

        let indoorMatch  = labels.first(where: { l in indoorKeywords.contains  { l.contains($0) } })
        let outdoorMatch = labels.first(where: { l in outdoorKeywords.contains { l.contains($0) } })
        let eventMatch   = labels.first(where: { l in eventKeywords.contains   { l.contains($0) } })

        var parts: [String] = []
        if let ev  = eventMatch   { parts.append("a \(ev) setting") }
        if let out = outdoorMatch { parts.append("an outdoor \(out) environment") }
        else if let ind = indoorMatch { parts.append("an indoor \(ind) setting") }
        else if color.isOutdoor   { parts.append("an outdoor setting") }

        guard !parts.isEmpty else { return nil }
        return "The scene takes place in \(parts.joined(separator: " within "))."
    }

    // MARK: - Sentence: Objects

    private func objectsSentence(labels: [String], personCount: Int) -> String? {
        let objectKeywords: [String] = [
            // Furniture / home
            "table", "chair", "sofa", "couch", "bed", "desk", "shelf", "counter", "stool",
            // Vehicles
            "car", "truck", "bus", "bicycle", "motorcycle", "boat", "airplane", "train",
            "scooter", "van", "jeep", "ferry", "kayak",
            // Nature objects
            "tree", "flower", "plant", "bush", "grass", "rock", "stone", "sand",
            "wave", "waterfall", "leaf", "branch",
            // Food and drink
            "food", "drink", "coffee", "tea", "cake", "pizza", "burger", "sandwich",
            "fruit", "vegetable", "ice cream", "beer", "wine", "cocktail", "sushi",
            "noodle", "rice", "bread", "salad", "dessert", "snack",
            // Tech and everyday
            "phone", "laptop", "computer", "camera", "book", "bag", "backpack", "purse",
            "umbrella", "bottle", "cup", "glass", "hat", "cap", "sunglasses", "watch",
            "headphones", "earphones", "keyboard", "monitor",
            // Entertainment / sport
            "ball", "toy", "guitar", "piano", "microphone", "drum", "speaker",
            "bat", "racket", "helmet", "surfboard", "skateboard", "skis",
            "snowboard", "tent", "campfire", "balloon",
            // Interior details
            "candle", "lamp", "light", "window", "door", "sign", "screen", "mirror",
            "curtain", "pillow", "blanket", "vase", "frame", "painting", "poster",
            // Outdoor / architecture
            "fence", "bench", "stairs", "bridge", "fountain", "statue", "pillar",
            "flag", "gate", "wall", "tower", "arch"
        ]
        let activityKeywords: [String] = [
            "eating", "drinking", "cooking", "running", "walking", "sitting",
            "standing", "jumping", "dancing", "playing", "smiling", "laughing",
            "hugging", "kissing", "reading", "writing", "working", "driving",
            "cycling", "swimming", "surfing", "climbing", "hiking", "sleeping",
            "watching", "performing", "exercising", "shopping", "celebrating",
            "posing", "holding", "waving", "pointing", "talking", "singing",
            "throwing", "catching", "cheering"
        ]

        let activities = labels.filter { l in activityKeywords.contains { l.contains($0) } }.prefix(2)
        let objects    = labels.filter { l in objectKeywords.contains    { l.contains($0) } }.prefix(5)

        let parts = Array(activities) + Array(objects)
        guard !parts.isEmpty else { return nil }
        return "Notable objects and details in the image: \(parts.joined(separator: ", "))."
    }

    // MARK: - Sentence: Background

    private func backgroundSentence(labels: [String], color: ColorInfo) -> String? {
        let bgKeywords = [
            "sky", "cloud", "mountain", "hill", "horizon", "forest", "tree",
            "sea", "ocean", "lake", "river", "beach", "sand", "desert",
            "snow", "fog", "mist", "sunset", "sunrise", "cityscape",
            "building", "skyline", "skyscraper", "bridge", "road",
            "grass", "field", "meadow", "coast", "shore", "wave",
            "cliff", "valley", "island", "harbor", "marina"
        ]
        let bgLabels = labels.filter { l in bgKeywords.contains { l.contains($0) } }.prefix(4)
        guard !bgLabels.isEmpty else { return nil }
        return "The environment features \(Array(bgLabels).joined(separator: ", "))."
    }

    // MARK: - Sentence: Composition

    private func compositionSentence(
        image: UIImage,
        personCount: Int,
        faces: [VNFaceObservation],
        humans: [VNHumanObservation],
        labels: [String]
    ) -> String {
        guard image.size.height > 0 else { return "The photo has a standard composition." }
        let ratio = image.size.width / image.size.height
        var parts: [String] = []

        if ratio > 1.7      { parts.append("a wide panoramic frame") }
        else if ratio > 1.2 { parts.append("a landscape-oriented frame") }
        else if ratio < 0.6 { parts.append("a tall portrait-oriented frame") }
        else                { parts.append("a square or near-square frame") }

        // Shot type from face (most reliable for close-up portrait detection)
        if let bigFace = faces.max(by: {
            $0.boundingBox.width * $0.boundingBox.height < $1.boundingBox.width * $1.boundingBox.height
        }) {
            let area = bigFace.boundingBox.width * bigFace.boundingBox.height
            if area > 0.25       { parts.append("a tight close-up portrait") }
            else if area > 0.08  { parts.append("a medium portrait shot") }
        } else if let bigHuman = humans.max(by: {
            $0.boundingBox.width * $0.boundingBox.height < $1.boundingBox.width * $1.boundingBox.height
        }) {
            let area = bigHuman.boundingBox.width * bigHuman.boundingBox.height
            if area > 0.20       { parts.append("a close-up or medium shot") }
            else if area > 0.06  { parts.append("a full-body or medium-distance shot") }
            else                  { parts.append("a wide shot with subjects at a distance") }
        }

        if labels.contains(where: { $0.contains("macro") || $0.contains("close up") || $0.contains("detail") }) {
            parts.append("a macro or detail-level perspective")
        }

        // Multiple subjects
        if personCount > 2 {
            parts.append("multiple subjects spread across the frame")
        }

        return "The image is composed as \(parts.joined(separator: ", "))."
    }

    // MARK: - Body Pose Activity Inference

    private func inferPoseActivities(from observations: [VNHumanBodyPoseObservation]) -> [String] {
        var activities: [String] = []

        for obs in observations.prefix(2) {
            // Standing vs seated: in Vision coords y=0 is bottom, y=1 is top.
            // For a standing person the neck sits considerably above the root (hip midpoint).
            if let neck = try? obs.recognizedPoint(.neck),
               let root = try? obs.recognizedPoint(.root),
               neck.confidence > 0.4, root.confidence > 0.4 {
                let verticalSpan = neck.location.y - root.location.y
                if verticalSpan > 0.20 {
                    activities.append("standing upright")
                } else if verticalSpan > 0.05 {
                    activities.append("seated or crouching")
                }
            }

            // Arms raised: either wrist is above its corresponding shoulder
            if let lWrist    = try? obs.recognizedPoint(.leftWrist),
               let rWrist    = try? obs.recognizedPoint(.rightWrist),
               let lShoulder = try? obs.recognizedPoint(.leftShoulder),
               let rShoulder = try? obs.recognizedPoint(.rightShoulder) {
                let leftRaised  = lWrist.confidence > 0.35 && lShoulder.confidence > 0.35
                    && lWrist.location.y > lShoulder.location.y + 0.04
                let rightRaised = rWrist.confidence > 0.35 && rShoulder.confidence > 0.35
                    && rWrist.location.y > rShoulder.location.y + 0.04
                if leftRaised || rightRaised { activities.append("arms or hands raised") }
            }

            // In motion: ankles far apart horizontally (stride / running gait)
            if let lAnkle = try? obs.recognizedPoint(.leftAnkle),
               let rAnkle = try? obs.recognizedPoint(.rightAnkle),
               lAnkle.confidence > 0.35, rAnkle.confidence > 0.35 {
                let xSpread = abs(lAnkle.location.x - rAnkle.location.x)
                if xSpread > 0.12 { activities.append("in motion (walking or running)") }
            }

            // Leaning forward: nose noticeably ahead of hips on x-axis (side-on view)
            if let nose = try? obs.recognizedPoint(.nose),
               let root = try? obs.recognizedPoint(.root),
               nose.confidence > 0.4, root.confidence > 0.4 {
                let lean = abs(nose.location.x - root.location.x)
                if lean > 0.15 { activities.append("leaning forward") }
            }
        }

        return activities.uniqued()
    }

    // MARK: - Color Analysis

    struct ColorInfo {
        let atmosphereSentence: String
        let lightingSentence: String
        let isOutdoor: Bool
        let brightness: CGFloat
    }

    private func colorAnalysis(for cgImage: CGImage) -> ColorInfo {
        let size = 64
        var buf = [UInt8](repeating: 0, count: size * size * 4)
        let space = CGColorSpaceCreateDeviceRGB()

        guard let ctx = CGContext(
            data: &buf, width: size, height: size,
            bitsPerComponent: 8, bytesPerRow: size * 4,
            space: space, bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue
        ) else {
            return ColorInfo(
                atmosphereSentence: "The image has a neutral color palette.",
                lightingSentence:   "The lighting appears moderate.",
                isOutdoor: false, brightness: 0.5
            )
        }
        ctx.draw(cgImage, in: CGRect(x: 0, y: 0, width: size, height: size))

        var r: CGFloat = 0, g: CGFloat = 0, b: CGFloat = 0, sat: CGFloat = 0
        let topRows  = size / 3
        var tR: CGFloat = 0, tG: CGFloat = 0, tB: CGFloat = 0
        let total    = CGFloat(size * size)
        let topTotal = CGFloat(topRows * size)

        for row in 0..<size {
            for col in 0..<size {
                let i  = (row * size + col) * 4
                let pr = CGFloat(buf[i])     / 255
                let pg = CGFloat(buf[i + 1]) / 255
                let pb = CGFloat(buf[i + 2]) / 255
                r += pr; g += pg; b += pb
                sat += max(pr, pg, pb) - min(pr, pg, pb)
                if row < topRows { tR += pr; tG += pg; tB += pb }
            }
        }
        r /= total; g /= total; b /= total; sat /= total
        tR /= topTotal; tG /= topTotal; tB /= topTotal
        let brightness    = (r + g + b) / 3
        let topBrightness = (tR + tG + tB) / 3

        // Sky detection: top-third is blue-dominant (daytime) or gold-dominant (golden hour)
        let hasBlueSky   = tB > tR * 1.10 && tB > tG * 0.90 && topBrightness > 0.38
        let hasGoldenSky = tR > tB * 1.10 && tG > tB * 0.80 && topBrightness > 0.42
        let isOutdoor    = hasBlueSky || hasGoldenSky

        let dominant: String
        if r > g * 1.2 && r > b * 1.2 {
            dominant = r > 0.5 && g > 0.3 ? "warm orange and golden" : "deep red and warm"
        } else if g > r * 1.2 && g > b * 1.2 {
            dominant = "green, evoking natural foliage or outdoor scenery"
        } else if b > r * 1.15 && b > g * 1.0 {
            dominant = "cool blue, suggesting sky, water, or a cool atmosphere"
        } else if r > 0.45 && g > 0.45 && b < 0.35 {
            dominant = "bright yellow"
        } else if r > 0.4 && b > 0.4 && g < 0.3 {
            dominant = "purple or magenta"
        } else if r > 0.45 && g > 0.38 && b > 0.38 {
            dominant = "light and airy pastel"
        } else {
            dominant = "neutral and balanced"
        }

        let satDesc: String
        if sat > 0.35      { satDesc = "highly saturated and vivid" }
        else if sat > 0.20 { satDesc = "moderately colorful" }
        else if sat > 0.08 { satDesc = "muted and understated" }
        else               { satDesc = "nearly monochrome or desaturated" }

        let atmosphereSentence =
            "The color palette is \(satDesc) — \(dominant) tones dominate the frame."

        let lightingSentence: String
        if hasGoldenSky {
            lightingSentence = "The lighting indicates golden hour — warm directional sunlight " +
                "characteristic of late afternoon or early morning, casting long soft shadows."
        } else if hasBlueSky && brightness > 0.50 {
            lightingSentence = "The scene is lit by broad outdoor daylight under an open sky."
        } else if brightness > 0.75 {
            lightingSentence = "The scene is very brightly lit — consistent with strong midday " +
                "sunlight, overcast even light, or high-key studio lighting."
        } else if brightness < 0.20 {
            lightingSentence = "The scene is very dark — suggesting a nighttime environment, " +
                "a dimly lit interior, or an intentional low-key/silhouette exposure."
        } else if brightness < 0.36 {
            lightingSentence = "The scene has low ambient light with a moody, underexposed quality."
        } else if r > b * 1.2 && brightness < 0.65 {
            lightingSentence = "The warm cast and moderate brightness suggest indoor tungsten " +
                "or candlelight, or a warm-toned sunset setting."
        } else if b > r * 1.1 && brightness < 0.55 {
            lightingSentence = "The cool-toned moderate light suggests overcast outdoor conditions or shade."
        } else {
            lightingSentence = "The lighting is naturally balanced with even, diffused illumination."
        }

        return ColorInfo(
            atmosphereSentence: atmosphereSentence,
            lightingSentence:   lightingSentence,
            isOutdoor:          isOutdoor,
            brightness:         brightness
        )
    }

    // MARK: - Vocabulary & Helpers

    private let blocklist: Set<String> = [
        "image", "photo", "picture", "photograph", "shot", "graphic",
        "object", "thing", "item", "element", "detail", "view",
        "color", "colour", "pattern", "texture", "shape", "still", "clip"
    ]

    /// All person-related label fragments — used to avoid duplicating with personCount sentence.
    private let personTerms = [
        "person", "people", "man", "woman", "child", "baby", "boy", "girl",
        "face", "portrait", "human", "selfie", "individual", "adult", "figure",
        "kid", "infant", "teen", "crowd", "pedestrian"
    ]

    /// Ordered list of (classification keyword → refined label) for single-person photos.
    private let personSpecificTerms: [(keyword: String, label: String)] = [
        ("infant",  "an infant"),
        ("baby",    "a baby"),
        ("toddler", "a toddler"),
        ("child",   "a child"),
        ("kid",     "a child"),
        ("boy",     "a boy"),
        ("girl",    "a girl"),
        ("woman",   "a woman"),
        ("man",     "a man"),
        ("teen",    "a teenager"),
        ("selfie",  "a person taking a selfie")
    ]

    private let classificationActivityKeywords = [
        "eating", "drinking", "running", "walking", "swimming", "dancing",
        "playing", "jumping", "sitting", "standing", "hiking", "cycling",
        "surfing", "skiing", "climbing", "posing", "working", "cooking",
        "celebrating", "laughing", "smiling", "hugging", "waving", "reading",
        "singing", "performing", "shopping", "exercising"
    ]

    private func readableLabel(_ identifier: String) -> String {
        // VN identifiers may be dot-separated hierarchies — use the most-specific leaf
        let leaf = identifier.split(separator: ".").last.map(String.init) ?? identifier
        return leaf
            .replacingOccurrences(of: "_", with: " ")
            .replacingOccurrences(of: "-", with: " ")
            .trimmingCharacters(in: .whitespaces)
            .lowercased()
    }

    private func fallbackSubjectSentence(for image: UIImage) -> String {
        guard image.size.height > 0 else { return "This photo captures a scene." }
        let ratio = image.size.width / image.size.height
        if ratio > 1.5  { return "This is a wide landscape-format photo." }
        if ratio < 0.67 { return "This is a tall portrait-format photo." }
        return "This photo captures a scene."
    }
}

// MARK: - Sequence + uniqued

private extension Array where Element: Hashable {
    func uniqued() -> [Element] {
        var seen = Set<Element>()
        return filter { seen.insert($0).inserted }
    }
}
