import Foundation
import UIKit
import Vision

enum ImageSemanticAnalyzer {
    static func description(for image: UIImage) -> String {
        let stats = image.colorStatistics
        let recognizedText = recognizedTextLines(in: image)

        var sentences: [String] = []
        sentences.append(sceneOverview(from: image, stats: stats))
        sentences.append(colorAndMoodSentence(from: stats))
        sentences.append(lightingSentence(from: stats))

        if !recognizedText.isEmpty {
            let textSample = recognizedText.prefix(3).joined(separator: ", ")
            sentences.append("Visible text in the image includes \(textSample).")
        }

        return sentences.joined(separator: " ")
    }

    private static func sceneOverview(from image: UIImage, stats: ColorStatistics) -> String {
        let aspectDescriptor: String
        if image.size.width > image.size.height * 1.15 {
            aspectDescriptor = "landscape-oriented"
        } else if image.size.height > image.size.width * 1.15 {
            aspectDescriptor = "portrait-oriented"
        } else {
            aspectDescriptor = "square or near-square"
        }

        let brightnessDescriptor = stats.brightness > 0.72 ? "bright" : (stats.brightness < 0.32 ? "dark" : "balanced")
        let saturationDescriptor = stats.saturation > 0.55 ? "vivid" : "muted"

        return "This is a \(brightnessDescriptor), \(saturationDescriptor), \(aspectDescriptor) image."
    }

    private static func colorAndMoodSentence(from stats: ColorStatistics) -> String {
        switch stats.dominantColor {
        case .red:
            return "Red and warm tones dominate the frame, which can suggest people, indoor light, or dramatic color accents."
        case .green:
            return "Green tones dominate the frame, which often suggests trees, plants, grass, or other natural scenery."
        case .blue:
            return "Blue and cool tones dominate the frame, which can suggest sky, water, open space, or a cooler atmosphere."
        case .yellow:
            return "Yellow tones are strong in the frame, which can suggest sunlight, bright artificial light, or warm highlights."
        case .orange:
            return "Orange and warm tones dominate the frame, which can suggest sunset light, indoor warmth, or golden-hour color."
        case .purple:
            return "Purple or magenta tones stand out in the frame, giving the scene a stylized or low-light appearance."
        case .neutral:
            return "The frame is mostly neutral in color, without one strong dominant hue."
        }
    }

    private static func lightingSentence(from stats: ColorStatistics) -> String {
        if stats.brightness > 0.68 && stats.dominantColor == .blue {
            return "The scene appears outdoors in daylight and may include open sky or water."
        }

        if stats.brightness > 0.7 {
            return "The scene appears well lit and likely captured in daylight or bright interior lighting."
        }

        if stats.brightness < 0.35 {
            return "The scene appears low light and may resemble a night scene, a dim interior, or evening lighting."
        }

        if stats.dominantColor == .green {
            return "The overall look suggests an outdoor natural setting."
        }

        if stats.dominantColor == .orange {
            return "The overall look suggests a warm scene that may resemble sunset or indoor ambient light."
        }

        return "The lighting appears moderate and evenly balanced."
    }

    private static func recognizedTextLines(in image: UIImage) -> [String] {
        guard let cgImage = image.cgImage else { return [] }

        let request = VNRecognizeTextRequest()
        request.recognitionLevel = .fast
        request.usesLanguageCorrection = false

        let handler = VNImageRequestHandler(cgImage: cgImage, options: [:])
        try? handler.perform([request])

        let observations = request.results ?? []
        let strings = observations
            .compactMap { $0.topCandidates(1).first?.string }
            .map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
            .filter { !$0.isEmpty }

        var uniqueStrings: [String] = []
        var seen = Set<String>()

        for string in strings {
            guard seen.insert(string).inserted else { continue }
            uniqueStrings.append(string)

            if uniqueStrings.count == 4 {
                break
            }
        }

        return uniqueStrings
    }
}

private enum DominantColor {
    case red
    case green
    case blue
    case yellow
    case orange
    case purple
    case neutral
}

private struct ColorStatistics {
    let brightness: CGFloat
    let saturation: CGFloat
    let dominantColor: DominantColor
}

private extension UIImage {
    var colorStatistics: ColorStatistics {
        guard let cgImage = cgImage else {
            return ColorStatistics(brightness: 0.5, saturation: 0.0, dominantColor: .neutral)
        }

        let width = 24
        let height = 24
        let bytesPerPixel = 4
        let bytesPerRow = width * bytesPerPixel
        var buffer = [UInt8](repeating: 0, count: width * height * bytesPerPixel)
        let colorSpace = CGColorSpaceCreateDeviceRGB()

        guard let context = CGContext(
            data: &buffer,
            width: width,
            height: height,
            bitsPerComponent: 8,
            bytesPerRow: bytesPerRow,
            space: colorSpace,
            bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue
        ) else {
            return ColorStatistics(brightness: 0.5, saturation: 0.0, dominantColor: .neutral)
        }

        context.draw(cgImage, in: CGRect(x: 0, y: 0, width: width, height: height))

        var redTotal: CGFloat = 0
        var greenTotal: CGFloat = 0
        var blueTotal: CGFloat = 0
        var saturationTotal: CGFloat = 0

        for index in stride(from: 0, to: buffer.count, by: bytesPerPixel) {
            let red = CGFloat(buffer[index]) / 255
            let green = CGFloat(buffer[index + 1]) / 255
            let blue = CGFloat(buffer[index + 2]) / 255
            let maxValue = max(red, green, blue)
            let minValue = min(red, green, blue)

            redTotal += red
            greenTotal += green
            blueTotal += blue
            saturationTotal += maxValue - minValue
        }

        let pixelCount = CGFloat(width * height)
        let red = redTotal / pixelCount
        let green = greenTotal / pixelCount
        let blue = blueTotal / pixelCount
        let brightness = (red + green + blue) / 3
        let saturation = saturationTotal / pixelCount

        let dominantColor: DominantColor
        if red > green * 1.15 && red > blue * 1.15 {
            dominantColor = red > 0.5 && green > 0.3 ? .orange : .red
        } else if green > red * 1.15 && green > blue * 1.15 {
            dominantColor = .green
        } else if blue > red * 1.15 && blue > green * 1.15 {
            dominantColor = .blue
        } else if red > 0.45 && green > 0.45 && blue < 0.35 {
            dominantColor = .yellow
        } else if red > 0.4 && blue > 0.4 {
            dominantColor = .purple
        } else {
            dominantColor = .neutral
        }

        return ColorStatistics(brightness: brightness, saturation: saturation, dominantColor: dominantColor)
    }
}
