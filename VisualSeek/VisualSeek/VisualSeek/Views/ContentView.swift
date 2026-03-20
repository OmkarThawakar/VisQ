import SwiftUI
import UIKit

enum AppTab: String, CaseIterable, Hashable {
    case library
    case search
    case composed
    case indexing
    case settings

    var title: String {
        switch self {
        case .library: return "Library"
        case .search: return "Search"
        case .composed: return "Remix"
        case .indexing: return "Index"
        case .settings: return "Studio"
        }
    }

    var systemImage: String {
        switch self {
        case .library: return "photo.on.rectangle.angled"
        case .search: return "magnifyingglass"
        case .composed: return "wand.and.stars.inverse"
        case .indexing: return "square.stack.3d.up.fill"
        case .settings: return "slider.horizontal.3"
        }
    }
}

enum VSTheme {
    private static func adaptive(light: UIColor, dark: UIColor) -> Color {
        Color(
            uiColor: UIColor { traits in
                traits.userInterfaceStyle == .dark ? dark : light
            }
        )
    }

    static let ink = adaptive(
        light: UIColor(red: 0.16, green: 0.16, blue: 0.19, alpha: 1),
        dark: UIColor(white: 0.97, alpha: 1)
    )
    static let mutedInk = adaptive(
        light: UIColor(red: 0.24, green: 0.25, blue: 0.29, alpha: 1),
        dark: UIColor(white: 0.82, alpha: 1)
    )
    static let sand = adaptive(
        light: UIColor(red: 0.98, green: 0.95, blue: 0.90, alpha: 1),
        dark: UIColor(red: 0.03, green: 0.04, blue: 0.05, alpha: 1)
    )
    static let clay = adaptive(
        light: UIColor(red: 0.93, green: 0.84, blue: 0.74, alpha: 1),
        dark: UIColor(red: 0.12, green: 0.10, blue: 0.11, alpha: 1)
    )
    static let apricot = Color(red: 0.90, green: 0.49, blue: 0.28)
    static let ember = Color(red: 0.77, green: 0.31, blue: 0.19)
    static let teal = Color(red: 0.19, green: 0.54, blue: 0.55)
    static let moss = Color(red: 0.45, green: 0.57, blue: 0.32)
    static let card = adaptive(
        light: UIColor(white: 1.0, alpha: 0.80),
        dark: UIColor(red: 0.13, green: 0.13, blue: 0.15, alpha: 0.92)
    )
    static let cardStrong = adaptive(
        light: UIColor(white: 1.0, alpha: 0.92),
        dark: UIColor(red: 0.17, green: 0.17, blue: 0.20, alpha: 0.96)
    )
    static let field = adaptive(
        light: UIColor(white: 1.0, alpha: 0.88),
        dark: UIColor(red: 0.16, green: 0.16, blue: 0.19, alpha: 0.96)
    )
    static let line = adaptive(
        light: UIColor(white: 0.0, alpha: 0.08),
        dark: UIColor(white: 1.0, alpha: 0.10)
    )
    static let shadow = adaptive(
        light: UIColor(white: 0.0, alpha: 0.08),
        dark: UIColor(white: 0.0, alpha: 0.28)
    )
    static let danger = Color(red: 0.73, green: 0.22, blue: 0.18)
}

struct VSAppBackground: View {
    var body: some View {
        ZStack {
            LinearGradient(
                colors: [VSTheme.sand, VSTheme.clay.opacity(0.95), Color.white],
                startPoint: .topLeading,
                endPoint: .bottomTrailing
            )

            Circle()
                .fill(
                    RadialGradient(
                        colors: [VSTheme.apricot.opacity(0.28), .clear],
                        center: .center,
                        startRadius: 12,
                        endRadius: 220
                    )
                )
                .frame(width: 360, height: 360)
                .offset(x: 150, y: -250)
                .blur(radius: 10)

            Circle()
                .fill(
                    RadialGradient(
                        colors: [VSTheme.teal.opacity(0.24), .clear],
                        center: .center,
                        startRadius: 10,
                        endRadius: 220
                    )
                )
                .frame(width: 320, height: 320)
                .offset(x: -170, y: 330)
                .blur(radius: 18)
        }
        .ignoresSafeArea()
    }
}

struct VSSectionCard<Content: View>: View {
    let padding: CGFloat
    let content: Content

    init(padding: CGFloat = 18, @ViewBuilder content: () -> Content) {
        self.padding = padding
        self.content = content()
    }

    var body: some View {
        content
            .padding(padding)
            .frame(maxWidth: .infinity, alignment: .leading)
            .background(VSTheme.card, in: RoundedRectangle(cornerRadius: 28, style: .continuous))
            .overlay {
                RoundedRectangle(cornerRadius: 28, style: .continuous)
                    .stroke(VSTheme.line, lineWidth: 1)
            }
            .shadow(color: VSTheme.shadow, radius: 18, y: 8)
    }
}

struct VSHeroHeader: View {
    let eyebrow: String
    let title: String
    let subtitle: String
    var trailingBadge: String?

    var body: some View {
        HStack(alignment: .top, spacing: 16) {
            VStack(alignment: .leading, spacing: 8) {
                Text(eyebrow.uppercased())
                    .font(.system(size: 11, weight: .semibold, design: .rounded))
                    .tracking(1.6)
                    .foregroundStyle(VSTheme.apricot)

                Text(title)
                    .font(.system(size: 34, weight: .bold, design: .serif))
                    .foregroundStyle(VSTheme.ink)

                Text(subtitle)
                    .font(.system(size: 15, weight: .medium, design: .rounded))
                    .foregroundStyle(VSTheme.mutedInk)
                    .fixedSize(horizontal: false, vertical: true)
            }

            Spacer(minLength: 0)

            if let trailingBadge {
                VSPill(text: trailingBadge, tint: VSTheme.teal)
            }
        }
    }
}

struct VSSectionHeading: View {
    let title: String
    let subtitle: String?

    var body: some View {
        VStack(alignment: .leading, spacing: 4) {
            Text(title)
                .font(.system(size: 20, weight: .bold, design: .rounded))
                .foregroundStyle(VSTheme.ink)

            if let subtitle {
                Text(subtitle)
                    .font(.system(size: 13, weight: .medium, design: .rounded))
                    .foregroundStyle(VSTheme.mutedInk)
            }
        }
    }
}

struct VSPill: View {
    let text: String
    var tint: Color = VSTheme.apricot

    var body: some View {
        Text(text)
            .font(.system(size: 12, weight: .semibold, design: .rounded))
            .foregroundStyle(tint)
            .padding(.horizontal, 12)
            .padding(.vertical, 8)
            .background(tint.opacity(0.12), in: Capsule())
            .overlay {
                Capsule()
                    .stroke(tint.opacity(0.25), lineWidth: 1)
            }
    }
}

struct VSMatchChipRow: View {
    let chips: [String]
    var tint: Color = VSTheme.teal

    var body: some View {
        if !chips.isEmpty {
            ScrollView(.horizontal, showsIndicators: false) {
                HStack(spacing: 8) {
                    ForEach(Array(chips.enumerated()), id: \.offset) { _, chip in
                        Text(chip)
                            .font(.system(size: 12, weight: .semibold, design: .rounded))
                            .foregroundStyle(tint)
                            .padding(.horizontal, 10)
                            .padding(.vertical, 7)
                            .background(tint.opacity(0.12), in: Capsule())
                            .overlay {
                                Capsule()
                                    .stroke(tint.opacity(0.22), lineWidth: 1)
                            }
                    }
                }
                .padding(.vertical, 1)
            }
            .scrollIndicators(.hidden)
        }
    }
}

struct VSMetricTile: View {
    let label: String
    let value: String
    var accent: Color = VSTheme.apricot

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text(label.uppercased())
                .font(.system(size: 11, weight: .semibold, design: .rounded))
                .tracking(1.3)
                .foregroundStyle(VSTheme.mutedInk)

            Text(value)
                .font(.system(size: 30, weight: .bold, design: .rounded))
                .foregroundStyle(VSTheme.ink)

            RoundedRectangle(cornerRadius: 999, style: .continuous)
                .fill(accent.opacity(0.22))
                .frame(width: 44, height: 6)
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .padding(16)
        .background(VSTheme.cardStrong, in: RoundedRectangle(cornerRadius: 24, style: .continuous))
    }
}

struct VSStatusBanner: View {
    let icon: String
    let text: String
    var tint: Color = VSTheme.teal

    var body: some View {
        HStack(alignment: .top, spacing: 12) {
            Image(systemName: icon)
                .font(.system(size: 14, weight: .bold))
                .foregroundStyle(tint)
                .frame(width: 28, height: 28)
                .background(tint.opacity(0.15), in: RoundedRectangle(cornerRadius: 10, style: .continuous))

            Text(text)
                .font(.system(size: 14, weight: .medium, design: .rounded))
                .foregroundStyle(VSTheme.ink)
                .frame(maxWidth: .infinity, alignment: .leading)
        }
        .padding(14)
        .background(VSTheme.cardStrong, in: RoundedRectangle(cornerRadius: 22, style: .continuous))
        .overlay {
            RoundedRectangle(cornerRadius: 22, style: .continuous)
                .stroke(tint.opacity(0.18), lineWidth: 1)
        }
    }
}

struct VSEmptyState: View {
    let title: String
    let subtitle: String
    let systemImage: String

    var body: some View {
        VSSectionCard {
            VStack(spacing: 16) {
                Image(systemName: systemImage)
                    .font(.system(size: 30, weight: .medium))
                    .foregroundStyle(VSTheme.apricot)
                    .frame(width: 72, height: 72)
                    .background(VSTheme.apricot.opacity(0.12), in: RoundedRectangle(cornerRadius: 24, style: .continuous))

                VStack(spacing: 8) {
                    Text(title)
                        .font(.system(size: 22, weight: .bold, design: .rounded))
                        .foregroundStyle(VSTheme.ink)

                    Text(subtitle)
                        .font(.system(size: 14, weight: .medium, design: .rounded))
                        .foregroundStyle(VSTheme.mutedInk)
                        .multilineTextAlignment(.center)
                }
            }
            .frame(maxWidth: .infinity)
            .padding(.vertical, 10)
        }
    }
}

struct VSPrimaryButtonStyle: ButtonStyle {
    var tint: Color = VSTheme.apricot

    func makeBody(configuration: Configuration) -> some View {
        configuration.label
            .font(.system(size: 15, weight: .bold, design: .rounded))
            .foregroundStyle(.white)
            .frame(maxWidth: .infinity)
            .padding(.horizontal, 18)
            .padding(.vertical, 16)
            .background(
                LinearGradient(
                    colors: [tint, tint.opacity(0.88), VSTheme.ember],
                    startPoint: .topLeading,
                    endPoint: .bottomTrailing
                ),
                in: RoundedRectangle(cornerRadius: 20, style: .continuous)
            )
            .scaleEffect(configuration.isPressed ? 0.98 : 1.0)
            .opacity(configuration.isPressed ? 0.94 : 1.0)
    }
}

struct VSSecondaryButtonStyle: ButtonStyle {
    var tint: Color = VSTheme.teal

    func makeBody(configuration: Configuration) -> some View {
        configuration.label
            .font(.system(size: 15, weight: .bold, design: .rounded))
            .foregroundStyle(tint)
            .frame(maxWidth: .infinity)
            .padding(.horizontal, 18)
            .padding(.vertical, 16)
            .background(tint.opacity(0.10), in: RoundedRectangle(cornerRadius: 20, style: .continuous))
            .overlay {
                RoundedRectangle(cornerRadius: 20, style: .continuous)
                    .stroke(tint.opacity(0.18), lineWidth: 1)
            }
            .scaleEffect(configuration.isPressed ? 0.985 : 1.0)
    }
}

struct VSDestructiveButtonStyle: ButtonStyle {
    func makeBody(configuration: Configuration) -> some View {
        configuration.label
            .font(.system(size: 15, weight: .bold, design: .rounded))
            .foregroundStyle(VSTheme.danger)
            .frame(maxWidth: .infinity)
            .padding(.horizontal, 18)
            .padding(.vertical, 16)
            .background(VSTheme.danger.opacity(0.10), in: RoundedRectangle(cornerRadius: 20, style: .continuous))
            .overlay {
                RoundedRectangle(cornerRadius: 20, style: .continuous)
                    .stroke(VSTheme.danger.opacity(0.18), lineWidth: 1)
            }
            .scaleEffect(configuration.isPressed ? 0.985 : 1.0)
    }
}

struct VSGlassField<Content: View>: View {
    let content: Content

    init(@ViewBuilder content: () -> Content) {
        self.content = content()
    }

    var body: some View {
        content
            .padding(.horizontal, 16)
            .padding(.vertical, 14)
            .background(VSTheme.field, in: RoundedRectangle(cornerRadius: 20, style: .continuous))
            .overlay {
                RoundedRectangle(cornerRadius: 20, style: .continuous)
                    .stroke(VSTheme.line, lineWidth: 1)
            }
    }
}

struct VSTabDock: View {
    @Binding var selectedTab: AppTab

    var body: some View {
        HStack(spacing: 8) {
            ForEach(AppTab.allCases, id: \.self) { tab in
                VSTabDockItem(
                    tab: tab,
                    isSelected: selectedTab == tab
                ) {
                    withAnimation(.spring(response: 0.35, dampingFraction: 0.82)) {
                        selectedTab = tab
                    }
                }
            }
        }
        .padding(10)
        .background(VSTheme.card, in: RoundedRectangle(cornerRadius: 28, style: .continuous))
        .overlay {
            RoundedRectangle(cornerRadius: 28, style: .continuous)
                .stroke(VSTheme.line, lineWidth: 1)
        }
        .shadow(color: VSTheme.shadow, radius: 20, y: 6)
    }
}

private struct VSTabDockItem: View {
    let tab: AppTab
    let isSelected: Bool
    let action: () -> Void

    private var foregroundColor: Color {
        isSelected ? .white : VSTheme.mutedInk
    }

    var body: some View {
        Button(action: action) {
            VStack(spacing: 7) {
                Image(systemName: tab.systemImage)
                    .font(.system(size: 16, weight: .bold))

                Text(tab.title)
                    .font(.system(size: 11, weight: .bold, design: .rounded))
                    .lineLimit(1)
                    .minimumScaleFactor(0.8)
            }
            .foregroundStyle(foregroundColor)
            .frame(maxWidth: .infinity)
            .padding(.vertical, 10)
            .background(backgroundView)
        }
        .buttonStyle(.plain)
    }

    @ViewBuilder
    private var backgroundView: some View {
        if isSelected {
            LinearGradient(
                colors: [VSTheme.apricot, VSTheme.ember],
                startPoint: .topLeading,
                endPoint: .bottomTrailing
            )
            .clipShape(RoundedRectangle(cornerRadius: 18, style: .continuous))
        } else {
            RoundedRectangle(cornerRadius: 18, style: .continuous)
                .fill(Color.clear)
        }
    }
}

struct ContentView: View {
    @State private var selectedTab: AppTab = .library

    var body: some View {
        ZStack {
            VSAppBackground()

            currentTabView
                .transition(.opacity)
                .animation(.easeInOut(duration: 0.22), value: selectedTab)
        }
        .safeAreaInset(edge: .bottom, spacing: 0) {
            VSTabDock(selectedTab: $selectedTab)
                .padding(.horizontal, 20)
                .padding(.top, 10)
                .padding(.bottom, 10)
                .background(Color.clear)
        }
    }

    @ViewBuilder
    private var currentTabView: some View {
        switch selectedTab {
        case .library:
            LibraryView()
        case .search:
            SearchView()
        case .composed:
            ComposedRetrievalView()
        case .indexing:
            IndexingView()
        case .settings:
            SettingsView()
        }
    }
}
