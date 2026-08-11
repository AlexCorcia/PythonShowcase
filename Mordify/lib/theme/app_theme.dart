import 'package:flutter/material.dart';
import 'package:google_fonts/google_fonts.dart';

const _seedColor = Color(0xFF00897B);

final lightColorScheme =
    ColorScheme.fromSeed(seedColor: _seedColor, brightness: Brightness.light);
final darkColorScheme =
    ColorScheme.fromSeed(seedColor: _seedColor, brightness: Brightness.dark);

ThemeData buildTheme(ColorScheme colorScheme) => ThemeData(
      useMaterial3: true,
      colorScheme: colorScheme,
      scaffoldBackgroundColor: colorScheme.surfaceContainerLowest,
      appBarTheme: AppBarTheme(
        backgroundColor: colorScheme.surface,
        foregroundColor: colorScheme.onSurface,
        titleTextStyle: TextStyle(
          color: colorScheme.onSurface,
          fontSize: 26,
          fontWeight: FontWeight.w700,
        ),
        scrolledUnderElevation: 1,
      ),
      tabBarTheme: TabBarThemeData(
        labelColor: colorScheme.primary,
        unselectedLabelColor: colorScheme.onSurfaceVariant,
        indicatorColor: colorScheme.primary,
        labelStyle: const TextStyle(fontWeight: FontWeight.w600),
      ),
      cardTheme: CardThemeData(
        elevation: 0,
        color: colorScheme.surfaceContainerLow,
        shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(16)),
      ),
      listTileTheme: ListTileThemeData(
        shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(12)),
      ),
      inputDecorationTheme: InputDecorationTheme(
        border: OutlineInputBorder(borderRadius: BorderRadius.circular(12)),
        filled: true,
        fillColor: colorScheme.surfaceContainerHighest.withValues(alpha: 0.4),
      ),
      dialogTheme: DialogThemeData(
        shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(20)),
      ),
      floatingActionButtonTheme: FloatingActionButtonThemeData(
        backgroundColor: colorScheme.primaryContainer,
        foregroundColor: colorScheme.onPrimaryContainer,
        elevation: 2,
      ),
      chipTheme: ChipThemeData(
        backgroundColor: colorScheme.surfaceContainerHigh,
        shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(20)),
      ),
    );

// ─────────────────────────────────────────────────────────────────────────
// Nocturne — the dark design system from the Claude Design handoff
// (design/handoff/). A quiet, compact dark interface: near-neutral blue-grey
// ground, Inter at medium weight, soft radii, and a single structural accent
// (blurple) used as a line/glow rather than a flood. See
// design/handoff/mordify-app-design-brief/project/_ds/*/readme.md for the
// full rationale.
// ─────────────────────────────────────────────────────────────────────────

/// Nocturne's dark ground - the app-wide dark scaffold background.
const nocturneBg = Color(0xFF161826);

/// The base surface a card/section sits on.
const nocturneSurface = Color(0xFF232532);

/// A more elevated surface tint, one step up from [nocturneSurface] (used
/// for e.g. the level/XP card on the home dashboard).
const nocturneSurfaceHigh = Color(0xFF2A2C3B);

/// Primary text color on the dark ground.
const nocturneText = Color(0xFFE9E9ED);

/// The single structural accent (a blurple) - reserved for checkboxes,
/// active nav/tab states, links and other structural UI. Never used for
/// gamification signals; see [mordifyAmber] for those.
const nocturneAccent = Color(0xFF9184D9);
const nocturneAccent2 = Color(0xFFA7A1DB);
const nocturneAccent100 = Color(0xFFF5F4FF);
const nocturneAccent300 = Color(0xFFD2CEFD);
const nocturneAccent800 = Color(0xFF423A6A);
const nocturneAccent900 = Color(0xFF2B2741);
const nocturneNeutral800 = Color(0xFF3F424D);

/// Divider/outline hairlines - 14% of [nocturneText] over the dark ground.
final nocturneDivider = nocturneText.withValues(alpha: 0.14);

/// Muted/secondary text - 55% of [nocturneText] over the dark ground.
final nocturneTextMuted = nocturneText.withValues(alpha: 0.55);

/// The one color reserved exclusively for gamification signals: points, XP,
/// streaks and badges. Pulled from the app icon's eye color. Never used for
/// structural UI (that's [nocturneAccent]'s job) and never flooded across
/// large areas - a line, a glow, a small fill on a badge/points moment.
const mordifyAmber = Color(0xFFF0A940);
final mordifyAmberDim = mordifyAmber.withValues(alpha: 0.18);

/// Muted, low-chroma folder/category colors in Nocturne's voice - used by
/// the starter folders and offered first in the folder color picker.
const nocturneSage = Color(0xFF8FAE86);
const nocturneTerracotta = Color(0xFFC98A68);
const nocturneDustyBlue = Color(0xFF7691B0);
const nocturneDustyPlum = Color(0xFF9A87B0);

final nocturneColorScheme = ColorScheme(
  brightness: Brightness.dark,
  surface: nocturneBg,
  onSurface: nocturneText,
  surfaceContainerLowest: nocturneBg,
  surfaceContainerLow: nocturneBg,
  surfaceContainer: nocturneSurface,
  surfaceContainerHigh: nocturneSurfaceHigh,
  surfaceContainerHighest: nocturneNeutral800,
  onSurfaceVariant: nocturneTextMuted,
  outline: nocturneDivider,
  outlineVariant: nocturneDivider,
  primary: nocturneAccent,
  onPrimary: nocturneBg,
  primaryContainer: nocturneAccent800,
  onPrimaryContainer: nocturneAccent100,
  secondary: nocturneAccent2,
  onSecondary: nocturneBg,
  secondaryContainer: nocturneAccent800,
  onSecondaryContainer: nocturneAccent100,
  tertiary: mordifyAmber,
  onTertiary: nocturneBg,
  tertiaryContainer: mordifyAmberDim,
  onTertiaryContainer: mordifyAmber,
  error: const Color(0xFFFFB4AB),
  onError: const Color(0xFF690005),
  errorContainer: const Color(0xFF93000A),
  onErrorContainer: const Color(0xFFFFDAD6),
  shadow: Colors.black,
  scrim: Colors.black,
  inverseSurface: nocturneText,
  onInverseSurface: nocturneBg,
  inversePrimary: nocturneAccent800,
);

ThemeData buildNocturneTheme() {
  final colorScheme = nocturneColorScheme;
  final baseText = GoogleFonts.interTextTheme(ThemeData(brightness: Brightness.dark).textTheme)
      .apply(bodyColor: nocturneText, displayColor: nocturneText);

  return ThemeData(
    useMaterial3: true,
    brightness: Brightness.dark,
    colorScheme: colorScheme,
    scaffoldBackgroundColor: nocturneBg,
    fontFamily: GoogleFonts.inter().fontFamily,
    textTheme: baseText,
    appBarTheme: AppBarTheme(
      backgroundColor: nocturneBg,
      foregroundColor: nocturneText,
      surfaceTintColor: Colors.transparent,
      titleTextStyle: GoogleFonts.inter(
        color: nocturneText,
        fontSize: 22,
        fontWeight: FontWeight.w500,
      ),
      scrolledUnderElevation: 0,
    ),
    navigationBarTheme: NavigationBarThemeData(
      backgroundColor: nocturneBg,
      surfaceTintColor: Colors.transparent,
      indicatorColor: nocturneAccent.withValues(alpha: 0.16),
      height: 64,
      labelTextStyle: WidgetStateProperty.resolveWith(
        (states) => GoogleFonts.inter(
          fontSize: 11,
          fontWeight: states.contains(WidgetState.selected) ? FontWeight.w600 : FontWeight.w400,
          color: states.contains(WidgetState.selected) ? nocturneText : nocturneTextMuted,
        ),
      ),
      iconTheme: WidgetStateProperty.resolveWith(
        (states) => IconThemeData(
          size: 22,
          color: states.contains(WidgetState.selected) ? nocturneAccent : nocturneTextMuted,
        ),
      ),
    ),
    tabBarTheme: TabBarThemeData(
      labelColor: nocturneAccent,
      unselectedLabelColor: nocturneTextMuted,
      indicatorColor: nocturneAccent,
      labelStyle: const TextStyle(fontWeight: FontWeight.w600),
    ),
    segmentedButtonTheme: SegmentedButtonThemeData(
      style: ButtonStyle(
        backgroundColor: WidgetStateProperty.resolveWith(
          (states) => states.contains(WidgetState.selected) ? Colors.transparent : Colors.transparent,
        ),
        foregroundColor: WidgetStateProperty.resolveWith(
          (states) => states.contains(WidgetState.selected) ? nocturneAccent : nocturneTextMuted,
        ),
        side: WidgetStateProperty.resolveWith(
          (states) => BorderSide(
            color: states.contains(WidgetState.selected) ? nocturneAccent : nocturneDivider,
          ),
        ),
        textStyle: WidgetStatePropertyAll(GoogleFonts.inter(fontSize: 13)),
        shape: WidgetStatePropertyAll(
          RoundedRectangleBorder(borderRadius: BorderRadius.circular(8)),
        ),
      ),
    ),
    cardTheme: CardThemeData(
      elevation: 0,
      color: nocturneSurface,
      surfaceTintColor: Colors.transparent,
      shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(14)),
    ),
    listTileTheme: ListTileThemeData(
      shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(8)),
      iconColor: nocturneTextMuted,
      textColor: nocturneText,
    ),
    checkboxTheme: CheckboxThemeData(
      shape: const CircleBorder(),
      side: BorderSide(color: nocturneDivider, width: 1.5),
      fillColor: WidgetStateProperty.resolveWith(
        (states) => states.contains(WidgetState.selected) ? nocturneAccent : Colors.transparent,
      ),
      checkColor: const WidgetStatePropertyAll(nocturneBg),
    ),
    switchTheme: SwitchThemeData(
      thumbColor: WidgetStateProperty.resolveWith(
        (states) => states.contains(WidgetState.selected) ? nocturneAccent : nocturneTextMuted,
      ),
      trackColor: WidgetStateProperty.resolveWith(
        (states) =>
            states.contains(WidgetState.selected) ? nocturneAccent.withValues(alpha: 0.4) : nocturneDivider,
      ),
      trackOutlineColor: const WidgetStatePropertyAll(Colors.transparent),
    ),
    radioTheme: RadioThemeData(
      fillColor: WidgetStateProperty.resolveWith(
        (states) => states.contains(WidgetState.selected) ? nocturneAccent : nocturneDivider,
      ),
    ),
    dividerTheme: DividerThemeData(color: nocturneDivider, space: 1),
    inputDecorationTheme: InputDecorationTheme(
      border: OutlineInputBorder(
        borderRadius: BorderRadius.circular(8),
        borderSide: BorderSide(color: nocturneDivider),
      ),
      enabledBorder: OutlineInputBorder(
        borderRadius: BorderRadius.circular(8),
        borderSide: BorderSide(color: nocturneDivider),
      ),
      focusedBorder: OutlineInputBorder(
        borderRadius: BorderRadius.circular(8),
        borderSide: const BorderSide(color: nocturneAccent),
      ),
      filled: true,
      fillColor: nocturneSurface,
      labelStyle: TextStyle(color: nocturneTextMuted),
    ),
    dialogTheme: DialogThemeData(
      backgroundColor: nocturneSurface,
      surfaceTintColor: Colors.transparent,
      shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(14)),
    ),
    bottomSheetTheme: const BottomSheetThemeData(
      backgroundColor: nocturneSurface,
      surfaceTintColor: Colors.transparent,
      shape: RoundedRectangleBorder(
        borderRadius: BorderRadius.vertical(top: Radius.circular(14)),
      ),
    ),
    floatingActionButtonTheme: const FloatingActionButtonThemeData(
      backgroundColor: nocturneSurfaceHigh,
      foregroundColor: nocturneAccent,
      elevation: 0,
    ),
    chipTheme: ChipThemeData(
      backgroundColor: nocturneSurfaceHigh,
      side: BorderSide(color: nocturneDivider),
      shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(6)),
      labelStyle: TextStyle(color: nocturneText),
    ),
    // Primary actions are an accent outline, never a fill.
    filledButtonTheme: FilledButtonThemeData(
      style: ButtonStyle(
        backgroundColor: WidgetStateProperty.resolveWith(
          (states) => states.contains(WidgetState.pressed)
              ? nocturneAccent.withValues(alpha: 0.22)
              : states.contains(WidgetState.hovered)
                  ? nocturneAccent.withValues(alpha: 0.12)
                  : Colors.transparent,
        ),
        foregroundColor: const WidgetStatePropertyAll(nocturneAccent),
        side: const WidgetStatePropertyAll(BorderSide(color: nocturneAccent)),
        shape: WidgetStatePropertyAll(
          RoundedRectangleBorder(borderRadius: BorderRadius.circular(8)),
        ),
        textStyle: WidgetStatePropertyAll(GoogleFonts.inter(fontWeight: FontWeight.w500)),
      ),
    ),
    outlinedButtonTheme: OutlinedButtonThemeData(
      style: OutlinedButton.styleFrom(
        foregroundColor: nocturneText,
        side: BorderSide(color: nocturneDivider),
        shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(8)),
      ),
    ),
    textButtonTheme: TextButtonThemeData(
      style: TextButton.styleFrom(
        foregroundColor: nocturneAccent,
      ),
    ),
    iconTheme: IconThemeData(color: nocturneTextMuted),
    dropdownMenuTheme: DropdownMenuThemeData(
      textStyle: TextStyle(color: nocturneText),
      menuStyle: MenuStyle(
        backgroundColor: const WidgetStatePropertyAll(nocturneSurface),
        shape: WidgetStatePropertyAll(
          RoundedRectangleBorder(borderRadius: BorderRadius.circular(8)),
        ),
      ),
    ),
    popupMenuTheme: PopupMenuThemeData(
      color: nocturneSurfaceHigh,
      shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(8)),
    ),
  );
}
