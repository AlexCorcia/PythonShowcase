import 'package:flutter/material.dart';

/// A user-defined, colored grouping tasks can be filed under.
class Folder {
  final String id;
  String name;
  int colorValue;
  bool isExpanded;

  Folder({
    required this.id,
    required this.name,
    required this.colorValue,
    this.isExpanded = true,
  });

  Color get color => Color(colorValue);

  Map<String, dynamic> toJson() => {
        'id': id,
        'name': name,
        'colorValue': colorValue,
        'isExpanded': isExpanded,
      };

  factory Folder.fromJson(Map<String, dynamic> json) => Folder(
        id: json['id'] as String,
        name: json['name'] as String,
        colorValue: json['colorValue'] as int,
        isExpanded: json['isExpanded'] as bool? ?? true,
      );
}

/// A fixed, curated palette so folder colors stay legible in both themes
/// without needing a full custom color picker. Kept low-chroma to match the
/// Nocturne dark theme's "keep chroma low outside the accent" rule - these
/// are folder/category colors, not the app's structural accent.
const List<Color> folderColorPalette = [
  Color(0xFF8FAE86), // sage
  Color(0xFFC98A68), // terracotta
  Color(0xFF7691B0), // dusty blue
  Color(0xFF9A87B0), // dusty plum
  Color(0xFFC97A93), // dusty rose
  Color(0xFFB0A25A), // muted gold
  Color(0xFF6FA0A0), // muted teal
  Color(0xFFA0876F), // taupe
];
