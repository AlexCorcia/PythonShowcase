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
/// without needing a full custom color picker.
const List<Color> folderColorPalette = [
  Color(0xFFE53935), // red
  Color(0xFFFB8C00), // orange
  Color(0xFFFDD835), // yellow
  Color(0xFF43A047), // green
  Color(0xFF00897B), // teal
  Color(0xFF1E88E5), // blue
  Color(0xFF5E35B1), // deep purple
  Color(0xFF8E24AA), // purple
  Color(0xFFD81B60), // pink
  Color(0xFF6D4C41), // brown
];
