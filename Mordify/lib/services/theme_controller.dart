import 'package:flutter/material.dart';

import 'settings_repository.dart';

class ThemeController extends ValueNotifier<ThemeMode> {
  ThemeController() : super(ThemeMode.system);

  final _settings = SettingsRepository();

  Future<void> load() async {
    value = await _settings.getThemeMode();
  }

  Future<void> setThemeMode(ThemeMode mode) async {
    value = mode;
    await _settings.setThemeMode(mode);
  }
}
