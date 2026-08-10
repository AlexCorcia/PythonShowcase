import 'package:flutter/material.dart';
import 'package:shared_preferences/shared_preferences.dart';

/// What the persistent status notification's body shows.
enum NotificationContentMode { countsOnly, taskNames, dueSoon, countsAndNames }

/// Single owner of small app-wide settings (persisted via SharedPreferences).
class SettingsRepository {
  static const _showStatusNotificationKey = 'mordify.showStatusNotification';
  static const _notificationContentModeKey = 'mordify.notificationContentMode';
  static const _themeModeKey = 'mordify.themeMode';

  Future<bool> getShowStatusNotification() async {
    final prefs = await SharedPreferences.getInstance();
    return prefs.getBool(_showStatusNotificationKey) ?? true;
  }

  Future<void> setShowStatusNotification(bool value) async {
    final prefs = await SharedPreferences.getInstance();
    await prefs.setBool(_showStatusNotificationKey, value);
  }

  Future<NotificationContentMode> getNotificationContentMode() async {
    final prefs = await SharedPreferences.getInstance();
    final raw = prefs.getString(_notificationContentModeKey);
    return NotificationContentMode.values.firstWhere(
      (m) => m.name == raw,
      orElse: () => NotificationContentMode.countsOnly,
    );
  }

  Future<void> setNotificationContentMode(NotificationContentMode mode) async {
    final prefs = await SharedPreferences.getInstance();
    await prefs.setString(_notificationContentModeKey, mode.name);
  }

  Future<ThemeMode> getThemeMode() async {
    final prefs = await SharedPreferences.getInstance();
    final raw = prefs.getString(_themeModeKey);
    return ThemeMode.values.firstWhere(
      (m) => m.name == raw,
      orElse: () => ThemeMode.system,
    );
  }

  Future<void> setThemeMode(ThemeMode mode) async {
    final prefs = await SharedPreferences.getInstance();
    await prefs.setString(_themeModeKey, mode.name);
  }
}
