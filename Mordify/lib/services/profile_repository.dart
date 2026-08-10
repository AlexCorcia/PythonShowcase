import 'package:shared_preferences/shared_preferences.dart';

const _pointsPerLevel = 100;

class LevelInfo {
  final int level;
  final int pointsIntoLevel;
  final int pointsForNextLevel;

  const LevelInfo({
    required this.level,
    required this.pointsIntoLevel,
    required this.pointsForNextLevel,
  });

  double get progress => pointsIntoLevel / pointsForNextLevel;
}

LevelInfo levelForPoints(int points) {
  return LevelInfo(
    level: points ~/ _pointsPerLevel + 1,
    pointsIntoLevel: points % _pointsPerLevel,
    pointsForNextLevel: _pointsPerLevel,
  );
}

class ProfileRepository {
  static const _totalPointsKey = 'mordify.totalPoints';
  static const _displayNameKey = 'mordify.displayName';
  static const defaultDisplayName = 'ML41';

  Future<int> getTotalPoints() async {
    final prefs = await SharedPreferences.getInstance();
    return prefs.getInt(_totalPointsKey) ?? 0;
  }

  /// Adds [delta] (can be negative, e.g. when a task is unchecked) to the
  /// running total and returns the new total.
  Future<int> addPoints(int delta) async {
    final prefs = await SharedPreferences.getInstance();
    final current = prefs.getInt(_totalPointsKey) ?? 0;
    final updated = (current + delta).clamp(0, 1 << 31);
    await prefs.setInt(_totalPointsKey, updated);
    return updated;
  }

  Future<String> getDisplayName() async {
    final prefs = await SharedPreferences.getInstance();
    return prefs.getString(_displayNameKey) ?? defaultDisplayName;
  }

  Future<void> setDisplayName(String name) async {
    final prefs = await SharedPreferences.getInstance();
    await prefs.setString(_displayNameKey, name);
  }
}
