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
  static const _unlockedBadgeIdsKey = 'mordify.unlockedBadgeIds';
  static const _badgesBaselinedKey = 'mordify.badgesBaselined';
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

  Future<void> setTotalPoints(int value) async {
    final prefs = await SharedPreferences.getInstance();
    await prefs.setInt(_totalPointsKey, value.clamp(0, 1 << 31));
  }

  Future<String> getDisplayName() async {
    final prefs = await SharedPreferences.getInstance();
    return prefs.getString(_displayNameKey) ?? defaultDisplayName;
  }

  Future<void> setDisplayName(String name) async {
    final prefs = await SharedPreferences.getInstance();
    await prefs.setString(_displayNameKey, name);
  }

  Future<Set<String>> getUnlockedBadgeIds() async {
    final prefs = await SharedPreferences.getInstance();
    return (prefs.getStringList(_unlockedBadgeIdsKey) ?? const []).toSet();
  }

  Future<void> setUnlockedBadgeIds(Set<String> ids) async {
    final prefs = await SharedPreferences.getInstance();
    await prefs.setStringList(_unlockedBadgeIdsKey, ids.toList());
  }

  /// Whether [getUnlockedBadgeIds] has ever been seeded from the badges a
  /// user already had unlocked when this feature shipped - without this,
  /// every badge already earned by an existing user would look "new" on
  /// their first launch post-upgrade and fire a celebration for each.
  Future<bool> hasBaselinedBadges() async {
    final prefs = await SharedPreferences.getInstance();
    return prefs.getBool(_badgesBaselinedKey) ?? false;
  }

  Future<void> markBadgesBaselined() async {
    final prefs = await SharedPreferences.getInstance();
    await prefs.setBool(_badgesBaselinedKey, true);
  }
}
