import '../models/completion_record.dart';
import '../models/folder.dart';
import '../models/task.dart';
import 'profile_repository.dart';

/// Everything a badge's unlock condition might need to look at.
class AchievementContext {
  final List<Task> tasks;
  final List<Folder> folders;
  final List<CompletionRecord> completionLog;
  final int totalPoints;

  const AchievementContext({
    required this.tasks,
    required this.folders,
    required this.completionLog,
    required this.totalPoints,
  });
}

class AchievementBadge {
  final String id;
  final String label;
  final String description;
  final bool unlocked;

  const AchievementBadge({
    required this.id,
    required this.label,
    required this.description,
    required this.unlocked,
  });
}

/// Whether [log] contains at least one completion on each of 7 consecutive
/// calendar days at some point. A simplified "perfect week": it rewards a
/// full week of daily activity rather than literally every due task being
/// completed every day (the latter isn't reconstructable retroactively once
/// tasks are edited/deleted).
bool _hasSevenDayRun(List<CompletionRecord> log) {
  if (log.isEmpty) return false;
  final days = log.map((r) => r.day).toSet().toList()..sort();
  var run = 1;
  for (var i = 1; i < days.length; i++) {
    final gap = days[i].difference(days[i - 1]).inDays;
    if (gap == 1) {
      run += 1;
      if (run >= 7) return true;
    } else if (gap > 1) {
      run = 1;
    }
  }
  return run >= 7;
}

typedef _UnlockCheck = bool Function(AchievementContext context);

class _BadgeDefinition {
  final String id;
  final String label;
  final String description;
  final _UnlockCheck isUnlocked;

  const _BadgeDefinition({
    required this.id,
    required this.label,
    required this.description,
    required this.isUnlocked,
  });
}

final List<_BadgeDefinition> _badgeDefinitions = [
  _BadgeDefinition(
    id: 'first_task',
    label: 'First Task',
    description: 'Complete any task for the first time.',
    isUnlocked: (c) => c.completionLog.isNotEmpty,
  ),
  _BadgeDefinition(
    id: 'streak_7',
    label: '7-Day Streak',
    description: 'Reach a 7-period streak on any task.',
    isUnlocked: (c) => c.tasks.any((t) => t.currentStreak >= 7),
  ),
  _BadgeDefinition(
    id: 'streak_30',
    label: '30-Day Streak',
    description: 'Reach a 30-period streak on any task.',
    isUnlocked: (c) => c.tasks.any((t) => t.currentStreak >= 30),
  ),
  _BadgeDefinition(
    id: 'points_100',
    label: '100 Points',
    description: 'Earn 100 total points.',
    isUnlocked: (c) => c.totalPoints >= 100,
  ),
  _BadgeDefinition(
    id: 'tasks_50',
    label: '50 Tasks',
    description: 'Complete tasks 50 times in total, across all of them.',
    isUnlocked: (c) => c.tasks.fold<int>(0, (sum, t) => sum + t.totalCompletions) >= 50,
  ),
  _BadgeDefinition(
    id: 'perfect_week',
    label: 'Perfect Week',
    description: 'Complete at least one task on each of 7 consecutive days.',
    isUnlocked: (c) => _hasSevenDayRun(c.completionLog),
  ),
  _BadgeDefinition(
    id: 'early_bird',
    label: 'Early Bird',
    description: 'Complete a task before 8 AM.',
    isUnlocked: (c) => c.completionLog.any((r) => r.completedAt.hour < 8),
  ),
  _BadgeDefinition(
    id: 'night_owl',
    label: 'Night Owl',
    description: 'Complete a task at or after 10 PM.',
    isUnlocked: (c) => c.completionLog.any((r) => r.completedAt.hour >= 22),
  ),
  _BadgeDefinition(
    id: 'folders_3',
    label: '3 Folders',
    description: 'Organize your tasks into 3 or more folders.',
    isUnlocked: (c) => c.folders.length >= 3,
  ),
  _BadgeDefinition(
    id: 'points_500',
    label: '500 Points',
    description: 'Earn 500 total points.',
    isUnlocked: (c) => c.totalPoints >= 500,
  ),
  _BadgeDefinition(
    id: 'streak_365',
    label: '365-Day Streak',
    description: 'Reach a 365-period streak on any task.',
    isUnlocked: (c) => c.tasks.any((t) => t.currentStreak >= 365),
  ),
  _BadgeDefinition(
    id: 'level_20',
    label: 'Level 20',
    description: 'Reach level 20.',
    isUnlocked: (c) => levelForPoints(c.totalPoints).level >= 20,
  ),
];

/// Badges are derived state, not persisted - always recomputed from the
/// current tasks/folders/completion log/points, the same way [levelForPoints]
/// derives level from points. Keeps this immune to desync bugs.
List<AchievementBadge> computeBadges(AchievementContext context) => [
      for (final def in _badgeDefinitions)
        AchievementBadge(
          id: def.id,
          label: def.label,
          description: def.description,
          unlocked: def.isUnlocked(context),
        ),
    ];
