import 'package:flutter/material.dart';

import '../models/completion_record.dart';
import '../models/folder.dart';
import '../models/task.dart';
import '../services/profile_repository.dart';
import '../theme/app_theme.dart';

const _monthAbbrev = [
  'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
  'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec',
];

const _dayInitials = ['M', 'T', 'W', 'T', 'F', 'S', 'S'];

DateTime _startOfWeek(DateTime date) {
  final day = DateTime(date.year, date.month, date.day);
  return day.subtract(Duration(days: day.weekday - 1));
}

class _WeeklyRecap {
  final int completed;
  final int points;
  final String? bestDay;

  const _WeeklyRecap({required this.completed, required this.points, this.bestDay});
}

/// The "Home" bottom-nav destination: a dashboard summarizing today's
/// progress, the current streak, level/XP, last week's recap and what's due
/// next - everything here is read-only, derived from state owned by the app
/// shell. Matches design/handoff's "Home Dashboard" mockup.
class HomeDashboardScreen extends StatefulWidget {
  final List<Task> tasks;
  final List<Folder> folders;
  final List<CompletionRecord> completionLog;

  const HomeDashboardScreen({
    super.key,
    required this.tasks,
    required this.folders,
    required this.completionLog,
  });

  @override
  State<HomeDashboardScreen> createState() => _HomeDashboardScreenState();
}

class _HomeDashboardScreenState extends State<HomeDashboardScreen> {
  final _profile = ProfileRepository();
  String _displayName = ProfileRepository.defaultDisplayName;
  int _totalPoints = 0;
  bool _loading = true;

  @override
  void initState() {
    super.initState();
    _load();
  }

  @override
  void didUpdateWidget(covariant HomeDashboardScreen oldWidget) {
    super.didUpdateWidget(oldWidget);
    _load();
  }

  Future<void> _load() async {
    final name = await _profile.getDisplayName();
    final points = await _profile.getTotalPoints();
    if (!mounted) return;
    setState(() {
      _displayName = name;
      _totalPoints = points;
      _loading = false;
    });
  }

  Folder? _folderForTask(Task task) {
    if (task.folderId == null) return null;
    for (final folder in widget.folders) {
      if (folder.id == task.folderId) return folder;
    }
    return null;
  }

  _WeeklyRecap _lastWeekRecap() {
    final now = DateTime.now();
    final currentWeekStart = _startOfWeek(now);
    final lastWeekStart = currentWeekStart.subtract(const Duration(days: 7));
    final entries = widget.completionLog
        .where((r) => !r.day.isBefore(lastWeekStart) && r.day.isBefore(currentWeekStart))
        .toList();
    if (entries.isEmpty) return const _WeeklyRecap(completed: 0, points: 0);

    final points = entries.fold<int>(0, (sum, r) => sum + (r.points ?? 0));
    final countByDay = <DateTime, int>{};
    for (final r in entries) {
      countByDay[r.day] = (countByDay[r.day] ?? 0) + 1;
    }
    final bestDayEntry = countByDay.entries.reduce((a, b) => a.value >= b.value ? a : b);
    return _WeeklyRecap(
      completed: entries.length,
      points: points,
      bestDay: weekdayNames[bestDayEntry.key.weekday - 1],
    );
  }

  @override
  Widget build(BuildContext context) {
    if (_loading) {
      return const Center(child: CircularProgressIndicator());
    }
    final theme = Theme.of(context);
    final colorScheme = theme.colorScheme;
    final now = DateTime.now();
    final todayKey = DateTime(now.year, now.month, now.day);

    final dueToday = widget.tasks.where((t) => t.category == TaskCategory.daily && t.isDueToday).toList();
    final doneToday = dueToday.where((t) => t.isDoneForCurrentPeriod).length;
    final goalFraction = dueToday.isEmpty ? 0.0 : doneToday / dueToday.length;

    final bestStreakTask = widget.tasks.isEmpty
        ? null
        : widget.tasks.reduce((a, b) => a.currentStreak >= b.currentStreak ? a : b);

    final activeDays = <DateTime>{for (final r in widget.completionLog) r.day};
    final last7 = [
      for (var i = 6; i >= 0; i--) todayKey.subtract(Duration(days: i)),
    ];

    final recap = _lastWeekRecap();

    final upNext = dueToday.where((t) => !t.isDoneForCurrentPeriod).toList()
      ..sort((a, b) {
        final aKey = a.hasReminder ? a.hour! * 60 + a.minute! : 24 * 60;
        final bKey = b.hasReminder ? b.hour! * 60 + b.minute! : 24 * 60;
        return aKey.compareTo(bKey);
      });
    final upNextTop = upNext.take(3).toList();

    return ListView(
      padding: const EdgeInsets.fromLTRB(20, 16, 20, 24),
      children: [
        Row(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Expanded(
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text(
                    '${weekdayNames[now.weekday - 1]}, ${_monthAbbrev[now.month - 1]} ${now.day}',
                    style: theme.textTheme.bodySmall?.copyWith(color: colorScheme.onSurfaceVariant),
                  ),
                  const SizedBox(height: 2),
                  Text(
                    'Nice pace, ${_displayName.split(' ').first}',
                    style: theme.textTheme.headlineSmall?.copyWith(fontWeight: FontWeight.w500),
                  ),
                ],
              ),
            ),
            Icon(Icons.notifications_none, color: colorScheme.onSurfaceVariant),
          ],
        ),
        const SizedBox(height: 18),
        Container(
          padding: const EdgeInsets.all(18),
          decoration: BoxDecoration(
            color: colorScheme.surfaceContainer,
            borderRadius: BorderRadius.circular(14),
          ),
          child: Row(
            children: [
              SizedBox(
                width: 100,
                height: 100,
                child: Stack(
                  alignment: Alignment.center,
                  children: [
                    SizedBox(
                      width: 100,
                      height: 100,
                      child: CircularProgressIndicator(
                        value: goalFraction == 0 ? 1 : goalFraction,
                        strokeWidth: 9,
                        backgroundColor: colorScheme.outlineVariant,
                        valueColor: AlwaysStoppedAnimation(
                          goalFraction == 0 ? colorScheme.outlineVariant : mordifyAmber,
                        ),
                      ),
                    ),
                    Column(
                      mainAxisSize: MainAxisSize.min,
                      children: [
                        Text('$doneToday/${dueToday.length}',
                            style: theme.textTheme.titleLarge?.copyWith(fontWeight: FontWeight.w600)),
                        Text('today',
                            style: theme.textTheme.labelSmall
                                ?.copyWith(color: colorScheme.onSurfaceVariant)),
                      ],
                    ),
                  ],
                ),
              ),
              const SizedBox(width: 20),
              Expanded(
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  mainAxisSize: MainAxisSize.min,
                  children: [
                    Text('Daily goal',
                        style: theme.textTheme.bodyMedium?.copyWith(fontWeight: FontWeight.w500)),
                    const SizedBox(height: 4),
                    Text(
                      dueToday.isEmpty
                          ? 'No daily tasks yet'
                          : doneToday >= dueToday.length
                              ? 'All done for today'
                              : '${dueToday.length - doneToday} task${dueToday.length - doneToday == 1 ? '' : 's'} left - keep it going',
                      style: theme.textTheme.bodySmall?.copyWith(color: colorScheme.onSurfaceVariant),
                    ),
                    if (bestStreakTask != null && bestStreakTask.currentStreak >= 2) ...[
                      const SizedBox(height: 6),
                      Row(
                        children: [
                          const Text('🔥', style: TextStyle(fontSize: 15)),
                          const SizedBox(width: 6),
                          Text('${bestStreakTask.currentStreak}-day streak',
                              style: theme.textTheme.bodyMedium?.copyWith(fontWeight: FontWeight.w500)),
                          if (bestStreakTask.freezesAvailable > 0) ...[
                            const SizedBox(width: 6),
                            Container(
                              padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 2),
                              decoration: BoxDecoration(
                                border: Border.all(color: nocturneAccent.withValues(alpha: 0.45)),
                                borderRadius: BorderRadius.circular(20),
                              ),
                              child: Text(
                                '${bestStreakTask.freezesAvailable} freeze',
                                style: theme.textTheme.labelSmall?.copyWith(color: nocturneAccent2),
                              ),
                            ),
                          ],
                        ],
                      ),
                    ],
                    const SizedBox(height: 8),
                    Row(
                      children: [
                        for (final day in last7)
                          Expanded(
                            child: Column(
                              children: [
                                Text(_dayInitials[day.weekday - 1],
                                    style: theme.textTheme.labelSmall
                                        ?.copyWith(color: colorScheme.onSurfaceVariant, fontSize: 9)),
                                const SizedBox(height: 3),
                                Container(
                                  width: 14,
                                  height: 14,
                                  decoration: BoxDecoration(
                                    shape: BoxShape.circle,
                                    color: activeDays.contains(day) ? mordifyAmber : Colors.transparent,
                                    border: Border.all(
                                      color: activeDays.contains(day) ? mordifyAmber : colorScheme.outlineVariant,
                                      width: 1.5,
                                    ),
                                  ),
                                ),
                              ],
                            ),
                          ),
                      ],
                    ),
                  ],
                ),
              ),
            ],
          ),
        ),
        const SizedBox(height: 14),
        Container(
          padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 14),
          decoration: BoxDecoration(
            color: colorScheme.surfaceContainer,
            borderRadius: BorderRadius.circular(14),
          ),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Row(
                mainAxisAlignment: MainAxisAlignment.spaceBetween,
                children: [
                  Text('Level ${levelForPoints(_totalPoints).level}',
                      style: theme.textTheme.bodyMedium?.copyWith(fontWeight: FontWeight.w500)),
                  Text(
                    '${levelForPoints(_totalPoints).pointsIntoLevel} / ${levelForPoints(_totalPoints).pointsForNextLevel} XP',
                    style: theme.textTheme.bodySmall?.copyWith(color: colorScheme.onSurfaceVariant),
                  ),
                ],
              ),
              const SizedBox(height: 8),
              ClipRRect(
                borderRadius: BorderRadius.circular(4),
                child: LinearProgressIndicator(
                  value: levelForPoints(_totalPoints).progress,
                  minHeight: 8,
                  backgroundColor: colorScheme.outlineVariant,
                  valueColor: AlwaysStoppedAnimation(mordifyAmber),
                ),
              ),
            ],
          ),
        ),
        const SizedBox(height: 14),
        Container(
          padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 14),
          decoration: BoxDecoration(
            color: colorScheme.surfaceContainerHigh,
            borderRadius: BorderRadius.circular(14),
          ),
          child: Row(
            mainAxisAlignment: MainAxisAlignment.spaceBetween,
            children: [
              Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text('Last week', style: theme.textTheme.bodyMedium?.copyWith(fontWeight: FontWeight.w500)),
                  const SizedBox(height: 2),
                  Text(
                    recap.completed == 0
                        ? 'No completions yet'
                        : '${recap.completed} tasks${recap.bestDay != null ? ' · best day ${recap.bestDay}' : ''}',
                    style: theme.textTheme.bodySmall?.copyWith(color: colorScheme.onSurfaceVariant),
                  ),
                ],
              ),
              Text('+${recap.points}',
                  style: theme.textTheme.titleMedium?.copyWith(color: mordifyAmber, fontWeight: FontWeight.w600)),
            ],
          ),
        ),
        if (upNextTop.isNotEmpty) ...[
          const SizedBox(height: 22),
          Text(
            'UP NEXT',
            style: theme.textTheme.labelMedium
                ?.copyWith(color: colorScheme.onSurfaceVariant, letterSpacing: 1.1),
          ),
          for (final task in upNextTop)
            Container(
              padding: const EdgeInsets.symmetric(vertical: 10),
              decoration: BoxDecoration(
                border: Border(bottom: BorderSide(color: colorScheme.outlineVariant)),
              ),
              child: Row(
                children: [
                  Container(
                    width: 8,
                    height: 8,
                    decoration: BoxDecoration(
                      shape: BoxShape.circle,
                      color: _folderForTask(task)?.color ?? colorScheme.primary,
                    ),
                  ),
                  const SizedBox(width: 14),
                  Expanded(child: Text(task.title, style: theme.textTheme.bodyMedium)),
                  Text(
                    task.hasReminder
                        ? TimeOfDay(hour: task.hour!, minute: task.minute!).format(context)
                        : 'Anytime',
                    style: theme.textTheme.bodySmall?.copyWith(color: colorScheme.onSurfaceVariant),
                  ),
                ],
              ),
            ),
        ],
      ],
    );
  }
}
