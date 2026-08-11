import 'package:flutter/material.dart';

import '../models/completion_record.dart';
import '../theme/app_theme.dart';

const _monthAbbrev = [
  'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
  'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec',
];

const _weekdayShort = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'];

DateTime _startOfWeek(DateTime date) {
  final day = DateTime(date.year, date.month, date.day);
  return day.subtract(Duration(days: day.weekday - 1));
}

class _WeekBucket {
  final DateTime start;
  final int completions;
  final int points;

  const _WeekBucket({required this.start, required this.completions, required this.points});
}

/// Trends derived from the completion log - pushed from [ProfileScreen]'s
/// "View stats & trends" link. Everything here is read-only and hand-rolled
/// (no chart dependency) to match the rest of the app's custom progress
/// rings/bars.
class StatsScreen extends StatelessWidget {
  final List<CompletionRecord> completionLog;

  const StatsScreen({super.key, required this.completionLog});

  List<_WeekBucket> _lastNWeeks(int n) {
    final now = DateTime.now();
    final currentWeekStart = _startOfWeek(now);
    final buckets = <_WeekBucket>[];
    for (var i = n - 1; i >= 0; i--) {
      final start = currentWeekStart.subtract(Duration(days: 7 * i));
      final end = start.add(const Duration(days: 7));
      final entries = completionLog.where((r) => !r.day.isBefore(start) && r.day.isBefore(end));
      buckets.add(_WeekBucket(
        start: start,
        completions: entries.length,
        points: entries.fold<int>(0, (sum, r) => sum + (r.points ?? 0)),
      ));
    }
    return buckets;
  }

  List<int> _completionsByWeekday() {
    final counts = List.filled(7, 0);
    for (final r in completionLog) {
      counts[r.completedAt.weekday - 1] += 1;
    }
    return counts;
  }

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    final colorScheme = theme.colorScheme;

    final totalPoints = completionLog.fold<int>(0, (sum, r) => sum + (r.points ?? 0));
    final weeks = _lastNWeeks(8);
    final byWeekday = _completionsByWeekday();
    final maxWeekly = weeks.map((w) => w.completions).fold<int>(0, (a, b) => a > b ? a : b);
    final maxWeekday = byWeekday.fold<int>(0, (a, b) => a > b ? a : b);

    final now = DateTime.now();
    final thisMonthEntries =
        completionLog.where((r) => r.completedAt.year == now.year && r.completedAt.month == now.month);
    final thisMonthCount = thisMonthEntries.length;

    return Scaffold(
      appBar: AppBar(title: const Text('Stats & Trends')),
      body: ListView(
        padding: const EdgeInsets.fromLTRB(20, 8, 20, 24),
        children: [
          Row(
            children: [
              Expanded(
                child: _SummaryCard(
                  label: 'Total completions',
                  value: '${completionLog.length}',
                  icon: Icons.check_circle,
                  color: colorScheme.primary,
                ),
              ),
              const SizedBox(width: 12),
              Expanded(
                child: _SummaryCard(
                  label: 'Lifetime points',
                  value: '$totalPoints',
                  icon: Icons.stars_rounded,
                  color: mordifyAmber,
                ),
              ),
            ],
          ),
          const SizedBox(height: 12),
          _SummaryCard(
            label: '${_monthAbbrev[now.month - 1]} completions',
            value: '$thisMonthCount',
            icon: Icons.calendar_month_outlined,
            color: colorScheme.secondary,
            fullWidth: true,
          ),
          const SizedBox(height: 24),
          Text('LAST 8 WEEKS',
              style: theme.textTheme.labelMedium
                  ?.copyWith(color: colorScheme.onSurfaceVariant, letterSpacing: 1.1)),
          const SizedBox(height: 12),
          Container(
            padding: const EdgeInsets.fromLTRB(16, 20, 16, 12),
            decoration: BoxDecoration(
              color: colorScheme.surfaceContainer,
              borderRadius: BorderRadius.circular(14),
            ),
            child: Column(
              children: [
                SizedBox(
                  height: 120,
                  child: Row(
                    crossAxisAlignment: CrossAxisAlignment.end,
                    children: [
                      for (final week in weeks)
                        Expanded(
                          child: Padding(
                            padding: const EdgeInsets.symmetric(horizontal: 4),
                            child: Tooltip(
                              message: '${week.completions} done · +${week.points}',
                              child: Column(
                                mainAxisAlignment: MainAxisAlignment.end,
                                children: [
                                  Text(
                                    week.completions == 0 ? '' : '${week.completions}',
                                    style: theme.textTheme.labelSmall
                                        ?.copyWith(color: colorScheme.onSurfaceVariant),
                                  ),
                                  const SizedBox(height: 4),
                                  Container(
                                    height: maxWeekly == 0
                                        ? 4
                                        : 6 + (week.completions / maxWeekly) * 84,
                                    decoration: BoxDecoration(
                                      color: week.completions == 0
                                          ? colorScheme.outlineVariant
                                          : mordifyAmber,
                                      borderRadius: BorderRadius.circular(4),
                                    ),
                                  ),
                                ],
                              ),
                            ),
                          ),
                        ),
                    ],
                  ),
                ),
                const SizedBox(height: 8),
                Row(
                  children: [
                    for (final week in weeks)
                      Expanded(
                        child: Text(
                          '${week.start.day}/${week.start.month}',
                          textAlign: TextAlign.center,
                          style: theme.textTheme.labelSmall
                              ?.copyWith(color: colorScheme.onSurfaceVariant, fontSize: 9),
                        ),
                      ),
                  ],
                ),
              ],
            ),
          ),
          const SizedBox(height: 24),
          Text('BY DAY OF WEEK',
              style: theme.textTheme.labelMedium
                  ?.copyWith(color: colorScheme.onSurfaceVariant, letterSpacing: 1.1)),
          const SizedBox(height: 12),
          Container(
            padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 16),
            decoration: BoxDecoration(
              color: colorScheme.surfaceContainer,
              borderRadius: BorderRadius.circular(14),
            ),
            child: Column(
              children: [
                for (var i = 0; i < 7; i++)
                  Padding(
                    padding: const EdgeInsets.symmetric(vertical: 5),
                    child: Row(
                      children: [
                        SizedBox(
                          width: 34,
                          child: Text(_weekdayShort[i], style: theme.textTheme.bodySmall),
                        ),
                        Expanded(
                          child: ClipRRect(
                            borderRadius: BorderRadius.circular(4),
                            child: LinearProgressIndicator(
                              value: maxWeekday == 0 ? 0 : byWeekday[i] / maxWeekday,
                              minHeight: 10,
                              backgroundColor: colorScheme.outlineVariant,
                              valueColor: const AlwaysStoppedAnimation(mordifyAmber),
                            ),
                          ),
                        ),
                        const SizedBox(width: 10),
                        SizedBox(
                          width: 22,
                          child: Text(
                            '${byWeekday[i]}',
                            textAlign: TextAlign.right,
                            style: theme.textTheme.bodySmall
                                ?.copyWith(color: colorScheme.onSurfaceVariant),
                          ),
                        ),
                      ],
                    ),
                  ),
              ],
            ),
          ),
          if (completionLog.isEmpty) ...[
            const SizedBox(height: 40),
            Center(
              child: Text(
                'Complete a few tasks to see your trends here',
                style: theme.textTheme.bodyMedium?.copyWith(color: colorScheme.outline),
              ),
            ),
          ],
        ],
      ),
    );
  }
}

class _SummaryCard extends StatelessWidget {
  final String label;
  final String value;
  final IconData icon;
  final Color color;
  final bool fullWidth;

  const _SummaryCard({
    required this.label,
    required this.value,
    required this.icon,
    required this.color,
    this.fullWidth = false,
  });

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    return Container(
      width: fullWidth ? double.infinity : null,
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: Theme.of(context).colorScheme.surfaceContainer,
        borderRadius: BorderRadius.circular(14),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Icon(icon, color: color),
          const SizedBox(height: 8),
          Text(value, style: theme.textTheme.headlineSmall?.copyWith(fontWeight: FontWeight.bold)),
          Text(label, style: theme.textTheme.bodySmall?.copyWith(color: theme.colorScheme.outline)),
        ],
      ),
    );
  }
}
