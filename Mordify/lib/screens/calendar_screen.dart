import 'package:flutter/material.dart';

import '../models/completion_record.dart';
import '../models/task.dart' show weekdayNames;
import '../theme/app_theme.dart' show mordifyAmber;

const _monthNames = [
  'January',
  'February',
  'March',
  'April',
  'May',
  'June',
  'July',
  'August',
  'September',
  'October',
  'November',
  'December',
];

const _weekdayLabels = ['M', 'T', 'W', 'T', 'F', 'S', 'S'];

/// Month-at-a-glance view of [CompletionRecord]s - tap any past (or today's)
/// day to see which tasks were completed on it. Future days are shown but
/// disabled, since there's nothing to look back on yet.
class CalendarScreen extends StatefulWidget {
  final List<CompletionRecord> completionLog;

  const CalendarScreen({super.key, required this.completionLog});

  @override
  State<CalendarScreen> createState() => _CalendarScreenState();
}

class _CalendarScreenState extends State<CalendarScreen> {
  late DateTime _visibleMonth;
  late DateTime _selectedDay;

  @override
  void initState() {
    super.initState();
    final now = DateTime.now();
    _visibleMonth = DateTime(now.year, now.month);
    _selectedDay = DateTime(now.year, now.month, now.day);
  }

  Map<DateTime, List<CompletionRecord>> _groupByDay() {
    final map = <DateTime, List<CompletionRecord>>{};
    for (final record in widget.completionLog) {
      map.putIfAbsent(record.day, () => []).add(record);
    }
    return map;
  }

  void _changeMonth(int delta) {
    setState(() => _visibleMonth = DateTime(_visibleMonth.year, _visibleMonth.month + delta));
  }

  String _selectedDayLabel(DateTime todayKey) {
    if (_selectedDay == todayKey) return 'Today';
    if (_selectedDay == todayKey.subtract(const Duration(days: 1))) return 'Yesterday';
    return '${weekdayNames[_selectedDay.weekday - 1]}, '
        '${_monthNames[_selectedDay.month - 1]} ${_selectedDay.day}';
  }

  String _formatTime(DateTime dt) {
    final hour = dt.hour % 12 == 0 ? 12 : dt.hour % 12;
    final minute = dt.minute.toString().padLeft(2, '0');
    final period = dt.hour < 12 ? 'AM' : 'PM';
    return '$hour:$minute $period';
  }

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    final colorScheme = theme.colorScheme;

    final now = DateTime.now();
    final todayKey = DateTime(now.year, now.month, now.day);
    final byDay = _groupByDay();

    final firstOfMonth = DateTime(_visibleMonth.year, _visibleMonth.month, 1);
    final daysInMonth = DateTime(_visibleMonth.year, _visibleMonth.month + 1, 0).day;
    final leadingEmpty = (firstOfMonth.weekday - DateTime.monday) % 7;
    final isCurrentMonth = _visibleMonth.year == todayKey.year && _visibleMonth.month == todayKey.month;

    final cells = <Widget>[
      for (var i = 0; i < leadingEmpty; i++) const SizedBox.shrink(),
      for (var day = 1; day <= daysInMonth; day++)
        _buildDayCell(context, day, byDay, todayKey),
    ];

    final selectedRecords = (byDay[_selectedDay] ?? const <CompletionRecord>[]).toList()
      ..sort((a, b) => b.completedAt.compareTo(a.completedAt));

    return ListView(
        padding: const EdgeInsets.fromLTRB(20, 16, 20, 24),
        children: [
          Row(
            mainAxisAlignment: MainAxisAlignment.spaceBetween,
            children: [
              IconButton(
                icon: const Icon(Icons.chevron_left),
                onPressed: () => _changeMonth(-1),
              ),
              Text(
                '${_monthNames[_visibleMonth.month - 1]} ${_visibleMonth.year}',
                style: theme.textTheme.titleLarge?.copyWith(fontWeight: FontWeight.bold),
              ),
              IconButton(
                icon: const Icon(Icons.chevron_right),
                onPressed: isCurrentMonth ? null : () => _changeMonth(1),
              ),
            ],
          ),
          const SizedBox(height: 4),
          Row(
            children: [
              for (final label in _weekdayLabels)
                Expanded(
                  child: Center(
                    child: Text(
                      label,
                      style: theme.textTheme.labelMedium?.copyWith(color: colorScheme.outline),
                    ),
                  ),
                ),
            ],
          ),
          const SizedBox(height: 4),
          GridView.count(
            crossAxisCount: 7,
            shrinkWrap: true,
            physics: const NeverScrollableScrollPhysics(),
            children: cells,
          ),
          const SizedBox(height: 24),
          Text(
            _selectedDayLabel(todayKey),
            style: theme.textTheme.titleMedium?.copyWith(fontWeight: FontWeight.bold),
          ),
          const SizedBox(height: 8),
          if (selectedRecords.isEmpty)
            Padding(
              padding: const EdgeInsets.symmetric(vertical: 24),
              child: Center(
                child: Text(
                  'No tasks completed',
                  style: theme.textTheme.bodyMedium?.copyWith(color: colorScheme.outline),
                ),
              ),
            )
          else
            Card(
              child: Column(
                children: [
                  for (final record in selectedRecords)
                    ListTile(
                      leading: CircleAvatar(
                        radius: 6,
                        backgroundColor: record.folderColorValue != null
                            ? Color(record.folderColorValue!)
                            : colorScheme.primary,
                      ),
                      title: Text(record.taskTitle),
                      subtitle: Text(
                        _formatTime(record.completedAt),
                        style: theme.textTheme.bodySmall?.copyWith(color: colorScheme.outline),
                      ),
                      trailing: record.points != null
                          ? Text(
                              '+${record.points}',
                              style: theme.textTheme.bodyMedium
                                  ?.copyWith(color: mordifyAmber, fontWeight: FontWeight.w600),
                            )
                          : null,
                    ),
                ],
              ),
            ),
        ],
      );
  }

  Widget _buildDayCell(
    BuildContext context,
    int day,
    Map<DateTime, List<CompletionRecord>> byDay,
    DateTime todayKey,
  ) {
    final theme = Theme.of(context);
    final colorScheme = theme.colorScheme;
    final date = DateTime(_visibleMonth.year, _visibleMonth.month, day);
    final records = byDay[date] ?? const <CompletionRecord>[];
    final isSelected = date == _selectedDay;
    final isToday = date == todayKey;
    final isFuture = date.isAfter(todayKey);

    final dotColors = <Color>{
      for (final r in records) r.folderColorValue != null ? Color(r.folderColorValue!) : colorScheme.primary,
    }.take(4).toList();

    return InkWell(
      onTap: isFuture ? null : () => setState(() => _selectedDay = date),
      borderRadius: BorderRadius.circular(12),
      child: Container(
        margin: const EdgeInsets.all(3),
        decoration: BoxDecoration(
          color: isSelected ? colorScheme.primaryContainer : null,
          border: isToday && !isSelected ? Border.all(color: colorScheme.primary, width: 1.5) : null,
          borderRadius: BorderRadius.circular(12),
        ),
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            Text(
              '$day',
              style: theme.textTheme.bodyMedium?.copyWith(
                color: isFuture
                    ? colorScheme.outlineVariant
                    : isSelected
                        ? colorScheme.onPrimaryContainer
                        : colorScheme.onSurface,
                fontWeight: isToday ? FontWeight.bold : FontWeight.normal,
              ),
            ),
            const SizedBox(height: 2),
            SizedBox(
              height: 6,
              child: Row(
                mainAxisAlignment: MainAxisAlignment.center,
                children: [
                  for (final color in dotColors)
                    Container(
                      width: 5,
                      height: 5,
                      margin: const EdgeInsets.symmetric(horizontal: 1),
                      decoration: BoxDecoration(color: color, shape: BoxShape.circle),
                    ),
                ],
              ),
            ),
          ],
        ),
      ),
    );
  }
}
