import '../models/task.dart';

/// Best-effort "when does this task's reminder next fire", for display
/// purposes only (e.g. the status notification's "due soon" mode). This is
/// intentionally a lighter, separate calculation from NotificationService's
/// internal alarm-scheduling math - that code is the real load-bearing logic
/// actual Android alarms are scheduled against (stateful timezone
/// precondition, exact-to-the-second correctness), while a display estimate
/// only needs to be right to within a few minutes.
DateTime? nextReminderOccurrence(Task task, {DateTime? now}) {
  if (!task.hasReminder) return null; // covers timesPerWeek too - never has hour/minute
  final n = now ?? DateTime.now();
  var candidate = DateTime(n.year, n.month, n.day, task.hour!, task.minute!);

  switch (task.frequency) {
    case TaskFrequency.daily:
      if (!candidate.isAfter(n)) candidate = candidate.add(const Duration(days: 1));
      return candidate;

    case TaskFrequency.weekdays:
      while (!candidate.isAfter(n) ||
          candidate.weekday == DateTime.saturday ||
          candidate.weekday == DateTime.sunday) {
        candidate = candidate.add(const Duration(days: 1));
      }
      return candidate;

    case TaskFrequency.weekly:
      final targetWeekday = task.weekday ?? DateTime.monday;
      while (candidate.weekday != targetWeekday || !candidate.isAfter(n)) {
        candidate = candidate.add(const Duration(days: 1));
      }
      return candidate;

    case TaskFrequency.monthly:
      final targetDay = task.dayOfMonth ?? 1;
      candidate = _monthlyOccurrenceOnOrAfter(n.year, n.month, targetDay, task.hour!, task.minute!);
      if (!candidate.isAfter(n)) {
        final nextMonth = DateTime(n.year, n.month + 1, 1);
        candidate = _monthlyOccurrenceOnOrAfter(
            nextMonth.year, nextMonth.month, targetDay, task.hour!, task.minute!);
      }
      return candidate;

    case TaskFrequency.interval:
      final anchor = task.anchorDate ?? n;
      final days = task.intervalDays ?? 1;
      var c = DateTime(anchor.year, anchor.month, anchor.day, task.hour!, task.minute!);
      while (!c.isAfter(n)) {
        c = c.add(Duration(days: days));
      }
      return c;

    case TaskFrequency.timesPerWeek:
      return null; // unreachable - hasReminder is always false for this frequency
  }
}

DateTime _monthlyOccurrenceOnOrAfter(int year, int month, int day, int hour, int minute) {
  final daysInMonth = DateTime(year, month + 1, 0).day;
  final clampedDay = day > daysInMonth ? daysInMonth : day;
  return DateTime(year, month, clampedDay, hour, minute);
}
