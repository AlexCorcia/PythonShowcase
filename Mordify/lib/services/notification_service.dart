import 'package:flutter_local_notifications/flutter_local_notifications.dart';
import 'package:flutter_timezone/flutter_timezone.dart';
import 'package:timezone/data/latest_all.dart' as tz;
import 'package:timezone/timezone.dart' as tz;

import '../models/task.dart';
import 'next_occurrence.dart';
import 'settings_repository.dart';

class NotificationService {
  NotificationService._();
  static final NotificationService instance = NotificationService._();

  final _plugin = FlutterLocalNotificationsPlugin();
  bool _timezoneReady = false;

  static const _reminderChannelId = 'mordify_reminders';
  static const _reminderChannelName = 'Task reminders';
  static const _reminderChannelDescription = 'Reminders for your Mordify tasks';

  static const _statusChannelId = 'mordify_status';
  static const _statusChannelName = 'Task status';
  static const _statusChannelDescription =
      'Always-visible summary of your Mordify tasks';
  static const _statusNotificationId = 999999;

  static const _nagChannelId = 'mordify_nag';
  static const _nagChannelName = 'Pending task nudges';
  static const _nagChannelDescription =
      'Nudges you when you still have tasks left to do today';
  static const _nagNotificationId = 999998;

  /// [requestPermissions] should only be true when called from the running
  /// app (foreground); background isolates have no Activity to show a
  /// permission prompt against.
  Future<void> init({bool requestPermissions = true}) async {
    if (!_timezoneReady) {
      tz.initializeTimeZones();
      final timeZoneName = await FlutterTimezone.getLocalTimezone();
      tz.setLocalLocation(tz.getLocation(timeZoneName.identifier));
      _timezoneReady = true;
    }

    const androidSettings = AndroidInitializationSettings('@mipmap/ic_launcher');
    await _plugin.initialize(
      settings: const InitializationSettings(android: androidSettings),
    );

    if (requestPermissions) {
      final androidPlugin = _plugin.resolvePlatformSpecificImplementation<
          AndroidFlutterLocalNotificationsPlugin>();
      await androidPlugin?.requestNotificationsPermission();
      await androidPlugin?.requestExactAlarmsPermission();
    }
  }

  static const _reminderDetails = NotificationDetails(
    android: AndroidNotificationDetails(
      _reminderChannelId,
      _reminderChannelName,
      channelDescription: _reminderChannelDescription,
      importance: Importance.high,
      priority: Priority.high,
    ),
  );

  Future<void> scheduleForTask(Task task) async {
    await cancelForTask(task);

    // No reminder configured (either the user opted out, or the task's
    // frequency - like timesPerWeek - has no fixed moment to notify about).
    final hour = task.hour;
    final minute = task.minute;
    if (hour == null || minute == null) return;

    if (task.frequency == TaskFrequency.weekdays) {
      for (var weekday = DateTime.monday;
          weekday <= DateTime.friday;
          weekday++) {
        final scheduledDate = _nextWeekdayOccurrence(weekday, hour, minute);
        await _plugin.zonedSchedule(
          id: task.notificationId(weekday),
          title: 'Mordify reminder',
          body: task.title,
          scheduledDate: scheduledDate,
          notificationDetails: _reminderDetails,
          androidScheduleMode: AndroidScheduleMode.exactAllowWhileIdle,
          matchDateTimeComponents: DateTimeComponents.dayOfWeekAndTime,
        );
      }
      return;
    }

    final DateTimeComponents? matchComponents;
    switch (task.frequency) {
      case TaskFrequency.daily:
        matchComponents = DateTimeComponents.time;
        break;
      case TaskFrequency.weekly:
        matchComponents = DateTimeComponents.dayOfWeekAndTime;
        break;
      case TaskFrequency.monthly:
        matchComponents = DateTimeComponents.dayOfMonthAndTime;
        break;
      case TaskFrequency.interval:
        // No native "every N days" match component; scheduled as a single
        // shot and re-armed for its next occurrence by the periodic
        // background tick (see BackgroundService).
        matchComponents = null;
        break;
      case TaskFrequency.weekdays:
        matchComponents = null; // handled above
        break;
      case TaskFrequency.timesPerWeek:
        return; // never has a reminder - no fixed moment to notify about
    }

    await _plugin.zonedSchedule(
      id: task.notificationId(),
      title: 'Mordify reminder',
      body: task.title,
      scheduledDate: _nextOccurrence(task, hour, minute),
      notificationDetails: _reminderDetails,
      androidScheduleMode: AndroidScheduleMode.exactAllowWhileIdle,
      matchDateTimeComponents: matchComponents,
    );
  }

  Future<void> cancelForTask(Task task) async {
    // Cancel the single-schedule id plus every possible weekday variant
    // (1..7) so this is safe to call regardless of the task's frequency.
    for (var variant = 0; variant <= 7; variant++) {
      await _plugin.cancel(id: task.notificationId(variant));
    }
  }

  /// Re-schedules every task's notification(s). Safe to call repeatedly;
  /// used both at app startup and on every background tick to self-heal
  /// after reboots and to re-arm one-shot [TaskFrequency.interval] alarms.
  Future<void> rescheduleAll(List<Task> tasks) async {
    for (final task in tasks) {
      try {
        await scheduleForTask(task);
      } catch (_) {
        // Don't let one bad task (e.g. revoked exact-alarm permission)
        // block scheduling the rest.
      }
    }
  }

  Future<void> updateStatusNotification(
    List<Task> tasks, {
    required NotificationContentMode mode,
  }) async {
    final daily = tasks
        .where((t) => t.category == TaskCategory.daily)
        .where((t) => t.isDueToday)
        .toList();
    final weekly =
        tasks.where((t) => t.category == TaskCategory.weekly).toList();
    final monthly =
        tasks.where((t) => t.category == TaskCategory.monthly).toList();

    final lines = mode == NotificationContentMode.dueSoon
        ? _dueSoonLines(tasks)
        : _categoryLines(mode, daily, weekly, monthly);

    final details = NotificationDetails(
      android: AndroidNotificationDetails(
        _statusChannelId,
        _statusChannelName,
        channelDescription: _statusChannelDescription,
        importance: Importance.low,
        priority: Priority.low,
        ongoing: true,
        autoCancel: false,
        onlyAlertOnce: true,
        showWhen: false,
        playSound: false,
        enableVibration: false,
        styleInformation: BigTextStyleInformation(lines.join('\n')),
      ),
    );

    await _plugin.show(
      id: _statusNotificationId,
      title: 'Mordify tasks',
      body: lines.join('  •  '),
      notificationDetails: details,
    );
  }

  List<String> _categoryLines(
    NotificationContentMode mode,
    List<Task> daily,
    List<Task> weekly,
    List<Task> monthly,
  ) {
    String lineFor(String label, List<Task> tasks) {
      final done = tasks.where((t) => t.isDoneForCurrentPeriod).length;
      final counts = '$label: $done/${tasks.length}';
      if (mode == NotificationContentMode.countsOnly) return counts;

      final pending = tasks.where((t) => !t.isDoneForCurrentPeriod).toList();
      final names = pending.isEmpty
          ? 'all done'
          : _joinTitles(pending.map((t) => t.title).toList());
      return mode == NotificationContentMode.taskNames
          ? '$label: $names'
          : '$label: $done/${tasks.length} — $names';
    }

    return [
      lineFor('Daily', daily),
      lineFor('Weekly', weekly),
      lineFor('Monthly', monthly),
    ];
  }

  List<String> _dueSoonLines(List<Task> tasks) {
    const window = Duration(hours: 3);
    final now = DateTime.now();

    final upcoming = <(Task, DateTime)>[];
    for (final task in tasks) {
      if (!task.hasReminder || task.isDoneForCurrentPeriod) continue;
      final dueAt = nextReminderOccurrence(task, now: now);
      if (dueAt == null) continue;
      if (dueAt.isBefore(now) || dueAt.difference(now) > window) continue;
      upcoming.add((task, dueAt));
    }
    upcoming.sort((a, b) => a.$2.compareTo(b.$2));

    if (upcoming.isEmpty) return ['Nothing due in the next 3 hours'];
    return [
      for (final (task, dueAt) in upcoming)
        '${dueAt.hour.toString().padLeft(2, '0')}:${dueAt.minute.toString().padLeft(2, '0')} — ${task.title}',
    ];
  }

  String _joinTitles(List<String> titles) {
    const maxShown = 6;
    if (titles.length <= maxShown) return titles.join(', ');
    final shown = titles.take(maxShown).join(', ');
    return '$shown, +${titles.length - maxShown} more';
  }

  Future<void> clearStatusNotification() =>
      _plugin.cancel(id: _statusNotificationId);

  Future<void> showNag(int pendingCount) async {
    const details = NotificationDetails(
      android: AndroidNotificationDetails(
        _nagChannelId,
        _nagChannelName,
        channelDescription: _nagChannelDescription,
        importance: Importance.high,
        priority: Priority.high,
      ),
    );
    final taskWord = pendingCount == 1 ? 'task' : 'tasks';
    await _plugin.show(
      id: _nagNotificationId,
      title: 'Mordify',
      body: 'You still have $pendingCount $taskWord to do today',
      notificationDetails: details,
    );
  }

  tz.TZDateTime _nextWeekdayOccurrence(int targetWeekday, int hour, int minute) {
    final now = tz.TZDateTime.now(tz.local);
    var scheduled = tz.TZDateTime(tz.local, now.year, now.month, now.day, hour, minute);
    while (scheduled.weekday != targetWeekday || !scheduled.isAfter(now)) {
      scheduled = scheduled.add(const Duration(days: 1));
    }
    return scheduled;
  }

  tz.TZDateTime _nextOccurrence(Task task, int hour, int minute) {
    final now = tz.TZDateTime.now(tz.local);
    var scheduled = tz.TZDateTime(
      tz.local,
      now.year,
      now.month,
      now.day,
      hour,
      minute,
    );

    switch (task.frequency) {
      case TaskFrequency.daily:
        if (!scheduled.isAfter(now)) {
          scheduled = scheduled.add(const Duration(days: 1));
        }
        break;
      case TaskFrequency.weekdays:
        // Not used directly; each weekday is scheduled individually.
        break;
      case TaskFrequency.weekly:
        return _nextWeekdayOccurrence(
            task.weekday ?? DateTime.monday, hour, minute);
      case TaskFrequency.monthly:
        final targetDay = task.dayOfMonth ?? 1;
        scheduled = _monthlyOccurrenceOnOrAfter(now, targetDay, hour, minute);
        if (!scheduled.isAfter(now)) {
          scheduled = _monthlyOccurrenceOnOrAfter(
            tz.TZDateTime(tz.local, now.year, now.month + 1, 1),
            targetDay,
            hour,
            minute,
          );
        }
        break;
      case TaskFrequency.interval:
        final anchor = task.anchorDate ?? DateTime.now();
        final intervalDays = task.intervalDays ?? 1;
        var candidate = tz.TZDateTime(
          tz.local,
          anchor.year,
          anchor.month,
          anchor.day,
          hour,
          minute,
        );
        while (!candidate.isAfter(now)) {
          candidate = candidate.add(Duration(days: intervalDays));
        }
        scheduled = candidate;
        break;
      case TaskFrequency.timesPerWeek:
        break; // never called - scheduleForTask returns early for this case
    }
    return scheduled;
  }

  /// Finds the given [day] within the month of [from], clamped to the last
  /// valid day of that month (e.g. day 31 in February becomes the 28th/29th).
  tz.TZDateTime _monthlyOccurrenceOnOrAfter(
    tz.TZDateTime from,
    int day,
    int hour,
    int minute,
  ) {
    final daysInMonth = DateTime(from.year, from.month + 1, 0).day;
    final clampedDay = day > daysInMonth ? daysInMonth : day;
    return tz.TZDateTime(tz.local, from.year, from.month, clampedDay, hour, minute);
  }
}
