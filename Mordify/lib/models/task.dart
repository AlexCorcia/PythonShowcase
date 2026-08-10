enum TaskFrequency {
  /// Every single day.
  daily,

  /// Monday through Friday only. Resets daily like [daily], but isn't due
  /// (and won't notify) on weekends.
  weekdays,

  /// A specific day of the week, resets once per week.
  weekly,

  /// A specific day of the month, resets once per month.
  monthly,

  /// Every N days starting from an anchor date (e.g. "every 2 days").
  interval,

  /// A target number of completions per week, with no specific day(s)
  /// attached (e.g. "walk the dogs 5 times a week"). Never has a reminder
  /// time - there's no fixed moment to remind about.
  timesPerWeek,
}

/// Which tab/bucket of the app a task belongs in, and which bucket its
/// pending count rolls up into on the persistent status notification.
enum TaskCategory { daily, weekly, monthly }

/// A single checklist step within a task (e.g. "Cleanse" under a "Skin Care"
/// task). Purely a visual checklist - it never has its own reminder and
/// never affects the parent task's own done-state.
class SubTask {
  final String id;
  String title;
  DateTime? lastCompletedAt;

  SubTask({required this.id, required this.title, this.lastCompletedAt});

  Map<String, dynamic> toJson() => {
        'id': id,
        'title': title,
        'lastCompletedAt': lastCompletedAt?.toIso8601String(),
      };

  factory SubTask.fromJson(Map<String, dynamic> json) => SubTask(
        id: json['id'] as String,
        title: json['title'] as String,
        lastCompletedAt: json['lastCompletedAt'] == null
            ? null
            : DateTime.parse(json['lastCompletedAt'] as String),
      );
}

DateTime _startOfWeek(DateTime date) {
  final day = DateTime(date.year, date.month, date.day);
  return day.subtract(Duration(days: day.weekday - 1));
}

/// Shared period-reset math: was [completedAt] within the current period for
/// [frequency] (today for daily/weekdays, this week for weekly/timesPerWeek,
/// this month for monthly, this N-day bucket for interval)? Used both by
/// [Task.isDoneForCurrentPeriod] and [Task.isSubtaskDone], so a subtask
/// resets on exactly the same cadence as its parent task.
bool _isWithinPeriod(
  TaskFrequency frequency,
  DateTime? completedAt, {
  int? intervalDays,
  DateTime? anchorDate,
}) {
  if (completedAt == null) return false;
  final now = DateTime.now();
  switch (frequency) {
    case TaskFrequency.daily:
    case TaskFrequency.weekdays:
      return completedAt.year == now.year &&
          completedAt.month == now.month &&
          completedAt.day == now.day;
    case TaskFrequency.weekly:
    case TaskFrequency.timesPerWeek:
      final startOfWeek = _startOfWeek(now);
      final completedDay =
          DateTime(completedAt.year, completedAt.month, completedAt.day);
      return !completedDay.isBefore(startOfWeek);
    case TaskFrequency.monthly:
      return completedAt.year == now.year && completedAt.month == now.month;
    case TaskFrequency.interval:
      final anchor = anchorDate ?? now;
      final days = intervalDays ?? 1;
      final currentPeriod = DateTime(now.year, now.month, now.day)
              .difference(DateTime(anchor.year, anchor.month, anchor.day))
              .inDays ~/
          days;
      final completedPeriod = DateTime(
                  completedAt.year, completedAt.month, completedAt.day)
              .difference(DateTime(anchor.year, anchor.month, anchor.day))
              .inDays ~/
          days;
      return completedPeriod == currentPeriod;
  }
}

/// Monotonic "business day" index (Mon-Fri only) - weekends collapse onto
/// the preceding Friday's value so a Friday -> Monday completion is exactly
/// one apart, same as any other consecutive weekday.
int _businessDayIndex(DateTime date) {
  final epochMonday = DateTime(2000, 1, 3);
  final day = DateTime(date.year, date.month, date.day);
  final diffDays = day.difference(epochMonday).inDays;
  final weeks = diffDays ~/ 7;
  final remainder = diffDays % 7; // 0=Mon .. 6=Sun
  final clamped = remainder > 4 ? 4 : remainder;
  return weeks * 5 + clamped;
}

/// A monotonically increasing "which period is this" index for [date] under
/// [frequency], such that consecutive periods (e.g. today and tomorrow for
/// daily, this week and next week for weekly) differ by exactly 1. Used to
/// detect an unbroken streak: a completion continues the streak only if its
/// period index is exactly one more than the previous completion's.
int _periodIndexFor(
  TaskFrequency frequency,
  DateTime date, {
  int? intervalDays,
  DateTime? anchorDate,
}) {
  switch (frequency) {
    case TaskFrequency.daily:
      return DateTime(date.year, date.month, date.day)
          .difference(DateTime(2000, 1, 1))
          .inDays;
    case TaskFrequency.weekdays:
      return _businessDayIndex(date);
    case TaskFrequency.weekly:
    case TaskFrequency.timesPerWeek:
      return _startOfWeek(date).difference(DateTime(2000, 1, 3)).inDays ~/ 7;
    case TaskFrequency.monthly:
      return date.year * 12 + date.month;
    case TaskFrequency.interval:
      final anchor = anchorDate ?? date;
      final days = intervalDays ?? 1;
      return DateTime(date.year, date.month, date.day)
              .difference(DateTime(anchor.year, anchor.month, anchor.day))
              .inDays ~/
          days;
  }
}

/// How much rarer a completion of [frequency] is than a daily one, so a
/// task you only get to complete occasionally pays out more per tap than
/// one you complete every day - otherwise a monthly task would earn less
/// over its whole lifetime than a daily task earns in a single week.
/// Roughly tracks "how many days between typical completions".
double _frequencyWeight(
  TaskFrequency frequency, {
  int? intervalDays,
  int? targetCount,
}) {
  switch (frequency) {
    case TaskFrequency.daily:
    case TaskFrequency.weekdays:
      return 1.0;
    case TaskFrequency.weekly:
      return 3.0;
    case TaskFrequency.monthly:
      return 6.0;
    case TaskFrequency.interval:
      return ((intervalDays ?? 1) / 2.0).clamp(1.0, 4.0);
    case TaskFrequency.timesPerWeek:
      // Fewer times/week target => each one is closer to a weekly task.
      return (5.0 / (targetCount ?? 3)).clamp(1.0, 3.0);
  }
}

/// Points awarded for one completion: a flat base, plus a bonus for how
/// long the task has existed (rewards tasks that have stuck around), plus a
/// bonus for the current streak (rewards consistency - a "racha") - the
/// whole thing then scaled by [_frequencyWeight] so rarer tasks pay more.
int _pointsFor({
  required DateTime createdAt,
  required int streak,
  required TaskFrequency frequency,
  int? intervalDays,
  int? targetCount,
}) {
  final ageDays = DateTime.now().difference(createdAt).inDays;
  final agePoints = (ageDays ~/ 7).clamp(0, 20);
  final streakPoints = streak.clamp(0, 30);
  final subtotal = 10 + agePoints + streakPoints;
  final weight =
      _frequencyWeight(frequency, intervalDays: intervalDays, targetCount: targetCount);
  return (subtotal * weight).round();
}

/// A recurring task the user wants to be reminded about and check off.
///
/// [weekday] (1 = Monday .. 7 = Sunday) is only meaningful for [TaskFrequency.weekly].
/// [dayOfMonth] (1..31) is only meaningful for [TaskFrequency.monthly].
/// [intervalDays] and [anchorDate] are only meaningful for [TaskFrequency.interval].
/// [targetCount], [weeklyCompletionCount] and [weeklyPeriodStart] are only
/// meaningful for [TaskFrequency.timesPerWeek].
class Task {
  final String id;
  String title;
  TaskFrequency frequency;

  /// Null means "no reminder" - the task is still tracked and shows up in
  /// lists/summaries, it just never schedules a notification.
  int? hour;
  int? minute;

  int? weekday;
  int? dayOfMonth;
  int? intervalDays;
  DateTime? anchorDate;
  int? targetCount;
  int? weeklyCompletionCount;
  DateTime? weeklyPeriodStart;
  String? folderId;
  DateTime? lastCompletedAt;
  List<SubTask> subtasks;

  /// When this task was created - the longer it's stuck around, the more
  /// points each completion is worth.
  DateTime createdAt;

  /// Consecutive completions with no gap ("racha"). Resets to 1 whenever a
  /// period is missed.
  int currentStreak;

  /// Lifetime completion count - how many times this task has been done,
  /// ever (never resets).
  int totalCompletions;

  /// Points awarded for the most recent completion, kept so unchecking can
  /// cleanly claw back exactly what was given (no completion history log).
  int? lastAwardedPoints;

  Task({
    required this.id,
    required this.title,
    required this.frequency,
    this.hour,
    this.minute,
    this.weekday,
    this.dayOfMonth,
    this.intervalDays,
    this.anchorDate,
    this.targetCount,
    this.weeklyCompletionCount,
    this.weeklyPeriodStart,
    this.folderId,
    this.lastCompletedAt,
    List<SubTask>? subtasks,
    DateTime? createdAt,
    this.currentStreak = 0,
    this.totalCompletions = 0,
    this.lastAwardedPoints,
  })  : subtasks = subtasks ?? [],
        createdAt = createdAt ?? DateTime.now();

  bool get hasReminder => hour != null && minute != null;

  /// A stable base id derived from [id], used to schedule/cancel platform
  /// notifications tied to this task. [variant] distinguishes the (up to 7)
  /// separate weekday alarms a [TaskFrequency.weekdays] task needs.
  int notificationId([int variant = 0]) =>
      ((id.hashCode & 0x00ffffff) * 10 + variant) & 0x7fffffff;

  bool get _isWeekend {
    final weekday = DateTime.now().weekday;
    return weekday == DateTime.saturday || weekday == DateTime.sunday;
  }

  /// Whether this task is actually due today (relevant for [TaskFrequency.weekdays]
  /// tasks, which don't apply on weekends).
  bool get isDueToday => frequency != TaskFrequency.weekdays || !_isWeekend;

  /// Completions so far in the current week, resolving a stale
  /// [weeklyPeriodStart] (from a previous week) to 0.
  int get completionsThisWeek {
    final periodStart = weeklyPeriodStart;
    if (periodStart == null) return 0;
    if (_startOfWeek(DateTime.now()) != _startOfWeek(periodStart)) return 0;
    return weeklyCompletionCount ?? 0;
  }

  /// Registers one completion towards this week's target (for
  /// [TaskFrequency.timesPerWeek] tasks), rolling over a stale period first.
  /// Returns the points awarded.
  int registerWeeklyCompletion() {
    final currentWeekStart = _startOfWeek(DateTime.now());
    if (weeklyPeriodStart == null ||
        _startOfWeek(weeklyPeriodStart!) != currentWeekStart) {
      weeklyCompletionCount = 0;
      weeklyPeriodStart = currentWeekStart;
    }
    weeklyCompletionCount = (weeklyCompletionCount ?? 0) + 1;
    totalCompletions += 1;
    final points = _pointsFor(
      createdAt: createdAt,
      streak: weeklyCompletionCount!,
      frequency: frequency,
      targetCount: targetCount,
    );
    lastAwardedPoints = points;
    return points;
  }

  /// Removes one completion from this week's tally, if any.
  void unregisterWeeklyCompletion() {
    if (completionsThisWeek <= 0) return;
    weeklyCompletionCount = (weeklyCompletionCount ?? 0) - 1;
    if (totalCompletions > 0) totalCompletions -= 1;
    lastAwardedPoints = null;
  }

  /// Marks this (non-timesPerWeek) task done for the current period,
  /// extending the streak if the previous completion was in the immediately
  /// preceding period, or starting a fresh streak otherwise. Returns the
  /// points awarded.
  int completeOnce() {
    final now = DateTime.now();
    final currentPeriod =
        _periodIndexFor(frequency, now, intervalDays: intervalDays, anchorDate: anchorDate);
    final previous = lastCompletedAt;
    final previousPeriod = previous == null
        ? null
        : _periodIndexFor(frequency, previous,
            intervalDays: intervalDays, anchorDate: anchorDate);

    currentStreak = (previousPeriod != null && currentPeriod - previousPeriod == 1)
        ? currentStreak + 1
        : 1;
    totalCompletions += 1;
    lastCompletedAt = now;

    final points = _pointsFor(
      createdAt: createdAt,
      streak: currentStreak,
      frequency: frequency,
      intervalDays: intervalDays,
    );
    lastAwardedPoints = points;
    return points;
  }

  /// Best-effort undo for [completeOnce] - there's no completion history
  /// log, so this just rolls back the counters rather than restoring the
  /// exact prior streak state.
  void undoComplete() {
    if (currentStreak > 0) currentStreak -= 1;
    if (totalCompletions > 0) totalCompletions -= 1;
    lastCompletedAt = null;
    lastAwardedPoints = null;
  }

  bool get isDoneForCurrentPeriod {
    if (frequency == TaskFrequency.timesPerWeek) {
      return completionsThisWeek >= (targetCount ?? 1);
    }
    return _isWithinPeriod(frequency, lastCompletedAt,
        intervalDays: intervalDays, anchorDate: anchorDate);
  }

  /// Whether [subtask] has been checked off within the parent task's current
  /// period - same cadence as [isDoneForCurrentPeriod], but independent of
  /// it: this task's own checkbox never derives from its subtasks.
  bool isSubtaskDone(SubTask subtask) => _isWithinPeriod(
        frequency,
        subtask.lastCompletedAt,
        intervalDays: intervalDays,
        anchorDate: anchorDate,
      );

  /// Which tab/bucket this task rolls up into.
  TaskCategory get category {
    switch (frequency) {
      case TaskFrequency.daily:
      case TaskFrequency.weekdays:
      case TaskFrequency.interval:
        return TaskCategory.daily;
      case TaskFrequency.weekly:
      case TaskFrequency.timesPerWeek:
        return TaskCategory.weekly;
      case TaskFrequency.monthly:
        return TaskCategory.monthly;
    }
  }

  /// Whether this task should count towards "still have things to do today"
  /// nagging/summary totals right now.
  bool get countsAsPendingToday => isDueToday && !isDoneForCurrentPeriod;

  Map<String, dynamic> toJson() => {
        'id': id,
        'title': title,
        'frequency': frequency.name,
        'hour': hour,
        'minute': minute,
        'weekday': weekday,
        'dayOfMonth': dayOfMonth,
        'intervalDays': intervalDays,
        'anchorDate': anchorDate?.toIso8601String(),
        'targetCount': targetCount,
        'weeklyCompletionCount': weeklyCompletionCount,
        'weeklyPeriodStart': weeklyPeriodStart?.toIso8601String(),
        'folderId': folderId,
        'lastCompletedAt': lastCompletedAt?.toIso8601String(),
        'subtasks': subtasks.map((s) => s.toJson()).toList(),
        'createdAt': createdAt.toIso8601String(),
        'currentStreak': currentStreak,
        'totalCompletions': totalCompletions,
        'lastAwardedPoints': lastAwardedPoints,
      };

  factory Task.fromJson(Map<String, dynamic> json) => Task(
        id: json['id'] as String,
        title: json['title'] as String,
        frequency: TaskFrequency.values.byName(json['frequency'] as String),
        hour: json['hour'] as int?,
        minute: json['minute'] as int?,
        weekday: json['weekday'] as int?,
        dayOfMonth: json['dayOfMonth'] as int?,
        intervalDays: json['intervalDays'] as int?,
        anchorDate: json['anchorDate'] == null
            ? null
            : DateTime.parse(json['anchorDate'] as String),
        targetCount: json['targetCount'] as int?,
        weeklyCompletionCount: json['weeklyCompletionCount'] as int?,
        weeklyPeriodStart: json['weeklyPeriodStart'] == null
            ? null
            : DateTime.parse(json['weeklyPeriodStart'] as String),
        folderId: json['folderId'] as String?,
        lastCompletedAt: json['lastCompletedAt'] == null
            ? null
            : DateTime.parse(json['lastCompletedAt'] as String),
        subtasks: (json['subtasks'] as List<dynamic>?)
            ?.map((e) => SubTask.fromJson(e as Map<String, dynamic>))
            .toList(),
        createdAt: json['createdAt'] == null
            ? null
            : DateTime.parse(json['createdAt'] as String),
        currentStreak: json['currentStreak'] as int? ?? 0,
        totalCompletions: json['totalCompletions'] as int? ?? 0,
        lastAwardedPoints: json['lastAwardedPoints'] as int?,
      );
}

const List<String> weekdayNames = [
  'Monday',
  'Tuesday',
  'Wednesday',
  'Thursday',
  'Friday',
  'Saturday',
  'Sunday',
];
