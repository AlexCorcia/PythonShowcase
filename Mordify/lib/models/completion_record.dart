/// A single "this task was checked off at this moment" event.
///
/// Snapshots the task's title and folder color at completion time, so the
/// calendar keeps reading correctly even after the task is later renamed,
/// moved to a different folder, or deleted entirely.
class CompletionRecord {
  final String id;
  final String taskId;
  final String taskTitle;
  final int? folderColorValue;
  final DateTime completedAt;
  final int? points;

  CompletionRecord({
    required this.id,
    required this.taskId,
    required this.taskTitle,
    this.folderColorValue,
    required this.completedAt,
    this.points,
  });

  /// Midnight of [completedAt]'s date - used to bucket records by day.
  DateTime get day => DateTime(completedAt.year, completedAt.month, completedAt.day);

  Map<String, dynamic> toJson() => {
        'id': id,
        'taskId': taskId,
        'taskTitle': taskTitle,
        'folderColorValue': folderColorValue,
        'completedAt': completedAt.toIso8601String(),
        'points': points,
      };

  factory CompletionRecord.fromJson(Map<String, dynamic> json) => CompletionRecord(
        id: json['id'] as String,
        taskId: json['taskId'] as String,
        taskTitle: json['taskTitle'] as String,
        folderColorValue: json['folderColorValue'] as int?,
        completedAt: DateTime.parse(json['completedAt'] as String),
        points: json['points'] as int?,
      );
}
