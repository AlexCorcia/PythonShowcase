import 'dart:convert';
import 'dart:io';

import 'package:file_selector/file_selector.dart';
import 'package:path_provider/path_provider.dart';
import 'package:share_plus/share_plus.dart';

import '../models/completion_record.dart';
import '../models/folder.dart';
import '../models/task.dart';

const _schemaVersion = 1;

/// Everything a restore needs - the parsed, validated contents of a backup
/// file.
class BackupData {
  final List<Task> tasks;
  final List<Folder> folders;
  final List<CompletionRecord> completionLog;
  final String displayName;
  final int totalPoints;

  const BackupData({
    required this.tasks,
    required this.folders,
    required this.completionLog,
    required this.displayName,
    required this.totalPoints,
  });
}

/// Thrown when a picked file isn't a Mordify backup (or is corrupted).
class BackupFormatException implements Exception {
  final String message;
  const BackupFormatException(this.message);

  @override
  String toString() => message;
}

/// Exports/imports a user's tasks, folders, completion history and profile
/// stats as a single JSON file - the stopgap for "don't lose my data" until
/// there's a real backend to sync against. The file is handed off via the
/// OS share sheet (export) and document picker (import) rather than written
/// to a fixed path, so the user decides where it lives (Drive, email, Files,
/// etc.) and no storage permission is needed.
class BackupService {
  Future<void> shareBackup({
    required List<Task> tasks,
    required List<Folder> folders,
    required List<CompletionRecord> completionLog,
    required String displayName,
    required int totalPoints,
  }) async {
    final payload = {
      'schemaVersion': _schemaVersion,
      'exportedAt': DateTime.now().toIso8601String(),
      'tasks': tasks.map((t) => t.toJson()).toList(),
      'folders': folders.map((f) => f.toJson()).toList(),
      'completionLog': completionLog.map((r) => r.toJson()).toList(),
      'profile': {'displayName': displayName, 'totalPoints': totalPoints},
    };

    final dir = await getTemporaryDirectory();
    final stamp = DateTime.now().toIso8601String().replaceAll(RegExp(r'[:.]'), '-');
    final file = File('${dir.path}/mordify-backup-$stamp.json');
    await file.writeAsString(const JsonEncoder.withIndent('  ').convert(payload));

    await SharePlus.instance.share(
      ShareParams(files: [XFile(file.path)], text: 'Mordify backup'),
    );
  }

  /// Opens a document picker for the user to choose a backup file, parses
  /// and validates it. Returns null if the user cancelled the picker.
  Future<BackupData?> pickAndParseBackup() async {
    final file = await openFile(
      acceptedTypeGroups: [
        const XTypeGroup(label: 'Mordify backup', extensions: ['json']),
      ],
    );
    if (file == null) return null;

    final raw = await file.readAsString();
    final Map<String, dynamic> json;
    try {
      json = jsonDecode(raw) as Map<String, dynamic>;
    } on FormatException {
      throw const BackupFormatException('That file is not valid JSON.');
    }

    if (json['tasks'] is! List || json['folders'] is! List) {
      throw const BackupFormatException("That file doesn't look like a Mordify backup.");
    }

    try {
      final tasks = (json['tasks'] as List)
          .map((e) => Task.fromJson(e as Map<String, dynamic>))
          .toList();
      final folders = (json['folders'] as List)
          .map((e) => Folder.fromJson(e as Map<String, dynamic>))
          .toList();
      final completionLog = ((json['completionLog'] as List?) ?? [])
          .map((e) => CompletionRecord.fromJson(e as Map<String, dynamic>))
          .toList();
      final profile = json['profile'] as Map<String, dynamic>?;

      return BackupData(
        tasks: tasks,
        folders: folders,
        completionLog: completionLog,
        displayName: profile?['displayName'] as String? ?? 'ML41',
        totalPoints: profile?['totalPoints'] as int? ?? 0,
      );
    } catch (_) {
      throw const BackupFormatException("That file doesn't look like a Mordify backup.");
    }
  }
}
