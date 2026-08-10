import 'dart:convert';

import 'package:shared_preferences/shared_preferences.dart';

import '../models/folder.dart';
import '../models/task.dart';

class TaskRepository {
  static const _storageKey = 'mordify.tasks';
  static const _foldersStorageKey = 'mordify.folders';
  static const _seededV1Key = 'mordify.seededDefaultTasksV1';
  static const _seededV2Key = 'mordify.seededDefaultTasksV2';
  static const _backfilledFoldersKey = 'mordify.backfilledFolders';
  static const _seededSkincareSubtasksKey = 'mordify.seededSkincareSubtasksV1';

  Future<List<Task>> loadTasks() async {
    final prefs = await SharedPreferences.getInstance();
    final raw = prefs.getString(_storageKey);
    if (raw == null || raw.isEmpty) return [];
    final list = jsonDecode(raw) as List<dynamic>;
    return list
        .map((e) => Task.fromJson(e as Map<String, dynamic>))
        .toList();
  }

  Future<void> saveTasks(List<Task> tasks) async {
    final prefs = await SharedPreferences.getInstance();
    final raw = jsonEncode(tasks.map((t) => t.toJson()).toList());
    await prefs.setString(_storageKey, raw);
  }

  Future<List<Folder>> loadFolders() async {
    final prefs = await SharedPreferences.getInstance();
    final raw = prefs.getString(_foldersStorageKey);
    if (raw == null || raw.isEmpty) return [];
    final list = jsonDecode(raw) as List<dynamic>;
    return list
        .map((e) => Folder.fromJson(e as Map<String, dynamic>))
        .toList();
  }

  Future<void> saveFolders(List<Folder> folders) async {
    final prefs = await SharedPreferences.getInstance();
    final raw = jsonEncode(folders.map((f) => f.toJson()).toList());
    await prefs.setString(_foldersStorageKey, raw);
  }

  Future<bool> hasSeededDefaults() async {
    final prefs = await SharedPreferences.getInstance();
    return prefs.getBool(_seededV1Key) ?? false;
  }

  Future<void> markDefaultsSeeded() async {
    final prefs = await SharedPreferences.getInstance();
    await prefs.setBool(_seededV1Key, true);
  }

  Future<bool> hasSeededDefaultsV2() async {
    final prefs = await SharedPreferences.getInstance();
    return prefs.getBool(_seededV2Key) ?? false;
  }

  Future<void> markDefaultsSeededV2() async {
    final prefs = await SharedPreferences.getInstance();
    await prefs.setBool(_seededV2Key, true);
  }

  Future<bool> hasBackfilledFolders() async {
    final prefs = await SharedPreferences.getInstance();
    return prefs.getBool(_backfilledFoldersKey) ?? false;
  }

  Future<void> markFoldersBackfilled() async {
    final prefs = await SharedPreferences.getInstance();
    await prefs.setBool(_backfilledFoldersKey, true);
  }

  Future<bool> hasSeededSkincareSubtasks() async {
    final prefs = await SharedPreferences.getInstance();
    return prefs.getBool(_seededSkincareSubtasksKey) ?? false;
  }

  Future<void> markSkincareSubtasksSeeded() async {
    final prefs = await SharedPreferences.getInstance();
    await prefs.setBool(_seededSkincareSubtasksKey, true);
  }
}
