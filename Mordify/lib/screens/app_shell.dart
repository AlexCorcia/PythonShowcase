import 'package:flutter/material.dart';
import 'package:uuid/uuid.dart';

import '../models/completion_record.dart';
import '../models/folder.dart';
import '../models/task.dart';
import '../services/achievements.dart';
import '../services/backup_service.dart';
import '../services/default_tasks.dart';
import '../services/notification_service.dart';
import '../services/profile_repository.dart';
import '../services/settings_repository.dart';
import '../services/task_repository.dart';
import '../services/theme_controller.dart';
import '../widgets/add_task_dialog.dart';
import '../widgets/folder_section.dart';
import '../widgets/points_burst_overlay.dart';
import '../widgets/settings_sheet.dart';
import '../widgets/task_tile.dart';
import 'achievements_screen.dart';
import 'calendar_screen.dart';
import 'home_dashboard_screen.dart';
import 'profile_screen.dart';

/// The frequency the "+" button creates by default in each Tasks segment.
const _defaultFrequencyByCategory = {
  TaskCategory.daily: TaskFrequency.daily,
  TaskCategory.weekly: TaskFrequency.weekly,
  TaskCategory.monthly: TaskFrequency.monthly,
};

const _categoryIcons = {
  TaskCategory.daily: Icons.wb_sunny_outlined,
  TaskCategory.weekly: Icons.calendar_view_week,
  TaskCategory.monthly: Icons.calendar_month_outlined,
};

/// Owns all app state (tasks, folders, completion log, settings, profile)
/// and hosts the bottom-nav shell (Home / Tasks / Calendar / Profile) - the
/// four destinations from design/handoff's mockups. Each destination is a
/// plain content widget (no nested Scaffold/AppBar); this shell provides the
/// single outer Scaffold, bottom nav, and (on the Tasks tab) the FAB.
class AppShell extends StatefulWidget {
  final ThemeController themeController;

  const AppShell({super.key, required this.themeController});

  @override
  State<AppShell> createState() => _AppShellState();
}

class _AppShellState extends State<AppShell> {
  final _repository = TaskRepository();
  final _settings = SettingsRepository();
  final _profile = ProfileRepository();
  int _selectedIndex = 0;
  TaskCategory _selectedCategory = TaskCategory.daily;
  List<Task> _tasks = [];
  List<Folder> _folders = [];
  List<CompletionRecord> _completionLog = [];
  bool _loading = true;
  bool _showStatusNotification = true;
  NotificationContentMode _notificationContentMode = NotificationContentMode.countsOnly;

  @override
  void initState() {
    super.initState();
    _load();
  }

  List<Folder> _mergeFolders(List<Folder> existing, List<Folder> additions) {
    final byId = {for (final f in existing) f.id: f};
    for (final f in additions) {
      byId.putIfAbsent(f.id, () => f);
    }
    return byId.values.toList();
  }

  Future<void> _load() async {
    var tasks = await _repository.loadTasks();
    var folders = await _repository.loadFolders();

    if (!await _repository.hasSeededDefaults()) {
      final defaults = buildDefaultTasks();
      tasks = [...tasks, ...defaults];
      folders = _mergeFolders(folders, buildDefaultFolders());
      await _repository.saveTasks(tasks);
      await _repository.saveFolders(folders);
      await _repository.markDefaultsSeeded();
      for (final task in defaults) {
        await _scheduleTaskSafely(task);
      }
    }

    if (!await _repository.hasSeededDefaultsV2()) {
      final defaultsV2 = buildDefaultTasksV2();
      tasks = [...tasks, ...defaultsV2];
      folders = _mergeFolders(folders, buildDefaultFolders());
      await _repository.saveTasks(tasks);
      await _repository.saveFolders(folders);
      await _repository.markDefaultsSeededV2();
      for (final task in defaultsV2) {
        await _scheduleTaskSafely(task);
      }
    }

    // Tasks seeded before folders existed won't have a folderId - backfill
    // the original starter tasks by title so they pick up their color too.
    // Runs independently of the seed waves above since it may need to catch
    // up on a device that already seeded v1/v2 before this migration existed.
    if (!await _repository.hasBackfilledFolders()) {
      folders = _mergeFolders(folders, buildDefaultFolders());
      const backfill = {
        'Skin Care - Morning': skincareFolderId,
        'Skin Care - Night': skincareFolderId,
        'Exfoliate': skincareFolderId,
        'Fichaje': workFolderId,
      };
      for (final task in tasks) {
        if (task.folderId == null && backfill.containsKey(task.title)) {
          task.folderId = backfill[task.title];
        }
      }
      await _repository.saveTasks(tasks);
      await _repository.saveFolders(folders);
      await _repository.markFoldersBackfilled();
    }

    // Same idea, for the skincare routine steps: gives the already-installed
    // Skin Care tasks their checklist steps even though buildDefaultTasks()
    // (used only for brand-new installs) already includes them.
    if (!await _repository.hasSeededSkincareSubtasks()) {
      const stepsByTitle = {
        'Skin Care - Morning': ['Clean', 'Niacinamide', 'Hydrate', 'Sun Screen'],
        'Skin Care - Night': ['Clean', 'Hydrate'],
      };
      for (final task in tasks) {
        final steps = stepsByTitle[task.title];
        if (steps != null && task.subtasks.isEmpty) {
          task.subtasks = [
            for (final step in steps) SubTask(id: const Uuid().v4(), title: step),
          ];
        }
      }
      await _repository.saveTasks(tasks);
      await _repository.markSkincareSubtasksSeeded();
    }

    final completionLog = await _repository.loadCompletionLog();
    final showStatus = await _settings.getShowStatusNotification();
    final contentMode = await _settings.getNotificationContentMode();

    setState(() {
      _tasks = tasks;
      _folders = folders;
      _completionLog = completionLog;
      _showStatusNotification = showStatus;
      _notificationContentMode = contentMode;
      _loading = false;
    });
    await _refreshStatusNotification();
  }

  Future<void> _persist() => _repository.saveTasks(_tasks);
  Future<void> _persistFolders() => _repository.saveFolders(_folders);
  Future<void> _persistCompletionLog() => _repository.saveCompletionLog(_completionLog);

  Folder? _folderForTask(Task task) {
    if (task.folderId == null) return null;
    for (final folder in _folders) {
      if (folder.id == task.folderId) return folder;
    }
    return null;
  }

  void _logCompletion(Task task, DateTime completedAt) {
    _completionLog.add(CompletionRecord(
      id: const Uuid().v4(),
      taskId: task.id,
      taskTitle: task.title,
      folderColorValue: _folderForTask(task)?.colorValue,
      completedAt: completedAt,
      points: task.lastAwardedPoints,
    ));
  }

  /// Undoes exactly one completion of [task] from the log - the most recent
  /// one, matching how [Task.undoComplete]/[Task.unregisterWeeklyCompletion]
  /// only ever claw back a single completion at a time.
  void _unlogLastCompletion(String taskId) {
    for (var i = _completionLog.length - 1; i >= 0; i--) {
      if (_completionLog[i].taskId == taskId) {
        _completionLog.removeAt(i);
        return;
      }
    }
  }

  // Notification-plugin calls hit a real platform channel, which can fail
  // (revoked permission, no handler registered e.g. in tests) - never let
  // that block the task list itself from loading/updating.
  Future<void> _scheduleTaskSafely(Task task) async {
    try {
      await NotificationService.instance.scheduleForTask(task);
    } catch (_) {}
  }

  Future<void> _refreshStatusNotification() async {
    try {
      if (_showStatusNotification) {
        await NotificationService.instance
            .updateStatusNotification(_tasks, mode: _notificationContentMode);
      } else {
        await NotificationService.instance.clearStatusNotification();
      }
    } catch (_) {}
  }

  Future<void> _toggleStatusNotification(bool value) async {
    setState(() => _showStatusNotification = value);
    await _settings.setShowStatusNotification(value);
    await _refreshStatusNotification();
  }

  Future<void> _setNotificationContentMode(NotificationContentMode mode) async {
    setState(() => _notificationContentMode = mode);
    await _settings.setNotificationContentMode(mode);
    await _refreshStatusNotification();
  }

  Future<void> _openSettings() {
    return showModalBottomSheet(
      context: context,
      isScrollControlled: true,
      builder: (_) => SettingsSheet(
        showStatusNotification: _showStatusNotification,
        onShowStatusNotificationChanged: _toggleStatusNotification,
        notificationContentMode: _notificationContentMode,
        onNotificationContentModeChanged: _setNotificationContentMode,
        themeController: widget.themeController,
        onExportBackup: _exportBackup,
        onImportBackup: _importBackup,
      ),
    );
  }

  Future<void> _exportBackup() async {
    try {
      await BackupService().shareBackup(
        tasks: _tasks,
        folders: _folders,
        completionLog: _completionLog,
        displayName: await _profile.getDisplayName(),
        totalPoints: await _profile.getTotalPoints(),
      );
    } catch (e) {
      if (!mounted) return;
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text('Could not export backup: $e')),
      );
    }
  }

  Future<void> _importBackup() async {
    BackupData? picked;
    try {
      picked = await BackupService().pickAndParseBackup();
    } catch (e) {
      if (!mounted) return;
      ScaffoldMessenger.of(context).showSnackBar(SnackBar(content: Text('$e')));
      return;
    }
    if (picked == null || !mounted) return;
    final data = picked;

    final confirmed = await showDialog<bool>(
      context: context,
      builder: (_) => AlertDialog(
        title: const Text('Replace all data?'),
        content: Text(
          'This backup has ${data.tasks.length} tasks and ${data.totalPoints} points. '
          "It will replace everything currently on this device and can't be undone.",
        ),
        actions: [
          TextButton(
            onPressed: () => Navigator.of(context).pop(false),
            child: const Text('Cancel'),
          ),
          FilledButton(
            onPressed: () => Navigator.of(context).pop(true),
            child: const Text('Replace'),
          ),
        ],
      ),
    );
    if (confirmed != true || !mounted) return;

    for (final task in _tasks) {
      try {
        await NotificationService.instance.cancelForTask(task);
      } catch (_) {}
    }

    await _repository.saveTasks(data.tasks);
    await _repository.saveFolders(data.folders);
    await _repository.saveCompletionLog(data.completionLog);
    await _profile.setDisplayName(data.displayName);
    await _profile.setTotalPoints(data.totalPoints);
    // The backup already reflects a used state - skip the starter-task seed
    // waves so they don't layer defaults on top of the restored data.
    await _repository.markDefaultsSeeded();
    await _repository.markDefaultsSeededV2();
    await _repository.markFoldersBackfilled();
    await _repository.markSkincareSubtasksSeeded();

    await _load();
    for (final task in data.tasks) {
      await _scheduleTaskSafely(task);
    }
    if (!mounted) return;
    Navigator.of(context).pop();
    ScaffoldMessenger.of(context).showSnackBar(
      const SnackBar(content: Text('Backup restored')),
    );
  }

  Future<void> _openAchievements() async {
    final points = await _profile.getTotalPoints();
    if (!mounted) return;
    Navigator.of(context).push(MaterialPageRoute(
      builder: (_) => AchievementsScreen(
        achievementContext: AchievementContext(
          tasks: _tasks,
          folders: _folders,
          completionLog: _completionLog,
          totalPoints: points,
        ),
      ),
    ));
  }

  Future<Folder?> _createFolder(String name, int colorValue) async {
    final folder = Folder(id: const Uuid().v4(), name: name, colorValue: colorValue);
    setState(() => _folders = [..._folders, folder]);
    await _persistFolders();
    return folder;
  }

  Future<void> _toggleFolderExpanded(Folder folder) async {
    setState(() => folder.isExpanded = !folder.isExpanded);
    await _persistFolders();
  }

  Future<void> _renameFolder(Folder folder) async {
    final controller = TextEditingController(text: folder.name);
    final newName = await showDialog<String>(
      context: context,
      builder: (_) => AlertDialog(
        title: const Text('Rename folder'),
        content: TextField(controller: controller, autofocus: true),
        actions: [
          TextButton(onPressed: () => Navigator.of(context).pop(), child: const Text('Cancel')),
          FilledButton(
            onPressed: () => Navigator.of(context).pop(controller.text.trim()),
            child: const Text('Save'),
          ),
        ],
      ),
    );
    if (newName == null || newName.isEmpty) return;
    setState(() => folder.name = newName);
    await _persistFolders();
  }

  Future<void> _changeFolderColor(Folder folder) async {
    final newColor = await showDialog<Color>(
      context: context,
      builder: (_) => AlertDialog(
        title: const Text('Folder color'),
        content: Wrap(
          spacing: 10,
          runSpacing: 10,
          children: [
            for (final color in folderColorPalette)
              GestureDetector(
                onTap: () => Navigator.of(context).pop(color),
                child: Container(
                  width: 32,
                  height: 32,
                  decoration: BoxDecoration(
                    color: color,
                    shape: BoxShape.circle,
                    border: folder.colorValue == color.toARGB32()
                        ? Border.all(color: Theme.of(context).colorScheme.onSurface, width: 3)
                        : null,
                  ),
                ),
              ),
          ],
        ),
        actions: [
          TextButton(onPressed: () => Navigator.of(context).pop(), child: const Text('Close')),
        ],
      ),
    );
    if (newColor == null) return;
    setState(() => folder.colorValue = newColor.toARGB32());
    await _persistFolders();
  }

  Future<void> _deleteFolder(Folder folder) async {
    final confirmed = await showDialog<bool>(
      context: context,
      builder: (_) => AlertDialog(
        title: const Text('Delete folder?'),
        content: Text('Tasks in "${folder.name}" will be kept, just unfiled.'),
        actions: [
          TextButton(onPressed: () => Navigator.of(context).pop(false), child: const Text('Cancel')),
          FilledButton(onPressed: () => Navigator.of(context).pop(true), child: const Text('Delete')),
        ],
      ),
    );
    if (confirmed != true) return;
    setState(() {
      for (final task in _tasks) {
        if (task.folderId == folder.id) task.folderId = null;
      }
      _folders.removeWhere((f) => f.id == folder.id);
    });
    await _persist();
    await _persistFolders();
  }

  Future<void> _reorderFolders(List<Folder> scoped, int oldIndex, int newIndex) async {
    final updatedScoped = List<Folder>.from(scoped);
    final moved = updatedScoped.removeAt(oldIndex);
    updatedScoped.insert(newIndex, moved);

    final scopedIds = scoped.map((f) => f.id).toSet();
    final newMaster = <Folder>[];
    var pointer = 0;
    for (final f in _folders) {
      if (scopedIds.contains(f.id)) {
        newMaster.add(updatedScoped[pointer]);
        pointer++;
      } else {
        newMaster.add(f);
      }
    }
    setState(() => _folders = newMaster);
    await _persistFolders();
  }

  Future<void> _reorderTasks(List<Task> scoped, int oldIndex, int newIndex) async {
    final updatedScoped = List<Task>.from(scoped);
    final moved = updatedScoped.removeAt(oldIndex);
    updatedScoped.insert(newIndex, moved);

    final scopedIds = scoped.map((t) => t.id).toSet();
    final newMaster = <Task>[];
    var pointer = 0;
    for (final t in _tasks) {
      if (scopedIds.contains(t.id)) {
        newMaster.add(updatedScoped[pointer]);
        pointer++;
      } else {
        newMaster.add(t);
      }
    }
    setState(() => _tasks = newMaster);
    await _persist();
  }

  Future<void> _addTask(TaskFrequency frequency) async {
    final result = await showDialog<TaskEditorResult>(
      context: context,
      builder: (_) => AddTaskDialog(
        initialFrequency: frequency,
        folders: _folders,
        onCreateFolder: _createFolder,
      ),
    );
    if (result == null || result.action != TaskEditorAction.save) return;

    final task = result.task;
    setState(() => _tasks.add(task));
    await _persist();
    await _scheduleTaskSafely(task);
    await _refreshStatusNotification();
  }

  Future<void> _editTask(Task task) async {
    final result = await showDialog<TaskEditorResult>(
      context: context,
      builder: (_) => AddTaskDialog(
        initialFrequency: task.frequency,
        existingTask: task,
        folders: _folders,
        onCreateFolder: _createFolder,
      ),
    );
    if (result == null) return;

    switch (result.action) {
      case TaskEditorAction.save:
        setState(() {});
        await _persist();
        await _scheduleTaskSafely(task);
        await _refreshStatusNotification();
        break;
      case TaskEditorAction.delete:
        await _deleteTask(task);
        break;
    }
  }

  Future<void> _toggleTask(Task task, bool? checked) async {
    if (checked == true) {
      final points = task.completeOnce();
      setState(() => _logCompletion(task, task.lastCompletedAt!));
      await _profile.addPoints(points);
      _showCompletionCelebration(points, streak: task.currentStreak, taskTitle: task.title);
    } else {
      final lost = task.lastAwardedPoints ?? 0;
      setState(() {
        task.undoComplete();
        _unlogLastCompletion(task.id);
      });
      if (lost > 0) await _profile.addPoints(-lost);
    }
    await _persist();
    await _persistCompletionLog();
    await _refreshStatusNotification();
  }

  Future<void> _incrementTask(Task task) async {
    final points = task.registerWeeklyCompletion();
    setState(() => _logCompletion(task, DateTime.now()));
    await _profile.addPoints(points);
    _showCompletionCelebration(points, streak: task.completionsThisWeek, taskTitle: task.title);
    await _persist();
    await _persistCompletionLog();
    await _refreshStatusNotification();
  }

  Future<void> _decrementTask(Task task) async {
    final lost = task.lastAwardedPoints ?? 0;
    setState(() {
      task.unregisterWeeklyCompletion();
      _unlogLastCompletion(task.id);
    });
    if (lost > 0) await _profile.addPoints(-lost);
    await _persist();
    await _persistCompletionLog();
    await _refreshStatusNotification();
  }

  void _showCompletionCelebration(int points, {required int streak, required String taskTitle}) {
    showCompletionCelebration(context, points: points, streak: streak, taskTitle: taskTitle);
  }

  Future<void> _deleteTask(Task task) async {
    setState(() => _tasks.remove(task));
    await _persist();
    try {
      await NotificationService.instance.cancelForTask(task);
    } catch (_) {}
    await _refreshStatusNotification();
  }

  Future<void> _toggleSubtask(Task task, SubTask subtask, bool? checked) async {
    setState(() => subtask.lastCompletedAt = checked == true ? DateTime.now() : null);
    await _persist();
  }

  Future<void> _reorderSubtasks(Task task, int oldIndex, int newIndex) async {
    setState(() {
      final item = task.subtasks.removeAt(oldIndex);
      task.subtasks.insert(newIndex, item);
    });
    await _persist();
  }

  Widget _buildTaskTile(Task task, [int? index]) {
    return TaskTile(
      key: ValueKey(task.id),
      task: task,
      dragIndex: index,
      onToggle: (checked) => _toggleTask(task, checked),
      onIncrement: () => _incrementTask(task),
      onDecrement: () => _decrementTask(task),
      onEdit: () => _editTask(task),
      onDelete: () => _deleteTask(task),
      onToggleSubtask: (subtask, checked) => _toggleSubtask(task, subtask, checked),
      onReorderSubtasks: (oldIndex, newIndex) => _reorderSubtasks(task, oldIndex, newIndex),
    );
  }

  Widget _buildEmptyState(TaskCategory category) {
    final theme = Theme.of(context);
    return Center(
      child: Column(
        mainAxisSize: MainAxisSize.min,
        children: [
          Icon(_categoryIcons[category], size: 56, color: theme.colorScheme.outlineVariant),
          const SizedBox(height: 12),
          Text('No tasks yet', style: theme.textTheme.titleMedium),
          const SizedBox(height: 4),
          Text(
            'Tap + to add one',
            style: theme.textTheme.bodyMedium?.copyWith(color: theme.colorScheme.outline),
          ),
        ],
      ),
    );
  }

  Widget _buildList(TaskCategory category) {
    final tasksInCategory = _tasks.where((t) => t.category == category).toList();

    if (tasksInCategory.isEmpty) {
      return _buildEmptyState(category);
    }

    final foldersWithTasks =
        _folders.where((f) => tasksInCategory.any((t) => t.folderId == f.id)).toList();
    final unfiled = tasksInCategory.where((t) => t.folderId == null).toList();

    return CustomScrollView(
      slivers: [
        const SliverPadding(padding: EdgeInsets.only(top: 4)),
        if (foldersWithTasks.isNotEmpty)
          SliverReorderableList(
            itemCount: foldersWithTasks.length,
            onReorderItem: (oldIndex, newIndex) =>
                _reorderFolders(foldersWithTasks, oldIndex, newIndex),
            itemBuilder: (context, index) {
              final folder = foldersWithTasks[index];
              final tasksForFolder =
                  tasksInCategory.where((t) => t.folderId == folder.id).toList();
              return FolderSection(
                key: ValueKey(folder.id),
                folder: folder,
                tasks: tasksForFolder,
                dragIndex: index,
                onToggleExpand: () => _toggleFolderExpanded(folder),
                onRename: () => _renameFolder(folder),
                onChangeColor: () => _changeFolderColor(folder),
                onDelete: () => _deleteFolder(folder),
                onReorderTask: (oldIndex, newIndex) =>
                    _reorderTasks(tasksForFolder, oldIndex, newIndex),
                taskBuilder: (task, index) => _buildTaskTile(task, index),
              );
            },
          ),
        if (unfiled.isNotEmpty) ...[
          SliverPadding(
            padding: const EdgeInsets.fromLTRB(24, 16, 16, 4),
            sliver: SliverToBoxAdapter(
              child: Text(
                'NO FOLDER',
                style: Theme.of(context).textTheme.labelMedium?.copyWith(
                      color: Theme.of(context).colorScheme.outline,
                      letterSpacing: 1.1,
                    ),
              ),
            ),
          ),
          SliverReorderableList(
            itemCount: unfiled.length,
            onReorderItem: (oldIndex, newIndex) =>
                _reorderTasks(unfiled, oldIndex, newIndex),
            itemBuilder: (context, index) => _buildTaskTile(unfiled[index], index),
          ),
        ],
        const SliverPadding(padding: EdgeInsets.only(bottom: 88)),
      ],
    );
  }

  Widget _buildTasksTab() {
    final theme = Theme.of(context);
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Padding(
          padding: const EdgeInsets.fromLTRB(20, 16, 20, 14),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Text('Tasks', style: theme.textTheme.headlineSmall?.copyWith(fontWeight: FontWeight.w500)),
              const SizedBox(height: 14),
              SegmentedButton<TaskCategory>(
                segments: const [
                  ButtonSegment(
                    value: TaskCategory.daily,
                    label: Text('Daily', softWrap: false, overflow: TextOverflow.visible),
                  ),
                  ButtonSegment(
                    value: TaskCategory.weekly,
                    label: Text('Weekly', softWrap: false, overflow: TextOverflow.visible),
                  ),
                  ButtonSegment(
                    value: TaskCategory.monthly,
                    label: Text('Monthly', softWrap: false, overflow: TextOverflow.visible),
                  ),
                ],
                selected: {_selectedCategory},
                showSelectedIcon: false,
                style: const ButtonStyle(
                  padding: WidgetStatePropertyAll(EdgeInsets.symmetric(horizontal: 14, vertical: 10)),
                  visualDensity: VisualDensity.compact,
                ),
                onSelectionChanged: (selected) =>
                    setState(() => _selectedCategory = selected.first),
              ),
            ],
          ),
        ),
        Expanded(child: _buildList(_selectedCategory)),
      ],
    );
  }

  @override
  Widget build(BuildContext context) {
    if (_loading) {
      return const Scaffold(body: Center(child: CircularProgressIndicator()));
    }
    return Scaffold(
      body: SafeArea(
        child: IndexedStack(
          index: _selectedIndex,
          children: [
            HomeDashboardScreen(tasks: _tasks, folders: _folders, completionLog: _completionLog),
            _buildTasksTab(),
            CalendarScreen(completionLog: _completionLog),
            ProfileScreen(
              tasks: _tasks,
              onOpenSettings: _openSettings,
              onViewAchievements: _openAchievements,
            ),
          ],
        ),
      ),
      bottomNavigationBar: NavigationBar(
        selectedIndex: _selectedIndex,
        onDestinationSelected: (index) => setState(() => _selectedIndex = index),
        destinations: const [
          NavigationDestination(
            icon: Icon(Icons.auto_awesome_outlined),
            selectedIcon: Icon(Icons.auto_awesome),
            label: 'Home',
          ),
          NavigationDestination(
            icon: Icon(Icons.layers_outlined),
            selectedIcon: Icon(Icons.layers),
            label: 'Tasks',
          ),
          NavigationDestination(
            icon: Icon(Icons.calendar_today_outlined),
            selectedIcon: Icon(Icons.calendar_today),
            label: 'Calendar',
          ),
          NavigationDestination(
            icon: Icon(Icons.person_outline),
            selectedIcon: Icon(Icons.person),
            label: 'Profile',
          ),
        ],
      ),
      floatingActionButton: _selectedIndex == 1
          ? FloatingActionButton.extended(
              onPressed: () => _addTask(_defaultFrequencyByCategory[_selectedCategory]!),
              icon: const Icon(Icons.add),
              label: const Text('Add task'),
            )
          : null,
    );
  }
}
