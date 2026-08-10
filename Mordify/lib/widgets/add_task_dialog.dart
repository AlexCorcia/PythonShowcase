import 'package:flutter/material.dart';
import 'package:uuid/uuid.dart';

import '../models/folder.dart';
import '../models/task.dart';
import 'create_folder_dialog.dart';

const _frequencyLabels = {
  TaskFrequency.daily: 'Daily',
  TaskFrequency.weekdays: 'Weekdays (Mon–Fri)',
  TaskFrequency.weekly: 'Weekly',
  TaskFrequency.monthly: 'Monthly',
  TaskFrequency.interval: 'Every N days',
  TaskFrequency.timesPerWeek: 'X times a week',
};

const _createFolderSentinel = '__create_folder__';

enum TaskEditorAction { save, delete }

/// The outcome of showing [AddTaskDialog]: either the (created or edited)
/// task to save, or a request to delete [task] (only possible when editing).
class TaskEditorResult {
  final TaskEditorAction action;
  final Task task;

  const TaskEditorResult.save(this.task) : action = TaskEditorAction.save;
  const TaskEditorResult.delete(this.task) : action = TaskEditorAction.delete;
}

/// Add or edit a task. Pass [existingTask] to edit it in place - its fields
/// are mutated directly and it's the value returned on submit, so the
/// caller doesn't need to distinguish create vs. update by identity.
class AddTaskDialog extends StatefulWidget {
  final TaskFrequency initialFrequency;
  final Task? existingTask;
  final List<Folder> folders;
  final Future<Folder?> Function(String name, int colorValue) onCreateFolder;

  const AddTaskDialog({
    super.key,
    required this.initialFrequency,
    required this.folders,
    required this.onCreateFolder,
    this.existingTask,
  });

  @override
  State<AddTaskDialog> createState() => _AddTaskDialogState();
}

class _AddTaskDialogState extends State<AddTaskDialog> {
  late final _titleController =
      TextEditingController(text: widget.existingTask?.title ?? '');
  late TaskFrequency _frequency =
      widget.existingTask?.frequency ?? widget.initialFrequency;
  late bool _hasReminder = widget.existingTask?.hasReminder ??
      (_frequency != TaskFrequency.timesPerWeek);
  late TimeOfDay _time = widget.existingTask?.hasReminder == true
      ? TimeOfDay(hour: widget.existingTask!.hour!, minute: widget.existingTask!.minute!)
      : const TimeOfDay(hour: 9, minute: 0);
  late int _weekday = widget.existingTask?.weekday ?? DateTime.monday;
  late int _dayOfMonth = widget.existingTask?.dayOfMonth ?? 1;
  late int _intervalDays = widget.existingTask?.intervalDays ?? 2;
  late int _targetCount = widget.existingTask?.targetCount ?? 3;
  late List<Folder> _folders = List.of(widget.folders);
  late String? _folderId = widget.existingTask?.folderId;

  // Deep copy so Cancel is a true no-op, matching how every other field here
  // stays local until _submit().
  late List<SubTask> _subtasks = widget.existingTask?.subtasks
          .map((s) => SubTask(id: s.id, title: s.title, lastCompletedAt: s.lastCompletedAt))
          .toList() ??
      [];
  final _newSubtaskController = TextEditingController();
  final _subtaskControllers = <String, TextEditingController>{};

  bool get _isEditing => widget.existingTask != null;

  @override
  void dispose() {
    _titleController.dispose();
    _newSubtaskController.dispose();
    for (final c in _subtaskControllers.values) {
      c.dispose();
    }
    super.dispose();
  }

  TextEditingController _controllerFor(SubTask s) =>
      _subtaskControllers.putIfAbsent(s.id, () => TextEditingController(text: s.title));

  void _addSubtask() {
    final title = _newSubtaskController.text.trim();
    if (title.isEmpty) return;
    setState(() {
      _subtasks = [..._subtasks, SubTask(id: const Uuid().v4(), title: title)];
      _newSubtaskController.clear();
    });
  }

  void _removeSubtask(int index) {
    setState(() {
      _subtaskControllers.remove(_subtasks[index].id)?.dispose();
      _subtasks = List.of(_subtasks)..removeAt(index);
    });
  }

  void _reorderSubtasks(int oldIndex, int newIndex) {
    setState(() {
      final updated = List<SubTask>.from(_subtasks);
      final moved = updated.removeAt(oldIndex);
      updated.insert(newIndex, moved);
      _subtasks = updated;
    });
  }

  Future<void> _pickTime() async {
    final picked = await showTimePicker(context: context, initialTime: _time);
    if (picked != null) setState(() => _time = picked);
  }

  Future<void> _handleFolderSelection(String? value) async {
    if (value != _createFolderSentinel) {
      setState(() => _folderId = value);
      return;
    }
    final created = await showDialog<Folder>(
      context: context,
      builder: (_) => const CreateFolderDialog(),
    );
    if (created == null) return;
    final saved = await widget.onCreateFolder(created.name, created.colorValue);
    if (saved == null) return;
    setState(() {
      _folders = [..._folders, saved];
      _folderId = saved.id;
    });
  }

  Future<void> _confirmDelete() async {
    final confirmed = await showDialog<bool>(
      context: context,
      builder: (_) => AlertDialog(
        title: const Text('Delete task?'),
        content: const Text("This can't be undone."),
        actions: [
          TextButton(onPressed: () => Navigator.of(context).pop(false), child: const Text('Cancel')),
          FilledButton(onPressed: () => Navigator.of(context).pop(true), child: const Text('Delete')),
        ],
      ),
    );
    if (confirmed != true || !mounted) return;
    Navigator.of(context).pop(TaskEditorResult.delete(widget.existingTask!));
  }

  void _submit() {
    final title = _titleController.text.trim();
    if (title.isEmpty) return;

    final now = DateTime.now();
    final hour = _hasReminder && _frequency != TaskFrequency.timesPerWeek
        ? _time.hour
        : null;
    final minute = _hasReminder && _frequency != TaskFrequency.timesPerWeek
        ? _time.minute
        : null;

    final task = widget.existingTask ?? Task(id: const Uuid().v4(), title: title, frequency: _frequency);
    task
      ..title = title
      ..frequency = _frequency
      ..hour = hour
      ..minute = minute
      ..weekday = _frequency == TaskFrequency.weekly ? _weekday : null
      ..dayOfMonth = _frequency == TaskFrequency.monthly ? _dayOfMonth : null
      ..intervalDays = _frequency == TaskFrequency.interval ? _intervalDays : null
      ..anchorDate = _frequency == TaskFrequency.interval
          ? (task.anchorDate ?? DateTime(now.year, now.month, now.day))
          : null
      ..targetCount = _frequency == TaskFrequency.timesPerWeek ? _targetCount : null
      ..folderId = _folderId
      ..subtasks = _subtasks;

    if (_frequency != TaskFrequency.timesPerWeek) {
      task
        ..weeklyCompletionCount = null
        ..weeklyPeriodStart = null;
    }

    Navigator.of(context).pop(TaskEditorResult.save(task));
  }

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    return AlertDialog(
      title: Text(_isEditing ? 'Edit task' : 'New task'),
      content: SingleChildScrollView(
        child: Column(
          mainAxisSize: MainAxisSize.min,
          crossAxisAlignment: CrossAxisAlignment.stretch,
          children: [
            TextField(
              controller: _titleController,
              autofocus: true,
              decoration: const InputDecoration(labelText: 'Task title'),
            ),
            const SizedBox(height: 16),
            DropdownButtonFormField<TaskFrequency>(
              initialValue: _frequency,
              decoration: const InputDecoration(labelText: 'Frequency'),
              items: [
                for (final frequency in TaskFrequency.values)
                  DropdownMenuItem(
                    value: frequency,
                    child: Text(_frequencyLabels[frequency]!),
                  ),
              ],
              onChanged: (v) => setState(() => _frequency = v ?? _frequency),
            ),
            const SizedBox(height: 16),
            if (_frequency == TaskFrequency.weekly) ...[
              DropdownButtonFormField<int>(
                initialValue: _weekday,
                decoration: const InputDecoration(labelText: 'Day of week'),
                items: [
                  for (var i = 0; i < weekdayNames.length; i++)
                    DropdownMenuItem(value: i + 1, child: Text(weekdayNames[i])),
                ],
                onChanged: (v) => setState(() => _weekday = v ?? _weekday),
              ),
              const SizedBox(height: 16),
            ],
            if (_frequency == TaskFrequency.monthly) ...[
              DropdownButtonFormField<int>(
                initialValue: _dayOfMonth,
                decoration: const InputDecoration(labelText: 'Day of month'),
                items: [
                  for (var d = 1; d <= 31; d++)
                    DropdownMenuItem(value: d, child: Text('$d')),
                ],
                onChanged: (v) => setState(() => _dayOfMonth = v ?? _dayOfMonth),
              ),
              const SizedBox(height: 16),
            ],
            if (_frequency == TaskFrequency.interval) ...[
              DropdownButtonFormField<int>(
                initialValue: _intervalDays,
                decoration: const InputDecoration(labelText: 'Repeat every'),
                items: [
                  for (var d = 2; d <= 14; d++)
                    DropdownMenuItem(value: d, child: Text('$d days')),
                ],
                onChanged: (v) => setState(() => _intervalDays = v ?? _intervalDays),
              ),
              const SizedBox(height: 16),
            ],
            if (_frequency == TaskFrequency.timesPerWeek) ...[
              DropdownButtonFormField<int>(
                initialValue: _targetCount,
                decoration: const InputDecoration(labelText: 'Times per week'),
                items: [
                  for (var c = 1; c <= 7; c++)
                    DropdownMenuItem(value: c, child: Text('$c')),
                ],
                onChanged: (v) => setState(() => _targetCount = v ?? _targetCount),
              ),
              const SizedBox(height: 8),
              const Text(
                'No specific day - just check it off whenever, up to the weekly target.',
                style: TextStyle(fontStyle: FontStyle.italic),
              ),
              const SizedBox(height: 16),
            ],
            if (_frequency == TaskFrequency.weekdays)
              const Padding(
                padding: EdgeInsets.only(bottom: 16),
                child: Text(
                  'Reminds you Monday through Friday only.',
                  style: TextStyle(fontStyle: FontStyle.italic),
                ),
              ),
            DropdownButtonFormField<String?>(
              initialValue: _folderId,
              decoration: const InputDecoration(labelText: 'Folder'),
              items: [
                const DropdownMenuItem(value: null, child: Text('No folder')),
                for (final folder in _folders)
                  DropdownMenuItem(
                    value: folder.id,
                    child: Row(
                      mainAxisSize: MainAxisSize.min,
                      children: [
                        Container(
                          width: 14,
                          height: 14,
                          decoration: BoxDecoration(
                            color: folder.color,
                            shape: BoxShape.circle,
                          ),
                        ),
                        const SizedBox(width: 8),
                        Text(folder.name),
                      ],
                    ),
                  ),
                const DropdownMenuItem(
                  value: _createFolderSentinel,
                  child: Text('+ New folder…'),
                ),
              ],
              onChanged: _handleFolderSelection,
            ),
            if (_frequency != TaskFrequency.timesPerWeek) ...[
              const SizedBox(height: 8),
              SwitchListTile(
                contentPadding: EdgeInsets.zero,
                title: const Text('Set a reminder'),
                value: _hasReminder,
                onChanged: (v) => setState(() => _hasReminder = v),
              ),
              if (_hasReminder)
                ListTile(
                  contentPadding: EdgeInsets.zero,
                  title: const Text('Reminder time'),
                  trailing: Text(_time.format(context)),
                  onTap: _pickTime,
                ),
            ],
            const SizedBox(height: 8),
            Text('Steps', style: theme.textTheme.labelLarge),
            const SizedBox(height: 4),
            if (_subtasks.isNotEmpty)
              ReorderableListView(
                shrinkWrap: true,
                physics: const NeverScrollableScrollPhysics(),
                buildDefaultDragHandles: false,
                onReorderItem: _reorderSubtasks,
                children: [
                  for (var i = 0; i < _subtasks.length; i++)
                    Row(
                      key: ValueKey(_subtasks[i].id),
                      children: [
                        ReorderableDragStartListener(
                          index: i,
                          child: const Padding(
                            padding: EdgeInsets.symmetric(horizontal: 4),
                            child: Icon(Icons.drag_handle),
                          ),
                        ),
                        Expanded(
                          child: TextField(
                            controller: _controllerFor(_subtasks[i]),
                            onChanged: (v) => _subtasks[i].title = v,
                            decoration: const InputDecoration(
                              isDense: true,
                              border: InputBorder.none,
                            ),
                          ),
                        ),
                        IconButton(
                          icon: const Icon(Icons.close, size: 18),
                          onPressed: () => _removeSubtask(i),
                        ),
                      ],
                    ),
                ],
              ),
            Row(
              children: [
                Expanded(
                  child: TextField(
                    controller: _newSubtaskController,
                    decoration: const InputDecoration(hintText: 'Add step'),
                    onSubmitted: (_) => _addSubtask(),
                  ),
                ),
                IconButton(icon: const Icon(Icons.add), onPressed: _addSubtask),
              ],
            ),
          ],
        ),
      ),
      actions: [
        Row(
          children: [
            if (_isEditing)
              IconButton(
                icon: Icon(Icons.delete_outline, color: theme.colorScheme.error),
                tooltip: 'Delete task',
                onPressed: _confirmDelete,
              ),
            const Spacer(),
            TextButton(
              onPressed: () => Navigator.of(context).pop(),
              child: const Text('Cancel'),
            ),
            const SizedBox(width: 8),
            FilledButton(
              onPressed: _submit,
              child: Text(_isEditing ? 'Save' : 'Add'),
            ),
          ],
        ),
      ],
    );
  }
}
