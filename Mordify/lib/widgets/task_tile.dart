import 'package:flutter/material.dart';

import '../models/task.dart';

const _frequencyIcons = {
  TaskFrequency.daily: Icons.wb_sunny_outlined,
  TaskFrequency.weekdays: Icons.work_outline,
  TaskFrequency.weekly: Icons.calendar_view_week,
  TaskFrequency.monthly: Icons.calendar_month_outlined,
  TaskFrequency.interval: Icons.repeat,
  TaskFrequency.timesPerWeek: Icons.tag,
};

class TaskTile extends StatefulWidget {
  final Task task;

  /// Local (per-list) drag index for reordering this task among its
  /// siblings. Null means this tile isn't in a reorderable context.
  final int? dragIndex;

  final ValueChanged<bool?> onToggle;
  final VoidCallback onIncrement;
  final VoidCallback onDecrement;
  final VoidCallback onEdit;
  final VoidCallback onDelete;
  final void Function(SubTask subtask, bool? checked) onToggleSubtask;
  final void Function(int oldIndex, int newIndex) onReorderSubtasks;

  const TaskTile({
    super.key,
    required this.task,
    this.dragIndex,
    required this.onToggle,
    required this.onIncrement,
    required this.onDecrement,
    required this.onEdit,
    required this.onDelete,
    required this.onToggleSubtask,
    required this.onReorderSubtasks,
  });

  @override
  State<TaskTile> createState() => _TaskTileState();
}

class _TaskTileState extends State<TaskTile> {
  bool _expanded = false;

  String get _subtitle {
    final task = widget.task;
    String timeLabel = '';
    if (task.hasReminder) {
      final time = TimeOfDay(hour: task.hour!, minute: task.minute!);
      timeLabel =
          ' at ${time.hour.toString().padLeft(2, '0')}:${time.minute.toString().padLeft(2, '0')}';
    }
    switch (task.frequency) {
      case TaskFrequency.daily:
        return task.hasReminder ? 'Every day$timeLabel' : 'Every day - no reminder';
      case TaskFrequency.weekdays:
        return 'Weekdays (Mon–Fri)$timeLabel';
      case TaskFrequency.weekly:
        final name = weekdayNames[(task.weekday ?? 1) - 1];
        return 'Every $name$timeLabel';
      case TaskFrequency.monthly:
        return 'Day ${task.dayOfMonth} of each month$timeLabel';
      case TaskFrequency.interval:
        return 'Every ${task.intervalDays} days$timeLabel';
      case TaskFrequency.timesPerWeek:
        return '${task.targetCount ?? 1}x a week, no specific day';
    }
  }

  @override
  Widget build(BuildContext context) {
    final task = widget.task;
    final done = task.isDoneForCurrentPeriod;
    final isCounter = task.frequency == TaskFrequency.timesPerWeek;
    final theme = Theme.of(context);
    final mutedColor = theme.colorScheme.onSurfaceVariant;
    final hasSubtasks = task.subtasks.isNotEmpty;

    return Dismissible(
      key: ValueKey(task.id),
      direction: DismissDirection.endToStart,
      background: Container(
        alignment: Alignment.centerRight,
        padding: const EdgeInsets.symmetric(horizontal: 20),
        color: theme.colorScheme.errorContainer,
        child: Icon(Icons.delete, color: theme.colorScheme.onErrorContainer),
      ),
      onDismissed: (_) => widget.onDelete(),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.stretch,
        children: [
          ListTile(
            onTap: widget.onEdit,
            leading: isCounter
                ? Icon(_frequencyIcons[task.frequency], color: mutedColor)
                : Checkbox(value: done, onChanged: widget.onToggle),
            title: Row(
              children: [
                Flexible(
                  child: Text(
                    task.title,
                    style: done
                        ? TextStyle(decoration: TextDecoration.lineThrough, color: mutedColor)
                        : null,
                  ),
                ),
                if (task.currentStreak >= 2) ...[
                  const SizedBox(width: 6),
                  Text('🔥${task.currentStreak}', style: theme.textTheme.bodySmall),
                ],
              ],
            ),
            subtitle: Row(
              children: [
                if (!isCounter) ...[
                  Icon(_frequencyIcons[task.frequency], size: 14, color: mutedColor),
                  const SizedBox(width: 4),
                ],
                Flexible(
                  child: Text(_subtitle,
                      style: theme.textTheme.bodySmall?.copyWith(color: mutedColor)),
                ),
              ],
            ),
            trailing: Row(
              mainAxisSize: MainAxisSize.min,
              children: [
                if (isCounter) ...[
                  IconButton(
                    icon: const Icon(Icons.remove_circle_outline),
                    onPressed: task.completionsThisWeek > 0 ? widget.onDecrement : null,
                  ),
                  Text('${task.completionsThisWeek}/${task.targetCount ?? 1}',
                      style: theme.textTheme.titleSmall),
                  IconButton(
                    icon: const Icon(Icons.add_circle_outline),
                    onPressed: done ? null : widget.onIncrement,
                  ),
                ],
                if (hasSubtasks)
                  IconButton(
                    icon: AnimatedRotation(
                      turns: _expanded ? 0.25 : 0,
                      duration: const Duration(milliseconds: 200),
                      child: const Icon(Icons.chevron_right),
                    ),
                    onPressed: () => setState(() => _expanded = !_expanded),
                  ),
                if (widget.dragIndex != null)
                  ReorderableDragStartListener(
                    index: widget.dragIndex!,
                    child: const Padding(
                      padding: EdgeInsets.symmetric(horizontal: 4),
                      child: Icon(Icons.drag_handle),
                    ),
                  ),
              ],
            ),
          ),
          if (_expanded && hasSubtasks)
            ReorderableListView(
              shrinkWrap: true,
              physics: const NeverScrollableScrollPhysics(),
              buildDefaultDragHandles: false,
              primary: false,
              padding: const EdgeInsets.only(left: 40, right: 8),
              onReorderItem: widget.onReorderSubtasks,
              children: [
                for (var i = 0; i < task.subtasks.length; i++)
                  ListTile(
                    key: ValueKey(task.subtasks[i].id),
                    dense: true,
                    leading: Checkbox(
                      value: task.isSubtaskDone(task.subtasks[i]),
                      onChanged: (checked) =>
                          widget.onToggleSubtask(task.subtasks[i], checked),
                    ),
                    title: Text(
                      task.subtasks[i].title,
                      style: task.isSubtaskDone(task.subtasks[i])
                          ? TextStyle(decoration: TextDecoration.lineThrough, color: mutedColor)
                          : null,
                    ),
                    trailing: ReorderableDragStartListener(
                      index: i,
                      child: const Icon(Icons.drag_handle, size: 20),
                    ),
                  ),
              ],
            ),
        ],
      ),
    );
  }
}
