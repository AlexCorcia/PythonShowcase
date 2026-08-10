import 'package:flutter/material.dart';

import '../models/folder.dart';
import '../models/task.dart';

class FolderSection extends StatelessWidget {
  final Folder folder;
  final List<Task> tasks;
  final int dragIndex;
  final VoidCallback onToggleExpand;
  final VoidCallback onRename;
  final VoidCallback onChangeColor;
  final VoidCallback onDelete;
  final void Function(int oldIndex, int newIndex) onReorderTask;
  final Widget Function(Task task, int index) taskBuilder;

  const FolderSection({
    super.key,
    required this.folder,
    required this.tasks,
    required this.dragIndex,
    required this.onToggleExpand,
    required this.onRename,
    required this.onChangeColor,
    required this.onDelete,
    required this.onReorderTask,
    required this.taskBuilder,
  });

  @override
  Widget build(BuildContext context) {
    final doneCount = tasks.where((t) => t.isDoneForCurrentPeriod).length;
    final theme = Theme.of(context);

    return Card(
      key: ValueKey('folder-card-${folder.id}'),
      margin: const EdgeInsets.symmetric(horizontal: 12, vertical: 6),
      clipBehavior: Clip.antiAlias,
      elevation: 0,
      shape: RoundedRectangleBorder(
        borderRadius: BorderRadius.circular(16),
        side: BorderSide(color: folder.color.withValues(alpha: 0.35)),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.stretch,
        children: [
          Material(
            color: folder.color.withValues(alpha: 0.14),
            child: InkWell(
              onTap: onToggleExpand,
              child: Padding(
                padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 10),
                child: Row(
                  children: [
                    AnimatedRotation(
                      turns: folder.isExpanded ? 0.25 : 0,
                      duration: const Duration(milliseconds: 200),
                      child: const Icon(Icons.chevron_right),
                    ),
                    const SizedBox(width: 4),
                    Container(
                      width: 12,
                      height: 12,
                      decoration: BoxDecoration(color: folder.color, shape: BoxShape.circle),
                    ),
                    const SizedBox(width: 10),
                    Expanded(
                      child: Text(
                        folder.name,
                        style: theme.textTheme.titleMedium?.copyWith(fontWeight: FontWeight.w600),
                      ),
                    ),
                    Container(
                      padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 3),
                      decoration: BoxDecoration(
                        color: theme.colorScheme.surface,
                        borderRadius: BorderRadius.circular(20),
                      ),
                      child: Text('$doneCount/${tasks.length}', style: theme.textTheme.labelMedium),
                    ),
                    PopupMenuButton<String>(
                      icon: const Icon(Icons.more_vert),
                      onSelected: (value) {
                        switch (value) {
                          case 'rename':
                            onRename();
                            break;
                          case 'color':
                            onChangeColor();
                            break;
                          case 'delete':
                            onDelete();
                            break;
                        }
                      },
                      itemBuilder: (_) => const [
                        PopupMenuItem(value: 'rename', child: Text('Rename')),
                        PopupMenuItem(value: 'color', child: Text('Change color')),
                        PopupMenuItem(value: 'delete', child: Text('Delete folder')),
                      ],
                    ),
                    ReorderableDragStartListener(
                      index: dragIndex,
                      child: const Padding(
                        padding: EdgeInsets.symmetric(horizontal: 4),
                        child: Icon(Icons.drag_handle),
                      ),
                    ),
                  ],
                ),
              ),
            ),
          ),
          AnimatedCrossFade(
            duration: const Duration(milliseconds: 200),
            crossFadeState:
                folder.isExpanded ? CrossFadeState.showFirst : CrossFadeState.showSecond,
            firstChild: ReorderableListView(
              shrinkWrap: true,
              physics: const NeverScrollableScrollPhysics(),
              buildDefaultDragHandles: false,
              primary: false,
              onReorderItem: onReorderTask,
              children: [
                for (var i = 0; i < tasks.length; i++) taskBuilder(tasks[i], i),
              ],
            ),
            secondChild: const SizedBox(width: double.infinity),
          ),
        ],
      ),
    );
  }
}
