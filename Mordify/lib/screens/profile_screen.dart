import 'package:flutter/material.dart';

import '../models/task.dart';
import '../services/profile_repository.dart';

class ProfileScreen extends StatefulWidget {
  final List<Task> tasks;

  const ProfileScreen({super.key, required this.tasks});

  @override
  State<ProfileScreen> createState() => _ProfileScreenState();
}

class _ProfileScreenState extends State<ProfileScreen> {
  final _repository = ProfileRepository();
  bool _loading = true;
  String _displayName = ProfileRepository.defaultDisplayName;
  int _totalPoints = 0;

  @override
  void initState() {
    super.initState();
    _load();
  }

  Future<void> _load() async {
    final name = await _repository.getDisplayName();
    final points = await _repository.getTotalPoints();
    setState(() {
      _displayName = name;
      _totalPoints = points;
      _loading = false;
    });
  }

  Future<void> _renameProfile() async {
    final controller = TextEditingController(text: _displayName);
    final newName = await showDialog<String>(
      context: context,
      builder: (_) => AlertDialog(
        title: const Text('Your name'),
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
    setState(() => _displayName = newName);
    await _repository.setDisplayName(newName);
  }

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    final colorScheme = theme.colorScheme;

    if (_loading) {
      return const Scaffold(body: Center(child: CircularProgressIndicator()));
    }

    final level = levelForPoints(_totalPoints);
    final totalCompletions =
        widget.tasks.fold<int>(0, (sum, t) => sum + t.totalCompletions);
    final bestStreak = widget.tasks.isEmpty
        ? 0
        : widget.tasks.map((t) => t.currentStreak).reduce((a, b) => a > b ? a : b);
    final streakTasks = widget.tasks.where((t) => t.currentStreak >= 2).toList()
      ..sort((a, b) => b.currentStreak.compareTo(a.currentStreak));

    return Scaffold(
      appBar: AppBar(title: const Text('Profile')),
      body: ListView(
        padding: const EdgeInsets.all(20),
        children: [
          Center(
            child: Column(
              children: [
                GestureDetector(
                  onTap: _renameProfile,
                  child: CircleAvatar(
                    radius: 44,
                    backgroundColor: colorScheme.primaryContainer,
                    child: Text(
                      _initials(_displayName),
                      style: theme.textTheme.headlineMedium
                          ?.copyWith(color: colorScheme.onPrimaryContainer, fontWeight: FontWeight.bold),
                    ),
                  ),
                ),
                const SizedBox(height: 12),
                GestureDetector(
                  onTap: _renameProfile,
                  child: Row(
                    mainAxisSize: MainAxisSize.min,
                    children: [
                      Text(_displayName, style: theme.textTheme.headlineSmall?.copyWith(fontWeight: FontWeight.bold)),
                      const SizedBox(width: 6),
                      Icon(Icons.edit, size: 18, color: colorScheme.outline),
                    ],
                  ),
                ),
                const SizedBox(height: 4),
                Text('Level ${level.level}', style: theme.textTheme.titleMedium?.copyWith(color: colorScheme.primary)),
              ],
            ),
          ),
          const SizedBox(height: 24),
          Card(
            child: Padding(
              padding: const EdgeInsets.all(16),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Row(
                    mainAxisAlignment: MainAxisAlignment.spaceBetween,
                    children: [
                      Text('Level ${level.level}', style: theme.textTheme.labelLarge),
                      Text('${level.pointsIntoLevel}/${level.pointsForNextLevel}',
                          style: theme.textTheme.labelLarge),
                    ],
                  ),
                  const SizedBox(height: 8),
                  ClipRRect(
                    borderRadius: BorderRadius.circular(8),
                    child: LinearProgressIndicator(
                      value: level.progress,
                      minHeight: 10,
                      backgroundColor: colorScheme.surfaceContainerHighest,
                    ),
                  ),
                ],
              ),
            ),
          ),
          const SizedBox(height: 16),
          Row(
            children: [
              Expanded(
                child: _StatCard(
                  icon: Icons.stars_rounded,
                  label: 'Points',
                  value: '$_totalPoints',
                  color: colorScheme.primary,
                ),
              ),
              const SizedBox(width: 12),
              Expanded(
                child: _StatCard(
                  icon: Icons.local_fire_department,
                  label: 'Best streak',
                  value: '$bestStreak',
                  color: Colors.deepOrange,
                ),
              ),
            ],
          ),
          const SizedBox(height: 12),
          Row(
            children: [
              Expanded(
                child: _StatCard(
                  icon: Icons.check_circle,
                  label: 'Completions',
                  value: '$totalCompletions',
                  color: Colors.teal,
                ),
              ),
              const SizedBox(width: 12),
              Expanded(
                child: _StatCard(
                  icon: Icons.checklist_rounded,
                  label: 'Tasks tracked',
                  value: '${widget.tasks.length}',
                  color: colorScheme.secondary,
                ),
              ),
            ],
          ),
          if (streakTasks.isNotEmpty) ...[
            const SizedBox(height: 24),
            Text('Active streaks', style: theme.textTheme.titleMedium),
            const SizedBox(height: 8),
            Card(
              child: Column(
                children: [
                  for (final task in streakTasks)
                    ListTile(
                      leading: const Text('🔥', style: TextStyle(fontSize: 20)),
                      title: Text(task.title),
                      trailing: Text(
                        '${task.currentStreak}',
                        style: theme.textTheme.titleMedium?.copyWith(color: Colors.deepOrange),
                      ),
                    ),
                ],
              ),
            ),
          ],
        ],
      ),
    );
  }

  String _initials(String name) {
    final trimmed = name.trim();
    if (trimmed.isEmpty) return '?';
    final parts = trimmed.split(RegExp(r'\s+'));
    if (parts.length == 1) {
      return parts.first.substring(0, parts.first.length.clamp(0, 2)).toUpperCase();
    }
    return (parts.first[0] + parts.last[0]).toUpperCase();
  }
}

class _StatCard extends StatelessWidget {
  final IconData icon;
  final String label;
  final String value;
  final Color color;

  const _StatCard({
    required this.icon,
    required this.label,
    required this.value,
    required this.color,
  });

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    return Card(
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Icon(icon, color: color),
            const SizedBox(height: 8),
            Text(value, style: theme.textTheme.headlineSmall?.copyWith(fontWeight: FontWeight.bold)),
            Text(label, style: theme.textTheme.bodySmall?.copyWith(color: theme.colorScheme.outline)),
          ],
        ),
      ),
    );
  }
}
