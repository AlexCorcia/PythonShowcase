import 'package:flutter/material.dart';

import '../models/task.dart';
import '../services/profile_repository.dart';
import '../theme/app_theme.dart' show mordifyAmber, mordifyAmberDim;

class ProfileScreen extends StatefulWidget {
  final List<Task> tasks;
  final VoidCallback onOpenSettings;
  final VoidCallback onViewAchievements;

  const ProfileScreen({
    super.key,
    required this.tasks,
    required this.onOpenSettings,
    required this.onViewAchievements,
  });

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

  @override
  void didUpdateWidget(covariant ProfileScreen oldWidget) {
    super.didUpdateWidget(oldWidget);
    // The bottom-nav shell keeps this screen alive via IndexedStack rather
    // than recreating it per visit, so points/level (loaded once into local
    // state) would otherwise go stale after every completion. widget.tasks
    // gets a new reference on every app-shell rebuild, which is a reliable
    // signal to refresh.
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
      return const Center(child: CircularProgressIndicator());
    }

    final level = levelForPoints(_totalPoints);
    final totalCompletions =
        widget.tasks.fold<int>(0, (sum, t) => sum + t.totalCompletions);
    final bestStreak = widget.tasks.isEmpty
        ? 0
        : widget.tasks.map((t) => t.currentStreak).reduce((a, b) => a > b ? a : b);
    final streakTasks = widget.tasks.where((t) => t.currentStreak >= 2).toList()
      ..sort((a, b) => b.currentStreak.compareTo(a.currentStreak));

    return ListView(
        padding: const EdgeInsets.fromLTRB(20, 16, 20, 24),
        children: [
          Row(
            mainAxisAlignment: MainAxisAlignment.spaceBetween,
            children: [
              Text('Profile', style: theme.textTheme.headlineSmall?.copyWith(fontWeight: FontWeight.w500)),
              Row(
                children: [
                  IconButton(
                    tooltip: 'Settings',
                    icon: const Icon(Icons.settings_outlined),
                    onPressed: widget.onOpenSettings,
                  ),
                ],
              ),
            ],
          ),
          Center(
            child: Column(
              children: [
                GestureDetector(
                  onTap: _renameProfile,
                  child: Container(
                    width: 84,
                    height: 84,
                    decoration: BoxDecoration(
                      shape: BoxShape.circle,
                      color: colorScheme.surfaceContainerHigh,
                      border: Border.all(color: mordifyAmber, width: 2),
                    ),
                    alignment: Alignment.center,
                    child: Text(
                      _initials(_displayName),
                      style: theme.textTheme.headlineMedium
                          ?.copyWith(color: colorScheme.onSurface, fontWeight: FontWeight.bold),
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
                const SizedBox(height: 6),
                Container(
                  padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 3),
                  decoration: BoxDecoration(
                    color: mordifyAmberDim,
                    borderRadius: BorderRadius.circular(20),
                  ),
                  child: Text('Level ${level.level}',
                      style: theme.textTheme.labelMedium?.copyWith(color: mordifyAmber)),
                ),
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
                      valueColor: const AlwaysStoppedAnimation(mordifyAmber),
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
                  color: mordifyAmber,
                ),
              ),
              const SizedBox(width: 12),
              Expanded(
                child: _StatCard(
                  icon: Icons.local_fire_department,
                  label: 'Best streak',
                  value: '$bestStreak',
                  color: mordifyAmber,
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
                  color: colorScheme.primary,
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
                        style: theme.textTheme.titleMedium?.copyWith(color: mordifyAmber),
                      ),
                    ),
                ],
              ),
            ),
          ],
          const SizedBox(height: 8),
          InkWell(
            onTap: widget.onViewAchievements,
            child: Padding(
              padding: const EdgeInsets.symmetric(vertical: 12, horizontal: 4),
              child: Row(
                mainAxisAlignment: MainAxisAlignment.spaceBetween,
                children: [
                  Text('View all badges',
                      style: theme.textTheme.bodyMedium?.copyWith(color: colorScheme.primary)),
                  Icon(Icons.chevron_right, color: colorScheme.primary),
                ],
              ),
            ),
          ),
        ],
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
