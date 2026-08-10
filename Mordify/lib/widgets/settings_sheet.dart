import 'package:flutter/material.dart';

import '../services/settings_repository.dart';
import '../services/theme_controller.dart';

const _contentModeLabels = {
  NotificationContentMode.countsOnly: 'Counts only',
  NotificationContentMode.taskNames: 'Task names',
  NotificationContentMode.dueSoon: 'Due soon',
  NotificationContentMode.countsAndNames: 'Counts + names',
};

class SettingsSheet extends StatefulWidget {
  final bool showStatusNotification;
  final ValueChanged<bool> onShowStatusNotificationChanged;
  final NotificationContentMode notificationContentMode;
  final ValueChanged<NotificationContentMode> onNotificationContentModeChanged;
  final ThemeController themeController;

  const SettingsSheet({
    super.key,
    required this.showStatusNotification,
    required this.onShowStatusNotificationChanged,
    required this.notificationContentMode,
    required this.onNotificationContentModeChanged,
    required this.themeController,
  });

  @override
  State<SettingsSheet> createState() => _SettingsSheetState();
}

class _SettingsSheetState extends State<SettingsSheet> {
  late bool _showStatusNotification = widget.showStatusNotification;
  late NotificationContentMode _mode = widget.notificationContentMode;

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    return SafeArea(
      child: Padding(
        padding: const EdgeInsets.fromLTRB(20, 20, 20, 24),
        child: Column(
          mainAxisSize: MainAxisSize.min,
          crossAxisAlignment: CrossAxisAlignment.stretch,
          children: [
            Text('Settings', style: theme.textTheme.titleLarge),
            const SizedBox(height: 12),
            SwitchListTile(
              contentPadding: EdgeInsets.zero,
              title: const Text('Status notification'),
              subtitle: const Text('Always-visible summary of your tasks'),
              value: _showStatusNotification,
              onChanged: (v) {
                setState(() => _showStatusNotification = v);
                widget.onShowStatusNotificationChanged(v);
              },
            ),
            const SizedBox(height: 4),
            Text('Notification content', style: theme.textTheme.labelLarge),
            AnimatedOpacity(
              opacity: _showStatusNotification ? 1 : 0.4,
              duration: const Duration(milliseconds: 150),
              child: IgnorePointer(
                ignoring: !_showStatusNotification,
                child: RadioGroup<NotificationContentMode>(
                  groupValue: _mode,
                  onChanged: (v) {
                    if (v == null) return;
                    setState(() => _mode = v);
                    widget.onNotificationContentModeChanged(v);
                  },
                  child: Column(
                    children: [
                      for (final mode in NotificationContentMode.values)
                        RadioListTile<NotificationContentMode>(
                          contentPadding: EdgeInsets.zero,
                          title: Text(_contentModeLabels[mode]!),
                          value: mode,
                        ),
                    ],
                  ),
                ),
              ),
            ),
            const SizedBox(height: 12),
            Text('Theme', style: theme.textTheme.labelLarge),
            const SizedBox(height: 8),
            ValueListenableBuilder<ThemeMode>(
              valueListenable: widget.themeController,
              builder: (context, mode, _) => SegmentedButton<ThemeMode>(
                segments: const [
                  ButtonSegment(value: ThemeMode.system, label: Text('System')),
                  ButtonSegment(value: ThemeMode.light, label: Text('Light')),
                  ButtonSegment(value: ThemeMode.dark, label: Text('Dark')),
                ],
                selected: {mode},
                onSelectionChanged: (selected) =>
                    widget.themeController.setThemeMode(selected.first),
              ),
            ),
          ],
        ),
      ),
    );
  }
}
