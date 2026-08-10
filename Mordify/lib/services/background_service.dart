import 'package:shared_preferences/shared_preferences.dart';
import 'package:workmanager/workmanager.dart';

import 'notification_service.dart';
import 'settings_repository.dart';
import 'task_repository.dart';

const String periodicTaskUniqueName = 'mordify-periodic-tick';
const String periodicTaskName = 'mordifyTick';

const String _lastNagAtKey = 'mordify.lastNagAt';

const Duration nagInterval = Duration(hours: 3);
const int nagActiveStartHour = 9;
const int nagActiveEndHour = 22; // exclusive: last check window is 21:xx

@pragma('vm:entry-point')
void callbackDispatcher() {
  Workmanager().executeTask((task, inputData) async {
    if (task == periodicTaskName) {
      await BackgroundService.runTick();
    }
    return true;
  });
}

class BackgroundService {
  static Future<void> initialize() async {
    await Workmanager().initialize(callbackDispatcher);
    await Workmanager().registerPeriodicTask(
      periodicTaskUniqueName,
      periodicTaskName,
      frequency: const Duration(minutes: 15),
      existingWorkPolicy: ExistingPeriodicWorkPolicy.keep,
    );
  }

  /// Runs on every periodic tick, both from the background isolate and
  /// (harmlessly) if ever invoked while the app is foregrounded: re-arms
  /// notification schedules, refreshes the persistent status notification,
  /// and decides whether it's time for a "still have tasks" nudge.
  static Future<void> runTick() async {
    await NotificationService.instance.init(requestPermissions: false);
    final tasks = await TaskRepository().loadTasks();

    await NotificationService.instance.rescheduleAll(tasks);

    final settings = SettingsRepository();
    final showStatus = await settings.getShowStatusNotification();
    if (showStatus) {
      final mode = await settings.getNotificationContentMode();
      await NotificationService.instance.updateStatusNotification(tasks, mode: mode);
    } else {
      await NotificationService.instance.clearStatusNotification();
    }

    final prefs = await SharedPreferences.getInstance();
    final now = DateTime.now();
    if (now.hour >= nagActiveStartHour && now.hour < nagActiveEndHour) {
      final lastNagRaw = prefs.getString(_lastNagAtKey);
      final lastNagAt = lastNagRaw == null ? null : DateTime.tryParse(lastNagRaw);
      final dueForNag =
          lastNagAt == null || now.difference(lastNagAt) >= nagInterval;

      if (dueForNag) {
        final pending = tasks.where((t) => t.countsAsPendingToday).toList();
        if (pending.isNotEmpty) {
          await NotificationService.instance.showNag(pending.length);
          await prefs.setString(_lastNagAtKey, now.toIso8601String());
        }
      }
    }
  }
}
