import 'package:flutter/material.dart';

import 'screens/home_screen.dart';
import 'services/background_service.dart';
import 'services/notification_service.dart';
import 'services/theme_controller.dart';
import 'theme/app_theme.dart';

void main() async {
  WidgetsFlutterBinding.ensureInitialized();
  await NotificationService.instance.init();
  await BackgroundService.initialize();
  final themeController = ThemeController();
  await themeController.load();
  runApp(MordifyApp(themeController: themeController));
}

class MordifyApp extends StatelessWidget {
  final ThemeController themeController;

  const MordifyApp({super.key, required this.themeController});

  @override
  Widget build(BuildContext context) {
    return ValueListenableBuilder<ThemeMode>(
      valueListenable: themeController,
      builder: (context, mode, _) {
        return MaterialApp(
          title: 'Mordify',
          themeMode: mode,
          theme: buildTheme(lightColorScheme),
          darkTheme: buildTheme(darkColorScheme),
          home: HomeScreen(themeController: themeController),
        );
      },
    );
  }
}
