import 'package:flutter_test/flutter_test.dart';
import 'package:shared_preferences/shared_preferences.dart';

import 'package:mordify/main.dart';
import 'package:mordify/services/theme_controller.dart';

void main() {
  testWidgets('Mordify shows the task tabs', (WidgetTester tester) async {
    SharedPreferences.setMockInitialValues({});
    await tester.pumpWidget(MordifyApp(themeController: ThemeController()));
    await tester.pumpAndSettle();

    expect(find.text('Mordify'), findsOneWidget);
    expect(find.text('Daily'), findsOneWidget);
    expect(find.text('Weekly'), findsOneWidget);
    expect(find.text('Monthly'), findsOneWidget);
  });
}
