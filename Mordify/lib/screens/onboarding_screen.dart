import 'package:flutter/material.dart';

import '../theme/app_theme.dart';

class _OnboardingPage {
  final IconData icon;
  final String title;
  final String body;

  const _OnboardingPage({required this.icon, required this.title, required this.body});
}

const _pages = [
  _OnboardingPage(
    icon: Icons.auto_awesome,
    title: 'Welcome to Mordify',
    body: 'Turn your daily, weekly and monthly habits into streaks, points and levels.',
  ),
  _OnboardingPage(
    icon: Icons.local_fire_department,
    title: 'Build a streak',
    body: 'Complete a task in each of its periods to grow its streak. Every 7-period streak '
        'banks a freeze that auto-covers exactly one missed period, so an occasional slip '
        "won't reset all your progress.",
  ),
  _OnboardingPage(
    icon: Icons.stars_rounded,
    title: 'Earn points, level up',
    body: 'Every completion earns points - rarer tasks and longer streaks pay more. Points '
        'add up to levels and unlock achievement badges along the way.',
  ),
  _OnboardingPage(
    icon: Icons.calendar_today,
    title: 'Look back anytime',
    body: 'The Calendar and Stats tabs keep a full history of everything you\'ve completed, '
        'so you can always see how a habit is trending.',
  ),
];

/// Shown once, on the very first launch after tasks are seeded - see
/// [_AppShellState._load]'s `isFirstRun` capture. Purely explanatory; skips
/// straight into the already-seeded starter tasks either way.
class OnboardingScreen extends StatefulWidget {
  const OnboardingScreen({super.key});

  @override
  State<OnboardingScreen> createState() => _OnboardingScreenState();
}

class _OnboardingScreenState extends State<OnboardingScreen> {
  final _controller = PageController();
  int _page = 0;

  @override
  void dispose() {
    _controller.dispose();
    super.dispose();
  }

  void _next() {
    if (_page == _pages.length - 1) {
      Navigator.of(context).pop();
      return;
    }
    _controller.nextPage(duration: const Duration(milliseconds: 280), curve: Curves.easeOut);
  }

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    final colorScheme = theme.colorScheme;
    final isLast = _page == _pages.length - 1;

    return Scaffold(
      body: SafeArea(
        child: Column(
          children: [
            Align(
              alignment: Alignment.topRight,
              child: TextButton(
                onPressed: () => Navigator.of(context).pop(),
                child: const Text('Skip'),
              ),
            ),
            Expanded(
              child: PageView.builder(
                controller: _controller,
                itemCount: _pages.length,
                onPageChanged: (i) => setState(() => _page = i),
                itemBuilder: (context, index) {
                  final page = _pages[index];
                  return Padding(
                    padding: const EdgeInsets.symmetric(horizontal: 32),
                    child: Column(
                      mainAxisAlignment: MainAxisAlignment.center,
                      children: [
                        Container(
                          width: 96,
                          height: 96,
                          decoration: BoxDecoration(
                            shape: BoxShape.circle,
                            color: mordifyAmberDim,
                            border: Border.all(color: mordifyAmber, width: 2),
                          ),
                          child: Icon(page.icon, color: mordifyAmber, size: 42),
                        ),
                        const SizedBox(height: 28),
                        Text(
                          page.title,
                          textAlign: TextAlign.center,
                          style: theme.textTheme.headlineSmall?.copyWith(fontWeight: FontWeight.w600),
                        ),
                        const SizedBox(height: 12),
                        Text(
                          page.body,
                          textAlign: TextAlign.center,
                          style: theme.textTheme.bodyMedium
                              ?.copyWith(color: colorScheme.onSurfaceVariant, height: 1.4),
                        ),
                      ],
                    ),
                  );
                },
              ),
            ),
            Row(
              mainAxisAlignment: MainAxisAlignment.center,
              children: [
                for (var i = 0; i < _pages.length; i++)
                  AnimatedContainer(
                    duration: const Duration(milliseconds: 200),
                    margin: const EdgeInsets.symmetric(horizontal: 4),
                    width: i == _page ? 20 : 6,
                    height: 6,
                    decoration: BoxDecoration(
                      color: i == _page ? mordifyAmber : colorScheme.outlineVariant,
                      borderRadius: BorderRadius.circular(3),
                    ),
                  ),
              ],
            ),
            Padding(
              padding: const EdgeInsets.fromLTRB(24, 20, 24, 24),
              child: SizedBox(
                width: double.infinity,
                child: FilledButton(
                  onPressed: _next,
                  child: Text(isLast ? 'Get started' : 'Next'),
                ),
              ),
            ),
          ],
        ),
      ),
    );
  }
}
