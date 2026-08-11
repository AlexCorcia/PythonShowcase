import 'package:flutter/material.dart';

import '../theme/app_theme.dart';

/// Fires a top-banner "Achievement unlocked" toast for one newly-earned
/// badge - lighter-weight than [showCompletionCelebration] since it can
/// stack behind a task-completion celebration that's already on screen.
/// Auto-dismisses after a beat, or immediately on tap.
void showBadgeUnlockCelebration(BuildContext context, {required String label}) {
  final overlay = Overlay.of(context);
  late OverlayEntry entry;
  entry = OverlayEntry(
    builder: (_) => _BadgeUnlockBanner(
      label: label,
      onDismissed: () => entry.remove(),
    ),
  );
  overlay.insert(entry);
}

class _BadgeUnlockBanner extends StatefulWidget {
  final String label;
  final VoidCallback onDismissed;

  const _BadgeUnlockBanner({required this.label, required this.onDismissed});

  @override
  State<_BadgeUnlockBanner> createState() => _BadgeUnlockBannerState();
}

class _BadgeUnlockBannerState extends State<_BadgeUnlockBanner>
    with SingleTickerProviderStateMixin {
  late final AnimationController _controller;
  bool _dismissing = false;

  @override
  void initState() {
    super.initState();
    _controller = AnimationController(vsync: this, duration: const Duration(milliseconds: 2200))
      ..addStatusListener((status) {
        if (status == AnimationStatus.completed) widget.onDismissed();
      })
      ..forward();
  }

  @override
  void dispose() {
    _controller.dispose();
    super.dispose();
  }

  void _dismiss() {
    if (_dismissing) return;
    _dismissing = true;
    _controller.animateTo(1, duration: const Duration(milliseconds: 180));
  }

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    final topInset = MediaQuery.of(context).padding.top;

    return Positioned(
      top: topInset + 8,
      left: 16,
      right: 16,
      child: GestureDetector(
        onTap: _dismiss,
        child: AnimatedBuilder(
          animation: _controller,
          builder: (context, _) {
            final t = _controller.value;
            // Slide/fade in over the first 15%, hold, then slide/fade out
            // over the last 18% - same easing shape as the completion
            // celebration's entrance/exit for a consistent feel.
            final enter = t < 0.15 ? Curves.easeOutBack.transform(t / 0.15) : 1.0;
            final exit = t > 0.82 ? (1 - (t - 0.82) / 0.18).clamp(0.0, 1.0) : 1.0;
            final slide = (1 - enter) * -24;

            return Opacity(
              opacity: exit,
              child: Transform.translate(
                offset: Offset(0, slide),
                child: Material(
                  color: Colors.transparent,
                  child: Container(
                    padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 12),
                    decoration: BoxDecoration(
                      color: nocturneSurfaceHigh,
                      borderRadius: BorderRadius.circular(14),
                      border: Border.all(color: mordifyAmber.withValues(alpha: 0.5)),
                      boxShadow: [
                        BoxShadow(
                          color: Colors.black.withValues(alpha: 0.25),
                          blurRadius: 12,
                          offset: const Offset(0, 4),
                        ),
                      ],
                    ),
                    child: Row(
                      children: [
                        Container(
                          width: 40,
                          height: 40,
                          decoration: BoxDecoration(
                            shape: BoxShape.circle,
                            color: mordifyAmberDim,
                          ),
                          child: const Icon(Icons.auto_awesome, color: mordifyAmber, size: 20),
                        ),
                        const SizedBox(width: 12),
                        Expanded(
                          child: Column(
                            crossAxisAlignment: CrossAxisAlignment.start,
                            mainAxisSize: MainAxisSize.min,
                            children: [
                              Text(
                                'Achievement unlocked',
                                style: theme.textTheme.labelSmall?.copyWith(color: mordifyAmber),
                              ),
                              Text(
                                widget.label,
                                style: theme.textTheme.bodyMedium
                                    ?.copyWith(color: nocturneText, fontWeight: FontWeight.w600),
                              ),
                            ],
                          ),
                        ),
                      ],
                    ),
                  ),
                ),
              ),
            );
          },
        ),
      ),
    );
  }
}
