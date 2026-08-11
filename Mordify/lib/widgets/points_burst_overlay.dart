import 'dart:math';

import 'package:flutter/material.dart';

import '../theme/app_theme.dart';

const _confettiColors = [
  mordifyAmber,
  nocturneAccent,
  nocturneSage,
  nocturneTerracotta,
];

/// Fires the full-screen completion celebration (dimmed backdrop, confetti,
/// checkmark, "+N" and streak) from design/handoff's "Task Completion
/// Moment" mockup. Auto-dismisses after a beat, or immediately on tap so it
/// never gets in the way of someone checking off several tasks in a row.
void showCompletionCelebration(
  BuildContext context, {
  required int points,
  required int streak,
  required String taskTitle,
}) {
  final overlay = Overlay.of(context);
  late OverlayEntry entry;
  entry = OverlayEntry(
    builder: (_) => _CompletionCelebration(
      points: points,
      streak: streak,
      taskTitle: taskTitle,
      onDismissed: () => entry.remove(),
    ),
  );
  overlay.insert(entry);
}

class _CompletionCelebration extends StatefulWidget {
  final int points;
  final int streak;
  final String taskTitle;
  final VoidCallback onDismissed;

  const _CompletionCelebration({
    required this.points,
    required this.streak,
    required this.taskTitle,
    required this.onDismissed,
  });

  @override
  State<_CompletionCelebration> createState() => _CompletionCelebrationState();
}

class _CompletionCelebrationState extends State<_CompletionCelebration>
    with SingleTickerProviderStateMixin {
  late final AnimationController _controller;
  late final List<_Particle> _particles;
  bool _dismissing = false;

  @override
  void initState() {
    super.initState();
    _controller = AnimationController(vsync: this, duration: const Duration(milliseconds: 1400))
      ..addStatusListener((status) {
        if (status == AnimationStatus.completed) widget.onDismissed();
      })
      ..forward();

    final random = Random();
    _particles = List.generate(24, (_) {
      return _Particle(
        left: random.nextDouble(),
        fallDelay: random.nextDouble() * 0.3,
        size: 5 + random.nextDouble() * 6,
        rot: random.nextDouble() * 2 * pi,
        color: _confettiColors[random.nextInt(_confettiColors.length)],
        round: random.nextBool(),
      );
    });
  }

  @override
  void dispose() {
    _controller.dispose();
    super.dispose();
  }

  void _dismiss() {
    if (_dismissing) return;
    _dismissing = true;
    // Skip straight to the fade-out tail of the animation instead of
    // jumping to full completion, which would pop the overlay with no
    // transition at all.
    _controller.animateTo(1, duration: const Duration(milliseconds: 180));
  }

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    final screenSize = MediaQuery.of(context).size;

    return GestureDetector(
      onTap: _dismiss,
      child: AnimatedBuilder(
        animation: _controller,
        builder: (context, _) {
          final t = _controller.value;
          final entrance = t < 0.15 ? Curves.easeOutBack.transform(t / 0.15) : 1.0;
          final exitOpacity = t > 0.82 ? (1 - (t - 0.82) / 0.18).clamp(0.0, 1.0) : 1.0;

          return Stack(
            children: [
              Positioned.fill(
                child: Container(color: nocturneBg.withValues(alpha: 0.6 * exitOpacity)),
              ),
              for (final p in _particles)
                _buildParticle(p, t, screenSize, exitOpacity),
              Center(
                child: Opacity(
                  opacity: exitOpacity,
                  child: Transform.scale(
                    scale: entrance,
                    child: Column(
                      mainAxisSize: MainAxisSize.min,
                      children: [
                        Container(
                          width: 76,
                          height: 76,
                          decoration: BoxDecoration(
                            shape: BoxShape.circle,
                            color: mordifyAmberDim,
                            border: Border.all(color: mordifyAmber, width: 2),
                          ),
                          child: const Icon(Icons.check, color: mordifyAmber, size: 34),
                        ),
                        const SizedBox(height: 10),
                        Text(
                          '+${widget.points}',
                          style: theme.textTheme.headlineMedium
                              ?.copyWith(color: mordifyAmber, fontWeight: FontWeight.w600),
                        ),
                        const SizedBox(height: 4),
                        Text(
                          '${widget.taskTitle} complete',
                          textAlign: TextAlign.center,
                          style: theme.textTheme.bodyMedium?.copyWith(color: nocturneText),
                        ),
                        if (widget.streak >= 2) ...[
                          const SizedBox(height: 6),
                          Row(
                            mainAxisSize: MainAxisSize.min,
                            children: [
                              const Text('🔥', style: TextStyle(fontSize: 15)),
                              const SizedBox(width: 5),
                              Text(
                                '${widget.streak}-day streak',
                                style: theme.textTheme.bodyMedium
                                    ?.copyWith(color: nocturneText, fontWeight: FontWeight.w500),
                              ),
                            ],
                          ),
                        ],
                      ],
                    ),
                  ),
                ),
              ),
            ],
          );
        },
      ),
    );
  }

  Widget _buildParticle(_Particle p, double t, Size screenSize, double exitOpacity) {
    final localT = ((t - p.fallDelay) / (1 - p.fallDelay)).clamp(0.0, 1.0);
    final top = localT * screenSize.height * 0.9;
    final opacity = (localT < 0.05 ? localT / 0.05 : (1 - localT).clamp(0.0, 1.0)) * exitOpacity;
    if (opacity <= 0) return const SizedBox.shrink();
    return Positioned(
      left: p.left * screenSize.width,
      top: top,
      child: Opacity(
        opacity: opacity,
        child: Transform.rotate(
          angle: p.rot * t * 4,
          child: Container(
            width: p.size,
            height: p.size * (p.round ? 1 : 1.6),
            decoration: BoxDecoration(
              color: p.color,
              shape: p.round ? BoxShape.circle : BoxShape.rectangle,
              borderRadius: p.round ? null : BorderRadius.circular(1.5),
            ),
          ),
        ),
      ),
    );
  }
}

class _Particle {
  final double left;
  final double fallDelay;
  final double size;
  final double rot;
  final Color color;
  final bool round;

  _Particle({
    required this.left,
    required this.fallDelay,
    required this.size,
    required this.rot,
    required this.color,
    required this.round,
  });
}
