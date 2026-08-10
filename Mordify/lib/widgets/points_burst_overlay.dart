import 'dart:math';

import 'package:flutter/material.dart';

const _confettiColors = [
  Color(0xFFFFC107), // amber
  Color(0xFFFF4081), // pink accent
  Color(0xFF40C4FF), // light blue accent
  Color(0xFF69F0AE), // green accent
  Color(0xFFE040FB), // purple accent
  Color(0xFFFF6E40), // deep orange accent
];

/// Fires a one-shot confetti burst + floating "+N points" label, inserted
/// directly into the nearest [Overlay] and removed automatically once its
/// animation finishes.
void showPointsBurst(BuildContext context, {required int points, required int streak}) {
  final overlay = Overlay.of(context);
  late OverlayEntry entry;
  entry = OverlayEntry(
    builder: (_) => _PointsBurst(
      points: points,
      streak: streak,
      onCompleted: () => entry.remove(),
    ),
  );
  overlay.insert(entry);
}

class _PointsBurst extends StatefulWidget {
  final int points;
  final int streak;
  final VoidCallback onCompleted;

  const _PointsBurst({
    required this.points,
    required this.streak,
    required this.onCompleted,
  });

  @override
  State<_PointsBurst> createState() => _PointsBurstState();
}

class _PointsBurstState extends State<_PointsBurst> with SingleTickerProviderStateMixin {
  late final AnimationController _controller;
  late final List<_Particle> _particles;

  @override
  void initState() {
    super.initState();
    _controller = AnimationController(vsync: this, duration: const Duration(milliseconds: 1100))
      ..addStatusListener((status) {
        if (status == AnimationStatus.completed) widget.onCompleted();
      })
      ..forward();

    final random = Random();
    _particles = List.generate(28, (_) {
      final angle = -pi / 2 + (random.nextDouble() - 0.5) * pi * 1.4;
      final speed = 140 + random.nextDouble() * 180;
      return _Particle(
        angle: angle,
        speed: speed,
        color: _confettiColors[random.nextInt(_confettiColors.length)],
        size: 5 + random.nextDouble() * 6,
        spin: (random.nextDouble() - 0.5) * 10,
      );
    });
  }

  @override
  void dispose() {
    _controller.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    final screenSize = MediaQuery.of(context).size;
    final origin = Offset(screenSize.width / 2, screenSize.height * 0.32);
    final theme = Theme.of(context);

    return IgnorePointer(
      child: Stack(
        children: [
          AnimatedBuilder(
            animation: _controller,
            builder: (context, _) => CustomPaint(
              size: screenSize,
              painter: _ConfettiPainter(
                particles: _particles,
                progress: _controller.value,
                origin: origin,
              ),
            ),
          ),
          AnimatedBuilder(
            animation: _controller,
            builder: (context, child) {
              final t = _controller.value;
              final scale = t < 0.2 ? Curves.easeOutBack.transform(t / 0.2) : 1.0;
              final opacity = t > 0.7 ? (1 - (t - 0.7) / 0.3).clamp(0.0, 1.0) : 1.0;
              final rise = t > 0.2 ? (t - 0.2) * 40 : 0.0;
              return Positioned(
                left: 0,
                right: 0,
                top: origin.dy - 40 - rise,
                child: Opacity(
                  opacity: opacity,
                  child: Transform.scale(scale: scale, child: child),
                ),
              );
            },
            child: Column(
              children: [
                Text(
                  '+${widget.points}',
                  textAlign: TextAlign.center,
                  style: theme.textTheme.displaySmall?.copyWith(
                    fontWeight: FontWeight.w900,
                    color: theme.colorScheme.primary,
                    shadows: [Shadow(color: theme.colorScheme.surface, blurRadius: 12)],
                  ),
                ),
                if (widget.streak >= 2)
                  Text(
                    '🔥 ${widget.streak} streak',
                    textAlign: TextAlign.center,
                    style: theme.textTheme.titleMedium?.copyWith(fontWeight: FontWeight.bold),
                  ),
              ],
            ),
          ),
        ],
      ),
    );
  }
}

class _Particle {
  final double angle;
  final double speed;
  final Color color;
  final double size;
  final double spin;

  _Particle({
    required this.angle,
    required this.speed,
    required this.color,
    required this.size,
    required this.spin,
  });
}

class _ConfettiPainter extends CustomPainter {
  final List<_Particle> particles;
  final double progress;
  final Offset origin;

  _ConfettiPainter({required this.particles, required this.progress, required this.origin});

  @override
  void paint(Canvas canvas, Size size) {
    const gravity = 420.0;
    for (final p in particles) {
      final t = progress;
      final dx = cos(p.angle) * p.speed * t;
      final dy = sin(p.angle) * p.speed * t + 0.5 * gravity * t * t;
      final opacity = (1 - t).clamp(0.0, 1.0);
      if (opacity <= 0) continue;

      final center = origin + Offset(dx, dy);
      final paint = Paint()..color = p.color.withValues(alpha: opacity);

      canvas.save();
      canvas.translate(center.dx, center.dy);
      canvas.rotate(p.spin * t * pi);
      canvas.drawRect(
        Rect.fromCenter(center: Offset.zero, width: p.size, height: p.size * 1.6),
        paint,
      );
      canvas.restore();
    }
  }

  @override
  bool shouldRepaint(covariant _ConfettiPainter oldDelegate) => true;
}
