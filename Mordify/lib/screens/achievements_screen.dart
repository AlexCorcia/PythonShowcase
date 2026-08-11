import 'package:flutter/material.dart';

import '../services/achievements.dart';
import '../theme/app_theme.dart';

/// Grid of unlockable badges - pushed from [ProfileScreen]'s "View all
/// badges" link. Unlock state is derived live from [AchievementContext],
/// never persisted.
class AchievementsScreen extends StatelessWidget {
  final AchievementContext achievementContext;

  const AchievementsScreen({super.key, required this.achievementContext});

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    final colorScheme = theme.colorScheme;
    final badges = computeBadges(achievementContext);
    final unlockedCount = badges.where((b) => b.unlocked).length;

    return Scaffold(
      appBar: AppBar(title: const Text('Achievements')),
      body: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Padding(
            padding: const EdgeInsets.fromLTRB(20, 4, 20, 12),
            child: Text(
              '$unlockedCount of ${badges.length} unlocked',
              style: theme.textTheme.bodySmall?.copyWith(color: colorScheme.onSurfaceVariant),
            ),
          ),
          Expanded(
            child: GridView.builder(
              padding: const EdgeInsets.fromLTRB(18, 0, 18, 20),
              gridDelegate: const SliverGridDelegateWithFixedCrossAxisCount(
                crossAxisCount: 3,
                mainAxisSpacing: 20,
                crossAxisSpacing: 12,
                childAspectRatio: 0.8,
              ),
              itemCount: badges.length,
              itemBuilder: (context, index) {
                final badge = badges[index];
                return InkWell(
                  borderRadius: BorderRadius.circular(12),
                  onTap: () => _showBadgeDetail(context, badge),
                  child: Opacity(
                    opacity: badge.unlocked ? 1 : 0.55,
                    child: Column(
                      children: [
                        Container(
                          width: 56,
                          height: 56,
                          decoration: BoxDecoration(
                            shape: BoxShape.circle,
                            color: badge.unlocked ? mordifyAmber : colorScheme.surfaceContainerHigh,
                            border: Border.all(
                              color: badge.unlocked ? mordifyAmber : colorScheme.outlineVariant,
                              width: 2,
                            ),
                          ),
                          child: Icon(
                            badge.unlocked ? Icons.auto_awesome : Icons.lock_outline,
                            color: badge.unlocked ? nocturneBg : colorScheme.onSurfaceVariant,
                            size: 22,
                          ),
                        ),
                        const SizedBox(height: 7),
                        Text(
                          badge.label,
                          textAlign: TextAlign.center,
                          style: theme.textTheme.labelSmall
                              ?.copyWith(color: colorScheme.onSurfaceVariant, height: 1.25),
                        ),
                      ],
                    ),
                  ),
                );
              },
            ),
          ),
        ],
      ),
    );
  }
}

void _showBadgeDetail(BuildContext context, AchievementBadge badge) {
  final colorScheme = Theme.of(context).colorScheme;
  showDialog<void>(
    context: context,
    builder: (context) => AlertDialog(
      title: Row(
        children: [
          Container(
            width: 40,
            height: 40,
            decoration: BoxDecoration(
              shape: BoxShape.circle,
              color: badge.unlocked ? mordifyAmber : colorScheme.surfaceContainerHigh,
              border: Border.all(
                color: badge.unlocked ? mordifyAmber : colorScheme.outlineVariant,
                width: 2,
              ),
            ),
            child: Icon(
              badge.unlocked ? Icons.auto_awesome : Icons.lock_outline,
              color: badge.unlocked ? nocturneBg : colorScheme.onSurfaceVariant,
              size: 18,
            ),
          ),
          const SizedBox(width: 12),
          Expanded(child: Text(badge.label)),
        ],
      ),
      content: Column(
        mainAxisSize: MainAxisSize.min,
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text(badge.description),
          if (!badge.unlocked) ...[
            const SizedBox(height: 10),
            Text(
              'Not yet unlocked',
              style: TextStyle(color: colorScheme.onSurfaceVariant, fontStyle: FontStyle.italic),
            ),
          ],
        ],
      ),
      actions: [
        TextButton(onPressed: () => Navigator.of(context).pop(), child: const Text('Close')),
      ],
    ),
  );
}
