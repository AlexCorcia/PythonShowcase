import 'package:uuid/uuid.dart';

import '../models/folder.dart';
import '../models/task.dart';

// Fixed ids for the folders Mordify ships with, so seeding can also retro-fit
// them onto tasks created by the very first seed wave.
const String skincareFolderId = 'folder-skincare';
const String fitnessFolderId = 'folder-fitness';
const String gymFolderId = 'folder-gym';
const String workFolderId = 'folder-work';

/// The starter tasks Mordify ships with, seeded once on first launch.
List<Task> buildDefaultTasks() {
  const uuid = Uuid();
  final today = DateTime.now();

  return [
    Task(
      id: uuid.v4(),
      title: 'Skin Care - Morning',
      frequency: TaskFrequency.daily,
      hour: 7,
      minute: 0,
      folderId: skincareFolderId,
      subtasks: [
        SubTask(id: uuid.v4(), title: 'Clean'),
        SubTask(id: uuid.v4(), title: 'Niacinamide'),
        SubTask(id: uuid.v4(), title: 'Hydrate'),
        SubTask(id: uuid.v4(), title: 'Sun Screen'),
      ],
    ),
    Task(
      id: uuid.v4(),
      title: 'Skin Care - Night',
      frequency: TaskFrequency.daily,
      hour: 21,
      minute: 0,
      folderId: skincareFolderId,
      subtasks: [
        SubTask(id: uuid.v4(), title: 'Clean'),
        SubTask(id: uuid.v4(), title: 'Hydrate'),
      ],
    ),
    Task(
      id: uuid.v4(),
      title: 'Exfoliate',
      frequency: TaskFrequency.interval,
      hour: 21,
      minute: 0,
      intervalDays: 2,
      anchorDate: DateTime(today.year, today.month, today.day),
      folderId: skincareFolderId,
    ),
    Task(
      id: uuid.v4(),
      title: 'Fichaje',
      frequency: TaskFrequency.weekdays,
      hour: 8,
      minute: 55,
      folderId: workFolderId,
    ),
  ];
}

/// Second wave of starter tasks (weekly/flexible ones), seeded once.
List<Task> buildDefaultTasksV2() {
  const uuid = Uuid();

  return [
    Task(
      id: uuid.v4(),
      title: 'Take the dogs for a walk',
      frequency: TaskFrequency.timesPerWeek,
      targetCount: 5,
      folderId: fitnessFolderId,
    ),
    Task(
      id: uuid.v4(),
      title: 'Smash training',
      frequency: TaskFrequency.daily,
      folderId: fitnessFolderId,
    ),
    Task(
      id: uuid.v4(),
      title: 'Tennis training',
      frequency: TaskFrequency.timesPerWeek,
      targetCount: 3,
      folderId: fitnessFolderId,
    ),
    Task(
      id: uuid.v4(),
      title: 'Meal prep for the week',
      frequency: TaskFrequency.weekly,
      weekday: DateTime.sunday,
      hour: 18,
      minute: 0,
    ),
    Task(
      id: uuid.v4(),
      title: 'Gym - Leg Day',
      frequency: TaskFrequency.timesPerWeek,
      targetCount: 1,
      folderId: gymFolderId,
    ),
    Task(
      id: uuid.v4(),
      title: 'Gym - Push',
      frequency: TaskFrequency.timesPerWeek,
      targetCount: 1,
      folderId: gymFolderId,
    ),
    Task(
      id: uuid.v4(),
      title: 'Gym - Pull',
      frequency: TaskFrequency.timesPerWeek,
      targetCount: 1,
      folderId: gymFolderId,
    ),
    Task(
      id: uuid.v4(),
      title: 'Gym - Extra',
      frequency: TaskFrequency.timesPerWeek,
      targetCount: 1,
      folderId: gymFolderId,
    ),
  ];
}

/// Folders backing the default tasks above. Safe to merge into an existing
/// folder list keyed by id (won't duplicate on repeat seeding).
List<Folder> buildDefaultFolders() => [
      Folder(id: skincareFolderId, name: 'Skincare', colorValue: 0xFF00897B),
      Folder(id: fitnessFolderId, name: 'Fitness', colorValue: 0xFFFB8C00),
      Folder(id: gymFolderId, name: 'Gym', colorValue: 0xFF5E35B1),
      Folder(id: workFolderId, name: 'Work', colorValue: 0xFF1E88E5),
    ];
