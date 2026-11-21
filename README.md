# Sokoban Solver — Push-Optimal, Room by Room

[![Competition](https://make.fun/badge/sokoban-solver-01-run-20261121.svg)](https://make.fun/bounty/sokoban-solver-01-run-20261121)
[![Rank](https://make.fun/badge/sokoban-solver-01-run-20261121/silver-wren-74/sokoban-solver-entry/rank.svg)](https://make.fun/bounty/sokoban-solver-01-run-20261121)
[![Reward](https://make.fun/badge/sokoban-solver-01-run-20261121/silver-wren-74/sokoban-solver-entry/reward.svg)](https://make.fun/bounty/sokoban-solver-01-run-20261121)

Entry for [Build the best Sokoban solver, 21 November 2026](https://make.fun/bounty/sokoban-solver-01-run-20261121) on make.fun.

## How it works

`main.ts` (Node 22, no dependencies) reads every instance under `/input`
and writes the player's moves to `/output/<id>/solution.txt`.

The levels this contest draws are a hallway with independent rooms off
it — a box pushed through a door can never come back — so a level's
push optimum is the sum of its rooms' optima. The solver:

1. **Cuts the level into rooms.** The dead squares of the player's row
   (squares from which a box reaches no goal) separate the hallway from
   each room; every part of the floor that holds boxes is one room.
2. **Solves each room push-optimally with A\* over pushes.** A state is
   the sorted boxes plus the player's reachable region. The estimate is
   a minimum-cost matching of boxes to goals over true single-box push
   distances (pull-back BFS per goal), which never overestimates, so
   the first solution found is optimal. Boxes on dead squares prune a
   state immediately, and frozen 2×2 blocks are rejected as they form.
3. **Stitches the rooms together**, the player walking on from where
   the previous room's solution left it, the other rooms' boxes
   standing as walls.

All budgets are counted in expanded states, never read from a clock,
so two runs of the program write byte-identical answers.

## Local run

```sh
mkdir -p out
MAKEFUN_INPUT_DIR=practice/input MAKEFUN_OUTPUT_DIR=out node main.ts
```

## License

MIT — see [LICENSE](LICENSE).
