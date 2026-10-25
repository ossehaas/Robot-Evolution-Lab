# crawler_q4

Four-legged crawler, 8 SG90 servos, ~181 g. Each leg has a yaw hip (swings the leg forward/back) and a pitch
knee (lifts the foot); legs are splayed 35 degrees towards the front and back. Reference robot for the
locomotion pipeline and a seed genome for evolution.

```bat
:: writes prototypes\crawler_q4\build\
lab -m core.morphology build prototypes\crawler_q4\genome.json
lab -m core.evolution.train --genome prototypes\crawler_q4\genome.json --out runs\crawler_q4
lab -m core.evolution.play --run_dir runs\crawler_q4
```

`build/` holds the simulation USD, URDF, print-ready STL parts (`build/print/stl`), the bill of materials
(`build/print/BOM.md`), the servo channel map for firmware and `preview.png`.
