# Data

## Dog motion capture (not in git)

The imitation experiments of the robot dog (journal step 31 onwards: `amp_*`, `robust_*` and everything built on
them) use the dog motion capture of AI4Animation as a style reference:

> He Zhang, Sebastian Starke, Taku Komura, Jun Saito. *Mode-Adaptive Neural Networks for Quadruped Motion Control.*
> ACM Transactions on Graphics 37(4), SIGGRAPH 2018. A German Shepherd captured in BVH format.

Licence: **CC BY-NC 4.0** (attribution, non-commercial). It may not be redistributed here, so get it yourself:

1. Download https://starke-consult.de/AI4Animation/SIGGRAPH_2018/MotionCapture.zip
2. Unpack the BVH files into `data/mocap/ai4animation_dog/bvh/`
3. Retarget them to the robot dog (normal speed, and with slowed copies for slow walking):

```
lab -m core.mocap.retarget --genome prototypes/puppy/genome.json --out data/mocap/puppy_dog_motion.npz
lab -m core.mocap.retarget --genome prototypes/puppy/genome.json --out data/mocap/puppy_dog_motion_slow.npz --playback 1 0.75 0.5
```

The batch scripts in `runs/exp2_dog/` expect these two files.
