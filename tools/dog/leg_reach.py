"""How high can each paw be lifted (pure geometry), for the joint ranges the policy may use?"""
import numpy as np
from core.morphology import Genome, build_robot

s = build_robot(Genome.load("prototypes/puppy/genome.json"))
J = {j.name: j for j in s.joints}
lm = s.link_map


def paw_world(leg):
    link = lm[f"{leg}_s2"]
    sph = next(c for c in link.collisions if c.kind == "sphere")
    return link.origin + sph.transform[:3, 3], sph.size[0]


def Ry(a):
    c, sn = np.cos(a), np.sin(a)
    return np.array([[c, 0, sn], [0, 1, 0], [-sn, 0, c]])


print(f"chassis centre at rest: {1000 * s.spawn_height:.0f} mm above ground")
for leg in ("L0", "L1"):
    H, K = J[f"{leg}_j1"].origin, J[f"{leg}_j2"].origin
    F, r = paw_world(leg)
    ground = F[2] - r
    print(f"{leg}: hip {1000 * (H[2] - ground):.0f} mm, knee {1000 * (K[2] - ground):.0f} mm above ground; thigh "
          f"{1000 * np.linalg.norm(K - H):.0f}, shin {1000 * np.linalg.norm(F - K):.0f} mm; hip +-{J[f'{leg}_j1'].upper:.2f}, "
          f"knee +-{J[f'{leg}_j2'].upper:.2f} rad")
    for scale in (0.5, 0.75, 1.0):
        best_lift, best = -1.0, None
        for a1 in np.linspace(-J[f"{leg}_j1"].upper * scale, J[f"{leg}_j1"].upper * scale, 61):
            for a2 in np.linspace(-J[f"{leg}_j2"].upper * scale, J[f"{leg}_j2"].upper * scale, 61):
                K2 = H + Ry(a1) @ (K - H)
                F2 = K2 + Ry(a1 + a2) @ (F - K)
                lift, fwd = F2[2] - F[2], F2[0] - F[0]
                if abs(fwd) < 0.06 and lift > best_lift:  # a useful step: the paw stays within 6 cm fore/aft
                    best_lift, best = lift, (a1, a2, fwd)
        print(f"   {int(scale * 100):3d}% of the joint range: paw lifts {1000 * best_lift:4.0f} mm "
              f"(hip {best[0]:+.2f}, knee {best[1]:+.2f} rad, paw {1000 * best[2]:+.0f} mm fore/aft)")
