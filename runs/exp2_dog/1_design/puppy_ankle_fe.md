# Ankle spring and soft paw: FE cross-check (prototypes\puppy\genome.json)

Material: E = 33.6 MPa (Polymaker PolyFlex TPU95 TDS V5.4, Young's modulus X-Y, ISO 37), nu = 0.45; linear elastic, small strain. Solver: this file (structured voxel/quad meshes, selective reduced integration).

## Ankle spring: blade t = 7.78 mm, L = 14.3 mm, width w = 20.0 mm, pivot 18.0 mm above the paw centre
Beam formula (Euler-Bernoulli, pure moment): k = E w t^3 / (12 L) = 1844 N*mm/rad; surface strain at the limit 0.4 rad: limit * t / (2 L) = 0.109
3D FE of the printed profile (fillets, bolt holes), blocks held on their far + front/back faces (tight socket) / far faces only (soft pocket):
  pitch 1542 / 1463 N*mm/rad  (beam x 0.84 / 0.79)
  roll  7058 / 6461 N*mm/rad  (4.6 x pitch)
  yaw   2493 / 2115 N*mm/rad  (1.6 x pitch)
  axial 217 N/mm, shear along x 40 N/mm, along y 44 N/mm
  peak strain at the limit: 0.094 / 0.089
Design value used by dog_paw.py: 1503 N*mm/rad (calibration 0.815), peak strain 0.100 at 0.4 rad

Tuning at fixed L (k ~ t^3), dog_paw.spring_k:
| t [mm] | k [N*m/rad] | strain at limit |
|---|---|---|
| 7.18 | 1.18 | 0.092 |
| 7.38 | 1.28 | 0.095 |
| 7.58 | 1.39 | 0.098 |
| 7.78 | 1.50 | 0.100 |
| 7.98 | 1.62 | 0.103 |
| 8.18 | 1.75 | 0.105 |
| 8.38 | 1.88 | 0.108 |

## Sole disc: 4 arc arms (55 deg), arms 4.23 mm thick, travel to the bump stop 1.5 mm
3D FE (rim held, pad pushed up): k = 10.0 N/mm with the tread sliding, 10.5 N/mm sticking; peak strain 0.117 at the stop (1.5 mm), 0.078 at 1 mm (~10 N)
Design value used by dog_paw.py: 10.0 N/mm, peak strain 0.117 at the stop

Tuning (k ~ t^2.2: short curved arms bend and twist), dog_paw.sole_k:
| arms t [mm] | k [N/mm] | strain at the stop |
|---|---|---|
| 3.00 | 4.7 | 0.099 |
| 3.50 | 6.6 | 0.106 |
| 4.00 | 8.8 | 0.114 |
| 4.23 | 10.0 | 0.117 |
| 4.50 | 11.4 | 0.121 |
| 5.00 | 14.4 | 0.128 |

## Rejected: a hollow round paw printed sole-down (axisymmetric FE, sphere R = 21 mm, 45 deg cavity roof)
| flat sole radius | wall 1.0 | 1.2 | 1.4 | 1.6 mm | [N/mm] |
|---|---|---|---|---|---|
| 6.4 mm | 30 | 37 | 45 | 53 | |
| 4.5 mm | 18 | 23 | 28 | 34 | |
| 3.0 mm | 11 | 15 | 19 | 23 | |
Reissner (point load on a thin sphere): 4.1, 6.0, 8.1, 10.6 N/mm. A sole-down print needs a flat of r >= 6 mm (steeper overhangs fail in TPU), which makes a 1.0-1.6 mm shell 3-5x stiffer than 10 N/mm (thinner walls would not last).
