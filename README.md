# 🧠🦾 SynapseArm

**A learning brain for a 6-DOF industrial arm.** It combines classical robotics (kinematics, computed-torque control, Routh–Hurwitz stability) with machine learning that makes the arm faster to solve, more robust to model error, and able to draw whatever you hand it.

![tests](https://img.shields.io/badge/tests-84%20passing-2a78d6) ![python](https://img.shields.io/badge/python-3.11%2B-2a78d6) ![deps](https://img.shields.io/badge/stack-NumPy%20·%20SciPy%20·%20scikit--learn%20·%20OpenCV-8a8984)

![Sketch to robot program](results/pathgen_demo.png)
<sub>A hand-drawn PNG becomes 832 `MoveL` instructions for an ABB controller. Pen-up travel is cut by 87%, and every target is IK-verified before it reaches the robot.</sub>

## Highlights

| Module | What it does | Result |
|---|---|---|
| **Neural IK** | MLP predicts joint angles, then damped least squares refines them | **99.5%** solve rate vs 38% for classical DLS; **10× faster** |
| **Bayesian-optimised control** | Gaussian-process BO tunes gains across ±30% model error, load, saturation and noise | **8–13× lower** tracking error. Exposes a textbook design that goes **unstable on 7.5%** of plants |
| **Evolution-strategies neural controller** | Neural policy trained by black-box RL on the same plants | A further **−35%** error and **0.5%** overshoot vs the best PID; settles **98%** of the time vs 50% |
| **GP dynamics learning** | Learns unmodelled friction, cogging and parameter error, then adds them as feed-forward | **15×** better tracking on a trajectory it never trained on, after one 20 s run |
| **Sketch → robot program** | Vision or plain English (Claude) → strokes → TSP ordering → reachability check → ABB RAPID | **−87%** pen-up travel, **−70%** targets at ≤0.1 mm deviation, 0 unreachable points |
| **Learned dexterity map** | Gradient-boosted surrogate of IK reachability and Jacobian conditioning picks where to put the workpiece | **11×** better worst-case dexterity; searches 6,000 placements in **5 s** vs ~55 min of brute-force IK |
| **6-DOF dynamics + fault detection** | Batched Newton–Euler dynamics; momentum observer fused with a learned model of normal behaviour | **100%** of collisions caught in **40 ms** at 1% false alarms; faults diagnosed, faulty joint located **100%** of the time |

Everything is reproducible from a script, benchmarked on **held-out** data, and covered by unit tests. The tests check the robotics against hand-derived closed forms, and the ML against known answers.

```mermaid
flowchart LR
    subgraph Core["robotics/ (classical core)"]
        DH[D-H kinematics<br/>FK · Jacobian · DLS IK]
        CT[Computed-torque control<br/>Routh–Hurwitz]
        RB[Rigid-body dynamics<br/>Newton–Euler]
    end
    subgraph ML["ml/"]
        NIK[Neural IK]
        BO[Bayesian optimisation]
        ES[Evolution-strategies policy]
        GP[GP residual dynamics]
        DX[Dexterity map]
        FD[Fault detection<br/>+ diagnosis]
    end
    subgraph PG["pathgen/"]
        LLM[Plain English · Claude] --> SRC
        SRC[Text · shapes · sketches] --> OPT[Simplify + TSP order] --> PLC[Auto placement] --> RCH[Reachability] --> RAPID[ABB RAPID .mod]
    end
    DH --> NIK --> RCH
    DH --> DX --> PLC
    CT --> BO & ES & GP
    RB --> FD
```

## Quick start

```bash
pip install -r requirements.txt
pytest                                                  # 84 tests, ~45 s

python scripts/draw.py "draw a 7-pointed star 60 mm"    # -> results/drawing.{mod,png,json}
python scripts/draw.py "write 'Hello robot' 25 mm"
python scripts/draw.py --image examples/sketch.png --width 160
python scripts/draw.py --image photo.jpg --mode edges   # photos: Canny edges instead of ink lines
python scripts/draw.py "write 'Hello' 30 mm" --auto-place          # best paper position & rotation
python scripts/draw.py "sign 'Ada' ~3 cm with a small star to the right" --llm   # needs ANTHROPIC_API_KEY

python scripts/train_neural_ik.py       # ~5-10 min   neural IK model + benchmark
python scripts/tune_controller.py       # ~1.5 min    Bayesian-optimised gains
python scripts/train_rl_controller.py   # ~5 min      evolution-strategies neural controller
python scripts/learn_dynamics.py        # ~2 min      GP dynamics learning
python scripts/train_dexterity_map.py   # ~4 min      learned dexterity map + placement demo
python scripts/fault_detection.py       # ~10 min     fleet simulation, fault detection + diagnosis
python scripts/plot_workspace.py        #             reachable workspace, coloured by manipulability
```

---

## 1 · Neural inverse kinematics

`ml/neural_ik.py`: numerical IK is accurate but needs a good starting guess. From a poor seed it's slow, lands on the wrong configuration, or fails near limits and singularities. A multilayer perceptron, trained on 150k poses generated from the arm's own forward kinematics (so no external data is needed), predicts the joints in microseconds. Damped least squares then polishes that guess to nanometre accuracy in a handful of iterations.

- **Branch-consistent training.** IK is one-to-many (elbow up/down, wrist flip). Trained on every solution, a regression network averages between them, so it's trained on one configuration branch. That's the same idea as an industrial controller's `confdata`.
- **Angle encoding.** Joints are predicted as (sin θ, cos θ), so the ±180° wrap of the wrist doesn't confuse the network.

Benchmark on 500 random reachable 6-DOF poses (`results/neural_ik_benchmark.json`):

| Strategy | Success rate | Median position error | Mean iterations | Time per solve |
|---|---|---|---|---|
| Neural net only | 0.0% | 60.9 mm | 0 | 0.01 ms |
| DLS from home pose | 38.0% | 8.17 µm | 66.3 | 13.61 ms |
| DLS from random seed | 22.0% | 516.6 mm | 80.5 | 16.42 ms |
| **Neural net + DLS (hybrid)** | **99.5%** | **1.2 nm** | **4.6** | **1.31 ms** |

The network alone is too coarse to drive the arm, but it's an excellent initial guess. That's how learned IK is used in practice.

## 2 · Robust controller tuning with Bayesian optimisation

`ml/bayes_opt.py` (Gaussian process + Expected Improvement, written from scratch on scikit-learn) and `ml/controller_tuning.py`. One axis is modelled as a mass-spring-damper (m = 7, b = 1, k = 9). Gains are tuned to hold up when the real plant **isn't** the model:

- true m, b, k each off by up to **±30%**
- an unknown **±3 N step load** at t = 10 s
- actuator saturation **|f| ≤ 30 N** with anti-windup, plus position and velocity sensor noise

Each candidate is simulated on 24 training plants at once (vectorised NumPy). The cost is mean + ½σ of IAE + control chatter + overshoot. Every design is then scored on **200 held-out plants**.

![Controller tuning](results/controller_tuning.png)

| Design | Gains (kp, kd, ki) | Mean IAE | Worst IAE | Overshoot | Settled ≤10 s | Error after load | Unstable plants |
|---|---|---|---|---|---|---|---|
| Classical computed-torque PD (ζ = 1, ωn = ½√(k/m)) | 0.321, 1.13, 0 | 14.56 | 46.17 | 42% | 2% | 1.135 | **7.5%** |
| BO: computed-torque + integral | 6.57, 2.47, 0.457 | 1.13 | 1.95 | 20% | 63% | 0.018 | **0%** |
| BO: model-free PID | 12.3, 8.7, 5.59 | 1.82 | 2.39 | 3% | 50% | 0.005 | **0%** |

- **The textbook design is fragile.** Its `+k·x` feed-forward cancels the *nominal* spring. If the true k is more than 25% lower than modelled, the net stiffness `m·kp + k_true − k_nom` goes negative and the loop is **unstable**. This is proven per plant with a Routh–Hurwitz test (`linear_stable`) and confirmed in simulation.
- **BO's designs stay stable on every unseen plant** and cut tracking error 8–13×.
- **BO is sample-efficient.** Averaged over 3 seeds, it matches random search's 40-simulation result in 15–18 simulations. That matters when each evaluation is a real hardware trial.

## 3 · Neural controller trained with evolution strategies

`ml/rl_policy.py`: the controller is a learned linear PID term plus a 16-unit tanh network. It's trained by **OpenAI-style evolution strategies** (antithetic perturbations, centred-rank fitness shaping, Adam). Each generation, the whole population is simulated on every training plant in **one batched simulation**, which takes about 2 s on a laptop CPU. The scenarios, cost and held-out test set are the same as in §2.

![ES controller](results/rl_controller.png)

| Controller (200 held-out plants) | Mean IAE | Worst IAE | Overshoot | Settled ≤10 s | Control effort (TV) |
|---|---|---|---|---|---|
| BO-tuned PID | 1.82 | 2.39 | 3.0% | 50% | 512 |
| **ES neural policy** | **1.19** | **1.80** | **0.5%** | **98%** | 511 |

The policy learned to drop its linear derivative term (kd ≈ 0) and do the damping nonlinearly through the network. It's faster without overshooting, at the same control effort. **Caveat:** unlike a PID, a neural policy has no Hurwitz certificate. Robustness here is shown empirically across 200 unseen plants, with 0 divergences. Formal guarantees (Lyapunov or IQC-based) would be the next step before any hardware use.

## 4 · Learning unmodelled dynamics with a Gaussian process

`ml/gp_dynamics.py`: a computed-torque controller is only as good as its model. Here the true axis has a heavier payload, 2.5× the viscous damping, a softer spring, **Coulomb friction** and position-dependent **cogging**. After each 20 s run, a GP learns the force the model misses, `r = f − (M a + B v + K x)`. Its mean is then added as feed-forward along the next reference, which is iterative learning control. Where the GP's predictive σ is high, the correction is faded out.

The key design choice is the **physics-structured kernel**:

`k = linear(x, v, a) + RBF(x) + RBF(v) + noise`

That is, parameter error + cogging(x) + friction(v). It's implemented from scratch, with hyperparameters fitted by marginal likelihood. Compared with a black-box 3-D RBF GP:

![GP dynamics](results/gp_dynamics.png)

| RMS tracking error | Training trajectory | **Unseen trajectory** |
|---|---|---|
| Nominal model only | 15.1 mm | 17.7 mm |
| + black-box RBF GP (5 runs) | 2.0 mm | 12.9 mm |
| **+ physics-structured GP (1 run)** | **0.31 mm** | **1.15 mm** |
| Perfect model (lower bound) | 0.17 mm | 0.23 mm |

The black-box GP memorises the training trajectory. The structured GP learns the *physics*: the right panel shows it recovering the friction curve, with uncertainty growing where it has no data. So it carries over to motions it has never seen.

## 5 · Sketch / text → ABB RAPID program

`pathgen/` goes from a request to a robot program in one call:

0. **Request planning** (`llm.py`): free-form requests ("sign 'Ada' about 3 cm tall with a small star to the right") go to Claude through **tool use with a strict JSON schema**. Every size and position it returns is validated and clamped before anything reaches the robot. Without an API key, a rule-based parser handles one or more clauses, and it's quote-aware, so "rock and roll" stays one word. The tests use a fake client, so CI needs no key.
1. **Sources** (`sources.py`): text (font outlines), parametric shapes, or **images**. Sketches are binarised (Otsu) and skeletonised to 1-px centre-lines, then the skeleton graph is traced into polylines. Photos go through Canny edge detection instead.
2. **Optimisation** (`optimize.py`): Ramer–Douglas–Peucker simplification within a tolerance. Then stroke ordering, solved as a generalised TSP (each stroke can be drawn either way): greedy nearest neighbour, then 2-opt with direction flips.
3. **Reachability** (`reach.py`): every target is mapped through the paper work-object and pen tool, solved with the §1 neural-IK hybrid, and followed with continuation so the joint path stays on one branch. It reports unreachable points, closeness to singularity, and the largest joint jump.
4. **RAPID** (`rapid.py`): a self-contained module with `tooldata`, `wobjdata` and `speeddata`. Coordinates are written relative to the work-object, so the program runs wherever the sheet is calibrated. Drawing moves use fine points at touchdown and small zones while drawing, and long programs are split into procedures.

| Job | Strokes | Targets (raw → simplified) | Pen-up travel (traced → greedy → 2-opt) | Unreachable | Max joint step |
|---|---|---|---|---|---|
| Hand-drawn sketch | 39 | 2693 → 793 | 2.49 m → 0.38 m → **0.33 m** | 0 | 4.9° |
| `write 'SynapseArm' 22 mm` | 14 | 644 → 363 | 0.19 m → 0.19 m → 0.18 m | 0 | 1.6° |
| `draw a 7-pointed star 60 mm` | 1 | 15 → 15 | n/a | 0 | 1.6° |

Example outputs, ready to load in RobotStudio, are in [`examples/`](examples/) (`*.mod`).

## 6 · Learned dexterity map and automatic placement

`ml/dexterity.py`: where should the workpiece go? Checking one placement properly means an IK solve for every target, so brute-force search over thousands of placements takes most of an hour. Instead:

1. **Label** 25k random tool-down points in the cell with the real IK, recording reachability and **dexterity**. Dexterity is the inverse condition number of the length-scaled Jacobian (0 = singular, 1 = isotropic). Unlike raw manipulability, it doesn't mix mm with radians, which would simply reward stretching the arm out.
2. **Learn** reachability (classifier) and dexterity (regressor) with gradient-boosted trees: **99.7%** reach accuracy (AUC 0.9999) and dexterity **R² 0.998** on 3,000 held-out IK-labelled points, at ~260k points/s.
3. **Search** 6,000 placements (x, y, z, rotation) with one batched prediction, maximising a soft-min of dexterity over the drawing while requiring confident reachability.
4. **Verify** the top 5 with real IK and keep the best verified one. The rotation carries through to the RAPID `wobjdata` quaternion.

![Dexterity map](results/dexterity_map.png)

On a 300 mm drawing, worst-case dexterity goes from **0.021** (default placement, close to a singularity) to **0.226**, with 0 unreachable points. The search takes **4.8 s**, against an estimated ~55 min of brute-force IK. Use it with `scripts/draw.py --auto-place`.

## 7 · Full 6-DOF dynamics, collision and fault detection

`robotics/dynamics.py`: the **recursive Newton–Euler algorithm**, vectorised over a batch of robots. It provides M(q), Coriolis/gravity bias and forward dynamics, plus reflected motor (armature) inertia, which dominates the wrist on real arms. It's tested against physics invariants: M symmetric positive-definite, gravity = ∇PE to 1e-8, Coriolis does no work, and energy conserved in free fall.

`ml/fault_detection.py` puts it to work on a **simulated fleet**. Each arm has its own friction (±30% unit to unit) and payload variation, and runs random moves under computed-torque control that knows only the rigid-body model. Four faults are injected at random times: **collisions** (40–150 N on the tool), an **unexpected payload** (1.5–4 kg), **joint wear** (friction ×2–3) and **actuator degradation** (60–80% of commanded torque). All detectors use only measured joint positions, velocities and commanded torque:

| Detector, 1% false alarms on unseen arms | Collision | Payload | Wear | Actuator | AUC |
|---|---|---|---|---|---|
| Inverse-dynamics residual | 99% · 44 ms | 97% · 39 ms | 79% · 71 ms | 61% · 59 ms | 0.940 |
| Momentum observer (rigid-body model) | 99% · 44 ms | 96% · 40 ms | 74% · 103 ms | 64% · 65 ms | 0.928 |
| Observer − learned normal model | 100% · 40 ms | 100% · 35 ms | 61% · 85 ms | 61% · 43 ms | 0.942 |
| **Fused (learned observer OR ID residual)** | **100% · 40 ms** | **100% · 33 ms** | **84% · 50 ms** | **69% · 43 ms** | **0.951** |

<sub>Detection rate · median delay from fault onset. 200 fault-free and 400 faulty test runs on arms never seen in training or calibration.</sub>

![Fault detection](results/fault_detection.png)

What made the difference:

- **Physics first, learning second.** The generalised-momentum observer estimates external torque without measuring acceleration. The gradient-boosted model learns only what physics leaves out (fleet-average friction), from fault-free runs. It's trained on the *de-filtered* target `r + ṙ/K`, and its prediction goes through the observer's own filter. That way the observer's lag at velocity reversals matches the model, instead of showing up as false spikes.
- **Calibrate on unseen arms.** An early version set per-joint scales on the training arms, where the learned residual is optimistically small. That inflated the alarm threshold, and the learned detector looked *worse* than the plain observer. Scales and thresholds now come from two separate held-out sets of arms.
- **Fuse complementary residuals.** The learned observer is best for sudden events (collisions, payload). The inverse-dynamics residual is best for slow friction wear on the big joints. An OR-fusion, with both thresholds tuned jointly so the union stays at 1% false alarms, beats each one on every fault type.
- **Diagnosis.** A classifier on the post-alarm residual signature names the fault 83% of the time: collisions 98% and payload 96%. Wear and actuator faults are the hard pair because both look like extra effort on one joint. The faulty joint itself is located **100%** of the time.

---

## Repository layout

```
robotics/    kinematics.py (D-H, FK, Jacobian, DLS IK, analytic wrist-centre IK), workspace.py, control.py
robotics/    dynamics.py (batched recursive Newton–Euler, armature, friction)
ml/          neural_ik.py · bayes_opt.py · controller_tuning.py · rl_policy.py · gp_dynamics.py
             dexterity.py · fault_detection.py
pathgen/     llm.py · sources.py · optimize.py · reach.py · rapid.py · pipeline.py
scripts/     one entry point per experiment (all reproducible, fixed seeds)
tests/       84 pytest tests (GitHub Actions on every push)
models/      neural IK, ES policy, dexterity map, fault residual + diagnosis models
results/     benchmark JSON + figures
examples/    sample sketch and generated RAPID programs
```

## Engineering notes

- **Kinematic model.** Standard D-H with a (mm) = [150, 760, 140, 0, 0, 0], α = [90°, 0, 90°, −90°, 90°, 0], d = [505, 0, 0, 795, 0, 0]. That's an offset shoulder and a spherical wrist, the typical ABB IRB geometry. FK is tested against the hand-derived closed form Px = c1(150 + 760c2 + 140c23 + 795s23), Pz = 505 + 760s2 + 140s23 − 795c23. The Jacobian is tested against finite differences.
- **Honest baselines.** Every learned method is compared with a reasonable classical one on the same data: DLS IK, a textbook computed-torque design, random search, a black-box GP, and a perfect-model oracle.
- **Held-out evaluation.** Controllers are tuned on 24 plants and scored on 200 different ones. The dynamics model is trained on one trajectory and scored on another.
- **Dynamics.** Link inertias are modelled as rods along the D-H segments, with IRB-class masses, a 2 kg payload and armature inertia. They're plausible values, not a datasheet; swapping in identified parameters changes numbers, not code.
- **Simulation-only.** Results come from simulated plants with noise, saturation and model error. They have not been run on hardware.

## Roadmap

- [x] Neural IK + hybrid solver
- [x] Bayesian-optimised robust gains
- [x] Evolution-strategies neural controller
- [x] GP residual dynamics + iterative learning
- [x] Sketch / text → RAPID with reachability checking
- [x] Learned dexterity map + automatic workpiece placement
- [x] 6-DOF rigid-body dynamics (batched Newton–Euler)
- [x] Collision & fault detection + diagnosis on joint torques
- [x] Natural-language requests via Claude (tool use, validated plans)
- [ ] Per-unit calibration of the learned normal model (closes the wear/actuator gap)
- [ ] GP / learned residual dynamics and ES control on the full 6-DOF arm
- [ ] Interactive browser demo: sketch in, RAPID out
