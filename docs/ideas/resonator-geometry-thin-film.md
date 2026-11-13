# Simulating the film on the real resonator geometry

Status: **idea / discussion notes** (started 2026-11-13). Nothing here is implemented.
Numbers marked *(estimate)* come from assumed parameters and need checking.

## Goal

Move the axisymmetric solver from "free surface over a flat bottom at $z=-h$" to the film
coating the actual resonator: a wedge disk on a pedestal, revolved around $r=0$. The film
covers the top of the disk, wraps around the wedge tip, the underside, the pedestal and the
substrate.

## The core difficulty: scale separation

| Quantity | Value |
|---|---|
| Resonator size $L$ | $\sim 1$–$6$ mm |
| Film thickness / excitation $h$ | $\sim 10$ nm |
| Aspect ratio $h/L$ | $\sim 10^{-5}$–$10^{-6}$ |
| Mode frequency | $\sim 180$ Hz, period $T\approx 5.5$ ms |

Assumed for the estimates below: $\rho\approx145$ kg/m³, $\sigma\approx3.5\times10^{-4}$ N/m,
$c_3\approx1$ m/s (inferred from 180 Hz on a mm-scale disk), so
$g_{\text{eff}}=c_3^2/h\approx10^{8}$ m/s². **To confirm:** actual $h_0$ and $\alpha_{\text{vdW}}$.

## Where the code stands (Nov 2026)

- The running axisymmetric path (`RadialVelocityCalculator`) uses the Bessel-series Green
  function, which encodes both the wall $r=R$ and the image at $z=-\text{depth}$. It cannot
  survive any geometry change.
- `EllipticGreenFunctions` is the free-space ring kernel (no image) and is not yet wired
  into the `G / dGdn / selfSingle / selfDouble` kernel interface.
- `BoundaryNodes`, the S/D assembly and `SelfTerms` are already kernel-agnostic.

## Parametrization: thickness along the substrate normal

$\eta(r)$ is not single-valued on the resonator. Use the substrate arclength $s$:

$$\mathbf X(s,t)=\mathbf X_w(s)+h(s,t)\,\mathbf n(s),\qquad
d\ell=\sqrt{(1-\kappa h)^2+h_s^2}\;ds,$$

$$h_t = q\,\frac{\sqrt{(1-\kappa h)^2+h_s^2}}{1-\kappa h}.$$

The current formulation is the special case $s=r$, $\kappa=0$. This breaks only where the
substrate radius of curvature is comparable to $h$ (sharp tip, concave pedestal corner).
Alternative: free Lagrangian markers $(r(\xi),z(\xi))$, which allow overturning but cluster.

## Options considered

| # | Option | Exact? | Main problem |
|---|---|---|---|
| 1 | Global dense BEM with substrate nodes | yes | $\sim10^5$–$10^6$ nodes, dense storage, ill-conditioned thin gap |
| 2 | Image kernel as exact reformulation (reflect collocation point across local tangent, keep substrate nodes for the residual) | yes | conditioning gain unproven on curved substrates |
| 3 | **Windowed BEM** using locality (below) | yes, to tolerance $e^{-\pi W/2h}$ | needs $\Delta s\lesssim h$ inside windows |
| 4 | Matrix-free GMRES, windows as preconditioner | yes | only needed if a single global solve is wanted |
| 5 | Fully nonlinear thin-film model (Serre–Green–Naghdi + vdW + capillarity) | no: error $(kh)^4/45$ in $\omega^2$ | fails where the surface varies on scale $h$; no overturning |
| 6 | Hybrid: BEM as short-domain reference, option 5 for ms dynamics | — | two codes |
| 7 | **Tip-only BEM with open ends** (below) | yes inside the region | needs open-boundary treatment and an exterior model |

Current direction: stay with BEM (options 3 and 7). Option 5 is parked.

## Locality of the solve

For a flat film of uniform thickness, $\hat q=k\tanh(kh)\,\hat\varphi$, i.e.

$$q(s)=-\partial_s\!\int K(s-s')\,\partial_{s'}\varphi(s')\,ds',\qquad
K(x)=\frac1\pi\ln\coth\frac{\pi|x|}{4h}\;\simeq\;\frac2\pi e^{-\pi|x|/2h}.$$

- Decay length $2h/\pi$; $\int K\,dx=h$ (shallow-water limit $q\to-h\varphi_{ss}$).
- Fully nonlinear surface, flat bottom: cut the film, put wrong data on the cut. The error
  $e$ is harmonic with $e=0$ on the surface and $\partial_ne=0$ on the bottom. Reflect across
  the bottom and compare with $\cos(\pi y/2H)\,e^{-\pi x/2H}$, $H>h_{\max}$:

$$|e|\le C\,\max_{\Gamma_c}|e|\;e^{-\pi d/2H}.$$

- Locality is **instantaneous** (elliptic solve), not dynamical, and it lives in the solution
  operator, **not** in the matrices $S$, $D$ (which decay only like $1/\text{distance}$).
- Locality is in arclength, not Euclidean distance (top and underside films near the tip do
  not couple).
- Buffer for error $\varepsilon$: $W=\frac{2H}{\pi}\ln\frac{C\max|e|}{\varepsilon}\approx15h$ for $10^{-10}$.
- Suggested cut data: $\phi=\varphi(s_c)$ constant across the thickness (mismatch $\sim h^2\varphi_{ss}$).

Open: constant in the bound for $q$; window blending; proof for curved substrates
(Saint-Venant energy estimate instead of reflection).

**Test:** global solve vs windows of varying $W$ on a flat bottom; error vs $W/h$ on a semilog
axis should have slope $\pi/2$, including at large amplitude.

## Time stepping

$$\omega^2=\Big(g_{\text{eff}}k+\frac\sigma\rho k^3\Big)\tanh(kh),\qquad
\ell_c=\sqrt{\frac{\sigma}{\rho g_{\text{eff}}}}\approx150\text{ nm}\approx15h\ \textit{(estimate)},$$

$$\omega\simeq c_3k\Big[1+\tfrac12\big(\ell_c^2-\tfrac{h^2}{3}\big)k^2\Big].$$

Capillary dispersion exceeds finite-depth dispersion by $3\ell_c^2/h^2\approx700$ *(estimate)*.

| $\Delta s$ | Stability $\Delta t$ | Steps/period (stability) | Steps/period (wave-resolving) |
|---|---|---|---|
| $h$ | $3\times10^{-10}$ s | $2\times10^{7}$ | $5\times10^{5}$ |
| $3h$ | $2\times10^{-9}$ s | $3\times10^{6}$ | $2\times10^{5}$ |
| $10h$ | $2\times10^{-8}$ s | $3\times10^{5}$ | $5\times10^{4}$ |

- **Stiffness** (capillary, removable): small-scale decomposition (Hou, Lowengrub & Shelley
  1994, arclength / tangent-angle form), IMEX, or an exponential integrator on the flat bottom.
- **Wave-resolving count** (not removable): steps per traversal $\sim N$, so cost per period $\sim N^2$.
- Narrowest expected structure $w\sim\ell_c\sqrt{h/a}$, i.e. $30$–$50\,h$ for $a\sim h$ *(estimate)*.
  The physics then asks for $\Delta s\sim3h$–$10h$, where the windowed BEM is weakest.

## Tip-only simulation with open ends

Simulate only $s\in[s_a,s_b]$ around the wedge tip, still axisymmetric (ring kernel with
$r(s)$). The ends are the same artificial cuts as the window construction.

- Mass in the region is not conserved, and should not be:
  $\dfrac{dM}{dt}=-\big[2\pi r\,h\,\bar u\big]_{s_a}^{s_b}$. The rest of the film is a reservoir.
- Open-boundary choices: (a) radiation / characteristic condition, using that the exterior is
  long-wave; (b) absorbing layer relaxing to $h_0$; (c) coupling to a cheap exterior model
  (linear shallow water or modal) that carries the global 180 Hz mode.
- With $N\sim10^3$–$10^4$ nodes the ms time scale becomes affordable.

## Hybrid: modal model for the flat disk + BEM for the tip

The team's modal model ("A non-linear dynamics treatise of optomechanical driving of a bath
of modes", April 2026) describes the flat disk as $N\approx15$ Bessel modes
$\psi_i(r)=J_0(\zeta_ir/R)/J_0(\zeta_i)$, $J_1(\zeta_i)=0$, with amplitudes $x_i=\eta_i(R)$,
van der Waals mode-interaction tensors $\chi_{ijk},\chi_{ijkl}$, and optical / opto-thermal
driving through $\sum_ix_i$. Its parameter table gives $R=3$ mm, $d=10$–$20$ nm,
$\alpha_{\text{vdW}}=2.6\times10^{-24}$ m⁵/s², displacements $\sim$ nm, $\Omega\sim$ kHz, $\Gamma\sim$ Hz.

Consequences for the estimates above: $a/h\sim0.1$, $Q\sim10^3$, shortest modal wavelength
$\approx0.4$ mm. The bulk solution is smooth; the bulk does not need a fine mesh or BEM.

Proposed split, with a cut at $r_c$ just inside the wedge:

- **Modes → tip:** thickness (or potential) at the cut,
  $h_c(t)=d+\sum_ix_i\,\psi_i(r_c)$.
- **Tip → modes:** volume flow rate $Q(t)$ from the disk into the tip. At linear order
  (derivation to be checked) it enters every mode equally,

$$\ddot x_i+\Gamma_i\dot x_i+\Omega_i^2x_i=\dots-\frac{\dot Q(t)}{\pi R^2},$$

  and requires adding the uniform mode $\ddot x_0=-\dot Q/(\pi R^2)$ (mean thickness), which
  the modal model currently lacks.
- **Time-scale separation:** crossing time of a $\sim30\ \mu$m tip region is $\sim30\ \mu$s,
  against modal periods $\gtrsim0.45$ ms. To first approximation the tip follows the modes
  quasi-statically.

Stages: (1) static tip equilibrium $h(s)$ under vdW + capillarity + optical force, as a
function of $h_c$ and optical power; (2) quasi-static tip map fed into the modal ODEs
(real optomechanical shift instead of linear $Gx$, flux $Q$); (3) fully dynamic BEM tip
co-simulated with the modes, only if the tip shows fast internal dynamics.

Open: slow convergence of a zero-flux modal basis when $Q\ne0$ (needs a lifting function);
opto-thermal (fountain) forcing is not potential flow and needs a separate thermal model;
whether the modal model should also carry the kinetic nonlinearities ($\eta u$ in
continuity, advection), which are the same order as the cubic vdW term.

## Literature pointers

- Grilli & Subramanya (1994), *Quasi-singular integrals in the modeling of nonlinear water
  waves in shallow water*, Eng. Anal. Bound. Elem. 13, 181–191 — BEM in high-aspect-ratio
  domains, near-singular integration. Open-access conference version: *Nonlinear wave
  modeling in very shallow water* (1993), WIT Trans. Modelling & Simulation 1,
  doi:10.2495/BE930131. Notes from that version:
  - 2D planar, gravity only, free-space log kernel with nodes on bottom and side walls.
  - Near-singular treatment is switched on by $d'=d/l$ (node-to-element distance over element
    length): modified Telles transformation for $0.05\le d'<1$, Lutz method for $d'<0.05$.
  - Domain aspect ratio 28–45; node spacing $0.1$–$0.5$ of the local depth, i.e. they still
    resolve the depth. Runs stop when the energy error reaches 1.5%.
  - Does not address conditioning at $\Delta s\gg h$, surface tension, or aspect ratios near ours.
- Grilli & Horrillo (1997), *Numerical generation and absorption of fully nonlinear periodic
  waves*, J. Eng. Mech. — absorbing beach (surface pressure + piston) in a BEM wave tank.
- Krishnasamy, Rizzo & Liu (1994), *Boundary integral equations for thin bodies*, IJNME 37 —
  degeneracy of the conventional BIE for thin bodies; later work by Liu et al. on thin
  shell-like domains and analytic near-singular integration.
- Shi et al. (2023), spectral coupled BEM — small BEM domain inside a cheap outer model.
- Hou, Lowengrub & Shelley (1994), *Removing the stiffness from interfacial flows with
  surface tension*, J. Comput. Phys. 114.
- Serre (1953); Green & Naghdi (1976) — fully nonlinear weakly dispersive model.

## Open questions

1. Actual $h_0$, $\alpha_{\text{vdW}}$, hence $\ell_c/h$.
2. Number of periods needed (tens, hundreds, full ring-down)?
3. Expected $a/h$.
4. Wedge tip rounding radius; treatment of the meniscus in the pedestal corner.
5. Equilibrium profile $h(s)$ on the curved substrate (separate, easier first milestone).
