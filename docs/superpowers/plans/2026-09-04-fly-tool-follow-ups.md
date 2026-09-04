# Fly tool: what was left undone

The Fly tool shipped in the `fly-tool` merge on 2026-09-04. It works end to end --
verified by driving the real application, not only by tests -- but the session that built
it hit its monthly spend limit partway through, which killed four agents mid-flight and
cut the review process short. This is the honest list of what that cost, so none of it is
mistaken for finished.

Suite at merge: **1020 passing**, from 896 at the branch point. After the 2026-09-04
defect pass: **1032**.

> **Update, 2026-09-04.** Everything under "Live mutants" and most of the smaller
> deferred items are now fixed, each verified by re-applying the mutant and confirming it
> fails. What genuinely remains is the unreviewed commits, the deliberate limitations,
> and three cosmetic items. Struck-through entries are done.

## Unreviewed commits

Three commits never got their code review. They are tested and the application runs, but
nobody read them with fresh eyes:

| Commit | What it is |
|---|---|
| `973023a` | `FlyTool` and craft launching |
| `541b0dc` | the six-row panel, scrubbers, hotkeys, Hohmann button |
| `9c4082b` | the overlay, and the semi-major-axis transfer fix |

The fix round on the craft rendering (`80aad32`) also has no re-review, and no
whole-branch review was ever run.

## Live mutants: three rendering behaviours are untested  -- FIXED 2026-09-04

**All three are closed.** Each mutant below was re-applied and confirmed to fail the
suite. The third needed a second attempt: deleting the craft branch does *not* stop a
craft being drawn -- it falls through to the ordinary material-sphere path -- so counting
pixels proved nothing. The thermal-tile colour is the discriminator, since only the
shuttle sprite paints it. Kept below for the record of what was wrong.

<details><summary>original finding</summary>


These were found by a reviewer that deliberately broke the code and checked whether the
suite noticed. It did not. Each of these edits still leaves the suite green:

1. **The shuttle silhouette is never drawn.** Force `_make_craft`'s `px < CRAFT_DETAIL_PX`
   branch to always take, so a craft is always a plain dot and no polygon is ever drawn.
   The sprite tests assert only that the surface is wider than 8 px and its bounding box
   wider than 2 -- both true of a circle.
   *Kill it with:* a test pinning the shape -- the opaque bounding box longer along the
   nose axis than across it, or a `_CRAFT_TILE` pixel below the centreline and a
   `_CRAFT_HULL` one above.

2. **The sprite is never rotated to its heading.** Change `if heading:` to `if False:`.
   The cache test asserts only `turned is not first`, which holds because the cache *key*
   differs regardless of the pixels.
   *Kill it with:* comparing `pygame.image.tostring` between heading 0 and heading pi/2.

3. **The entire craft branch in `render/draw.py` is uncovered.** Deleting the whole
   `elif is_craft:` blit passes, and so does replacing the scaled floor with a bare
   unscaled pixel count -- so the project's "pixel constants scale by
   `internal_height / REF_HEIGHT`" invariant is asserted nowhere for `CRAFT_MIN_PX`.
   *Kill it with:* a headless `WorldRenderer.draw` test with a craft at two buffer
   heights, asserting the drawn size scales.

</details>

## Known limitations, deliberate rather than accidental

- **A transfer reaches the target's orbit, not a rendezvous.** The plan is two-body; the
  world is n-body. Over a 256-day transfer the craft drifts, so Earth-to-Mars arrives with
  the textbook delta-v (3,080 + 2,758 m/s) but about 0.6 AU from Mars. Real missions solve
  this with midcourse corrections, which would be the natural next feature -- and a good
  one, because it is exactly what real mission design does.
- **The fast end of the time dial is not deliverable.** At `10 yr/s` the HUD honestly
  reports `running at 1%`, so the 435-day wait for a launch window is a slow watch. This
  is the pre-existing timestep finding, not something the Fly tool introduced: a single
  global block timestep is forced to the tightest pair in the system. Fixing it
  (individual or hierarchical timesteps) is the single largest improvement available to
  this simulator.
- **Heading is world-frame velocity.** Correct for every state the Fly tool can currently
  reach, because craft are launched heliocentric by design. If planetary capture is ever
  added, a craft in orbit around a planet will point along the planet's heliocentric
  track instead of along its own orbit.
- ~~**`flyby` returns `e = inf`** for a stationary craft where the limit is `e = 1`.~~
  **Fixed 2026-09-04**: returns `e = 1.0`.
- ~~**`hohmann` has no zero guard** on `r1`/`r2`.~~ **Fixed 2026-09-04**: returns `None`,
  and `_plan_transfer` leaves the plan alone. The crash actually fired earlier, in
  `_mean_radius`, which is guarded too. `orbits.dominant_attractor` had the same
  zero-separation division and is fixed alongside.

## Smaller deferred items

- ~~`tests/test_craft.py`: the units round-trip assertion is a **tautology**.~~
  **Fixed 2026-09-04**: replaced with one that pins `AUD_TO_MS` against `AU_M /
  SEC_PER_DAY` and against a literal. Verified by setting the factor to `AU_KM /
  SEC_PER_DAY` -- the plausible slip -- which the test now catches.
- ~~`maneuver.closest_approach`: the refinement pass's resolution is unpinned.~~
  **Checked 2026-09-04**: did not reproduce -- the reported minimum is exact and really is
  a minimum. Pinned by a regression test rather than fixed.
- ~~`maneuver.closest_approach`: the `samples` argument is silently floored to 8.~~
  **Checked 2026-09-04**: did not reproduce -- a larger `samples` does find a closer
  approach. Pinned by a regression test.
- ~~`maneuver.flyby`: `side=-0.0` rotates clockwise.~~ **Fixed 2026-09-04**: the sign test
  is now `side >= 0.0` rather than `copysign`, which reads the sign bit.
- `maneuver.hohmann`: `max(0.0, wait)` is unreachable defensive code -- the sign branch
  already guarantees a non-negative wait.
- ~~`maneuver.plan_conics` assumes the plan is sorted by `due_time`.~~
  **Fixed 2026-09-04**: it sorts, so a caller may hand one over in any order.
- `units.craft_apparent_radius()` is a zero-logic wrapper around its own constant. If it
  never becomes kind- or zoom-dependent, delete it and use the constant.
- At the exact instant a launch window opens, an outward transfer reports a full synodic
  period (779.67 days) rather than 0. Measure-zero and semantically defensible; recorded
  so nobody rediscovers it as a bug.

## A process note worth keeping

Two bugs in this feature were found only by **running the application and looking at it**,
after the full suite was green:

- the Hohmann solver was aimed at Mars's *instantaneous* distance, so it missed by 1.9 AU;
- the craft's render floor was 9 px against the Sun's own cap of 6, so a shuttle drew
  wider than the Sun and swallowed the Earth it launched from.

Neither was catchable by the tests as written. The screenshots are what caught them.
