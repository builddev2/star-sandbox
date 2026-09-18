# Graphics and resolution upgrade

*2026-09-18.  Approved in conversation; sections 1–3 reviewed one at a time.*

The sandbox draws hard-edged pixel art: three offset flat circles per body, a
flat white disc with a stepped halo for the Sun, 300 single-pixel stars, one-pixel
aliased trails, a non-antialiased system font, and one internal pixel per logical
point -- a 2x2 block on the Retina panel it runs on.  This design replaces the
pixel-art pipeline with smooth, shaded, antialiased rendering at native Retina
resolution, without changing the architecture: cached procedural sprites drawn
into one software surface, a `DisplayContext` that is the only thing that knows
what a window is, and every invariant in CLAUDE.md intact.

## Findings the design rests on

Measured on the 16" MacBook Pro (macOS 26, 3456x2234 panel), 2026-09-18.

- **pygame 2.6.1 cannot open a high-DPI window.**  The string `HIGHDPI` does not
  appear in its `display.c`, `window.c` or `_sdl2/video.pyx`.  No flag, hint or
  path reaches device pixels.
- **pygame-ce 2.5.8 can.**  `pygame.Window(size, resizable=True,
  allow_high_dpi=True)` plus `pygame._sdl2.video.Renderer(window)`:
  `renderer.coordinates_from_window((1152, 720))` reports `(2304, 1440)`;
  fullscreen reports `(3456, 2168)`.  The window *surface* stays in points; only
  the renderer sees pixels, so the frame is uploaded through a streaming
  `Texture`.
- **It is fast enough.**  A realistic frame at 2304x1440 -- fill, 600 stars, 15
  trails in 8 antialiased chunks, 15 alpha sprites, 30 labels, `texture.update`,
  `present` -- costs 7.8 ms median, 10.3 ms p95 (draw 3.5, upload 3.4).  At
  fullscreen the pixel count is 2.25x that.  With the 12 ms physics budget
  saturated (ten years a second) fullscreen Retina lands near 40 fps, which is why
  the resolution toggle survives (see `G` below).
- **It is stable.**  Resize, fullscreen and back, and dropping every reference
  then pumping events for a second: no crash.  The segfault CLAUDE.md records is
  specific to `Window.from_display_module()`, which this design never calls.
- **It works on the dummy driver**, so the test suite keeps running headless with
  a software renderer.
- **`import pygame` is unchanged.**  pygame-ce is a drop-in fork.  `aaline`,
  `aalines` and `aacircle` exist (with `width` on `aaline`/`aacircle`, not on
  `aalines`); `aapolygon` and `aaellipse` do not.
- **The default font is secretly small.**  `pygame.font.Font(None, 18)` scales
  the built-in face by 0.6875, giving a 12 px line.  Atkinson Hyperlegible
  Regular at size 12 gives a 15 px line and is *narrower* (528 px against 623 for
  the status-bar hint), so the interface's smallest window does not grow.

## 1. Display and resolution

Files: `src/app/config.py`, `src/app/input.py`, `src/app/main.py`,
`src/app/state.py`, `src/app/ui.py`, `src/render/hud.py`, `requirements.txt`,
`run.command`.

**Dependency.**  `requirements.txt` says `pygame-ce>=2.5.8`.  `run.command`
recreates `.venv` when it finds the old package: pygame and pygame-ce install
into the same `pygame/` directory and cannot coexist, so the launcher runs
`pip uninstall -y pygame` before installing requirements whenever
`pip show pygame` succeeds.  CLAUDE.md's command block is updated to say so.

**`DisplayContext` owns three objects.**

- `self.window`: one `pygame.Window`, `resizable=True`, `allow_high_dpi=True`,
  created once per process.  Fullscreen is `window.set_fullscreen(True)` /
  `window.set_windowed()`; presets set `window.size`.
- `self.renderer`: `Renderer(window, vsync=True)`.  The only thing that reads
  device pixels.
- `self.texture`: a streaming `Texture` the size of the internal buffer,
  rebuilt by `_adopt` whenever that size changes.

`self.surface` is a plain software `Surface((width, height))`.  Everything
downstream draws exactly as it does now.  `present()` is
`texture.update(surface)`, `renderer.clear()`, `texture.draw()`,
`renderer.present()`.  `pygame.display.set_mode` is gone; `self.display` is
gone; `_scaled`, `origin` and the letterbox are gone -- the texture always fills
the window, and the GPU does any scaling with linear filtering.
`pygame.display.get_desktop_sizes()` still serves the presets (it needs only
`pygame.display.init()`).

**`s` inverts its meaning.**  It was points per internal pixel (1, 2, 3 -- the
coarse grids).  It becomes **internal pixels per point**:

- `ctx.native` is the ratio the renderer reports,
  `coordinates_from_window(window.size)[0] / window.size[0]`, rounded to an
  int and floored at 1.  2 on a Retina display, 1 elsewhere and on the dummy
  driver.  Read from the renderer after every open, never guessed.
- `GRID_CHOICES` is gone.  `usable_grids` is `(native, 1)` when `native > 1`,
  else `(1,)`.  `G` cycles it: a **Retina / Standard** toggle.  `set_grid` refuses
  anything not in `usable_grids`.
- `internal_size(w, h, s) = (w * s, h * s)`.  `font_px(s) = TARGET_TEXT_POINTS *
  s`.  `min_internal(s) = (MIN_INTERNAL[0] * s, MIN_INTERNAL[1] * s)`.
  `grid_fits` becomes trivially true and is removed along with its tests; the
  window floor is `MIN_WINDOW = MIN_INTERNAL` in points as before.
- `upscale` (points per internal pixel) is `1 / s`.  Nothing outside tests reads
  it; keep it for them.

**Mouse.**  `to_internal(pos) = (pos[0] * s, pos[1] * s)`; `map_event` scales
`pos` and `rel` the same way; `mouse_pos` uses it.  `_identity` is `s == 1`.

**Resize.**  `pygame.Window` raises `WINDOWSIZECHANGED`, not `VIDEORESIZE`.
`input.handle_event` listens for `WINDOWSIZECHANGED` (its size is `event.x`,
`event.y`) and does what the `VIDEORESIZE` branch did.  The once-a-frame safety
poll in `step_once` compares `ctx.window.size` with `ctx.window_size`.
`resize()` keeps the push-back-to-the-floor behaviour by assigning
`window.size`.

**Sizes.**  Nothing else changes.  `widgets.Metrics` derives every interface
size from the font, and every world pixel constant scales by
`height / scale.REF_HEIGHT`, so at `s = 2` the interface is the same physical
size with twice the pixels and the drawn-radius floors are the same physical
size too.

**Text size.**  `TARGET_TEXT_POINTS` becomes 12 and is documented as the font's
point size at `s = 1` (the old 18 was a default-font size that rendered at 12
px).  `MIN_FONT_PX` goes; nothing shrinks the font any more.

**`G` in the interface.**  `hud.KEY_HELP` reads `("G", "resolution")`; the
`view.grid` control's tooltip reads "Retina / standard resolution"; it is
enabled when `len(ctx.usable_grids) > 1` (unchanged); `active` when
`ctx.s == ctx.native and ctx.native > 1`.  The control id stays `view.grid` so
the parity test and `NEEDS_DISPLAY` are untouched.

**Shutdown.**  `App.run` calls `ctx.close()` (which drops the texture and
renderer and calls `window.destroy()`) before `pygame.quit()`.

**Invariant kept.**  Nothing anywhere calls `Window.from_display_module()`.
The CLAUDE.md bullet about `pygame._sdl2.video.Window` is rewritten to say
exactly that: the context owns the one `Window`; `from_display_module()` is the
call that crashes.

## 2. Bodies: shaded spheres and looks

Files: new `src/render/shade.py`, `src/render/sprites.py`, `src/render/draw.py`,
`src/body.py`, `src/world.py`, `src/scenarios.py`, `src/save.py`,
`src/edits.py`.

### 2.1 `shade.py` -- one shader for every planet, moon and fragment

Pure numpy; the only pygame it touches is `pygame.surfarray` /
`pygame.image.frombuffer` to hand back a `SRCALPHA` surface.  Entry point:

```python
def sphere(px, lit_angle, tones, style, seed, atmosphere=None, glow=None) -> Surface
```

For a disc of radius `px` (surface `2*px+1` square, centre `px`):

1. **Coverage alpha.**  `alpha = clip(px + 0.5 - r, 0, 1)` where `r` is each
   pixel's distance from the centre: an antialiased rim at every size.  For
   `px < 2` the shader still runs; `draw.py`'s `set_at` branch is removed and a
   2 px sprite is what a distant body is.
2. **Normals and lighting.**  `z = sqrt(1 - (r/px)^2)` gives the sphere normal.
   The light vector is the star direction in the plane, **tilted
   `LIGHT_TILT = 40°` toward the viewer**: `L = (cos a cos t, -sin a cos t,
   sin t)`.  A true top-down view lit in-plane is a permanent half-moon; the
   tilt lights ~70 % of the disc, the same kind of cheat the drawn-radius floor
   is, and the docstring says so.  Diffuse is `max(n·L, 0)` with the terminator
   softened by `smoothstep(-0.08, 0.25, n·L)`; the night side keeps
   `AMBIENT = 0.08`; limb darkening multiplies by `z ** 0.6`.
3. **Albedo.**  A seeded value-noise fBm (`shade.fbm(shape, seed, octaves=4)`,
   vectorised numpy, no new dependency) shaped by `style`:
   - `cratered`: mottled base between `dark` and `mid`; 6..12 crater rims
     (a bright ring with a dark floor) at seeded positions, radii 0.06..0.2 px.
   - `banded`: latitude bands (`y / px` through a seeded 1-D profile) blended
     between the three tones, warped by low-frequency noise, with one storm
     oval at a seeded latitude.
   - `cracked`: smooth base near `mid`; bright fractures where a ridged noise
     (`1 - |fbm|`) exceeds a threshold, in `light`.
   - `turbulent`: cellular convection -- high-frequency fbm thresholded into
     cells, `light` at cell centres, `dark` at boundaries.
   Tones are `Material.tones` unless a look overrides them (2.3).  The
   blackbody blend for hot bodies is applied to the three tones before shading,
   exactly where `_make_body` does it today.
4. **Atmosphere rim.**  When `atmosphere` (an RGB) is given: a Fresnel band
   `(1 - z) ** 3` scaled by 0.55, added to the colour on the lit side and
   fading to 0.15 on the night side, plus a soft halo one to three pixels
   outside the rim carried in alpha.
5. **Size cap.**  `SHADE_MAX_PX = 512`.  Above it the shader runs at 512 and
   the result is `smoothscale`d to the requested size.  `draw.py`'s
   `MAX_SPRITE_HEIGHTS` flat-circle fallback stays for bodies bigger than the
   screen.

`sprites._make_body` becomes a thin wrapper that resolves tones/style/
atmosphere from the material and look, applies the glow blend, and calls
`shade.sphere`.  `_draw_detail`, `_mask_to_circle`, `MID_OFFSET`,
`HIGHLIGHT_OFFSET`, `HIGHLIGHT_SCALE` and `DETAIL_THRESHOLD` are deleted.
`SpriteCache.get` gains a `look` argument that joins the cache key.  Buckets,
LRU and `glow_key` are unchanged.

### 2.2 Stars

`shade.star(px, colour) -> Surface`, still blitted with `BLEND_ADD`, falloff in
colour never alpha (that rule stands).  The disc is limb-darkened
(`0.4 + 0.6 * z`) in `palette.blackbody_rgb(T)`, so a red dwarf is red and a
blue giant blue with no extra code; faint granulation from fbm at 6 % contrast;
the corona is `GLOW_STRENGTH * (px / r) ** 2` outside the disc, clipped at
`GLOW_SCALE` radii, with three to five seeded soft streamers of 30 % that
strength.  The temperature for the colour is `world.temperature[i]`, which for
a star is pinned to `units.star_temperature(mass)`.

### 2.3 Looks

`Body.look` is a string or `None`, set in `Body.__init__` beside `rings`,
`spin`, `disk` and `seed`: appearance only, never read by physics.

`sprites.LOOKS` maps a look name to `(tones, style, atmosphere)`:

| look | tones | style | atmosphere |
|---|---|---|---|
| `mercury` | greys | cratered | none |
| `venus` | pale sulphur yellows | banded (cloud deck, low contrast) | (232, 214, 160) |
| `terrestrial` | ocean blue / land green-brown / cloud white | `terrestrial` | (120, 170, 255) |
| `mars` | rust / dark basin / pale cap | cratered | (220, 170, 130), faint |
| `moon` | greys | cratered | none |
| `jupiter` | cream / tan / rust bands | banded, with the storm oval in rust | (200, 180, 150) |
| `saturn` | pale gold bands | banded, low contrast | (210, 190, 150) |
| `uranus` | cyan | banded, very low contrast | (150, 220, 230) |
| `neptune` | deep blue | banded, one dark storm | (90, 130, 230) |

`terrestrial` is a fifth style in `shade.sphere`: continents where fbm exceeds
a threshold (land tone, with a darker mountain tone where it exceeds a higher
one), ocean elsewhere, white polar caps beyond `|y/px| > 0.85`, and cloud
streaks from a second, higher-frequency fbm at 55 % alpha.  Anything with a
look that is not in `LOOKS` draws as if `look` were `None`.

**Rules.**

- `scenarios.PLANETS` and `MOONS` gain a look column: Mercury `mercury`, Venus
  `venus`, Earth `terrestrial`, Mars `mars`, Jupiter `jupiter`, Saturn
  `saturn`, Uranus `uranus`, Neptune `neptune`, Moon `moon`; the Galilean moons
  and Titan `None` (their materials already read right).  `World.add` gains
  `look=None`.
- Anything the child throws, and every merge, graze, fragment and tidal
  debris, has `look = None` and draws from its material.
- `edits.set_material` and `cycle_material` set `look = None` before writing:
  the material is now what the body is made of.  A locked material refuses the
  whole call, look included, as today.
- `save.py` writes `look` and reads it with `rec.get("look")`, so files saved
  before this design still load.  `from_data` validates it as a string or
  `None` like every other field.
- Nothing else inherits, copies or clears a look.

### 2.4 Black hole and craft

Composition unchanged.  `_make_black_hole` and `_make_craft` are called at
`2 * px` and the result is `pygame.transform.smoothscale`d to the size the
caller asked for -- supersampling, because the disk rings are ellipses and the
hull is polygons, which pygame cannot antialias.  Both stay blitted normally,
never `BLEND_ADD`.  Cache keys unchanged.

### 2.5 Rings

`sprites.ring_surface(px, tones, seed)` returns **two** surfaces, `(back,
front)`.  The ring is a flattened annulus (`BH_DISK_SQUASH`-style squash of
0.30) from 1.4 to 2.3 planet radii, radial density from a seeded 1-D fbm with a
Cassini-style gap at 1.95 radii, alpha 0.35..0.85, tones from the planet's own
`light`/`mid`.  `back` holds the half with `y < centre`, `front` the other
half; the planet's shadow is cut out of `back` as a dark band along the
anti-star direction.  `draw.py` blits `back`, the sprite, then `front`.  The
cache holds both under one key.

## 3. Sky, trails, text, effects, chrome

### 3.1 Starfield (`render/starfield.py`)

Pre-rendered once per buffer size into `self.sky`, a surface the size of the
buffer, built in `__init__` (which `App._build_render_stack` already calls on
every size change, so nothing caches a size across frames).  Star count is
`round(300 * width * height / (640 * 360))`, floored at 100 and capped at 3000.
Magnitudes from a power law (`brightness = 40 + 215 * u ** 3`, `u` uniform);
colours from `palette.blackbody_rgb` at temperatures log-uniform over
3000..15000 K, desaturated 50 % toward grey; the brightest 4 % drawn as
`aacircle`s of radius 1..2 px with a 30 % halo, the rest as single pixels.  A
Milky Way haze: a diagonal band (seeded angle, width 0.35 of the diagonal) of
low-frequency fbm at 6..10 % brightness in a warm grey, laid down before the
stars.  `draw` blits `sky` up to four times at the wrapped parallax offset
instead of 300 `set_at`s.

### 3.2 Trails and previews

`draw._draw_trails` uses `pygame.draw.aalines` per chunk.  In `app/tools.py`
every `pygame.draw.lines` becomes `aalines`, every `line` becomes `aaline`, and
every `circle` with `width=1` becomes `aacircle`; filled circles stay.  Widths
stay 1 px (`aalines` has no width).  The C-int overflow guards around the
preview stay exactly where they are.  `render/effects.py` rings use `aacircle`.

### 3.3 Text (`render/pixelfont.py`)

The class name, module name and the `height` / `width` / `render` / `draw`
interface stay -- the docstring promised the implementation could be swapped
without touching a caller.  `PixelFont(px)` loads `assets/fonts/
AtkinsonHyperlegible-Regular.ttf` (OFL; `assets/fonts/OFL.txt` alongside) at
point size `px`, antialiased; if the file is missing it falls back to
`pygame.font.Font(None, round(px / 0.6875))` so the fallback renders at the
same height.  The module docstring is rewritten: it explains the fallback and
why antialiasing is now on.

### 3.4 Effects (`render/effects.py`)

Flash, sparks, plume, flare and swallow particles are drawn from a
`_soft_disc(radius, colour)` cache -- a radial-gradient `SRCALPHA` sprite,
alpha `(1 - r/R) ** 1.5`, blitted `BLEND_ADD` -- instead of `pygame.draw.circle`.
Shock rings use `aacircle`.  Velocities, lifetimes, `MAX_PARTICLES` and the
wall-clock rule are untouched.

### 3.5 Chrome (`render/widgets.py`, `render/hud.py`)

No redesign.  Curved icon strokes (`draw_icon` arcs and circles) use `aaline` /
`aacircle`; straight strokes and rectangles stay.  `Metrics` is untouched.

## 4. Testing

The suite is 1253 tests and must end at least that size.  Every test below is
mutation-checked before the plan ships: name the one-line deletion or edit
that makes it fail.

**Display (`tests/test_window.py`, `tests/test_config.py`, `tests/test_smoke.py`).**
The `DisplayContext` fixture stays `DisplayContext(size=(800, 600))` on the
dummy driver.  Rewritten or new:

- `native` is 1 on the dummy driver and `usable_grids == (1,)`.
- With `native` monkeypatched to 2 (a fixture that sets `ctx.native = 2` and
  re-adopts): `usable_grids == (2, 1)`; `set_grid(2)` gives a buffer exactly
  twice the window each way and a texture the same size; `to_internal((10, 7))
  == (20, 14)`; `map_event` scales `pos` and `rel`; `set_grid(3)` is refused
  and lands on 2; `cycle_grid` toggles 2 -> 1 -> 2.
- `font_px(2) == 2 * font_px(1)`; `min_internal(2) == 2 * min_internal(1)`.
- A resize rebuilds the texture to the new buffer size.
- `present()` runs without error on the dummy driver after a resize (the
  texture and surface agree).
- `handle_event` on a `WINDOWSIZECHANGED` event resizes the context and calls
  `state.on_resize`; a `VIDEORESIZE` event is ignored.
- `test_layout.py` parametrises over `s in (1, 2)` instead of `(1, 2, 3)`; the
  floor stays `config.min_internal(s)`.
- `test_layering.py` gains: `grep` of `src/` for `from_display_module` finds
  nothing.

**Shader (`tests/test_shade.py`, new).**

- The rim is antialiased: at `px = 20` the alpha at `r = px` is strictly between
  0 and 255, at `r = px - 2` it is 255, at `r = px + 2` it is 0.
- Lighting: with `lit_angle = 0` the pixel at `(+0.6 px, 0)` is brighter than
  the pixel at `(-0.6 px, 0)`; with `lit_angle = pi` the order flips.
- The night side is not black: the darkest pixel inside `r < 0.9 px` has
  luminance above 0.
- Limb darkening: at `lit_angle = 0` the pixel at `(0, 0.95 px)` is darker than
  at `(0, 0)`.
- Determinism: the same arguments give bit-identical surfaces; a different seed
  gives a different one.
- Every style, and every look in `LOOKS`, renders at `px = 3`, `px = 40` and
  `px = 600` (which exercises the `SHADE_MAX_PX` upscale) without error and
  with the right surface size.
- The atmosphere halo makes a pixel just outside the rim non-transparent when
  an atmosphere is given and transparent when not.
- `star`: the centre is brighter than the limb; a pixel at `1.5 px` is non-black
  (the corona) and one at `GLOW_SCALE * px + 2` is black; a 3000 K star's red
  channel exceeds its blue and a 20000 K star's blue exceeds its red.

**Looks (`tests/test_sprites.py`, `tests/test_save.py`, `tests/test_edits.py`,
`tests/test_scenarios.py`).**

- Earth in the solar-system scenario has `look == "terrestrial"`, Io has `None`.
- `SpriteCache.get` with `look="terrestrial"` and with `look=None` for the same
  material returns different surfaces, and the cache key differs.
- `set_material` clears the look; a locked material leaves both alone.
- A merge's remnant and a shatter's fragments have `look None` even when the
  target had one.
- Save round-trips `look`; a record without a `look` key loads with `None`.
- `ring_surface` returns two surfaces of equal size; the back has opaque
  pixels only in its top half and the front only in its bottom half.

**Sky, trails, text, effects.**

- `Starfield(640, 360).sky.get_size() == (640, 360)`; the count scales with
  area within the floor and cap; a `draw` covers every pixel of a blank surface
  with something non-`SPACE` at least once (the blits wrap).
- `PixelFont(12).height` is between 13 and 17 (the bundled face) and a
  rendered glyph contains a pixel that is neither the text colour nor
  transparent (antialiasing is on).  With the font path monkeypatched away,
  the fallback's height is within 3 px of the bundled face's.
- `effects._soft_disc(8, (255, 200, 100))` has alpha 255 at the centre and 0 at
  the corner, and the cache returns the same object twice.
- `test_draw.py`: the `set_at` branch is gone -- a body at `px = 1.5` blits a
  sprite (the renderer's cache has one entry after the draw).

**Smoke.**  `test_smoke.py` keeps driving `App.step_once` headless and adds one
frame at `s = 2` (monkeypatched native) to prove the whole loop, the texture
upload and `present` agree on sizes.

## 5. Rollout

One branch, `graphics-upgrade`, three phases in this order, each a commit
series that leaves the suite green:

1. **Display** (section 1) -- the dependency swap and `DisplayContext`, with
   the interface otherwise untouched, so the whole app runs at Retina on the
   old sprites.  The smoke suite proves it.
2. **Bodies** (section 2) -- `shade.py`, looks, stars, rings, hole and craft
   supersampling.
3. **Sky, trails, text, effects, chrome** (section 3).

After each phase the app is run for real (`python -m src.app.main`) and
screenshots at the four views used during review -- system, Earth, Sun,
Jupiter -- are compared against the ones taken before the work.  Phase 1 must
show identical composition at twice the pixels; phases 2 and 3 are judged by
eye and by the tests above.

CLAUDE.md is updated at the end of each phase: the commands block (pygame-ce,
the launcher's uninstall step), the display invariants (`s` is pixels per
point, the `from_display_module` rule, the resolution toggle), the shading
design decisions (the 40° light tilt, the size cap, looks and when they are
cleared, rings in two halves, supersampling for the hole and craft, the
pre-rendered sky).

## Out of scope, on purpose

- **Planet rotation** (scrolling the albedo with `Body.spin`).  It defeats the
  sprite cache and needs a per-frame compositor; a natural follow-up now that
  the albedo and the lighting are computed separately.
- **A GPU pipeline** (moderngl).  ~130 `pygame.draw` call sites in the HUD,
  widgets, tools and effects would move; `WorldRenderer` keeps the door open.
- **A HUD redesign**, new physics, or any change to what a control does.
- **Coarse pixel grids** (`s = 2, 3` in the old sense).  Pixel art is the
  thing being replaced.
