"""A world and a session as plain data, and back again.

The physics layer's half of saving: numpy arrays and Body objects in, a dict
of lists, floats, strings and bools out, and the reverse.  Nothing here
touches a file -- ``app/saves.py`` does that -- and nothing here imports
pygame; ``tests/test_layering.py`` asserts it.

**``from_data`` validates everything before it touches the world.**  A load
that fails halfway would leave the child with half a solar system, so every
record is parsed into a checked tuple first and the world is only cleared
once all of them are good.

**Kind is loaded as saved, never re-derived from mass.**  A black hole and a
craft are one-way kinds (``edits._refresh_kind``) that mass alone cannot
reproduce.

**Temperature is loaded as saved and not settled.**  It is state, and a
planet he heated on purpose must still be hot when he comes back.
"""

from dataclasses import asdict, dataclass, replace
from datetime import datetime

from . import materials
from .body import KIND_BLACK_HOLE, KIND_CRAFT, KIND_DEBRIS, KIND_PLANET, KIND_STAR
from .maneuver import Maneuver

#: Bump when the file layout changes.  A file with any other version is
#: refused rather than guessed at.
VERSION = 1

KINDS = frozenset((KIND_PLANET, KIND_STAR, KIND_DEBRIS, KIND_BLACK_HOLE, KIND_CRAFT))


class SaveFormatError(ValueError):
    """The data is not a save this build can read."""


@dataclass(frozen=True)
class Session:
    """Everything the application saves that is not the world.

    Defined here rather than in ``app/state.py`` so the physics layer never
    has to see ``AppState``.  ``au_across`` is the camera's view width in AU
    -- never ``log_zoom``, which is pixels per AU and would show a different
    stretch of sky in a different window.
    """

    scenario: str
    time_index: int
    paused: bool
    show_trails: bool
    real_size: bool
    show_labels: bool
    explain_enabled: bool
    show_body_list: bool
    selected_id: object
    centre: tuple
    follow_id: object
    au_across: float


# --- writing ------------------------------------------------------------------------

def to_data(world, session, name, scenario_label, now=None):
    """The whole save as a JSON-able dict: header, world, session."""
    bodies = []
    for i in range(world.n):
        b = world.bodies[i]
        bodies.append({
            "id": int(b.id),
            "name": str(b.name),
            "kind": b.kind,
            "material": b.material.key,
            "mass": float(world.mass[i]),
            "radius": float(world.radius[i]),
            "pos": [float(world.pos[i, 0]), float(world.pos[i, 1])],
            "vel": [float(world.vel[i, 0]), float(world.vel[i, 1])],
            "temperature": float(world.temperature[i]),
            "born_day": float(b.born_day),
            "event_id": int(b.event_id),
            "no_collide_until": float(b.no_collide_until),
            "seed": int(b.seed),
            "rings": bool(b.rings),
            "spin": float(b.spin),
            "disk": float(b.disk),
            "look": b.look,
            "locked_fields": sorted(b.locked_fields),
            "fuel_dv": float(b.fuel_dv),
            "plan": [{"due_time": float(n.due_time), "prograde": float(n.prograde),
                      "radial": float(n.radial)} for n in b.plan],
        })
    when = datetime.now() if now is None else now
    sess = asdict(session)
    sess["centre"] = [float(session.centre[0]), float(session.centre[1])]
    return {
        "version": VERSION,
        "name": str(name),
        "saved_at": when.isoformat(timespec="minutes"),
        "summary": {"scenario_label": str(scenario_label),
                    "days": float(world.time), "bodies": int(world.n)},
        "world": {"time": float(world.time), "next_id": int(world.next_id),
                  "bodies": bodies},
        "session": sess,
    }


# --- reading ------------------------------------------------------------------------

def _opt_int(value):
    return None if value is None else int(value)


def _parse_body(rec):
    """One record into a checked dict of the exact types ``World.add`` and
    ``Body`` want.  Raises SaveFormatError on anything off."""
    try:
        kind = rec["kind"]
        if kind not in KINDS:
            raise SaveFormatError("unknown kind %r" % (kind,))
        material = materials.BY_KEY.get(rec["material"], materials.ROCK)
        pos, vel = rec["pos"], rec["vel"]
        return {
            "id": int(rec["id"]),
            "name": str(rec["name"]),
            "kind": kind,
            "material": material,
            "mass": float(rec["mass"]),
            "radius": float(rec["radius"]),
            "pos": (float(pos[0]), float(pos[1])),
            "vel": (float(vel[0]), float(vel[1])),
            "temperature": float(rec["temperature"]),
            "born_day": float(rec["born_day"]),
            "event_id": int(rec["event_id"]),
            "no_collide_until": float(rec["no_collide_until"]),
            "seed": int(rec["seed"]),
            "rings": bool(rec["rings"]),
            "spin": float(rec["spin"]),
            "disk": float(rec["disk"]),
            "look": None if rec.get("look") is None else str(rec["look"]),
            "locked_fields": frozenset(str(f) for f in rec["locked_fields"]),
            "fuel_dv": float(rec["fuel_dv"]),
            "plan": tuple(Maneuver(float(n["due_time"]), float(n["prograde"]),
                                   float(n["radial"])) for n in rec["plan"]),
        }
    except SaveFormatError:
        raise
    except (KeyError, TypeError, ValueError, IndexError) as exc:
        raise SaveFormatError("bad body record: %s" % exc) from exc


def _parse_session(s):
    try:
        centre = s["centre"]
        return Session(
            scenario=str(s["scenario"]),
            time_index=int(s["time_index"]),
            paused=bool(s["paused"]),
            show_trails=bool(s["show_trails"]),
            real_size=bool(s["real_size"]),
            show_labels=bool(s["show_labels"]),
            explain_enabled=bool(s["explain_enabled"]),
            show_body_list=bool(s["show_body_list"]),
            selected_id=_opt_int(s["selected_id"]),
            centre=(float(centre[0]), float(centre[1])),
            follow_id=_opt_int(s["follow_id"]),
            au_across=float(s["au_across"]),
        )
    except (KeyError, TypeError, ValueError, IndexError) as exc:
        raise SaveFormatError("bad session: %s" % exc) from exc


def from_data(data, world):
    """Replace ``world``'s contents from ``data`` and return the Session.

    Validates first, mutates second: on any SaveFormatError the world is
    exactly as it was.  Does not settle temperature and does not refresh
    the integrator -- the caller owns ``Simulation.refresh()``.
    """
    if not isinstance(data, dict) or data.get("version") != VERSION:
        raise SaveFormatError("not a version %d save" % VERSION)
    try:
        w = data["world"]
        records = w["bodies"]
        time = float(w["time"])
        next_id = int(w["next_id"])
        session_data = data["session"]
        if not isinstance(records, list):
            raise SaveFormatError("bodies is not a list")
    except (KeyError, TypeError, ValueError) as exc:
        raise SaveFormatError("bad save: %s" % exc) from exc
    if len(records) > world.capacity:
        raise SaveFormatError("%d bodies, capacity %d" % (len(records), world.capacity))
    parsed = [_parse_body(rec) for rec in records]
    ids = [p["id"] for p in parsed]
    if len(set(ids)) != len(ids):
        raise SaveFormatError("duplicate body id")
    session = _parse_session(session_data)

    world.clear()
    world.time = time
    for p in parsed:
        body = world.add(p["name"], p["material"], p["mass"], p["radius"],
                         p["pos"], p["vel"], kind=p["kind"], seed=p["seed"],
                         temperature=p["temperature"])
        body.id = p["id"]
        body.born_day = p["born_day"]
        body.event_id = p["event_id"]
        body.no_collide_until = p["no_collide_until"]
        body.rings = p["rings"]
        body.spin = p["spin"]
        body.disk = p["disk"]
        body.look = p["look"]
        body.locked_fields = p["locked_fields"]
        body.fuel_dv = p["fuel_dv"]
        body.plan = p["plan"]
    world.rekey(next_id)

    present = set(ids)
    return replace(session,
                   selected_id=session.selected_id if session.selected_id in present else None,
                   follow_id=session.follow_id if session.follow_id in present else None)
