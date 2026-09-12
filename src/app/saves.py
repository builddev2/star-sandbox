"""The three slot files.

``saves/slot1.json`` .. ``slot3.json`` in a ``saves/`` directory beside
``run.command``, so a parent can find them and back them up.  A write goes
to ``slotN.json.tmp`` and is moved into place with ``os.replace``, so a
crash mid-write leaves the old save rather than half of a new one.

Listing reads each file's header -- name, when, a summary -- and never
deserialises the world; a file that cannot be read is listed as unreadable
rather than raised, because the dialog still has to draw the other two.
"""

import json
import os
from dataclasses import dataclass

from .. import save

SLOTS = 3
UNREADABLE_NAME = "Can't read"


@dataclass(frozen=True)
class SlotInfo:
    """What the dialog shows for a filled slot."""

    slot: int
    name: str
    saved_at: str
    summary: dict
    readable: bool


def default_directory():
    """``saves/`` at the repository root: two levels up from this package."""
    here = os.path.dirname(os.path.abspath(__file__))       # src/app
    return os.path.join(os.path.dirname(os.path.dirname(here)), "saves")


class SlotStore:
    def __init__(self, directory):
        self.directory = directory

    def path(self, slot):
        if not 1 <= int(slot) <= SLOTS:
            raise ValueError("slot %r is not 1..%d" % (slot, SLOTS))
        return os.path.join(self.directory, "slot%d.json" % int(slot))

    def writable(self):
        """Can a save land here?  Creates the directory if it can."""
        try:
            os.makedirs(self.directory, exist_ok=True)
        except OSError:
            return False
        return os.access(self.directory, os.W_OK)

    def list(self):
        out = []
        for slot in range(1, SLOTS + 1):
            if not os.path.exists(self.path(slot)):
                out.append(None)
                continue
            try:
                data = self.read(slot)
                out.append(SlotInfo(slot, str(data["name"]), str(data["saved_at"]),
                                    dict(data["summary"]), True))
            except (OSError, ValueError, KeyError, TypeError):
                out.append(SlotInfo(slot, UNREADABLE_NAME, "", {}, False))
        return out

    def read(self, slot):
        """The whole document.  Raises OSError, or a ValueError subclass
        (json's decode error, or SaveFormatError for a version we do not know)."""
        with open(self.path(slot), encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict) or data.get("version") != save.VERSION:
            raise save.SaveFormatError("not a version %d save" % save.VERSION)
        return data

    def write(self, slot, data):
        os.makedirs(self.directory, exist_ok=True)
        final = self.path(slot)
        tmp = final + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f)
        os.replace(tmp, final)

    def rename(self, slot, name):
        data = self.read(slot)
        data["name"] = str(name)
        self.write(slot, data)
