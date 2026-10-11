"""Business Skills: how-to guidance the skills profile's model loads on demand.

Each Skill is one ``SKILL.md`` in the repository's trusted ``skills`` directory:
a front matter of exactly ``name``, ``description`` and ``version``, then a body
of steps.  The server reads and checks every file once, at startup; nothing is
installed from the network, an upload or model output, and nothing is re-read
while the service runs.  A Skill only tells the model how to go about a task:
identity, permissions, budgets, approval and the SQL checks stay with the
server whatever its text says.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
import re


SKILL_DIR = Path(__file__).resolve().parents[1] / "skills"
MAX_SKILL_LOADS = 2
# In UTF-8 bytes, not characters: two bodies at this size still fit the
# 24,000-byte context next to the worst case of everything else.
MAX_SKILL_BODY_BYTES = 2_400
# The index of every description stays in the system message, which has
# little room left.
MAX_SKILL_DESCRIPTION_CHARS = 20
# Starts a loaded Skill's system message; QUERYSHIELD_DATA marks untrusted data.
SKILL_MESSAGE_PREFIX = "QUERYSHIELD_SKILL"
_NAME = re.compile(r"[a-z][a-z0-9-]{2,23}")
_VERSION = re.compile(r"v[1-9][0-9]{0,2}")
_FIELDS = frozenset({"name", "description", "version"})
_FENCE = "---"


class SkillFileError(ValueError):
    """A Skill file the server refuses to run with; the message names the file, never its text."""

    code = "invalid_skill_file"


@dataclass(frozen=True)
class Skill:
    name: str
    description: str
    version: str
    body: str
    # Of the file's bytes, recorded with every load.
    sha256: str

    @property
    def version_tag(self) -> str:
        return f"{self.name}@{self.version}"

    @property
    def message(self) -> str:
        """The system message that carries the body once the model loaded it."""

        return f"{SKILL_MESSAGE_PREFIX} name={self.name}; version={self.version}; server_guidance\n{self.body}"


@dataclass(frozen=True)
class SkillLibrary:
    """The checked Skills, sorted by name."""

    skills: tuple[Skill, ...]

    @property
    def ids(self) -> frozenset[str]:
        return frozenset(skill.name for skill in self.skills)

    @property
    def versions(self) -> tuple[str, ...]:
        """What a run's RunConfig.skill_versions records: each Skill's name and version."""

        return tuple(skill.version_tag for skill in self.skills)

    def index(self) -> dict[str, str]:
        return {skill.name: skill.description for skill in self.skills}

    def get(self, name: str) -> Skill:
        return next(skill for skill in self.skills if skill.name == name)


def load_skill_library(directory: Path = SKILL_DIR) -> SkillLibrary:
    """Read and check every Skill; one bad entry fails the whole library, none is skipped."""

    entries = sorted(directory.iterdir()) if directory.is_dir() else []
    if not entries:
        raise SkillFileError(f"{directory.name}: no Skill directories")
    return SkillLibrary(tuple(_read_skill(entry) for entry in entries))


def _read_skill(entry: Path) -> Skill:
    path = entry / "SKILL.md"
    if not path.is_file():
        raise SkillFileError(f"{entry.name}: not a directory with a SKILL.md")
    raw = path.read_bytes()
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise SkillFileError(f"{entry.name}/SKILL.md: not UTF-8") from exc
    fields, body = _front_matter(text, entry.name)
    if fields["name"] != entry.name or not _NAME.fullmatch(fields["name"]):
        raise SkillFileError(f"{entry.name}/SKILL.md: name must match its directory and [a-z][a-z0-9-]{{2,23}}")
    if not _VERSION.fullmatch(fields["version"]):
        raise SkillFileError(f"{entry.name}/SKILL.md: version must look like v1")
    if len(fields["description"]) > MAX_SKILL_DESCRIPTION_CHARS:
        raise SkillFileError(f"{entry.name}/SKILL.md: description exceeds {MAX_SKILL_DESCRIPTION_CHARS} characters")
    if not body or len(body.encode("utf-8")) > MAX_SKILL_BODY_BYTES:
        raise SkillFileError(f"{entry.name}/SKILL.md: body must be 1 to {MAX_SKILL_BODY_BYTES} UTF-8 bytes")
    if "QUERYSHIELD_" in body:
        raise SkillFileError(f"{entry.name}/SKILL.md: body must not contain a server marker")
    return Skill(fields["name"], fields["description"], fields["version"], body, sha256(raw).hexdigest())


def _front_matter(text: str, label: str) -> tuple[dict[str, str], str]:
    """Exactly three ``key: value`` lines between two ``---`` lines, then the body.

    Not YAML: three plain strings need no type coercion, anchors or tags.
    """

    lines = text.split("\n")
    if "\r" in text or lines[0] != _FENCE or _FENCE not in lines[1:]:
        raise SkillFileError(f"{label}/SKILL.md: must start with a front matter between --- lines (LF only)")
    end = lines.index(_FENCE, 1)
    fields: dict[str, str] = {}
    for line in lines[1:end]:
        key, separator, value = line.partition(": ")
        if not separator or key not in _FIELDS or key in fields or not value or value != value.strip():
            raise SkillFileError(f"{label}/SKILL.md: front matter lines must be name, description and version, once each")
        fields[key] = value
    if set(fields) != _FIELDS:
        raise SkillFileError(f"{label}/SKILL.md: front matter lines must be name, description and version, once each")
    return fields, "\n".join(lines[end + 1:]).strip()


__all__ = [
    "MAX_SKILL_BODY_BYTES",
    "MAX_SKILL_DESCRIPTION_CHARS",
    "MAX_SKILL_LOADS",
    "SKILL_DIR",
    "SKILL_MESSAGE_PREFIX",
    "Skill",
    "SkillFileError",
    "SkillLibrary",
    "load_skill_library",
]
