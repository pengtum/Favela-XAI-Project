"""Configuration loading and path resolution.

Every script of the repository obtains its paths and parameters from
``config.yaml`` at the repository root:

    from favela_xai.config import load_config
    cfg = load_config()
    out = cfg.output_dir(__file__)              # outputs/<folder of this script>/
    for site in cfg.sites:                       # the ten sites
        path = cfg.raw_path("buildings_2013")    # data/raw/...
        shap = cfg.product("shap_gpkg", site=site)
    n_trees = cfg.get("model_main.n_estimators")

Rules implemented here:
* Paths in ``config.yaml`` are relative to the repository root and are returned
  as absolute ``pathlib.Path`` objects. No absolute path is written in code.
* Outputs of a script located in ``<root>/<folder>/`` are written to
  ``outputs/<folder>/``.
* ``demo.enabled: true`` (or ``FAVELA_XAI_DEMO=1``) restricts the run to the
  sites and paths of the ``demo`` section.
* ``FAVELA_XAI_CONFIG`` may point to another YAML file with the same keys; its
  relative paths are then resolved against the folder of that file, so keep a
  copy of config.yaml at the repository root when editing parameters.
"""
from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any, Iterable

import yaml

# The repository root is two levels above src/favela_xai/.
_PACKAGE_DIR = Path(__file__).resolve().parent
REPO_ROOT = _PACKAGE_DIR.parents[1]
DEFAULT_CONFIG = REPO_ROOT / "config.yaml"

_MISSING = object()


class ConfigError(RuntimeError):
    """Raised when a required configuration entry or input file is missing."""


def _truthy(value: str | None) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


class Config:
    """Read-only view of ``config.yaml`` with path helpers.

    Parameters are read with ``cfg.get("section.key")`` or ``cfg["section"]``.
    """

    def __init__(self, data: dict, root: Path, source: Path):
        self._data = data
        self.root = Path(root).resolve()
        self.source = Path(source).resolve()
        env_demo = os.environ.get("FAVELA_XAI_DEMO")
        self.demo = _truthy(env_demo) if env_demo is not None else bool(self.get("demo.enabled", False))

    # ------------------------------------------------------------------ access
    def __getitem__(self, key: str) -> Any:
        return self._data[key]

    def __contains__(self, key: str) -> bool:
        return self.get(key, _MISSING) is not _MISSING

    def get(self, dotted: str, default: Any = _MISSING) -> Any:
        """Return a value by dotted key, e.g. ``cfg.get("model_main.n_estimators")``."""
        node: Any = self._data
        for part in dotted.split("."):
            if isinstance(node, dict) and part in node:
                node = node[part]
            else:
                if default is _MISSING:
                    raise ConfigError(f"'{dotted}' is not defined in {self.source.name}")
                return default
        return copy.deepcopy(node)

    def section(self, name: str) -> dict:
        """Return a copy of a top-level section as a plain dict."""
        value = self.get(name)
        if not isinstance(value, dict):
            raise ConfigError(f"'{name}' is not a section in {self.source.name}")
        return value

    # ------------------------------------------------------------------ sites
    @property
    def all_sites(self) -> list[str]:
        """Keys of the ten sites, in the order of config.yaml (alphabetical)."""
        return [s["key"] for s in self._data["sites"]]

    @property
    def sites(self) -> list[str]:
        """Sites to process: the ten sites, or ``demo.sites`` when demo mode is on."""
        if self.demo:
            return list(self.get("demo.sites"))
        return self.all_sites

    def site_info(self, site: str) -> dict:
        """Metadata of one site: key, name, latex, window, extract_records."""
        for s in self._data["sites"]:
            if s["key"] == site:
                return copy.deepcopy(s)
        raise ConfigError(f"unknown site '{site}'")

    def site_name(self, site: str) -> str:
        return self.site_info(site)["name"]

    def site_latex(self, site: str) -> str:
        return self.site_info(site)["latex"]

    def site_window(self, site: str) -> tuple[float, float, float, float]:
        xmin, ymin, xmax, ymax = self.site_info(site)["window"]
        return float(xmin), float(ymin), float(xmax), float(ymax)

    def display_order(self, name: str, sites: Iterable[str] | None = None) -> list[str]:
        """A manuscript row/panel order restricted to the sites being processed."""
        keep = set(self.sites if sites is None else sites)
        return [s for s in self.get(f"display_orders.{name}") if s in keep]

    # ------------------------------------------------------------------ paths
    def resolve(self, relative: str | os.PathLike) -> Path:
        """Absolute path of a repository-relative path."""
        p = Path(relative)
        return p if p.is_absolute() else (self.root / p).resolve()

    def raw_path(self, name: str, required: bool = True) -> Path | None:
        """Path of a raw input (``paths.raw.<name>``, or ``demo.raw.<name>`` in demo mode).

        With ``required=True`` a missing entry or file raises ConfigError with a
        pointer to data/README.md.
        """
        rel = None
        if self.demo:
            rel = self.get(f"demo.raw.{name}", None)
        if rel is None:
            rel = self.get(f"paths.raw.{name}", None)
        if rel is None:
            if required:
                raise ConfigError(
                    f"paths.raw.{name} is not set in {self.source.name}. "
                    "See data/README.md for where to obtain this input and set its path.")
            return None
        path = self.resolve(rel)
        if required and not path.exists():
            raise ConfigError(
                f"input not found: {path}\n(paths.raw.{name} in {self.source.name}; "
                "see data/README.md for the download link and the expected location)")
        return path

    @property
    def outputs_root(self) -> Path:
        key = "demo.outputs_root" if self.demo else "paths.outputs_root"
        return self.resolve(self.get(key))

    def output_dir(self, script_or_folder: str | os.PathLike, create: bool = True) -> Path:
        """Output folder of a script: outputs/<folder>/ for <root>/<folder>/<script>.py.

        ``script_or_folder`` is normally ``__file__``; a repository-relative
        folder such as ``"main/02_classification"`` is also accepted.
        """
        p = Path(script_or_folder)
        if p.suffix == ".py":
            p = p.resolve().parent
        elif not p.is_absolute():
            p = (self.root / p).resolve()
        else:
            p = p.resolve()
        try:
            rel = p.relative_to(self.root)
        except ValueError as exc:
            raise ConfigError(f"{p} is not inside the repository {self.root}") from exc
        out = self.outputs_root / rel
        if create:
            out.mkdir(parents=True, exist_ok=True)
        return out

    def product(self, name: str, site: str | None = None, create_parent: bool = False, **fmt: Any) -> Path:
        """Path of a shared intermediate product (``products.<name>``).

        ``{site}`` and other placeholders are filled from the keyword arguments.
        """
        template = self.get(f"products.{name}")
        if site is not None:
            fmt["site"] = site
        try:
            rel = template.format(**fmt)
        except KeyError as exc:
            raise ConfigError(f"products.{name} needs the placeholder {exc}") from exc
        path = self.outputs_root / rel
        if create_parent:
            path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def require_product(self, name: str, site: str | None = None, **fmt: Any) -> Path:
        """Like :meth:`product`, but raise a clear error if the file does not exist yet."""
        path = self.product(name, site=site, **fmt)
        if not path.exists():
            raise ConfigError(
                f"{path} does not exist. It is written by an earlier step of the pipeline "
                f"(products.{name} in {self.source.name}); run that step first.")
        return path

    def elevation_raster(self) -> Path:
        """User-supplied 5 m terrain model (not distributed; IPP on request)."""
        path = self.raw_path("elevation_raster", required=False)
        if path is None or not path.exists():
            raise ConfigError(
                "The 5 m LiDAR digital terrain model is not distributed with this repository; "
                "it is available from the Instituto Municipal de Urbanismo Pereira Passos (IPP) on "
                "request. Set paths.raw.elevation_raster in config.yaml to your local GeoTIFF "
                "(EPSG:31983).")
        return path

    def __repr__(self) -> str:
        mode = "demo" if self.demo else "full"
        return f"Config({self.source}, mode={mode}, sites={len(self.sites)})"


def _default_config_path() -> Path:
    """config.yaml of this repository.

    With the editable install of environment.yml the package lives in
    <root>/src/favela_xai, so <root>/config.yaml is found directly. Otherwise the
    current working directory and its parents are searched.
    """
    if DEFAULT_CONFIG.exists():
        return DEFAULT_CONFIG
    for folder in [Path.cwd(), *Path.cwd().parents]:
        candidate = folder / "config.yaml"
        if candidate.exists() and (folder / "src" / "favela_xai").is_dir():
            return candidate
    raise ConfigError("config.yaml not found; run from inside the repository or set FAVELA_XAI_CONFIG")


def load_config(path: str | os.PathLike | None = None) -> Config:
    """Load ``config.yaml`` (or ``$FAVELA_XAI_CONFIG``, or ``path``).

    The repository root is the folder that contains the configuration file, so
    every relative path in it is resolved against that folder.
    """
    if path is None:
        env = os.environ.get("FAVELA_XAI_CONFIG")
        path = Path(env) if env else _default_config_path()
    path = Path(path).resolve()
    if not path.exists():
        raise ConfigError(f"configuration file not found: {path}")
    with open(path, encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    return Config(data, root=path.parent, source=path)
