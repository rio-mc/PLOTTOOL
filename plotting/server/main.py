from __future__ import annotations

import base64
import io
from dataclasses import asdict, is_dataclass
from typing import Any, Dict, List, Literal, Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel, Field

import matplotlib
matplotlib.use("Agg")  # must come before Figure import

from matplotlib.figure import Figure
from matplotlib import font_manager

from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware

from starlette.responses import PlainTextResponse
from starlette.types import ASGIApp, Receive, Scope, Send

from plotting.spec import PlotFamily, PlotSpec
from plotting.builder import build_series_data, draw
from plotting.plot_types import available_families, label_for_family, meta_for, types_for_family
from plotting.codegen import generate_plot_code


# -----------------------
# Request models
# -----------------------

class RenderRequest(BaseModel):
    spec: Dict[str, Any]

    format: Literal["png", "jpg", "pdf", "svg"] = "png"
    dpi: int = Field(300, ge=72, le=1200)
    jpg_quality: int = Field(95, ge=1, le=95)

    elev: Optional[float] = None
    azim: Optional[float] = None
    roll: Optional[float] = None  # fix typo


class CodegenRequest(BaseModel):
    spec: Dict[str, Any]


# -----------------------
# App creation
# -----------------------

app = FastAPI(title="Plotting Service")


# -----------------------
# Middleware: max body size
# -----------------------

class MaxBodySizeMiddleware:
    def __init__(self, app: ASGIApp, max_bytes: int) -> None:
        self.app = app
        self.max_bytes = max_bytes

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        received = 0

        async def limited_receive() -> dict:
            nonlocal received
            message = await receive()
            if message["type"] == "http.request":
                body = message.get("body", b"") or b""
                received += len(body)
                if received > self.max_bytes:
                    response = PlainTextResponse("Request body too large", status_code=413)
                    await response(scope, receive, send)
                    return {"type": "http.disconnect"}
            return message

        await self.app(scope, limited_receive, send)


app.add_middleware(MaxBodySizeMiddleware, max_bytes=1_000_000)


# -----------------------
# Middleware: rate limiting (optional but recommended)
# -----------------------

limiter = Limiter(key_func=get_remote_address, default_limits=["120/minute"])
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
app.add_middleware(SlowAPIMiddleware)


# -----------------------
# Helpers
# -----------------------

def _parse_spec(spec_dict: Dict[str, Any]) -> PlotSpec:
    try:
        return PlotSpec.from_dict(spec_dict).normalised()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid spec: {type(e).__name__}: {e}")


def _issues_as_strings(result: Any) -> List[str]:
    out: List[str] = []
    for iss in getattr(result, "issues", []) or []:
        series_index = getattr(iss, "series_index", None)
        msg = str(getattr(iss, "message", "") or "").strip()
        if not msg:
            continue
        if series_index is None:
            out.append(msg)
        else:
            s = int(series_index) + 1
            axis = str(getattr(iss, "axis", "") or "").strip()
            prefix = f"Series {s}" + (f" {axis}" if axis else "")
            out.append(f"{prefix}: {msg}")
    return out


def _plotmeta_to_dict(pm: Any) -> Dict[str, Any]:
    d = asdict(pm) if is_dataclass(pm) else dict(pm.__dict__)
    controls = d.get("controls") or []
    d["controls"] = [asdict(c) if is_dataclass(c) else dict(c.__dict__) for c in controls]
    fam = d.get("family")
    if hasattr(fam, "value"):
        d["family"] = fam.value
    return d


def _render_to_bytes(spec: PlotSpec, req: RenderRequest) -> tuple[bytes, str, List[str]]:
    fig = Figure()
    try:
        is_3d = bool(getattr(meta_for(spec.plot_type), "requires_z", False))
        ax = fig.add_subplot(111, projection="3d" if is_3d else None)

        result = build_series_data(spec)
        draw(ax, spec, result.series)

        if getattr(ax, "name", "") == "3d":
            if req.elev is not None or req.azim is not None or req.roll is not None:
                elev = float(req.elev if req.elev is not None else 30.0)
                azim = float(req.azim if req.azim is not None else -60.0)
                roll = float(req.roll if req.roll is not None else 0.0)
                try:
                    ax.view_init(elev=elev, azim=azim, roll=roll)
                except TypeError:
                    ax.view_init(elev=elev, azim=azim)

        fmt = req.format.lower()
        buf = io.BytesIO()

        save_kwargs: Dict[str, Any] = {"bbox_inches": "tight"}
        if fmt in ("png", "jpg"):
            save_kwargs["dpi"] = int(req.dpi)
        if fmt == "jpg":
            save_kwargs["format"] = "jpeg"
            save_kwargs["quality"] = int(req.jpg_quality)
        else:
            save_kwargs["format"] = fmt

        fig.savefig(buf, **save_kwargs)
        payload = buf.getvalue()

        mime = {
            "png": "image/png",
            "jpg": "image/jpeg",
            "pdf": "application/pdf",
            "svg": "image/svg+xml",
        }[fmt]

        return payload, mime, _issues_as_strings(result)

    finally:
        fig.clear()


# -----------------------
# Routes
# -----------------------

@app.get("/health")
def health():
    return {"ok": True}


@app.post("/render")
@limiter.limit("20/minute")  # apply per-endpoint limit
def render(req: RenderRequest):
    spec = _parse_spec(req.spec)
    try:
        payload, mime, issues = _render_to_bytes(spec, req)
        headers: Dict[str, str] = {}
        if issues:
            headers["X-Plot-Issues"] = " | ".join(issues)[:1500]
        return Response(content=payload, media_type=mime, headers=headers)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Render failed: {type(e).__name__}: {e}")


@app.post("/render_json")
@limiter.limit("20/minute")
def render_json(req: RenderRequest):
    spec = _parse_spec(req.spec)
    try:
        payload, mime, issues = _render_to_bytes(spec, req)
        return JSONResponse(
            {
                "mime": mime,
                "format": req.format.lower(),
                "payload_base64": base64.b64encode(payload).decode("ascii"),
                "issues": issues,
            }
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Render failed: {type(e).__name__}: {e}")


@app.post("/codegen")
def codegen(req: CodegenRequest):
    spec = _parse_spec(req.spec)
    try:
        code = generate_plot_code(spec)
        return JSONResponse({"code": code})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Codegen failed: {type(e).__name__}: {e}")


@app.post("/export/code")
def export_code(req: CodegenRequest):
    spec = _parse_spec(req.spec)
    try:
        code = generate_plot_code(spec)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Codegen failed: {type(e).__name__}: {e}")
    headers = {"Content-Disposition": 'attachment; filename="plot.py"'}
    return Response(content=code, media_type="text/x-python", headers=headers)


@app.get("/meta/families")
def meta_families():
    return JSONResponse([{"value": f.value, "label": label_for_family(f)} for f in available_families()])


@app.get("/meta/types")
def meta_types(family: str):
    try:
        fam = PlotFamily(family)
    except Exception:
        raise HTTPException(status_code=400, detail=f"Unknown family: {family}")

    out = []
    for pt in types_for_family(fam):
        pm = meta_for(pt)
        out.append({"value": pt.value, "label": getattr(pm, "label", pt.value), "meta": _plotmeta_to_dict(pm)})
    return JSONResponse(out)


@app.get("/meta/fonts")
def meta_fonts():
    names = sorted({f.name for f in font_manager.fontManager.ttflist if getattr(f, "name", None)})
    return JSONResponse(names)
