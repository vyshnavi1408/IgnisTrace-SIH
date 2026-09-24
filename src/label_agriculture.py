
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
import ee 


DNBR_DROP = 0.10

PRE_NBR_WOODY = 0.25          
PRE_DAYS, POST_DAYS = 16, 8
MIN_POST_SCENES = 1       
POSITIVE, NEGATIVE, ABSTAIN = 1, 0, -1     
WILDFIRE = 2                                

CLOUD_MAX = 40            
COLLECTION = "COPERNICUS/S2_SR_HARMONIZED"


BURNED_AREA = "MODIS/061/MCD64A1"
BA_TOL_DAYS = 6                 
BA_YEAR_DAYS = 366              
BA_QA_LAND_VALID_UNSHORTENED = 3  
                                  
BA_QA_MASK = 0b111
SCALE_M = 20              

RULE_INPUT_COLUMNS = ("nbr_pre", "nbr_post", "dnbr", "post_scenes", "agri_label",
                      "type", "ba_burn", "ba_seen",            
                      "s2_class")                              


VEG_FIRE_TYPE = 0


def windows(fire: pd.Timestamp) -> tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp]:
    """(start, fire, end) covering the pre and post look-back/forward windows."""
    return (pd.Timestamp(fire) - pd.Timedelta(days=PRE_DAYS),
            pd.Timestamp(fire),
            pd.Timestamp(fire) + pd.Timedelta(days=POST_DAYS))


def decide(dnbr, post_scenes, drop: float = DNBR_DROP,
           min_scenes: int = MIN_POST_SCENES, producer_type=None,
           pre=None, woody: float = PRE_NBR_WOODY):
    """The rule, as a pure function so it can be tested without Earth Engine.

    Three conditions, no more, and all of them on data this file already fetches:
      * the producer's `type` has to say vegetation fire (gate: a flare dents NBR too);
      * a drop of `drop` in a window holding >= `min_scenes` clear scenes makes it a burn;
      * the LEVEL of nbr before the burn then splits crop clearing (herbaceous) from
        wildfire (woody). `pre=None` -- no usable pre window -- never invents a wildfire,
        it stays agriculture, because that is the class the scar evidence actually supports.
    `producer_type=None` (no `type` column in the file) means no gate, so this still runs on
    files that lack it.
    """
    if producer_type is not None and not pd.isna(producer_type):
        if int(producer_type) != VEG_FIRE_TYPE:
            return ABSTAIN                 
    if post_scenes is None or (isinstance(post_scenes, float) and np.isnan(post_scenes)):
        return ABSTAIN
    if pd.isna(dnbr) or post_scenes < min_scenes:
        return ABSTAIN                      
    if dnbr < drop:
        return NEGATIVE                     # looked, saw no scar: not a vegetation fire
    if pre is not None and not pd.isna(pre) and pre >= woody:
        return WILDFIRE                     # scar + woody fuel
    return POSITIVE                         # scar over herbaceous fuel (or fuel unknown)


# ============================================================
# Earth Engine
# ============================================================
QA_CLOUD_SHADOW = (1 << 10) | (1 << 11)     # 3072. NOT int("1100000000", 2), which is 768
                                            # (bits 8-9) and masks nothing relevant --
                                            # with that bug clouds survive and a cloudy
                                            # "no change" reads as a confident NEGATIVE.


def _require_ee():
    """Authenticate or fail with the instruction, not a raw EEException.

    Shared by fetch() and --probe because both are entry points a person can hit first,
    and an unauthenticated client raises "client library not initialized" from deep
    inside a size() call, which reads like a bug in the query rather than a missing login.
    """
    import ee
    try:
        ee.Initialize(project='ignistrace-sih')                             # idempotent when already authenticated
    except Exception as exc:
        raise RuntimeError(
            "Earth Engine isn't authenticated here. Run `earthengine authenticate` once "
            "in a terminal (it opens a browser), then retry.\n"
            f"  ee.Initialize said: {str(exc)[:160]}") from exc


def _prep(img):
    """QA60 cloud + cloud-shadow mask, then NBR = (NIR - SWIR) / (NIR + SWIR).

    The copyProperties line is the whole function, not a footnote. `.map()` hands back a
    NEW image whose properties are empty, and `filterDate()` keys on `system:time_start` --
    so a filterDate placed after the map matches nothing and reports an empty collection.
    Every pre/post window in this file is filtered on the mapped collection, which is why a
    run over 60 clear-sky January dates came back "no post imagery" 60 times: not weather,
    not clouds, not the account. Measured by --probe as `mapped 5, pre 0, post 0`, which is
    arithmetically impossible for a real window and is the signature of this bug.
    """
    bad = img.select("QA60").bitwiseAnd(QA_CLOUD_SHADOW)
    nbr = (img.updateMask(bad.eq(0))
              .normalizedDifference(["B8", "B12"])
              .rename("nbr"))
    return nbr.copyProperties(img, ["system:time_start", "system:index", "CLOUDY_PIXEL_PERCENTAGE"])


def _collection(fire: pd.Timestamp, pts, cloud_max: int = CLOUD_MAX):
    """The image collection spanning both look windows, already cloud-cut, NOT yet masked.

    Deliberately returned before _prep: every consumer filters dates on this, and `.map()`
    strips properties, so a mapped collection must never be the thing a filterDate is asked
    of. The mask is applied per window inside _window(), after its date filter.

    cloud_max=100 disables the whole-scene filter entirely, which is what --probe uses to
    find out whether CLOUDY_PIXEL_PERCENTAGE is the step that empties the collection.
    """
    import ee
    s, _, e = windows(fire)
    ic = (ee.ImageCollection(COLLECTION)
          .filterBounds(pts)
          .filterDate(s.isoformat(), (e + pd.Timedelta(days=1)).isoformat()))
    if cloud_max < 100:
        ic = ic.filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", cloud_max))
    return ic


def probe(fire, pts, cloud_max: int = CLOUD_MAX, n_points: int | None = None) -> dict:
    """size() after each filter step, for ONE date. A zero says which step killed it.

    Read it like this: `bounds` zero means the dataset has no imagery over those points at
    all (wrong AOI, or the account can't see the collection); `window` zero with a non-zero
    `bounds` means the date arithmetic is wrong; `cloud` zero with a non-zero `window` means
    the CLOUDY_PIXEL_PERCENTAGE filter is what empties it -- raise --cloud-max to 100.
    """
    import ee
    f = pd.Timestamp(fire)
    s, _, e = windows(f)
    post_b = e + pd.Timedelta(days=1)
    base = ee.ImageCollection(COLLECTION).filterBounds(pts)
    win = base.filterDate(s.isoformat(), post_b.isoformat())
    # n_points is passed in, not asked of `pts`: an ee.FeatureCollection has no len().
    out = {"acq_date": f.date(), "points": int(n_points) if n_points is not None else -1,
           "bounds": base.size().getInfo(), "window": win.size().getInfo()}
    cld = win if cloud_max >= 100 else win.filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", cloud_max))
    out["cloud"] = cld.size().getInfo()
    mapped = cld.map(_prep)
    out["mapped"] = mapped.size().getInfo()
    (pre_a, pre_b), (post_a, _) = ranges(f)
    out["pre"] = mapped.filterDate(pre_a.isoformat(), pre_b.isoformat()).size().getInfo()
    out["post"] = mapped.filterDate(post_a.isoformat(), post_b.isoformat()).size().getInfo()
    return out



def ranges(fire):
    """(pre_start, pre_end_exclusive), (post_start, post_end_exclusive).

    Single source of truth on purpose: fetch() probes these same pairs with size()
    before asking for a reduction, so the probe can never disagree with the window it
    is guarding. An earlier version wrote the pre end as `fire - 1 day` in one place and
    `fire` (exclusive) in the other, which silently dropped a lone scene on the day
    before the fire.
    """
    s, f, e = windows(pd.Timestamp(fire))
    return (s, f), (f + pd.Timedelta(days=1), e + pd.Timedelta(days=1))


def _ba_collection(fire: pd.Timestamp, pts, span_days: int = 20):
    """The monthly burned-area images that could contain this fire (+/- `span_days`).

    A month-wide pad is deliberate: BurnDate is stored per calendar month, so a fire on the
    2nd can be mapped in the previous month's image and a tolerance of 6 days must be able to
    reach across the tile boundary.
    """
    import ee
    f = pd.Timestamp(fire)
    return (ee.ImageCollection(BURNED_AREA)
            .filterBounds(pts)
            .filterDate((f - pd.Timedelta(days=span_days)).isoformat(),
                        (f + pd.Timedelta(days=span_days + 1)).isoformat()))


def _ba_image(fire: pd.Timestamp, pts, kind: str):
    """One per-pixel 0/1 image: 'burn' = a burn mapped within BA_TOL_DAYS of this date,
    'seen' = the cell was processable at all. max() over the padded months: ANY month in
    the pad voting yes is enough. Same filter-then-map order as _window(), same reason."""
    import ee
    f = pd.Timestamp(fire)
    ic = _ba_collection(f, pts)
    if kind == "seen":
        return (ic.select("QA")
                .map(lambda q: q.bitwiseAnd(BA_QA_MASK)
                     .eq(BA_QA_LAND_VALID_UNSHORTENED).rename("ba_seen"))
                .max())
    doy = int(f.dayofyear)
    # bd.gt(0) is not decoration: 0 means UNBURNED, and for a fire on 3 January the bare
    # |0 - 3| <= 6 test would label every unburned pixel in India a wildfire.
    return (ic.select("BurnDate")
            .map(lambda img: _ba_near(img.select("BurnDate"), doy))
            .max())


def ba_flags(fire, pts):
    """ba_burn/ba_seen reduced over the date's points, or None if the asset did not resolve.

    None means UNREADABLE, never "no burn" -- which is why it is its own return value and
    not a zero. Same doctrine as the S2 windows: size() before any reduce, because a reduce
    over an empty collection is a bandless image and EE raises "Image has no bands".
    """
    import ee
    if _ba_collection(fire, pts).size().getInfo() == 0:
        return None
    burn = _ba_image(fire, pts, "burn")
    seen = _ba_image(fire, pts, "seen")
    return (burn.addBands(seen).rename(["ba_burn", "ba_seen"])   # rename on the COMBINED
            .reduceRegions(collection=pts,                       # image, same as build_query
                           reducer=ee.Reducer.mean(), scale=500))  # 500 m: native pixel


def _ba_near(bd, doy: int):
    """|BurnDate - doy| <= BA_TOL_DAYS, on a 366-day circle, and only for a real burn date."""
    import ee
    diff = bd.subtract(doy).abs()
    # .Or, not .or -- the latter is a Python keyword, and logicalOr does not exist on Image.
    near = diff.lte(BA_TOL_DAYS).Or(diff.gte(BA_YEAR_DAYS - BA_TOL_DAYS))    # wraps 31 Dec/1 Jan
    return bd.gt(0).And(near).rename("ba_burn")


def _window(ic, a: pd.Timestamp, b: pd.Timestamp, name: str, kind: str):
    """masked-NBR median() / count() of one window, built from a PRE-map collection.

    Never call this on an empty collection:
    EE reduces zero images to a BANDLESS image, `rename` then dies with
    "Image has no bands" (routine when the whole 8-day post window is cloud-filtered
    out, e.g. monsoon dates). Callers check .size() first -- see fetch()."""
    # date first, THEN _prep -- the order is the point. A filterDate on a mapped
    # collection silently matches nothing, because the mapped image has lost the
    # system:time_start property the filter reads.
    sel = ic.filterDate(pd.Timestamp(a).isoformat(), pd.Timestamp(b).isoformat()).map(_prep)
    return (sel.count() if kind == "count" else sel.median()).rename(name)


def build_query(fire: pd.Timestamp, pts: "ee.FeatureCollection", want_pre: bool = True,
                cloud_max: int = CLOUD_MAX):
    """The per-date request: NBR before, NBR after, and how many clear post scenes."""
    import ee
    f = pd.Timestamp(fire)
    ic = _collection(f, pts, cloud_max)
    (pre_a, pre_b), (post_a, post_b) = ranges(f)
    bands, names = [], []
    if want_pre:
        bands.append(_window(ic, pre_a, pre_b, "nbr_pre", "median"))
        names.append("nbr_pre")
    bands.append(_window(ic, post_a, post_b, "nbr_post", "median"))
    names.append("nbr_post")
    bands.append(_window(ic, post_a, post_b, "post_scenes", "count"))
    names.append("post_scenes")
    return (ee.Image(bands).rename(names)                    # rename on the COMBINED
            .reduceRegions(collection=pts,                    # image: an unnamed band
                           reducer=ee.Reducer.mean(),        # list would give band_1..n
                           scale=SCALE_M))                   # and the props would not match


def _points(part: pd.DataFrame):
    """EE points for a chunk of detections, carrying _i so results can be joined back.

    Do NOT rebuild this from `itertuples()`. pandas renames any column that starts with
    an underscore to a positional field (`_i` -> `_1`), so `r._i` raises
    AttributeError: 'Pandas' object has no attribute '_i'. Column access instead.
    """
    import ee
    ids = part["_i"].to_numpy()
    lat = part["latitude"].to_numpy(float)
    lon = part["longitude"].to_numpy(float)
    return ee.FeatureCollection([
        ee.Feature(ee.Geometry.Point([float(lon[k]), float(lat[k])]), {"_i": int(ids[k])})
        for k in range(len(part))])


def fetch(detections: pd.DataFrame, chunk: int = 400, verbose: bool = True,
          debug: bool = False, cloud_max: int = CLOUD_MAX) -> tuple[pd.DataFrame, pd.DataFrame]:
    """One region query per acquisition DATE, over all detections sharing it.

    Returns (stats, probes). `stats` carries a row for EVERY point, including the ones
    for which nothing was asked for, and a `s2_note` saying which case applies. That
    distinction is the whole point: a dropped row and a cloud-masked pixel both read as
    NaN downstream, but one means "no imagery" and the other means "no burn", and you
    need to be able to tell them apart when every label comes back -1.
    """
    _require_ee()
    d = detections.copy().reset_index(drop=True)
    d["_i"] = d.index
    rows, probes = [], []
    empty_post = empty_pre = masked = ba_dead = 0
    by_date = list(d.groupby("acq_date"))
    for n, (date, g) in enumerate(by_date, 1):
        f = pd.Timestamp(date)
        ids = [int(i) for i in g["_i"]]
        (pre_a, pre_b), (post_a, post_b) = ranges(f)
        ic = _collection(f, _points(g), cloud_max)
        # Two cheap size() probes decide whether a reduce is even legal to ask for:
        # median()/count() over ZERO images returns a bandless image and EE then
        # raises "Image has no bands" out of rename()/reduceRegions().
        pre_n = ic.filterDate(pre_a.isoformat(), pre_b.isoformat()).size().getInfo()
        post_n = ic.filterDate(post_a.isoformat(), post_b.isoformat()).size().getInfo()
        # MCD64A1 is sampled for EVERY date, even when the S2 post window is empty: the
        # point of a second opinion from a different sensor is that it does not share
        # S2's clouds. A missing pair means the asset did not resolve, NOT "no burn".
        ba_fc = ba_flags(f, _points(g))
        ba = {}
        if ba_fc is None:
            ba_dead += 1
        else:
            for ft in ba_fc.getInfo().get("features", []):
                p = ft.get("properties")
                if p and p.get("_i") is not None:
                    ba[int(p["_i"])] = (p.get("ba_burn"), p.get("ba_seen"))
        bb = [v[0] for v in ba.values() if v[0] is not None]
        ss = [v[1] for v in ba.values() if v[1] is not None]
        probes.append({"acq_date": f.date(), "points": len(ids),
                       "pre_scenes": pre_n, "post_scenes": post_n,
                       "ba_burn": round(float(np.mean(bb)), 3) if bb else None,
                       "ba_seen": round(float(np.mean(ss)), 3) if ss else None})
        if post_n == 0:                    # nothing to compare against -> no label, no crash
            empty_post += 1
            rows += [{"_i": i, "nbr_pre": None, "nbr_post": None, "post_scenes": 0,
                      "s2_note": "post window had no scene under the cloud filter",
                      "ba_burn": ba.get(i, (None, None))[0],
                      "ba_seen": ba.get(i, (None, None))[1]} for i in ids]
            continue
        if pre_n == 0:
            empty_pre += 1                 # post only: dnbr stays NaN -> ABSTAIN downstream
        got = set()
        for a in range(0, len(g), chunk):
            part = g.iloc[a:a + chunk]
            q = build_query(f, _points(part), want_pre=pre_n > 0, cloud_max=cloud_max)
            for ft in q.getInfo().get("features", []):
                p = ft.get("properties")
                if p and p.get("_i") is not None:
                    p["s2_note"] = "no clear pre window" if pre_n == 0 else "ok"
                    p["ba_burn"], p["ba_seen"] = ba.get(int(p["_i"]), (None, None))
                    got.add(int(p["_i"]))
                    rows.append(p)
        miss = [i for i in ids if i not in got]      # asked, pixel came back masked
        masked += len(miss)
        rows += [{"_i": i, "nbr_pre": None, "nbr_post": None, "post_scenes": None,
                  "s2_note": "pixel cloud/shadow-masked in every post scene",
                  "ba_burn": ba.get(i, (None, None))[0],
                  "ba_seen": ba.get(i, (None, None))[1]} for i in miss]
        if verbose and (n % 20 == 0 or n == len(by_date)):
            print(f"  {n}/{len(by_date)} dates  ({len(rows):,} points)", flush=True)
    if verbose:
        print(f"  why: {empty_post} of {len(by_date)} dates had NO post scene at all, "
              f"{empty_pre} had no pre scene, {masked} point(s) were masked in every post scene")
        if by_date and empty_post == len(by_date):
            print("  every date came back with zero scenes -> that is not weather, that is a\n"
                  "    filter or an AOI/date problem. Re-run with --probe 5 and compare the\n"
                  "    columns: the first one that hits 0 names the culprit. (--cloud-max 100\n"
                  "    removes the scene-level cloud filter if `window` is non-zero and\n"
                  "    `cloud` is zero.)")
        if ba_dead:
            print(f"  MCD64A1: no image on {ba_dead} of {len(by_date)} dates (asset unreadable"
                  " in this project, or date outside 2000-11..2026-07) -> ba columns stay empty")
        if debug and probes:
            print(pd.DataFrame(probes).head(40).to_string(index=False))
    return pd.DataFrame(rows), pd.DataFrame(probes)


def run_probe(detections: pd.DataFrame, n: int = 5,
              cloud_max: int = CLOUD_MAX) -> pd.DataFrame:
    """Probe the first n dates of a (possibly filtered) selection. Kept separate from
    main() so the plumbing -- the _i column _points() needs, one row per date -- is
    testable without an Earth Engine account."""
    rows = []
    for date, g in list(detections.groupby("acq_date"))[:n]:
        gg = g.reset_index(drop=True).copy()
        gg["_i"] = gg.index                       # _points() reads this; without it, KeyError
        rows.append(probe(date, _points(gg), cloud_max, len(gg)))
    return pd.DataFrame(rows)


def screen_score(df: pd.DataFrame) -> pd.Series:
    
    out = pd.Series(0.0, index=df.index)
    if "type" in df:
        out += (pd.to_numeric(df["type"], errors="coerce") == VEG_FIRE_TYPE).astype(float)
    cell = (np.round(df["latitude"].to_numpy() * 20).astype("int64").astype(str) + "_" +
            np.round(df["longitude"].to_numpy() * 20).astype("int64").astype(str))
    work = df.assign(_cell=cell)
    per_day = work.groupby(["_cell", "acq_date"]).size()
    per_cell = work.groupby("_cell").acq_date.nunique()
    
    rows = pd.MultiIndex.from_arrays([work["_cell"].to_numpy(), work["acq_date"].to_numpy()])
    crowded = per_day.reindex(rows).to_numpy()
    repeat = per_cell.reindex(work["_cell"].to_numpy()).to_numpy()
    out += (crowded >= 2).astype(float)
    out += (repeat <= 4).astype(float)
    assert len(crowded) == len(df) and not np.all(pd.isna(crowded)), "cell-day join lost its keys"
    return out


def prioritize(df: pd.DataFrame, n: int) -> pd.DataFrame:
    
    if n <= 0 or n >= len(df):
        return df
    sc = screen_score(df)
    work = df.assign(_score=sc.to_numpy())
    work["_rank"] = work.groupby("acq_date")._score.rank(ascending=False, method="first")
    best = work.groupby("acq_date")._score.max().rename("_best")
    
    dates = pd.DataFrame({"acq_date": best.index, "_best": best.to_numpy()})
    dates["_month"] = pd.to_datetime(dates["acq_date"]).dt.month
    dates["_drank"] = dates.groupby("_month")._best.rank(ascending=False, method="first")
    work["_drank"] = work["acq_date"].map(dates.set_index("acq_date")._drank)
    pick = work.sort_values(["_rank", "_drank"], kind="mergesort").head(n)
    return pick.drop(columns=["_score", "_rank", "_drank"])


def sample(df: pd.DataFrame, n: int) -> pd.DataFrame:
    
    if not n or n >= len(df):
        return df.reset_index(drop=True)
    k = max(1, len(df) // n)
    return df.iloc[::k].head(n).reset_index(drop=True)



def assign_agri_s2(detections: pd.DataFrame, stats: pd.DataFrame | None = None,
                   drop: float = DNBR_DROP, min_scenes: int = MIN_POST_SCENES,
                   verbose: bool = True, debug: bool = False,
                   cloud_max: int = CLOUD_MAX, woody: float = PRE_NBR_WOODY) -> tuple[pd.DataFrame, dict]:
    
    out = detections.copy().reset_index(drop=True)
    out["_i"] = out.index
    if stats is None:
        stats, _ = fetch(out, verbose=verbose, debug=debug, cloud_max=cloud_max)
    join_cols = [c for c in ("nbr_pre", "nbr_post", "post_scenes",
                             "ba_burn", "ba_seen", "s2_note")
                 if stats is not None and c in stats.columns]
    if stats is not None and len(stats):
        stats = stats.copy()
        stats["_i"] = pd.to_numeric(stats["_i"], errors="coerce").astype("Int64")
        for c in ("nbr_pre", "nbr_post", "post_scenes", "ba_burn", "ba_seen"):
            if c in stats:
                stats[c] = pd.to_numeric(stats[c], errors="coerce")
        out = out.merge(stats[["_i"] + join_cols], on="_i", how="left")
    for c in ("nbr_pre", "nbr_post", "post_scenes", "ba_burn", "ba_seen"):
        if c not in out:
            out[c] = np.nan
    if "s2_note" not in out:
        out["s2_note"] = "not fetched"
    if "dnbr" not in out:
        out["dnbr"] = out["nbr_pre"] - out["nbr_post"]
    if "type" in out.columns:
        ptype = pd.to_numeric(out["type"], errors="coerce")
        gated = ptype.notna() & (ptype.astype("Int64") != VEG_FIRE_TYPE).fillna(False)
    else:
        ptype, gated = None, pd.Series(False, index=out.index)
    pre_col = out["nbr_pre"] if "nbr_pre" in out else pd.Series(np.nan, index=out.index)
    out["agri_label"] = [decide(a, b, drop, min_scenes, t, p, woody)
                         for a, b, t, p in zip(out["dnbr"], out["post_scenes"],
                                               ptype if ptype is not None else [None] * len(out),
                                               pre_col)]
    out["s2_class"] = out.agri_label.map({POSITIVE: "agriculture", WILDFIRE: "wildfire",
                                          NEGATIVE: "no_scar", ABSTAIN: "abstain"})
    if gated.any():
        out.loc[gated & (out.agri_label == ABSTAIN), "s2_note"] = \
            f"producer type != {VEG_FIRE_TYPE} (not a vegetation fire)"
    
    has_burn = pd.Series(False, index=out.index)
    demote = pd.Series(False, index=out.index)
    conflict = pd.Series(False, index=out.index)
    if "ba_burn" in out and "ba_seen" in out:
        b = pd.to_numeric(out["ba_burn"], errors="coerce")
        sv = pd.to_numeric(out["ba_seen"], errors="coerce")
        has_burn = (b > 0).fillna(False)
        was_seen = (sv > 0).fillna(False)
        demote = (out.agri_label == WILDFIRE) & was_seen & ~has_burn
        conflict = (out.agri_label == NEGATIVE) & has_burn
        out.loc[demote, "agri_label"] = POSITIVE
        out.loc[demote, "s2_note"] = \
            "BA veto: cell observed, no burn mapped -> too small to be a wildfire event"
        out.loc[conflict, "agri_label"] = ABSTAIN
        out.loc[conflict, "s2_note"] = "conflict: MCD64A1 maps a burn where the S2 scar says none"
        out["s2_class"] = out.agri_label.map({POSITIVE: "agriculture", WILDFIRE: "wildfire",
                                              NEGATIVE: "no_scar", ABSTAIN: "abstain"})
    out = out.drop(columns=["_i"])

    ab = out.agri_label == ABSTAIN
    ps = out["post_scenes"]
    st = {"n": int(len(out)), "positive": int((out.agri_label == POSITIVE).sum()),
          "negative": int((out.agri_label == NEGATIVE).sum()),
          "abstain": int(ab.sum()), "dnbr_threshold": drop, "woody_nbr": woody,
          "wildfire": int((out.agri_label == WILDFIRE).sum()),
          "ba_confirmed": int(((out.agri_label == WILDFIRE) & has_burn).sum()),
          "ba_demoted": int(demote.sum()),
          "ba_conflict": int(conflict.sum()),
          "abstain_ba_conflict": int((ab & conflict).sum()),
          "ba_burn_on_agri": int(((out.agri_label == POSITIVE) & has_burn).sum()),
          "abstain_producer_type": int((ab & gated).sum()),
          "abstain_no_post_imagery": int((ab & ~gated & (ps == 0)).sum()),
          "abstain_pixel_masked": int((ab & ~gated & ps.isna()).sum()),
          "abstain_no_pre_imagery": int((ab & ~gated & (ps > 0) & out.dnbr.isna()).sum())}
    q = pd.to_numeric(out["dnbr"], errors="coerce").dropna()
    st["dnbr_measured"] = int(len(q))
    if verbose:
        n = max(st["n"], 1)
        print("\n" + "=" * 58 + "\nAGRI LABELS FROM SENTINEL-2 SCARS\n" + "=" * 58)
        st["wildfire"] = int((out.agri_label == WILDFIRE).sum())
        for k in ("n", "positive", "wildfire", "negative", "abstain"):
            print(f"  {k:<10s}: {st[k]:>7,}  ({st[k]/n:.1%})")
        print(f"  dnbr >= {drop} with >= {min_scenes} clear post scene -> positive")
        print(f"  why -1    : {st['abstain_producer_type']:,} producer says not a veg fire | "
              f"{st['abstain_no_post_imagery']:,} no post imagery | "
              f"{st['abstain_pixel_masked']:,} pixel masked | "
              f"{st['abstain_no_pre_imagery']:,} no pre imagery")
        if "ba_burn" in out:
            print(f"  ba (MCD64A1): {st['ba_confirmed']:,} wildfire confirmed | "
                  f"{st['ba_demoted']:,} demoted (cell seen, no burn) | "
                  f"{st['ba_conflict']:,} conflicts -> abstain | "
                  f"{st['ba_burn_on_agri']:,} big burns kept agri")
        scarred = out[(out.agri_label == POSITIVE) | (out.agri_label == WILDFIRE)]
        fq = pd.to_numeric(scarred["nbr_pre"], errors="coerce").dropna() if len(scarred) else pd.Series(dtype=float)
        if len(fq):
            lo = int((fq < woody).sum()); hi = int((fq >= woody).sum())
            print(f"  fuel nbr_pre (scarred rows only, n={len(fq):,}): median {fq.median():+.3f} "
                  f"p10 {fq.quantile(0.1):+.3f} p90 {fq.quantile(0.9):+.3f}")
            print(f"  split at {woody}: {lo:,} agriculture / {hi:,} wildfire")
            spread = float(fq.max() - fq.min())
            if hi == 0 or lo == 0:
                print(f"  -> the split puts EVERYTHING on one side of {woody}, so fuel is not"
                      "\n     separating anything here. Do not ship a wildfire class from this"
                      "\n     number; use land cover (WorldCover class 10 tree vs 40 cropland)"
                      "\n     as the gate instead, or move the split into the distribution:")
                print(f"        --pre-nbr {round(float(fq.median()), 2)}")
            elif fq.std() / max(abs(fq.mean()), 1e-9) < 0.25:
                print(f"  -> WARNING: nbr_pre is tight (CV {float(fq.std())/abs(float(fq.mean())):.2f}),"
                      " one blob rather than two\n     humps. A wildfire class cut from this is"
                      " fragile: quote the split in the write-up\n     and spot-check a few points"
                      " on a map before you trust the accuracy.")
        if len(q):
            print(f"  dnbr      : n={len(q):,} median {q.median():+.3f} "
                  f"p90 {q.quantile(0.9):+.3f} max {q.max():+.3f}")
            if st["positive"] == 0:
                suggest = max(0.03, round(float(q.quantile(0.95)), 3))
                print(f"  -> the LARGEST scar you have is {q.max():.3f} vs threshold {drop}."
                      f"\n     This is a threshold question. Try --drop {suggest}.")
        else:
            wet = 0.0
            if "acq_date" in out.columns:
                mo = pd.to_datetime(out["acq_date"], errors="coerce").dt.month.dropna()
                if len(mo):
                    wet = float(mo.isin([6, 7, 8, 9]).mean())
            if wet > 0.5:
                print(f"  -> dnbr has no values and {wet:.0%} of this selection is Jun-Sep. That is"
                      "\n     the monsoon: narrow it to --months 11,12,1,2,3,4 before anything else.")
            else:
                print("  -> dnbr has NO values and this selection is NOT monsoon-heavy, so clouds do"
                      "\n     not explain it. Do not touch --drop and do not re-pick the season --"
                      "\n     read the filter ladder, which names the filter that empties the window:")
                print("        <same flags> --probe 5      (6 size() calls per date, no queries)"
                      "\n        bounds == 0 everywhere  -> your EE project cannot read " + COLLECTION +
                      "\n        window > 0, cloud == 0  -> the scene filter: re-run --cloud-max 100")
        if st["negative"] == 0 and st["positive"] == 0:
            print("  WARNING: no labels at all -> nothing to train the classes with.")
        print("=" * 58)
    return out, st


def assert_no_s2_leak(feature_cols) -> None:
    bad = sorted(set(map(str, feature_cols)) & set(RULE_INPUT_COLUMNS))
    if bad:
        raise ValueError("LEAK GUARD: these Sentinel-2 columns made the labels and cannot "
                         "be model inputs: " + ", ".join(bad))


def main() -> None:
    ap = argparse.ArgumentParser(description="Sentinel-2 agricultural-burn weak labels.")
    ap.add_argument("--csv", default="data/MODIS_archive_cleaned.csv")
    ap.add_argument("--out", default="outputs/agri_s2_labels.csv")
    ap.add_argument("--box", default=None,
                    help="lat0,lat1,lon0,lon1 to narrow the run, e.g. Andhra: 12.4,19.1,76.0,85.1")
    ap.add_argument("--months", default=None, help="comma list, e.g. 1,2,3,4 for the AP peak")
    ap.add_argument("--drop", type=float, default=DNBR_DROP)
    ap.add_argument("--limit", type=int, default=0,
                    help="N rows spread across the file, for a smoke test")
    ap.add_argument("--cloud-max", type=int, default=CLOUD_MAX,
                    help="max scene CLOUDY_PIXEL_PERCENTAGE; 100 disables the scene filter")
    ap.add_argument("--pre-nbr", type=float, default=PRE_NBR_WOODY,
                    help="nbr_pre at/above this is woody fuel -> wildfire instead of agriculture")
    ap.add_argument("--probe", type=int, default=0,
                    help="N dates: print size() after each filter step, then stop")
    ap.add_argument("--debug", action="store_true",
                    help="print scenes found per date -- answers 'why is everything -1'")
    ap.add_argument("--dry-run", action="store_true", help="print the plan, touch nothing")
    a = ap.parse_args()

    df = pd.read_csv(a.csv)
    df["acq_date"] = pd.to_datetime(df["acq_date"]).dt.normalize()
    if a.box:
        la0, la1, lo0, lo1 = [float(x) for x in a.box.split(",")]
        df = df[df.latitude.between(la0, la1) & df.longitude.between(lo0, lo1)]
    if a.months:
        ms = [int(x) for x in a.months.split(",")]
        df = df[df.acq_date.dt.month.isin(ms)]
    if a.limit:
        # Score BEFORE subsetting: the neighbour counts are a property of the whole
        # selection, so a score recomputed on 60 rows would find no crowded evenings at all.
        sc_all = screen_score(df)
        df = prioritize(df, a.limit)      # NOT head(): head() is four monsoon dates here
        n_strong = int((sc_all.loc[df.index] >= 2).sum())
        print(f"  --limit {a.limit}: took the {n_strong} evenings that look most like crop "
              f"clearing (crowded cell, one-off spot),\n"
              f"  one per date first, so the smoke test still visits {df.acq_date.nunique()} "
              f"dates. Ordering spends the\n  queries where a scar is plausible; it never "
              f"sets a label -- the scar does.")
    df = df.reset_index(drop=True)

    nd = df.acq_date.nunique()
    print(f"{len(df):,} detections on {nd} dates "
          f"({df.acq_date.min().date()}..{df.acq_date.max().date()}) "
          f"-> {nd} queries + {2 * nd} size probes")
    monsoon = float(df.acq_date.dt.month.isin([6, 7, 8, 9]).mean()) if len(df) else 0.0
    if monsoon > 0.5:
        print(f"  WARNING: {monsoon:.0%} of these detections are Jun-Sep, when Sentinel-2 "
              f"is monsoon-clouded\n           and almost every pixel abstains by design. "
              f"Add --months 11,12,1,2,3,4.")

    if a.dry_run:
        s, f, e = windows(df.acq_date.min())
        print(f"  sample window for {f.date()}: {s.date()}..{f.date()} pre, "
              f"{(f + pd.Timedelta(days=1)).date()}..{e.date()} post")
        print(f"  dataset COPERNICUS/S2_SR_HARMONIZED, bands B8/B12, QA60 mask, "
              f"scale {SCALE_M} m, cloud<={CLOUD_MAX}%, dnbr>={a.drop}")
        print(f"  lon {df.longitude.min():.2f}..{df.longitude.max():.2f}  "
              f"lat {df.latitude.min():.2f}..{df.latitude.max():.2f}")
        print("  months present: " + ", ".join(
            f"{m}:{c}" for m, c in sorted(df.acq_date.dt.month.value_counts().items())))
        print("  dry run: nothing requested")
        return

    if a.probe:
        n = min(a.probe, df.acq_date.nunique())
        print(f"\nprobing {n} date(s) against {COLLECTION}, cloud-max {a.cloud_max}")
        try:
            _require_ee()
        except RuntimeError as exc:
            print(f"\n{exc}")                         # same clean exit as the label path
            return
        pb = run_probe(df, n, a.cloud_max)
        print(pb.to_string(index=False))
        zero_bounds = int((pb.bounds == 0).sum())
        print(f"\n  bounds==0 on {zero_bounds}/{len(pb)} dates -> "
              + ("the dataset has NO imagery over that AOI in these windows: check --box and "
                 "that your Earth Engine project can read " + COLLECTION
                 + " (a quota/billing-disabled project returns empty, not an error)."
                 if zero_bounds == len(pb) else
                 "bounds are fine; look at which later column goes to 0."))
        print(f"  window>0 but cloud==0 on {int(((pb.window>0)&(pb.cloud==0)).sum())} dates -> "
              f"the scene cloud filter is the culprit: re-run with --cloud-max 100")
        return

    lab, st = assign_agri_s2(df, drop=a.drop, debug=a.debug, cloud_max=a.cloud_max,
                             woody=a.pre_nbr)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    lab.to_csv(a.out, index=False)
    with open(os.path.splitext(a.out)[0] + "_manifest.json", "w") as fh:
        json.dump(st, fh, indent=2)
    print(f"wrote {a.out} ({len(lab):,} rows)")


if __name__ == "__main__":
    main()
