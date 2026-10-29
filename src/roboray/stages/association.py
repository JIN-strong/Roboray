"""A2 multi-person hand association: wearer-priority check + inter-frame nearest neighbor + entity continuity.

spec 2026-10-26 §3.3: left/right labels only break ties, never act as hard constraints —
the entity stays continuous when the wearer crosses hands or turns. ``switches`` counts
entity re-assignment events, not x-order changes (the old x-rank semantics remain in
fusion.associate_wearer for the no-candidates fallback).
"""

import numpy as np

__all__ = ["associate_candidates", "wearer_score"]


def wearer_score(cand: dict, width: float, height: float,
                 prior_center=(0.5, 0.85), prior_weight=1.0, size_weight=0.5) -> float:
    """Wearer-prior score: proximity to the lower-center of the frame (high) + bbox area
    (large) — in an ego view the wearer's hands are near and typically in the lower sides
    of the frame. Normalized to ~[0, 2] magnitude."""
    x1, y1, x2, y2 = cand["bbox"]
    cx, cy = (x1 + x2) / 2 / width, (y1 + y2) / 2 / height
    dist = float(np.hypot(cx - prior_center[0], cy - prior_center[1]))
    area = float(max(0.0, (x2 - x1)) * max(0.0, (y2 - y1))) / (width * height)
    return -prior_weight * dist + size_weight * min(1.0, area * 8.0)


def _match(prev_xy, cand, width, motion_thresh_px):
    """Center distance <= threshold -> distance; otherwise None (no match)."""
    x1, y1, x2, y2 = cand["bbox"]
    d = float(np.hypot((x1 + x2) / 2 - prev_xy[0], (y1 + y2) / 2 - prev_xy[1]))
    return d if d <= motion_thresh_px else None


def _motion_match(prev_xy, cands, slot, width, motion_thresh_px, motion_weight,
                  tie_tol, excluded=frozenset()):
    """Entity-slot nearest neighbor: composite cost motion_weight·dist + (1−score) is
    minimized subject to center distance <= threshold.

    Exclusivity: candidates already claimed by another slot (``excluded``) are skipped,
    so the two slots never bind the same candidate. Left/right labels only break ties:
    when the best candidate is on the opposite side from the slot and a same-side candidate
    is within ``tie_tol`` of it (near-tie), pick the lowest-cost same-side candidate — this
    avoids entity switches when x-order swaps during crossed hands; when the cost gap is
    clear, motion nearest-neighbor still decides (side is not a hard constraint).
    """
    matched = []
    for j, c in enumerate(cands):
        if j in excluded:
            continue
        d = _match(prev_xy, c, width, motion_thresh_px)
        if d is not None:
            matched.append((motion_weight * d + (1.0 - float(c.get("score", 0.5))), j, c))
    if not matched:
        return None
    matched.sort(key=lambda x: x[0])
    best_cost = matched[0][0]
    same_side = [x for x in matched
                 if x[0] <= best_cost + tie_tol and x[2].get("side") == slot]
    chosen = min(same_side, key=lambda x: x[0]) if same_side else matched[0]
    return chosen[1]


def associate_candidates(candidates, width, height,
                         prior_center=(0.5, 0.85), prior_weight=1.0,
                         size_weight=0.5, motion_weight=1.0,
                         tie_tol=15.0, reroute_margin=0.3):
    """Bind candidates frame by frame to the wearer's left/right "entity slots".

    - First frame (or when the previous frame has no entities): each slot picks the best
      by wearer_score (side tie-break).
    - Later frames: each entity slot takes the nearest neighbor in the candidates
      (minimal composite cost distance + motion_weight·(1−score) with center distance
      <= threshold; side tie-break on near-ties); unmatched slots are reselected this
      frame by the prior, counting one switch.
    - Wearer-priority reroute: when a slot is locked onto a low-quality (non-wearer)
      target and an unassigned candidate is clearly better by wearer_score
      (> reroute_margin), drop the low-quality target, reselect, and count a switch.
    Output dict: joints_2d (T,2,21,3), present (T,2), switches [(frame, slot)],
    labels (T,2) (candidate index in the current frame's list, −1 = unobserved — for QC).
    """
    t = len(candidates)
    out_j2 = np.zeros((t, 2, 21, 3), np.float32)
    out_present = np.zeros((t, 2), bool)
    labels = np.full((t, 2), -1, np.int32)
    switches: list = []
    motion_thresh = 0.15 * float(width)
    ents: list = [None, None]          # previous frame (cx, cy) per slot

    def _slot_score(j, slot):
        c = cands[j]
        return (wearer_score(c, width, height, prior_center, prior_weight, size_weight)
                + (0.05 if c.get("side") == slot else 0.0))

    for i, frame in enumerate(candidates):
        cands = list(frame or [])
        assigned = {}
        for slot in (0, 1):
            if ents[slot] is None or not cands:
                continue
            j = _motion_match(ents[slot], cands, slot, width, motion_thresh,
                              motion_weight, tie_tol,
                              excluded=frozenset(assigned.values()))
            if j is not None:
                assigned[slot] = j
        # Wearer-priority reroute: an established entity locked onto a low-quality target
        # while an unassigned candidate is clearly better -> reselect
        for slot in (0, 1):
            if slot not in assigned or ents[slot] is None:
                continue
            pool = [j for j in range(len(cands)) if j not in assigned.values()]
            if not pool:
                continue
            j_best = max(pool, key=lambda j: _slot_score(j, slot))
            if _slot_score(j_best, slot) > _slot_score(assigned[slot], slot) + reroute_margin:
                assigned[slot] = j_best
                switches.append((i, slot))
        # Unmatched slot -> reselect this frame by the prior (side tie-break)
        for slot in (0, 1):
            if slot not in assigned and cands:
                pool = [j for j in range(len(cands)) if j not in assigned.values()]
                if pool:
                    best = max(pool, key=lambda j: _slot_score(j, slot))
                    assigned[slot] = best
                    if ents[slot] is not None:
                        switches.append((i, slot))
        for slot, j in assigned.items():
            c = cands[j]
            out_j2[i, slot] = c["joints_2d"]
            out_present[i, slot] = True
            labels[i, slot] = j
            x1, y1, x2, y2 = c["bbox"]
            ents[slot] = ((x1 + x2) / 2, (y1 + y2) / 2)
    return {"joints_2d": out_j2, "present": out_present,
            "switches": switches, "labels": labels}
