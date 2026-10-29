"""A2 multi-person hand association: wearer-priority check + temporal nearest neighbor + entity continuity."""
import numpy as np

from roboray.stages.association import associate_candidates


def _cand(cx, cy, size, side, score=0.9, seed=0):
    rng = np.random.default_rng(seed)
    j2 = np.zeros((21, 3), np.float32)
    j2[:, 0] = cx + rng.normal(0, 0.01, 21) * size
    j2[:, 1] = cy + rng.normal(0, 0.01, 21) * size
    j2[:, 2] = score
    j3 = np.zeros((21, 3), np.float32); j3[:, 2] = 0.4
    return {"bbox": np.array([cx - size / 2, cy - size / 2,
                              cx + size / 2, cy + size / 2], np.float32),
            "side": side, "score": score, "joints_2d": j2, "joints_3d_cam": j3}


def test_wearer_selected_over_smaller_upper_bystander():
    W, H = 640, 480
    frames = [[_cand(200, 400, 120, 0), _cand(430, 390, 120, 1),
               _cand(320, 100, 50, 0, seed=1)]] * 6      # bystander: small and high
    out = associate_candidates(frames, W, H)
    assert out["present"].all()
    # Wearer selected: joint-mean x should be near 200/430, not 320
    xs = out["joints_2d"][:, :, :, 0].mean(axis=2)
    assert abs(xs[:, 0].mean() - 200) < 30 and abs(xs[:, 1].mean() - 430) < 30
    assert out["switches"] == []


def test_bystander_intrusion_keeps_identity():
    W, H = 640, 480
    frames = [[_cand(200, 400, 120, 0), _cand(430, 390, 120, 1)]] * 5
    # Frames 6-8: bystander intrudes near the wearer's right hand (closer to the previous
    # position is the wearer itself)
    for k in range(3):
        frames.append([_cand(200, 400, 120, 0, seed=k),
                       _cand(430, 390, 120, 1, seed=k),
                       _cand(445, 395, 60, 1, seed=k + 2)])
    frames.append([_cand(200, 400, 120, 0), _cand(430, 390, 120, 1)])
    out = associate_candidates(frames, W, H)
    assert out["switches"] == []                          # bystander does not trigger a switch
    xs = out["joints_2d"][:, :, :, 0].mean(axis=2)
    assert abs(xs[:, 1].mean() - 430) < 30                # right-hand entity stays continuous


def test_wrist_cross_keeps_entity_continuity():
    """Crossed hands: x-order swaps but the entity stays continuous (an x-rank baseline
    would misjudge >= 1 switch)."""
    W, H = 640, 480
    frames = []
    for t in range(10):
        x1 = 200 + 24 * t          # left hand 200 -> 416
        x2 = 430 - 24 * t          # right hand 430 -> 214 (crosses around t≈5)
        frames.append([_cand(x1, 400, 120, 0, seed=t), _cand(x2, 390, 120, 1, seed=t + 1)])
    out = associate_candidates(frames, W, H)
    assert out["switches"] == []
    x_track0 = out["joints_2d"][:, 0, :, 0].mean(axis=1)
    assert (np.diff(x_track0) > 0).all()                 # slot 0 always tracks the original left entity


def test_first_frame_only_bystander_then_reroute():
    W, H = 640, 480
    frames = [[_cand(320, 100, 50, 0, seed=3)]]          # first frame has only the bystander
    frames += [[_cand(320, 100, 50, 0, seed=4),
                _cand(200, 400, 120, 0, seed=5), _cand(430, 390, 120, 1, seed=6)]] * 4
    out = associate_candidates(frames, W, H)
    assert len(out["switches"]) >= 1                     # wearer enters -> reselect (counted truthfully)
    xs = out["joints_2d"][:, :, :, 0].mean(axis=2)
    assert abs(xs[-1, 0].mean() - 200) < 30              # final state selects the wearer


def test_exclusivity_two_slots_never_bind_same_candidate():
    """When the right hand jumps far away, slot 1 must not silently rebind to the left-hand
    entity (exclusivity guard)."""
    W, H = 640, 480
    frames = []
    for t in range(4):   # two hands coexisting close together
        frames.append([_cand(300, 400, 120, 0, seed=t), _cand(330, 390, 120, 1, seed=t + 1)])
    # Frame 5: right hand jumps far away (outside the motion threshold) -> only left is matchable
    frames.append([_cand(300, 400, 120, 0, seed=9)])
    out = associate_candidates(frames, W, H)
    # Exclusivity: no frame may bind both slots to the same candidate index
    lab = out["labels"]
    for i in range(len(frames)):
        both = lab[i, 0] >= 0 and lab[i, 1] >= 0
        if both:
            assert lab[i, 0] != lab[i, 1], f"frame {i} both slots bound to candidate {lab[i]}"
