"""MARS 19-joint skeleton definition for the kinematic Output Head.

Source: MARS (An & Ogras, ACM TECS 2021), Table 2 & Table 3; Kinect V2 joint
hierarchy. The 19 joints = Kinect V2 (25) minus the 6 hand joints
(HandLeft/Right, HandTipLeft/Right, ThumbLeft/Right).

Index order = Kinect-native order with the 6 hand joints removed, which matches
the per-joint ordering in Table 3 (SpineShoulder appears last as index 18).
If your label array uses a different order, rebuild PARENT from PARENT_BY_NAME
(see build_parent_from_names) — the parent map *by name* is order-independent.
"""

JOINT_NAMES = [
    "SpineBase",      # 0  (root)
    "SpineMid",       # 1
    "Neck",           # 2
    "Head",           # 3
    "ShoulderLeft",   # 4
    "ElbowLeft",      # 5
    "WristLeft",      # 6
    "ShoulderRight",  # 7
    "ElbowRight",     # 8
    "WristRight",     # 9
    "HipLeft",        # 10
    "KneeLeft",       # 11
    "AnkleLeft",      # 12
    "FootLeft",       # 13
    "HipRight",       # 14
    "KneeRight",      # 15
    "AnkleRight",     # 16
    "FootRight",      # 17
    "SpineShoulder",  # 18
]
N_JOINTS = 19
ROOT = 0

# parent index for each joint (-1 = root)
PARENT = [-1, 0, 18, 2, 18, 4, 5, 18, 7, 8, 0, 10, 11, 12, 0, 14, 15, 16, 1]

# topological order: a parent always appears before its children.
# IMPORTANT: integrate bones in THIS order, not in index order
# (SpineShoulder=18 is parent of 2/4/7 but has the last index).
TOPO = [0, 1, 10, 14, 18, 11, 15, 2, 4, 7, 12, 16, 3, 5, 8, 13, 17, 6, 9]

# symmetric bone pairs (childLeft, childRight) for the bone-symmetry prior
SYM_PAIRS = [(4, 7), (5, 8), (6, 9), (10, 14), (11, 15), (12, 16), (13, 17)]

# parent map by NAME (robust if the label order differs)
PARENT_BY_NAME = {
    "SpineBase": None, "SpineMid": "SpineBase", "SpineShoulder": "SpineMid",
    "Neck": "SpineShoulder", "Head": "Neck",
    "ShoulderLeft": "SpineShoulder", "ElbowLeft": "ShoulderLeft", "WristLeft": "ElbowLeft",
    "ShoulderRight": "SpineShoulder", "ElbowRight": "ShoulderRight", "WristRight": "ElbowRight",
    "HipLeft": "SpineBase", "KneeLeft": "HipLeft", "AnkleLeft": "KneeLeft", "FootLeft": "AnkleLeft",
    "HipRight": "SpineBase", "KneeRight": "HipRight", "AnkleRight": "KneeRight", "FootRight": "AnkleRight",
}


def build_parent_from_names(joint_order):
    """Rebuild the PARENT index list for an arbitrary joint ordering."""
    idx = {n: i for i, n in enumerate(joint_order)}
    return [-1 if PARENT_BY_NAME[n] is None else idx[PARENT_BY_NAME[n]] for n in joint_order]


def topo_order(parent):
    """Return joints in parent-before-child order for a given parent list."""
    order, seen = [], set()
    def visit(j):
        if j in seen:
            return
        if parent[j] >= 0:
            visit(parent[j])
        seen.add(j); order.append(j)
    for j in range(len(parent)):
        visit(j)
    return order


def validate():
    assert len(JOINT_NAMES) == N_JOINTS
    assert len(PARENT) == N_JOINTS
    assert sorted(TOPO) == list(range(N_JOINTS))
    pos = {j: i for i, j in enumerate(TOPO)}
    for j, p in enumerate(PARENT):
        if p >= 0:
            assert pos[p] < pos[j], f"parent {p} after child {j} in TOPO"
    assert build_parent_from_names(JOINT_NAMES) == PARENT
    return True


if __name__ == "__main__":
    print("skeleton valid:", validate())
    print("TOPO:", TOPO)
