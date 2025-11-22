import numpy as np
from pathlib import Path

def load_skeleton(path: str | Path):
    """Given path to skeleton file return dict:
    {
        "num_frames": int,
        "num_persons": int,
        "persons": [np.ndarray(T,V=25,C=3), ...]
    }
    """
    path = Path(path)

    with path.open('r') as f:
        lines = f.readlines()

    it = iter(lines)
    num_frames = int(next(it).strip())

    PTVC = []

    for _ in range(num_frames):
        num_people = int(next(it).strip())

        if num_people == 0:
            continue

        for person_id in range(num_people):
            if person_id >= len(PTVC):
                PTVC.append([])

            _ = next(it)
            num_joints = int(next(it).strip())

            joints = []

            for _ in range(num_joints):
                line = next(it).split()
                x, y, z = float(line[0]), float(line[1]), float(line[2])
                joints.append([x, y, z])

            PTVC[person_id].append(joints)

    persons = []
    for person_frames in PTVC:
        person_frames = np.array(person_frames, dtype=np.float32)
        persons.append(person_frames)

    return {
        "num_frames": num_frames,
        "num_persons": len(persons),
        "persons": persons
    }
