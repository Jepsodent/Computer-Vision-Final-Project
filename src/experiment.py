from src.utils import build_sequences
import os
import numpy as np

base_path = 'Multi class/train'
dataset_metadata = [
    # === DROWSY SECTION (Label: 1) ===
    # SleepyCombination Folder
    {"folder": "drowsy/sleepyCombination", "prefix": "001_glasses_sleepyCombination", "start": 599, "end": 2747, "label": 1},
    {"folder": "drowsy/sleepyCombination", "prefix": "001_noglasses_sleepyCombination", "start": 330, "end": 2710, "label": 1},
    {"folder": "drowsy/sleepyCombination", "prefix": "002_glasses_sleepyCombination", "start": 314, "end": 2878, "label": 1},
    {"folder": "drowsy/sleepyCombination", "prefix": "002_noglasses_sleepyCombination", "start": 239, "end": 2656, "label": 1},
    {"folder": "drowsy/sleepyCombination", "prefix": "005_glasses_nonsleepyCombination", "start": 650, "end": 744, "label": 1},
    {"folder": "drowsy/sleepyCombination", "prefix": "005_glasses_sleepyCombination", "start": 287, "end": 2921, "label": 1},
    {"folder": "drowsy/sleepyCombination", "prefix": "005_noglasses_sleepyCombination", "start": 249, "end": 3191, "label": 1},
    {"folder": "drowsy/sleepyCombination", "prefix": "006_glasses_sleepyCombination", "start": 128, "end": 2848, "label": 1},

    # SlowBlinkWithNodding Folder
    {"folder": "drowsy/slowBlinkWithNodding", "prefix": "001_glasses_slowBlinkWithNodding", "start": 464, "end": 1920, "label": 1},
    {"folder": "drowsy/slowBlinkWithNodding", "prefix": "001_noglasses_slowBlinkWithNodding", "start": 398, "end": 1851, "label": 1},
    {"folder": "drowsy/slowBlinkWithNodding", "prefix": "002_glasses_slowBlinkWithNodding", "start": 395, "end": 2035, "label": 1},
    {"folder": "drowsy/slowBlinkWithNodding", "prefix": "002_noglasses_slowBlinkWithNodding", "start": 209, "end": 1809, "label": 1},
    {"folder": "drowsy/slowBlinkWithNodding", "prefix": "005_glasses_slowBlinkWithNodding", "start": 344, "end": 2252, "label": 1},
    {"folder": "drowsy/slowBlinkWithNodding", "prefix": "005_noglasses_slowBlinkWithNodding", "start": 223, "end": 2252, "label": 1},
    {"folder": "drowsy/slowBlinkWithNodding", "prefix": "006_glasses_slowBlinkWithNodding", "start": 316, "end": 508, "label": 1},

    # Yawning Folder
    {"folder": "drowsy/yawning", "prefix": "001_glasses_yawning", "start": 339, "end": 1849, "label": 1},
    {"folder": "drowsy/yawning", "prefix": "001_noglasses_yawning", "start": 196, "end": 1871, "label": 1},
    {"folder": "drowsy/yawning", "prefix": "002_glasses_yawning", "start": 432, "end": 2066, "label": 1},
    {"folder": "drowsy/yawning", "prefix": "002_noglasses_yawning", "start": 268, "end": 1770, "label": 1},
    {"folder": "drowsy/yawning", "prefix": "005_glasses_yawning", "start": 298, "end": 2297, "label": 1},
    {"folder": "drowsy/yawning", "prefix": "005_noglasses_yawning", "start": 407, "end": 2639, "label": 1},

    # === NOT dROWSY SECTION (Label: 0) ===
    {"folder": "notdrowsy", "prefix": "001_glasses_nonsleepyCombination", "start": 0, "end": 3329, "label": 0},
    {"folder": "notdrowsy", "prefix": "001_glasses_sleepyCombination", "start": 0, "end": 598, "label": 0},
    {"folder": "notdrowsy", "prefix": "001_glasses_slowBlinkWithNodding", "start": 0, "end": 1516, "label": 0},
    {"folder": "notdrowsy", "prefix": "001_glasses_yawning", "start": 0, "end": 1558, "label": 0},
    {"folder": "notdrowsy", "prefix": "001_noglasses_nonsleepyCombination", "start": 0, "end": 2731, "label": 0},
    {"folder": "notdrowsy", "prefix": "001_noglasses_sleepyCombination", "start": 0, "end": 329, "label": 0},
    {"folder": "notdrowsy", "prefix": "001_noglasses_slowBlinkWithNodding", "start": 0, "end": 1719, "label": 0},
    {"folder": "notdrowsy", "prefix": "001_noglasses_yawning", "start": 0, "end": 195, "label": 0},
    
    {"folder": "notdrowsy", "prefix": "002_glasses_nonsleepyCombination", "start": 0, "end": 2933, "label": 0},
    {"folder": "notdrowsy", "prefix": "002_glasses_sleepyCombination", "start": 0, "end": 313, "label": 0},
    {"folder": "notdrowsy", "prefix": "002_glasses_slowBlinkWithNodding", "start": 0, "end": 501, "label": 0},
    {"folder": "notdrowsy", "prefix": "002_glasses_yawning", "start": 0, "end": 431, "label": 0},
    {"folder": "notdrowsy", "prefix": "002_noglasses_nonsleepyCombination", "start": 0, "end": 2678, "label": 0},
    {"folder": "notdrowsy", "prefix": "002_noglasses_sleepyCombination", "start": 0, "end": 238, "label": 0},
    {"folder": "notdrowsy", "prefix": "002_noglasses_slowBlinkWithNodding", "start": 0, "end": 669, "label": 0},
    {"folder": "notdrowsy", "prefix": "002_noglasses_yawning", "start": 0, "end": 1487, "label": 0},

    {"folder": "notdrowsy", "prefix": "005_glasses_nonsleepyCombination", "start": 0, "end": 3072, "label": 0},
    {"folder": "notdrowsy", "prefix": "005_glasses_sleepyCombination", "start": 0, "end": 286, "label": 0},
    {"folder": "notdrowsy", "prefix": "005_glasses_slowBlinkWithNodding", "start": 0, "end": 343, "label": 0},
    {"folder": "notdrowsy", "prefix": "005_glasses_yawning", "start": 0, "end": 2089, "label": 0},
    {"folder": "notdrowsy", "prefix": "005_noglasses_nonsleepyCombination", "start": 0, "end": 3301, "label": 0},
    {"folder": "notdrowsy", "prefix": "005_noglasses_sleepyCombination", "start": 0, "end": 248, "label": 0},
    {"folder": "notdrowsy", "prefix": "005_noglasses_slowBlinkWithNodding", "start": 0, "end": 222, "label": 0},
    {"folder": "notdrowsy", "prefix": "005_noglasses_yawning", "start": 0, "end": 1748, "label": 0},

    {"folder": "notdrowsy", "prefix": "006_glasses_nonsleepyCombination", "start": 0, "end": 2867, "label": 0},
    {"folder": "notdrowsy", "prefix": "006_glasses_sleepyCombination", "start": 0, "end": 2759, "label": 0},
    {"folder": "notdrowsy", "prefix": "006_glasses_slowBlinkWithNodding", "start": 0, "end": 1021, "label": 0},
]


def generate_seq(variant="original"):
    all_X = []
    all_y = []
    all_groups = []
    for case in dataset_metadata:
        folder_path = os.path.join(base_path, case["folder"])
        X, y , groups = build_sequences(
            folder_path=folder_path,
            prefix=case['prefix'],
            start=case['start'],
            end=case['end'],
            label=case['label'],
            variant=variant,
            seq_len=30
        )
        if len(X) > 0:
            all_X.extend(X)
            all_y.extend(y)
            all_groups.extend(groups)
    print("Total Sequences: ", len(all_X))
    return np.array(all_X), np.array(all_y) , np.array(all_groups)

