
TARGET_GENRES = ["Blues", "Country", "Electronic", "Jazz","Rock"]

DRUM_NAMES = ["BD", "SD", "CH", "OH", "RD", "CR", "LT", "MT", "HT"]

DRUMINDEX_to_PITCH = {
    0: 36, # BD Bass
    1: 38, # SD Snare
    2: 42, # CH Closed Hi-Hat
    3: 46, # OH Open Hi-Hat
    4: 51, # RD Ride
    5: 49, # CR Crash 
    6: 41, # LT Low Tom
    7: 45, # MT Mid Tom
    8: 48  # HT High Tom
}

PITCH_DRUMINDEX = {
    35: 0, 36: 0,                             # BD
    38: 1, 40: 1,                             # SD
    37: 1,                                    # SD (side-stick)
    39: 1,                                    # SD (roll/flam/clap)
    42: 2, 44: 2,                             # CH
    54: 2, 69: 2, 70: 2,                      # CH (tambourine, shaker)
    46: 3,                                    # OH

    49: 5, 52: 5, 57: 5,                      # CR
    51: 4, 53: 4, 56: 4, 59: 4, 80: 4,        # RD (56 cowbell, 80 mute triangle)
    41: 6, 43: 6, 45: 7, 47: 7, 48: 8, 50: 8, # toms low to high -- LT, MT, HT
    60: 8, 61: 8, 62: 7, 63: 7, 64: 6,        # bongos high to low (use HT, MT, LT)
    67: 8, 68: 7                              # agogos high to low (use HT, MT)
    }

GET_DEFAULTS = {
    "midi_path" : "../data/Groove_Monkee_Mega_Pack_GM/",
    "model_path" : "../output/",
    "model_file" : "../output/Groove_Monkee_Mega_Pack_GM.h5",
    "X_data" : "../data/Groove_Monkee_Mega_Pack_GM_X.npy",
    "y_data" : "../data/Groove_Monkee_Mega_Pack_GM_y.npy"
}