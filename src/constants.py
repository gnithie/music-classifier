TARGET_GENRES = ["Blues", "Country", "Electronic", "Jazz","Rock"]   #list for Genre Labels
TARGET_SECTIONS = ['Chorus', "Fills", "Intro", "Outro", "Verse"]    #list for Section Labels

DRUM_NAMES = ["BD", "SD", "CH", "OH", "RD", "CR", "LT", "MT", "HT"] #list for Drum names

#dictionary for drum index to pitch
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

#dictionary for pitch to drum index
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

#default values for hyperparamters
GET_DEFAULTS = {
    #path for midi data
    "midi_path" : "../data/Groove_Monkee_Mega_Pack_GM/",
    "midi_path_full" : "../data/Groove_Monkee_Mega_Pack_GM_Full/",

    #output folder path
    "model_path" : "../output/",

    #model file names based on their type
    "model_file_G" : "../output/BaseGenre_TLSection/model/Model_G_",
    "model_file_TL_S" : "../output/BaseGenre_TLSection/model/Model_TL_S_",
    "model_file_S" : "../output/BaseSection_TLGenre_40Epochs/model/Model_S_",
    "model_file_TL_G" : "../output/BaseSection_TLGenre_40Epochs/model/Model_TL_G_",
    "model_file_MO" : "../output/MultiOutput_3/Model_MO_",
    
    #file details for Genre 
    "X" : "../data/Groove_Monkee_Mega_Pack_GM_X.npy",
    "y" : "../data/Groove_Monkee_Mega_Pack_GM_y.npy",

    #file details for Section 
    "X1" : "../data/Groove_Monkee_Mega_Pack_GM_Full_X1.npy",
    "y1" : "../data/Groove_Monkee_Mega_Pack_GM_Full_y1.npy",

    #file details for Multioutput model 
    "X_both" : "../data/Groove_Monkee_Mega_Pack_GM_both_X.npy",
    "y1_both" : "../data/Groove_Monkee_Mega_Pack_GM_both_y1.npy",
    "y2_both" : "../data/Groove_Monkee_Mega_Pack_GM_both_y2.npy", 

    #cnn hyperparameters default values
    "filters" : (32,64,128,256),
    "kernel_size" : (3,3),
    "padding" : 'same',
    "strides" : 1,
    "activation" : 'relu',
    "pool_size" : (2,2),
    "dense" : (256, 128, 32),
    "dropout1" : None,
    "dropout2" : None,
    "target_size" : 5
}