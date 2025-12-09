"""
Maps AudioSet-style multi-label strings from metadata
into a small set of environment classes.

Final coarse classes (8):
- animal
- crowd
- indoor
- other_ambient
- outdoor_nature
- vehicle_ground
- vehicle_water
- water
"""

CATEGORY_MAP = {
    # --------------------------
    # WATER (includes weather now)
    # --------------------------
    "Water": "water",
    "Waterfall": "water",
    "Stream": "water",
    "Ocean": "water",
    "Waves, surf": "water",
    "Rain": "water",
    "Rain on surface": "water",
    "Raindrop": "water",
    "Water tap, faucet": "water",
    "Sink (filling or washing)": "water",
    # weather merged into water
    "Thunder": "water",
    "Thunderstorm": "water",

    # --------------------------
    # ANIMAL / NON-HUMAN
    # --------------------------
    "Bird": "animal",
    "Bird vocalization": "animal",
    "Bird call": "animal",
    "Bird song": "animal",
    "Chirp": "animal",
    "Tweet": "animal",
    "Coo": "animal",
    "Pigeon": "animal",
    "Dove": "animal",
    "Bee": "animal",
    "Wasp": "animal",
    "Insect": "animal",
    "Fly, housefly": "animal",
    "Livestock": "animal",
    "Farm animals": "animal",
    "Cattle": "animal",
    "Wild animals": "animal",

    # --------------------------
    # CROWD / PEOPLE AS SCENE
    # --------------------------
    "Crowd": "crowd",
    "Crowd noise": "crowd",
    "Cheering": "crowd",
    "Children shouting": "crowd",
    "Battle cry": "crowd",
    "Hubbub, speech noise, speech babble": "crowd",
    "Hubbub": "crowd",

    # --------------------------
    # INDOOR (merge quiet + public)
    # --------------------------
    "Inside, small room": "indoor",
    "Inside, large room or hall": "indoor",
    "Inside, public space": "indoor",

    # --------------------------
    # OUTDOOR – NATURE / RURAL (merge wind)
    # --------------------------
    "Outside, rural or natural": "outdoor_nature",
    "Rustling leaves": "outdoor_nature",
    "Field recording": "outdoor_nature",
    "Wind": "outdoor_nature",
    "Wind noise (microphone)": "outdoor_nature",

    # --------------------------
    # VEHICLES (GROUND + URBAN)
    # --------------------------
    "Car passing by": "vehicle_ground",
    "Motor vehicle (road)": "vehicle_ground",
    "Traffic noise": "vehicle_ground",
    "Roadway noise": "vehicle_ground",
    "Rail transport": "vehicle_ground",
    "Train": "vehicle_ground",
    "Subway, metro, underground": "vehicle_ground",
    "Railroad car, train wagon": "vehicle_ground",
    "Train wheels squealing": "vehicle_ground",
    "Clickety-clack": "vehicle_ground",

    # formerly outdoor_urban now merged into vehicle_ground
    "Outside, urban or manmade": "vehicle_ground",
    "Traffic noise, roadway noise": "vehicle_ground",

    # extra vehicle/engine keys from metadata
    "Vehicle": "vehicle_ground",
    "Car": "vehicle_ground",
    "Engine": "vehicle_ground",
    "Idling": "vehicle_ground",
    "Fixed-wing aircraft, airplane": "vehicle_ground",
    "Aircraft": "vehicle_ground",
    "Helicopter": "vehicle_ground",
    "Jet engine": "vehicle_ground",
    "Propeller, airscrew": "vehicle_ground",
    "Race car, auto racing": "vehicle_ground",
    "Skidding": "vehicle_ground",
    "Tire squeal": "vehicle_ground",
    "Bus": "vehicle_ground",
    "Heavy engine (low frequency)": "vehicle_ground",
    "Medium engine (mid frequency)": "vehicle_ground",
    "Light engine (high frequency)": "vehicle_ground",

    # --------------------------
    # VEHICLES (WATER)
    # --------------------------
    "Boat, Water vehicle": "vehicle_water",
    "Rowboat, canoe, kayak": "vehicle_water",
    "Sailboat, sailing ship": "vehicle_water",
    "Motorboat, speedboat": "vehicle_water",
    "Ship": "vehicle_water",

    # --------------------------
    # OTHER AMBIENT / NOISE
    # --------------------------
    "Environmental noise": "other_ambient",
    "Silence": "other_ambient",
}




def map_to_balanced(label_string: str):
    """
    Map a semicolon-separated AudioSet label_names string
    to one of the unified classes.
    """
    parts = [p.strip() for p in label_string.split(";")]

    # 1) Exact match
    for p in parts:
        if p in CATEGORY_MAP:
            return CATEGORY_MAP[p]

    # 2) Partial text match
    for p in parts:
        for key in CATEGORY_MAP:
            if key.lower() in p.lower():
                return CATEGORY_MAP[key]

    return None
