#!/usr/bin/env python3
"""
build_taxonomy.py

Generate taxonomy_events.json from FSD50K vocabulary.csv.

The taxonomy maps canonical event class names to:
  - "specific": list of raw FSD50K label names that map to this class
  - "parents":  list of broader parent class names (for hierarchical labeling)

Since FSD50K uses AudioSet label names directly, each vocabulary entry
becomes its own canonical class. Parent relationships are inferred from
AudioSet's ontology structure embedded in the vocabulary.

Output:
  data/events/taxonomy_events.json

Usage:
  python scripts/build_taxonomy.py
  python scripts/build_taxonomy.py --fsd-root data/events/FSD50K --out data/events/taxonomy_events.json
"""

import argparse
import csv
import json
from pathlib import Path


# ── Broad parent groupings ────────────────────────────────────────────────────
# Maps raw FSD50K label names to a broad parent category.
# This gives the model hierarchical signal without needing the full AudioSet graph.

PARENT_MAP = {
    # Human sounds
    "Speech": "Human sounds",
    "Male speech, man speaking": "Human sounds",
    "Female speech, woman speaking": "Human sounds",
    "Child speech, kid speaking": "Human sounds",
    "Conversation": "Human sounds",
    "Narration, monologue": "Human sounds",
    "Babbling": "Human sounds",
    "Whispering": "Human sounds",
    "Laughter": "Human sounds",
    "Baby laughter": "Human sounds",
    "Giggle": "Human sounds",
    "Snicker": "Human sounds",
    "Belly laugh": "Human sounds",
    "Chuckle, chortle": "Human sounds",
    "Crying, sobbing": "Human sounds",
    "Baby cry, infant cry": "Human sounds",
    "Whimper": "Human sounds",
    "Wail, moan": "Human sounds",
    "Screaming": "Human sounds",
    "Shout": "Human sounds",
    "Bellow": "Human sounds",
    "Yell": "Human sounds",
    "Singing": "Human sounds",
    "choir": "Human sounds",
    "Humming": "Human sounds",
    "Whistling": "Human sounds",
    "Breathing": "Human sounds",
    "Cough": "Human sounds",
    "Sneeze": "Human sounds",
    "Sniff": "Human sounds",
    "Clapping": "Human sounds",
    "Finger snapping": "Human sounds",
    "Hands": "Human sounds",
    "Walk, footsteps": "Human sounds",
    "Run": "Human sounds",
    "Chewing, mastication": "Human sounds",
    "Biting": "Human sounds",
    "Gargling": "Human sounds",
    "Hiccup": "Human sounds",
    "Burping, eructation": "Human sounds",
    "Throat clearing": "Human sounds",
    "Snoring": "Human sounds",
    "Panting": "Human sounds",

    # Music
    "Music": "Music",
    "Musical instrument": "Music",
    "Plucked string instrument": "Music",
    "Guitar": "Music",
    "Electric guitar": "Music",
    "Bass guitar": "Music",
    "Acoustic guitar": "Music",
    "Steel guitar, slide guitar": "Music",
    "Tapping (guitar technique)": "Music",
    "Strum": "Music",
    "Banjo": "Music",
    "Sitar": "Music",
    "Mandolin": "Music",
    "Ukulele": "Music",
    "Piano": "Music",
    "Keyboard (musical)": "Music",
    "Organ": "Music",
    "Electronic organ": "Music",
    "Hammond organ": "Music",
    "Synthesizer": "Music",
    "Sampler": "Music",
    "Drum": "Music",
    "Drum kit": "Music",
    "Drum machine": "Music",
    "Snare drum": "Music",
    "Rimshot": "Music",
    "Drum roll": "Music",
    "Bass drum": "Music",
    "Timpani": "Music",
    "Tabla": "Music",
    "Cymbal": "Music",
    "Hi-hat": "Music",
    "Wood block": "Music",
    "Tambourine": "Music",
    "Rattle (instrument)": "Music",
    "Maraca": "Music",
    "Gong": "Music",
    "Tubular bells": "Music",
    "Mallet percussion": "Music",
    "Marimba, xylophone": "Music",
    "Glockenspiel": "Music",
    "Vibraphone": "Music",
    "Steelpan": "Music",
    "Orchestra": "Music",
    "Brass instrument": "Music",
    "Trumpet": "Music",
    "Trombone": "Music",
    "French horn": "Music",
    "Tuba": "Music",
    "Bowed string instrument": "Music",
    "Violin, fiddle": "Music",
    "Cello": "Music",
    "Double bass": "Music",
    "Wind instrument, woodwind instrument": "Music",
    "Flute": "Music",
    "Saxophone": "Music",
    "Clarinet": "Music",
    "Harp": "Music",
    "Bell": "Music",
    "Jingle bell": "Music",
    "Bicycle bell": "Music",
    "Tuning fork": "Music",
    "Chime": "Music",
    "Wind chime": "Music",
    "Change ringing (campanology)": "Music",
    "Harmonica": "Music",
    "Accordion": "Music",
    "Bagpipes": "Music",
    "Didgeridoo": "Music",
    "Shofar": "Music",
    "Theremin": "Music",
    "Singing bowl": "Music",
    "Scratching (performance technique)": "Music",
    "Pop music": "Music",
    "Hip hop music": "Music",
    "Beatboxing": "Music",
    "Rock music": "Music",
    "Heavy metal": "Music",
    "Punk rock": "Music",
    "Grunge": "Music",
    "Progressive rock": "Music",
    "Rock and roll": "Music",
    "Psychedelic rock": "Music",
    "Rhythm and blues": "Music",
    "Soul music": "Music",
    "Reggae": "Music",
    "Country": "Music",
    "Swing music": "Music",
    "Bluegrass": "Music",
    "Funk": "Music",
    "Folk music": "Music",
    "Middle Eastern music": "Music",
    "Jazz": "Music",
    "Disco": "Music",
    "Classical music": "Music",
    "Opera": "Music",
    "Electronic music": "Music",
    "House music": "Music",
    "Techno": "Music",
    "Dubstep": "Music",
    "Drum and bass": "Music",
    "Electronica": "Music",
    "Electronic dance music": "Music",
    "Ambient music": "Music",
    "Soundtrack music": "Music",
    "Lullaby": "Music",
    "Video game music": "Music",
    "Christmas music": "Music",
    "Dance music": "Music",

    # Animal sounds
    "Animal": "Animal sounds",
    "Domestic animals, pets": "Animal sounds",
    "Dog": "Animal sounds",
    "Bark": "Animal sounds",
    "Bow-wow": "Animal sounds",
    "Growling": "Animal sounds",
    "Whimper (dog)": "Animal sounds",
    "Cat": "Animal sounds",
    "Purr": "Animal sounds",
    "Meow": "Animal sounds",
    "Hiss": "Animal sounds",
    "Caterwaul": "Animal sounds",
    "Livestock, farm animals, working animals": "Animal sounds",
    "Horse": "Animal sounds",
    "Clip-clop": "Animal sounds",
    "Neigh, whinny": "Animal sounds",
    "Cattle, bovinae": "Animal sounds",
    "Moo": "Animal sounds",
    "Cowbell": "Animal sounds",
    "Pig": "Animal sounds",
    "Oink": "Animal sounds",
    "Goat": "Animal sounds",
    "Sheep": "Animal sounds",
    "Bleat": "Animal sounds",
    "Bird": "Animal sounds",
    "Bird vocalization, bird call, bird song": "Animal sounds",
    "Chirp, tweet": "Animal sounds",
    "Squawk": "Animal sounds",
    "Pigeon, dove": "Animal sounds",
    "Coo": "Animal sounds",
    "Crow": "Animal sounds",
    "Caw": "Animal sounds",
    "Owl": "Animal sounds",
    "Hoot": "Animal sounds",
    "Wild animals": "Animal sounds",
    "Roaring cats (lions, tigers)": "Animal sounds",
    "Roar": "Animal sounds",
    "Insect": "Animal sounds",
    "Cricket": "Animal sounds",
    "Mosquito": "Animal sounds",
    "Fly, housefly": "Animal sounds",
    "Bee, wasp, etc.": "Animal sounds",
    "Frog": "Animal sounds",
    "Croak": "Animal sounds",
    "Snake": "Animal sounds",
    "Rattle": "Animal sounds",
    "Whale vocalization": "Animal sounds",
    "Fowl": "Animal sounds",
    "Chicken, rooster": "Animal sounds",
    "Crowing, cock-a-doodle-doo": "Animal sounds",
    "Cluck": "Animal sounds",
    "Gobble": "Animal sounds",
    "Turkey": "Animal sounds",
    "Canidae, dogs, wolves": "Animal sounds",
    "Howl": "Animal sounds",
    "Yip": "Animal sounds",

    # Vehicles
    "Vehicle": "Vehicle sounds",
    "Car": "Vehicle sounds",
    "Motor vehicle (road)": "Vehicle sounds",
    "Truck": "Vehicle sounds",
    "Bus": "Vehicle sounds",
    "Motorcycle": "Vehicle sounds",
    "Traffic noise, roadway noise": "Vehicle sounds",
    "Car passing by": "Vehicle sounds",
    "Race car, auto racing": "Vehicle sounds",
    "Tire squeal": "Vehicle sounds",
    "Skidding": "Vehicle sounds",
    "Bicycle": "Vehicle sounds",
    "Rail transport": "Vehicle sounds",
    "Train": "Vehicle sounds",
    "Train whistle": "Vehicle sounds",
    "Train horn": "Vehicle sounds",
    "Railroad car, train wagon": "Vehicle sounds",
    "Subway, metro, underground": "Vehicle sounds",
    "Aircraft": "Vehicle sounds",
    "Fixed-wing aircraft, airplane": "Vehicle sounds",
    "Helicopter": "Vehicle sounds",
    "Boat, Water vehicle": "Vehicle sounds",
    "Motorboat, speedboat": "Vehicle sounds",
    "Ship": "Vehicle sounds",
    "Rowboat, canoe, kayak": "Vehicle sounds",
    "Sailboat, sailing ship": "Vehicle sounds",

    # Nature / environment
    "Water": "Nature sounds",
    "Rain": "Nature sounds",
    "Raindrop": "Nature sounds",
    "Rain on surface": "Nature sounds",
    "Stream": "Nature sounds",
    "Waterfall": "Nature sounds",
    "Ocean": "Nature sounds",
    "Waves, surf": "Nature sounds",
    "Wind": "Nature sounds",
    "Rustling leaves": "Nature sounds",
    "Thunderstorm": "Nature sounds",
    "Thunder": "Nature sounds",
    "Fire": "Nature sounds",
    "Crackle": "Nature sounds",

    # Mechanical / tools
    "Engine": "Mechanical sounds",
    "Idling": "Mechanical sounds",
    "Accelerating, revving, vroom": "Mechanical sounds",
    "Door": "Mechanical sounds",
    "Doorbell": "Mechanical sounds",
    "Ding-dong": "Mechanical sounds",
    "Sliding door": "Mechanical sounds",
    "Slam": "Mechanical sounds",
    "Knock": "Mechanical sounds",
    "Tap": "Mechanical sounds",
    "Squeak": "Mechanical sounds",
    "Cupboard open or close": "Mechanical sounds",
    "Drawer open or close": "Mechanical sounds",
    "Dishes, pots, and pans": "Mechanical sounds",
    "Cutlery, silverware": "Mechanical sounds",
    "Chopping (food)": "Mechanical sounds",
    "Frying (food)": "Mechanical sounds",
    "Microwave oven": "Mechanical sounds",
    "Blender": "Mechanical sounds",
    "Sink (filling or washing)": "Mechanical sounds",
    "Bathtub (filling or washing)": "Mechanical sounds",
    "Hair dryer": "Mechanical sounds",
    "Vacuum cleaner": "Mechanical sounds",
    "Sewing machine": "Mechanical sounds",
    "Mechanical fan": "Mechanical sounds",
    "Air conditioning": "Mechanical sounds",
    "Lawn mower": "Mechanical sounds",
    "Power tool": "Mechanical sounds",
    "Drill": "Mechanical sounds",
    "Jackhammer": "Mechanical sounds",
    "Chainsaw": "Mechanical sounds",
    "Typewriter": "Mechanical sounds",
    "Computer keyboard": "Mechanical sounds",
    "Writing": "Mechanical sounds",
    "Alarm": "Mechanical sounds",
    "Alarm clock": "Mechanical sounds",
    "Smoke detector, smoke alarm": "Mechanical sounds",
    "Fire alarm": "Mechanical sounds",
    "Telephone": "Mechanical sounds",
    "Telephone bell ringing": "Mechanical sounds",
    "Ringtone": "Mechanical sounds",
    "Telephone dialing, DTMF": "Mechanical sounds",
    "Busy signal": "Mechanical sounds",
    "Coin (dropping)": "Mechanical sounds",
    "Zipper (clothing)": "Mechanical sounds",
    "Scissors": "Mechanical sounds",
    "Electric shaver, electric razor": "Mechanical sounds",
    "Printer": "Mechanical sounds",
    "Camera": "Mechanical sounds",
    "Single-lens reflex camera": "Mechanical sounds",

    # Explosions / impacts
    "Explosion": "Impact sounds",
    "Gunshot, gunfire": "Impact sounds",
    "Burst, pop": "Impact sounds",
    "Eruption": "Impact sounds",
    "Boom": "Impact sounds",
    "Bang": "Impact sounds",
    "Thwack, smack": "Impact sounds",
    "Whip": "Impact sounds",
    "Slap, smack": "Impact sounds",
    "Breaking": "Impact sounds",
    "Shatter": "Impact sounds",
    "Glass": "Impact sounds",
    "Clatter": "Impact sounds",
    "Crumpling, crinkling": "Impact sounds",
    "Tearing": "Impact sounds",
    "Crushing": "Impact sounds",

    # Crowd / events
    "Crowd": "Crowd sounds",
    "Hubbub, speech noise, speech babble": "Crowd sounds",
    "Children playing": "Crowd sounds",
    "Cheering": "Crowd sounds",
    "Applause": "Crowd sounds",
    "Chatter": "Crowd sounds",

    # Silence / ambient
    "Silence": "Ambient",
    # Missing labels — added after vocabulary scan
    "Boiling": "Nature sounds",
    "Dishes, pots, and pans": "Mechanical sounds",
    "Human voice": "Human sounds",
    "Mechanisms": "Mechanical sounds",
    "Packing tape, duct tape": "Mechanical sounds",
    "Percussion": "Music",
    "Ratchet, pawl": "Mechanical sounds",
    "Respiratory sounds": "Human sounds",
    "Sawing": "Mechanical sounds",
    "Screech": "Impact sounds",
    "Siren": "Vehicle sounds",
    "Speech synthesizer": "Human sounds",
    "Thump, thud": "Impact sounds",
    "Tick": "Mechanical sounds",
    "Tick-tock": "Mechanical sounds",
    "Toilet flush": "Mechanical sounds",
    "Tools": "Mechanical sounds",
    "Typing": "Mechanical sounds",
    "Water tap, faucet": "Mechanical sounds",
    "Whoosh, swoosh, swish": "Nature sounds",
    "Wood": "Mechanical sounds",

    "Buzz": "Mechanical sounds",
    "Chink, clink": "Impact sounds",
    "Church bell": "Music",
    "Clock": "Mechanical sounds",
    "Crack": "Impact sounds",
    "Crash cymbal": "Music",
    "Dishes, pots, and pans": "Mechanical sounds",
    "Domestic sounds, home sounds": "Mechanical sounds",
    "Drip": "Nature sounds",
    "Engine starting": "Vehicle sounds",
    "Fart": "Human sounds",
    "Female singing": "Human sounds",
    "Fill (with liquid)": "Mechanical sounds",
    "Fireworks": "Impact sounds",
    "Gasp": "Human sounds",
    "Gull, seagull": "Animal sounds",
    "Gurgling": "Nature sounds",
    "Hammer": "Mechanical sounds",
    "Human group actions": "Human sounds",
    "Ice cream truck, ice cream van": "Vehicle sounds",
    "Keys jangling": "Mechanical sounds",
    "Liquid": "Nature sounds",
    "Male singing": "Human sounds",
    "Marimba, xylophone": "Music",
    "Pig": "Animal sounds",
    "Plop": "Nature sounds",
    "Pour": "Nature sounds",
    "Power windows, electric windows": "Vehicle sounds",
    "Printer": "Mechanical sounds",
    "Reversing beeps": "Vehicle sounds",
    "Rimshot": "Music",
    "Rustle": "Nature sounds",
    "Scratch": "Mechanical sounds",
    "Shuffling cards": "Mechanical sounds",
    "Sigh": "Human sounds",
    "Single-lens reflex camera": "Mechanical sounds",
    "Skateboard": "Vehicle sounds",
    "Slap, smack": "Impact sounds",
    "Slosh": "Nature sounds",
    "Snort": "Human sounds",
    "Splash, splatter": "Nature sounds",
    "Squish": "Nature sounds",
    "Static": "Mechanical sounds",
    "Stomach rumble": "Human sounds",
    "Thunk": "Impact sounds",
    "Trickle, dribble": "Nature sounds",
    "Vehicle horn, car horn, honking": "Vehicle sounds",

    "Environmental noise": "Ambient",
    "Pink noise": "Ambient",
    "White noise": "Ambient",
    "Field recording": "Ambient",
}


def parse_args():
    ap = argparse.ArgumentParser(description="Generate taxonomy_events.json from FSD50K vocabulary.")
    ap.add_argument("--fsd-root", type=Path, default=None,
                    help="FSD50K root dir (default: auto-detected)")
    ap.add_argument("--out", type=Path, default=None,
                    help="Output path (default: data/events/taxonomy_events.json)")
    return ap.parse_args()


def normalize_label(s: str) -> str:
    """Normalize FSD50K label to match PARENT_MAP keys.
    Converts underscores to spaces and _and_ to ', '.
    e.g. 'Bird_vocalization_and_bird_call_and_bird_song'
      -> 'Bird vocalization, bird call, bird song'
    """
    s = s.replace("_and_", ", ")
    s = s.replace("_(", " (").replace("_)", ")")
    s = s.replace("_", " ")
    return s


def main():
    args = parse_args()

    project_root = Path(__file__).resolve().parent.parent
    fsd_root     = args.fsd_root or project_root / "data" / "events" / "FSD50K"
    out_path     = args.out      or project_root / "data" / "events" / "taxonomy_events.json"
    vocab_path   = fsd_root / "FSD50K.ground_truth" / "vocabulary.csv"

    if not vocab_path.exists():
        raise FileNotFoundError(
            f"vocabulary.csv not found at {vocab_path}\n"
            "Run events_download.py first."
        )

    # Load all label names from vocabulary
    label_names = []
    with vocab_path.open(newline="", encoding="utf-8") as f:
        for row in csv.reader(f):
            if not row or len(row) < 2:
                continue
            name = row[1].strip()
            if name and name.lower() != "label_name":
                label_names.append(name)

    print(f"[INFO] Loaded {len(label_names)} labels from vocabulary.csv")

    # Build taxonomy — each label is its own canonical class
    taxonomy = {}
    unmapped = []

    for name in label_names:
        normalized = normalize_label(name)
        parent = PARENT_MAP.get(normalized) or PARENT_MAP.get(name)
        if parent is None:
            unmapped.append(name)
            parent_list = ["Other Event"]
        else:
            parent_list = [parent]

        # Store under normalized name; specific includes both forms for matching
        canonical = normalize_label(name)
        taxonomy[canonical] = {
            "specific": [name, canonical],   # raw + normalized
            "parents":  parent_list,
        }

    # Add parent classes as canonical entries too
    all_parents = set(p for info in taxonomy.values() for p in info["parents"])
    for parent in all_parents:
        if parent not in taxonomy:
            taxonomy[parent] = {
                "specific": [],
                "parents":  [],
            }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(taxonomy, f, indent=2)

    print(f"[INFO] Wrote taxonomy with {len(taxonomy)} entries -> {out_path}")
    if unmapped:
        print(f"[WARN] {len(unmapped)} labels not in PARENT_MAP (assigned to 'Other Event'):")
        for u in unmapped[:20]:
            print(f"  - {u}")
        if len(unmapped) > 20:
            print(f"  ... and {len(unmapped) - 20} more")
    else:
        print("[INFO] All labels mapped successfully.")

    print("\n[INFO] Parent category counts:")
    from collections import Counter
    parent_counts = Counter(
        p for info in taxonomy.values()
        for p in info["parents"]
    )
    for parent, count in sorted(parent_counts.items(), key=lambda x: -x[1]):
        print(f"  {count:4d}  {parent}")


if __name__ == "__main__":
    main()