"""Choose the high-ID rows worth buying back with the spare budget.

Budget arithmetic (merged armB ckpt-367, tied embeddings, hidden=2048):
    2,213,241,664 params - 2,000,000,000 = 213,241,664 -> 104,122 rows must go
    embedding is 248,320 rows -> keep <= 144,198
    144,198 - KEEP_LOW - 33 added = the extras allowance

Priority order:
  1. every high-ID token EgoLongQA actually touches (input + generated output + traces)
  2. domain vocabulary for the hidden test set, targeted at the eval's own categories --
     travel/sightseeing/shopping/gardening/hiking/fashion, NOT EgoProactive's cooking/DIY list

Why the domain list looks different here: our measured tail is dominated by PROPER NOUNS from
travel footage (Hostel, Gonzalez, Perez, Aks) rather than a closed set of procedural nouns. Proper
nouns are open-ended, so domain words buy less than they did on EgoProactive; the honest fallback
is the byte path, which is exact on the input side. We still spend the budget -- it is free -- but
the claim is "cheap insurance", not "closes the gap".

Usage:
  python src/vocab_pruning/build_extras_egolongqa.py --keep-low 143000 --cap 1165
"""
import argparse, json, os
from transformers import AutoProcessor

# EgoLongQA categories: Daily Activities, Sightseeing, Travel-Tourism, Shopping, Hiking-Outdoors,
# Gardening, Pets/social gatherings, Outdoor Sports, Events, Fashion Advice.
DOMAIN = """
hostel hostels motel motel guesthouse airbnb concierge lobby foyer atrium mezzanine turnstile
funicular tram trams tramway metro subway platform kiosk newsstand ticketing turnpike promenade
esplanade boardwalk pier jetty quay marina harbour harbor wharf dockside ferry gondola
cathedral basilica chapel cloister abbey minaret mosque synagogue shrine pagoda temple
courtyard piazza plaza boulevard avenue alleyway cobblestone cobbled facade balustrade
colonnade portico archway turret parapet battlement rampart citadel fortress
museum gallery exhibit exhibition rotunda atrium souvenir postcard guidebook itinerary
trailhead switchback ridgeline summit escarpment ravine gully gorge canyon plateau
meadow marsh wetland moor heath bracken thicket copse hedgerow bramble
tarp rucksack backpack canteen thermos carabiner crampon trekking hiking campsite
kayak canoe paddleboard snorkel wetsuit rink skate skateboard scooter helmet
trellis pergola planter allotment compost mulch topsoil perennial annual seedling
shrub hedge lawnmower secateurs pruner watering hose sprinkler greenhouse
receipt checkout aisle trolley cart barcode scanner cashier till voucher coupon
boutique storefront mannequin fitting garment blouse cardigan trousers denim
knitwear linen suede corduroy chiffon sequin embroidery lapel cuff hemline
leash kennel harness collar litter aquarium terrarium hutch
buffet marquee gazebo bunting confetti banner podium lectern
escalator elevator lift concourse terminal departures arrivals baggage carousel
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="work/merged")
    ap.add_argument("--keep-low", type=int, default=143000)
    ap.add_argument("--added-start", type=int, default=248044)
    ap.add_argument("--cap", type=int, default=1165, help="extras allowance (see arithmetic above)")
    ap.add_argument("--observed", default="reports/rare_ids_egolongqa.json")
    ap.add_argument("--heldout",
                    default="data/eval_qaego4d500.jsonl,data/egoconv_train_eval.jsonl",
                    help="comma-separated jsonl corpora NOT used to derive the keep-set")
    ap.add_argument("--genvocab", default="",
                    help="comma-separated capture_output_vocab.py outputs (generated-token ids)")
    ap.add_argument("--out", default="reports/extra_ids_egolongqa.json")
    ap.add_argument("--total-params", type=int, default=2_213_241_664)
    ap.add_argument("--emb-rows", type=int, default=248_320)
    ap.add_argument("--hidden", type=int, default=2048)
    a = ap.parse_args()

    tok = AutoProcessor.from_pretrained(a.model).tokenizer
    observed = [int(x) for x in json.load(open(a.observed))]
    extras, seen = list(observed), set(observed)

    # Highest-priority rows: tokens the model ACTUALLY GENERATES (capture_output_vocab.py). The
    # output side has no byte fallback -- a missing row there changes the word, not its spelling --
    # so these matter more than any input-side token. Must come from a capture at the SERVING
    # max_pixels on the SERVING machine; a capture at another resolution describes generations the
    # shipped model never produces.
    n_before_gen = len(seen)
    for path in [p.strip() for p in a.genvocab.split(",") if p.strip()]:
        if not os.path.exists(path):
            continue
        for t in (int(x) for x in json.load(open(path))["high_ids"]):
            if a.keep_low <= t < a.added_start and t not in seen:
                seen.add(t)
                extras.append(t)
    n_gen = len(seen) - n_before_gen

    # Held-out egocentric corpora the keep-set was NOT derived from. Measured rare-token rate on
    # these is 0.017-0.106%, and they cost ~80 rows out of an 800-row surplus -- a better use of
    # the budget than more guessed domain words, because they are real text from the same task
    # family rather than a hand-written list.
    n_before_heldout = len(seen)
    for path in a.heldout.split(","):
        path = path.strip()
        if not path or not os.path.exists(path):
            continue
        for line in open(path):
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            for f in ("question", "mcq_options"):
                for t in tok.encode(str(r.get(f, "")), add_special_tokens=False):
                    if a.keep_low <= t < a.added_start and t not in seen:
                        seen.add(t)
                        extras.append(t)
    n_heldout = len(seen) - n_before_heldout

    for w in DOMAIN.split():
        for form in (w, " " + w, w.capitalize(), " " + w.capitalize(),
                     w + "s", " " + w + "s"):
            for t in tok.encode(form, add_special_tokens=False):
                if a.keep_low <= t < a.added_start and t not in seen:
                    seen.add(t)
                    extras.append(t)

    dropped = max(0, len(extras) - a.cap)
    extras = sorted(extras[:a.cap])
    n_final = a.keep_low + len(extras) + 33
    params = a.total_params - (a.emb_rows - n_final) * a.hidden

    print(f"observed (EgoLongQA corpus): {len(observed)}")
    print(f"GENERATED-token rows added:  {n_gen}")
    print(f"held-out corpora added:      {n_heldout}")
    print(f"domain-added:                {len(seen) - len(observed) - n_heldout - n_gen}"
          + (f"   (dropped {dropped} over cap)" if dropped else ""))
    print(f"extras kept:                 {len(extras)}  of allowance {a.cap}")
    print(f"final vocab rows:            {n_final:,}  (was {a.emb_rows:,})")
    print(f"pruned params:               {params:,} = {params/1e9:.4f} B  "
          f"[{'UNDER' if params < 2e9 else 'OVER'} 2B, margin {(2e9-params)/1e6:+.2f} M]")
    assert params < 2e9, "over 2B -- lower --keep-low or --cap"
    json.dump(extras, open(a.out, "w"))
    print(f"-> {a.out}")


main()
