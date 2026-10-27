# K1 klompen v10: smooth upturned toe

v10 replaces v9's short hook with one long, smooth upward curve:
- **Sole.** The robot's foot ends at x≈133, so from the end of the front pad land (x=135) the sole curves up in a single arc into the tip, at about Z 43.
- **Top line.** It flows down from the toe-cap crown and flattens into the tip, so the toe points slightly upward with no knob or dip.
- **Size.** 229 × 108 × 67 mm.

Everything else is as in v7, which builds on v6 and keeps its 10 mm arch sole and 2.4 mm upper shell. v7 changes:

- **Domed toe cap.** The top of the toe cap now bulges up in the middle and curves down to a higher, more upturned nose (Z≈33, was 29). The crown is also rounder across. 
- **Arch.** The heel block has a steep front face, the arch stays flat at its full 7 mm depth for longer, and it steps up crisply at the ball. The sole edges are rounder along the waist. The sole under the arch is still 10 mm minimum.
- **More carving**, in the style of the reference clog:
  - Rosette on top of the toe cap.
  - A smaller rosette on each side of the toe cap.
  - A wolf-tooth zigzag border across the toe cap, following the opening.
  - Three diamond chips down the nose.
  - The scalloped side line now ends before the side rosettes.
  - All grooves are 1.4 mm wide and 0.7 mm deep.

Unchanged from v6 and v7:
- **Pad pockets:** 4 mm deep, existing pads unscaled. The front pads are handed and the left and right heel pads come from the same heel mold.
- **Mounting:** three Ø5.5 bores with Ø9.2 underside access, and 4 mm of sole above the screw heads.
- **Entry:** the toe-first donning path. Tip the heel up about 8°, slide the toe under the vamp, drop the heel, then fit the screws from below and glue in the pads.

## Print (PETG)

Print `Print_Shoes/K1_Left_Klomp_One_Piece.stl` and `K1_Right_Klomp_One_Piece.stl` at 100% scale.

Settings for ≤50% infill:
- **Perimeters:** ≥5 walls. With a 0.4 mm nozzle, 6 walls fill the 2.4 mm upper completely.
- **Top/bottom:** ≥6 layers, so the sole skin and the area around the bores print solid.

The vamp interior is an overhang. Either print upright with tree supports inside the toe, or stand the shoe on its heel, toe up, with a brim.

Untested: physical fit, PETG strength in walking, cables and other links.

To rebuild: `python build.py`, using the packages in `requirements.txt`.
