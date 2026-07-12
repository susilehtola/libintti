# libintti logo

`logo.svg` — the project logo as a self-contained SVG (512×512 viewBox,
transparent outside the patch). All lettering is outlined, so it needs no
fonts installed and renders identically everywhere.

Elements: a military patch (the *intti* pun — the army tent), camouflage
shared between the tent and the ring, a stove pipe with smoke, and the
integral as the centrepiece.

## Regenerating

`logo_body.svg` holds the artwork with `<!--INTEGRAL-->` and `<!--LETTERING-->`
placeholders; `build_logo.py` fills them by extracting glyph outlines with
fontTools and emitting the stencil notches:

```sh
cd assets && python3 build_logo.py     # writes logo.svg
rsvg-convert -w 1024 logo.svg -o logo-1024.png
```

Fonts used *at build time only* (their outlines are baked into the SVG):

| element   | font            |
|-----------|-----------------|
| wordmark  | DejaVu Sans Bold (+ hand-placed stencil bridges) |
| subtitle  | QTMilitary (stencil caps) |
| integral  | STIX Two Math |

Note: do not use Inkscape's "object to path" on the source — its font
fallback mangles the STIX integral glyph. `build_logo.py` extracts the
outlines directly from the font files instead.

## Palette

| role            | hex |
|-----------------|-----|
| cream (ink)     | `#F2EFE2` |
| field (dark olive) | `#2F3B21` |
| rim (dark)      | `#212A18` |
| ring (grey)     | `#A7AEA4` |
| camo olive      | `#6E7A4A` |
| camo dark       | `#414B2B` |
| camo brown      | `#6A5334` |
| camo khaki      | `#99A06E` |
