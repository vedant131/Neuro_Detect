# REVIEW — NeuroDetect film (pre-render)

Length 182.5 s, 1920×1080, 30 fps. `check`: 0 errors.

## Audio
- Narration: HeyGen TTS, 10 clips (167.8 s), placed per scene; scenes retimed to word timings.
- Music: HeyGen library track (153 s) crossfade-extended to 183.5 s (`assets/bgm/bed.wav`), vol 0.14, carved under the voice (carve.mjs, strength 0.8).
- SFX (bundled): riser 2.2 s (hook), whoosh 40.5 s (title), bass impact 138.3 s (82.6% lands), chime 176.4 s (end card).

## Captions
- 96 groups from HeyGen word timings, 3–6 words, one at a time, bottom band (y 948), no overlaps.

## Accuracy
- All numbers from `Full Code/models/metrics.json`. Datasets shown separately (ds004504 36/29; ds003478 121). FAA labelled preliminary / indicative only; depression 54.9% shown against the 50% chance line.
- Scene 8 says "within seconds" (measured ≈ 6.5 s upload→result, sample sub-001), not "about a second".
- Illustrative elements marked: Scene 2 figure, slice votes in Scenes 4/6, 2-D sketch of the feature space.

## Known, accepted warnings
- Dimmed electrode labels (Scenes 3, 5) are deliberately low-contrast.
- Some mono labels are 22–26 px (below the 28 px phone floor); readable at 1080p.
- Brief still holds in Scene 4 (the trace freezes once sliced) while narration continues.
