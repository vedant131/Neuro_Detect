# SPEC — NeuroDetect film

Concept: the film is a signal being read — every scene is drawn on the same dark EEG chart-paper, and the cyan trace is the thread that carries from the hook to the product.

## LOOK
- Ground #060B18, panel #0D1630 (1px border #1E2A4A), grid lines #12203D at 40 px.
- Ink #F4F7FB, muted #8A97B5. Cyan #22D3EE (Alzheimer's path / Fz O1 O2), violet #A78BFA (mood path / F3 F4), amber #F5B544 (cautions, AD class), red never.
- Type: Outfit 800 for headlines (88–140 px, tracking −0.02em), Outfit 400/600 body 30–40 px; IBM Plex Mono 22–26 px for labels and units only.
- Glow: text-shadow/box-shadow ≤ 24 px blur at ≤ 45% alpha, on cyan/violet accents only.
- Particles: seeded (mulberry32), ≤ 900 points, drawn from timeline time.

## MOTION
- Entrances: power3.out 0.6–0.9 s, y 30 px + opacity. Exits: power2.in 0.4 s.
- Numbers count up over 1.6 s with power2.out. Bars grow with expo.out 1.2 s.
- One camera-like drift per scene (scale 1 → 1.04 over the scene, ease none).
- Scene seams: 0.5 s fade through ground.

## SHOTS (root time)
| Scene | Slot | Key frame |
|---|---|---|
| s1-hook | 0–15 | point-cloud brain over a live trace; "THE BRAIN LEAVES SIGNALS." → "CAN WE LEARN TO READ THEM?" |
| s2-problem | 15–38 | stylised silhouette left, three fact cards right, turn to EEG + "EARLIER INSIGHT. GREATER ACCESS." |
| s3-intro | 38–53 | top-view 10–20 head, 19 dots dim, Fz/O1/O2 cyan + F3/F4 violet, title + subtitle |
| s4-pipeline | 53–82 | one trace transformed step by step; 6 step chips |
| s5-modules | 82–105 | split: θ/α + θ coherence (cyan) vs FAA (violet, amber "PRELIMINARY · INDICATIVE ONLY") |
| s6-classifier | 105–125 | cosine plane with two centroids, slice votes, 4 spec tiles |
| s7-results | 125–147 | 82.6% hero, 3 side metrics; 3 vs 19 electrodes; depression 54.9% vs chance line |
| s8-product | 147–165 | real app screenshots in a browser frame, callouts |
| s9-responsible | 165–175 | three statements |
| s10-end | 175–180 | end card |

## BANNED
- Any clinical/diagnostic claim; "detects depression"; a single 3-class model; hardware/device imagery; photos of real people; tilted screenshots; red alarm colours; stock "AI brain" glow clichés over text.

## QA
- Every number matches models/metrics.json. Datasets shown as separate (ds004504 36 AD / 29 HC; ds003478 121 BDI).
- Text inside 80% title-safe; captions band bottom 160 px kept clear of key copy.
