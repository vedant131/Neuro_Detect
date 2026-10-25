# SCRIPT — locked narration (one narrator)

Every number traces to the final report / metrics.json. ≈ 360 words → ≈ 165 s of voice inside 180 s of picture.

## s1-hook (0–15)
The brain never stops talking. Billions of cells, signalling in tiny pulses of electricity. Those signals leave a trace. Can we learn to read them?

## s2-problem (15–38)
Alzheimer's disease is the most common cause of dementia, and it is often recognised late. The tests that confirm it are costly, and specialists are scarce. In low-income countries, there is about one neurology specialist for every million people. An EEG is different. It is cheap, painless, and already in most hospitals.

## s3-intro (38–53)
NeuroDetect reads one short, resting EEG recording. It needs just five of the usual nineteen sensors. Three for Alzheimer's-type changes, and two for a mood marker.

## s4-pipeline (53–82)
Here is the journey of one recording. The signal is filtered to between half a hertz and forty-five hertz, and resampled to two hundred and fifty samples a second. Then it is cut into ten-second slices. Slices spoiled by blinks or movement are thrown away. Every clean slice becomes a few numbers, called biomarkers, and each slice is judged on its own.

## s5-modules (82–105)
Two paths run side by side. The Alzheimer's path compares slow theta waves with alpha waves at the back of the head, and checks how closely the front and back of the brain stay in step. The second path measures frontal alpha asymmetry, a depression-related marker. It is preliminary, and shown as indicative only.

## s6-classifier (105–125)
The classifier is deliberately simple. It learns one typical healthy pattern and one typical Alzheimer's pattern, and asks which a slice is closer to, using cosine distance. Every slice casts a vote. The whole model holds just sixteen learned parameters, about one kilobyte, and runs on an ordinary laptop.

## s7-results (125–147)
Tested on sixty-five people it had never seen, and repeated ten times, the Alzheimer's module reached eighty-two point six percent balanced accuracy, and correctly cleared ninety-four percent of healthy people. Three electrodes beat the full nineteen. The depression marker scored fifty-five percent: close to chance.

## s8-product (147–165)
And it is a working prototype. Upload a recording, and within seconds you get a plain-language verdict, the vote of every slice, the measurements behind it, and a P D F report.

## s9-responsible (165–175)
NeuroDetect is a screening aid, not a diagnosis. Before it is used in clinics, it needs testing on data from other hospitals.

## s10-end (175–180)
NeuroDetect. Making brain-wave screening more accessible.
