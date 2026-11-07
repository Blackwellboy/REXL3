## Highlights

- DeepSeek-V4.1-Flash-1017 (EXL3 1.55 bits per weight) on one GB10: 262,144-token window, one slot, speculative decoding and prefix cache on, image input off (`dsv41-single-262k`, port 8940). The two-node profile is not part of this release. Added by #75.

## Known issues

- At temperature 0 the model can repeat a phrase until the reply cap; use the sampling defaults for long reasoning. In the release quality run the hard-reasoning set scored 16 of 20, one case under its regression bar, and was accepted for v0.1.0.
