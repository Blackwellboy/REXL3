## Highlights

- GLM-5.3-Flash (EXL3 2.25 bits per weight) with speculation from the pack's NextN layer, prefix cache and image input on, in two profiles of one recipe: `glm53-flash-single` on one GB10 (262,144-token window, one slot) and `glm53-flash-tp2-1m` on two GB10 nodes with tensor parallelism (1,048,576-token window), both on port 8950. Added by #69.
