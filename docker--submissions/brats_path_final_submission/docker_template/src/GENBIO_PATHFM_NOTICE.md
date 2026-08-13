# Notice

This project vendors `genbio_pathfm_model.py`, a copy of the model
definition code from the official [`genbio-ai/genbio-pathfm`](https://github.com/genbio-ai/genbio-pathfm)
repository (`genbio_pathfm/model.py`), and loads GenBio-PathFM model weights
(`model.pth`, obtained separately from
[huggingface.co/genbio-ai/genbio-pathfm](https://huggingface.co/genbio-ai/genbio-pathfm))
into that vendored code as one of three foundation-model feature extractors
in this ensemble.

This is licensed under the GenBio AI Community License Agreement,
Copyright © GENBIO.AI, INC. All Rights Reserved. See
`GENBIO_PATHFM_LICENSE.txt` in this directory for the full text.

**Powered by GenBio AI.**

## Modifications from the original

- Only the model-definition code (`VisionTransformer`, `GenBio_PathFM_Inference`,
  and their supporting layers) was copied, unmodified, from
  `genbio_pathfm/model.py`.
- No modifications were made to the copied code itself. It is loaded and
  called from `src/brats_path_extract.py` in place of the
  `transformers.AutoModel(..., trust_remote_code=True)` path documented as
  "Option 1" in the upstream README, using the upstream README's own
  "Option 2: pip package" usage pattern instead.

Use of GenBio-PathFM (weights, code, and any outputs) under this package is
subject to the Non-Commercial restriction and other terms of the GenBio AI
Community License -- see the LICENSE.txt in this directory before
distributing or using outputs commercially.
