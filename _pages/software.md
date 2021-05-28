---
layout: archive
permalink: /software/
author_profile: true
title: "Software"
header:
  overlay_image: /assets/images/research/NCI-600px.jpg
  overlay_filter: 0.5
  caption: "Photo credit: [**Unsplash**](https://unsplash.com)"
---

Below are the software/pipelines I have designed and coded for different research projects.

## [SelfingMutAccum](https://github.com/EddieKHHo/SelfingMutAccum){:target="blank"}

SelfingMutAccum simulates a self-fertilizing population who genomes contain a mixture of loci experiencing time constant selection (C-loci) and loci experiencing selection that fluctuates over time (L-loci). Selection fluctuations are auto-correlated in time and can change in strength and direction. Data is continually outputted at as the simulation runs.

Language: C++

## [simMutAccumTE](https://github.com/EddieKHHo/simMutAccumTE){:target="blank"}

simMutAccumTE simulates a mutation accumulation (MA) experiment by inserting and/or deleting a number of transposable elements (TEs) in a diploid genome of a focal MA line while leaving the ancestral (ANC) genomes intact. After mutations are simulated in the given genome,  paired-end reads are generated for all lines for downstream bioinformatic processing.

Language: Python

Dependencies: [pIRS](https://github.com/galaxy001/pirs){:target="blank"}, [seqtk](https://github.com/lh3/seqtk){:target="blank"}

## [simMutAccumSV](https://github.com/EddieKHHo/simMutAccumSV){:target="blank"}

simMutAccumSV simulates a mutation accumulation (MA) experiment by introducing structural variants (deletion, duplications, and inversions) into a diploid genome of a focal MA lines while leaving the ancestral genomes intact. After mutations are simulated in the given genome,  paired-end reads are generated for all lines for downstream bioinformatic processing.

Language: Python

Dependencies: [pIRS](https://github.com/galaxy001/pirs){:target="blank"}, [SVsim](https://github.com/GregoryFaust/SVsim){:target="blank"}

