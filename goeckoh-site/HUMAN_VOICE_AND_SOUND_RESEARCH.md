# Human Voice & Sound — Research Reference

> Scientific foundation under the Goeckoh DSP engine. Each section links the acoustics/physiology/perception back to where it lives in the code (`dsp_engine.py`, `realtime_loop.py`, `phoneme_atlas.json`, the Bark-space classifier, the breath layer, and the Self-Correction Hypothesis).
> Compiled 2026-06-16.
>
> **Provenance:** synthesized primarily from secondary research summaries; the linked primary sources are topically authoritative but were not all individually fetched and verified line-by-line. The DIVA dual-control architecture and its babbling→imitation stages (§8.3) were confirmed against the cited primary source; the PNAS-2024 corollary-discharge specifics (§8.1) are from the paper's abstract/summary (full text paywalled) and should be re-checked against the PDF before being quoted in any grant, clinical, or public-facing material.

---

## 0. The one-paragraph map

Speech is **a buzz shaped by a tube**. The vocal folds chop airflow into a periodic buzz (the **source**, carrying pitch = F0); the throat/mouth/nose act as a resonant **filter** that boosts certain frequencies (**formants**, carrying vowel identity). The ear decomposes this on a roughly logarithmic frequency axis (**critical bands / Bark scale**) — which is exactly why Goeckoh measures formant error in Bark, not Hz. And the brain *predicts* the sound of its own voice before producing it (**corollary discharge / efference copy**); when prediction matches reality, the auditory cortex is suppressed (it "expected that"). That predict-compare-correct loop (formalized by the **DIVA model**) is the academic backbone of the **Self-Correction Hypothesis**: feed the child their own voice, corrected, inside the prediction window, and the loop re-converges.

---

## 1. Source–Filter Theory — the master framework

Speech production is two independent stages (Fant, 1960):

1. **Source** — air from the lungs forces the **vocal folds** (two muscular folds in the larynx) to open and snap shut repeatedly (**phonation**). This produces a periodic pulse train — a buzz. The repetition rate is the **fundamental frequency (F0)**, perceived as **pitch**. The source spectrum is harmonically rich: energy at F0 and all its integer multiples, rolling off ~−12 dB/octave.
2. **Filter** — the **vocal tract** (pharynx + oral cavity + nasal cavity), a tube ~17 cm long in adult males, has **resonances** that amplify frequency bands near its natural modes and attenuate others. These resonances are the **formants**.

**Key consequence — independence:** the source determines *pitch & voice quality*; the filter determines *timbre & phonetic identity (which vowel)*. You can change one without the other.

> **This independence is the entire reason Goeckoh works.** `DSPEngine.process_frame` shifts F1/F2 (the **filter**) toward target vowels while leaving the **excitation** (F0, voice quality, prosody) untouched. Result: the corrected output still *sounds like the child* (F0 unchanged +0.0 Hz on all 5 benchmark vowels) but articulates the intended vowel. That is source-filter separation exploited deliberately. See `engine/dsp_engine.py` LPC analysis → formant shift → re-synthesis through corrected filter.

- LPC is "a close approximation of the reality of speech production" precisely because it *is* the source-filter model: buzz at the end of a tube. ([Wikipedia – LPC](https://en.wikipedia.org/wiki/Linear_predictive_coding))
- [Source–Filter Theory of Speech (Oxford RE)](https://oxfordre.com/linguistics/display/10.1093/acrefore/9780199384655.001.0001/acrefore-9780199384655-e-894) · [MIT 24.915 lecture (PDF)](https://ocw.mit.edu/courses/24-915-linguistic-phonetics-fall-2015/8f4e9d5d8ea634dbbf0e2da8b92675ea_MIT24_915F15_lec4.pdf) · [VoiceScience lexicon](https://www.voicescience.org/lexicon/source-filter-theory/)

---

## 2. The Source — Pitch (F0), harmonics, voice quality

### 2.1 Fundamental frequency & pitch
- F0 = vocal-fold vibration rate; the **lowest** frequency component. Set by vocal-fold **length, thickness, tension**.
- Typical speaking F0:
  - **Adult male** ≈ 116 Hz (range ~93–135 Hz)
  - **Adult female** ≈ 205 Hz (range ~162–238 Hz)
  - **Pre-pubescent child** ≈ 223 Hz (sex-independent)
- Puberty: testosterone lengthens/thickens male folds → the characteristic pitch drop.
- **Missing fundamental:** the brain reconstructs pitch from the harmonic *pattern* even when F0 itself is filtered out — pitch is inferred from harmonic spacing, not the literal lowest frequency.

> Goeckoh extracts F0 by **autocorrelation, 80–500 Hz** (`_extract_f0`) — a band that comfortably spans child + adult voices. F0 is *measured but preserved*, used as a voice-identity invariant, not a correction target.

- [What frequency is the human voice? (ScienceInsights)](https://scienceinsights.org/what-frequency-is-the-human-voice/) · [VoiceScience – Frequency](https://www.voicescience.org/lexicon/frequency/)

### 2.2 Harmonics
- Integer multiples of F0. They give the voice its rich texture and are the "carrier" the formants sculpt. Intelligible speech needs spectrum out to **~8 kHz**, far above F0.

### 2.3 Voice quality (the source's "personality") — jitter, shimmer, HNR, spectral tilt
- **Jitter** — cycle-to-cycle *frequency* perturbation (F0 instability). Elevated in pathological/aging/dysarthric voices.
- **Shimmer** — cycle-to-cycle *amplitude* perturbation.
- **HNR (harmonics-to-noise ratio, dB)** — ratio of periodic (harmonic) energy to turbulent noise from the glottis. Normal adults ≈ 11–13 dB on sustained vowels (≥7.4 dB minimum). Lower HNR = breathier/rougher/more pathological.
- **Spectral tilt** — overall slope of the spectral envelope; flatter = pressed/loud, steeper = breathy. Carries the **lip-rounding fingerprint**.

> These are exactly the **"bubble" psychoacoustic features** Goeckoh found could *tag collapsed back vowels that F1/F2 cannot see* (`bubble_detect.py`, `attempt_analysis.py`): spectral_tilt + HNR + ZCR pushed front/back LDA from 73%→77%. **Spectral tilt carries the rounding cue** that F1/F2/F3 lose in dysarthria — the project's first label-free signal to beat baseline. The literature confirms tilt encodes phonation/rounding; this is not a coincidence.

- [Comprehensive review of jitter, shimmer, HNR (Koffi 2025, PDF)](https://repository.stcloudstate.edu/cgi/viewcontent.cgi?params=/context/stcloud_ling/article/1155/&path_info=1_Koffi2025ComprehensiveReviewOfJitterShimmerHNR.pdf) · [HNR as index of vocal aging](https://www.sciencedirect.com/science/article/abs/pii/S0892199702001236)

---

## 3. The Filter — Formants & the vowel space

### 3.1 What formants are
- **Formants (F1, F2, F3, …)** = vocal-tract resonance peaks. The **lowest two carry nearly all vowel identity**:
  - **F1 ↔ tongue height / jaw openness** (inverse): low F1 = high/close vowel (/i/, /u/); high F1 = low/open vowel (/a/).
  - **F2 ↔ tongue front/back** (advancement): high F2 = front (/i/, /æ/); low F2 = back (/u/, /ɑ/).
  - **F3 + higher** ↔ finer detail, rhotics (/r/), speaker identity.
- A vowel is essentially a **point in (F1, F2) space**.

### 3.2 The acoustic vowel space — Hillenbrand et al. (1995)
- The canonical dataset: F1–F4 in ~1.5k recordings, **139 speakers** (men, women, children), 12 vowels each.
- Plotting all vowels in F1/F2 yields the classic **vowel quadrilateral**. Categories form ellipses; **adjacent vowels overlap** (/i/–/ɪ/ small overlap; /æ/–/ɛ/ large). So even *healthy* vowel identity is statistical, not crisp — a key caveat for any classifier.
- Free public data: `homepages.wmich.edu/~hillenbr/voweldata.html`.

> Goeckoh's `phoneme_atlas.json` is exactly this concept made operational: each phoneme = (F1, F2, F3) target + per-formant std. **Crucial project finding:** Hillenbrand targets are *not always right for a given speaker population*. Hillenbrand `IY` (342/2322) caused a classification regression on TORGO males who naturally cluster at `IH`; the demo-WAV-calibrated `IY` (571/1819) was kept. Lesson encoded in memory: **use canonical targets for remapping, population/speaker-matched targets for classification.**

- [Hillenbrand 1995 vowel data](https://www.ling.upenn.edu/courses/cogs501/Hillenbrand.html) · [Formants of vowels (EduHK)](https://corpus.eduhk.hk/english_pronunciation/index.php/2-2-formants-of-vowels/) · [Vowel space development in children (PMC)](https://pmc.ncbi.nlm.nih.gov/articles/PMC2597712/)

### 3.3 Children's vowel space
- Children have **shorter vocal tracts → higher formants** (and higher F0). The vowel space is *larger and shifts down/left* with age as the tract lengthens. Any atlas/normalization must be **age/speaker-aware** — a generic adult atlas mis-scales a child.

---

## 4. Consonants & dynamics (where vowel-only models stop)

Vowels are quasi-stationary; consonants are **events**. Three descriptors:
- **Place of articulation** — *where* the constriction is (bilabial, alveolar, velar, glottal…). Acoustically **variable and fragile**, especially in connected speech.
- **Manner** — *how* air flows (stop, fricative, nasal, approximant, lateral). Acoustically **salient and robust** — manner survives where place blurs; it drives perception.
- **Voicing** — are the folds vibrating (/b/ vs /p/)?

**Coarticulation:** articulators move continuously, so a phoneme's acoustics bleed into its neighbors. Consonant onsets are marked by sharp **acoustic landmarks** (stop release, frication onset, nasal murmur).

> Relevance to Goeckoh: the engine is **vowel-centric** (formant correction on voiced frames) and explicitly gates out non-vowel frames (`if 150 < f1 < 1100 and f2 > f1+200`). Coarticulation is the **untested temporal lever** the project flagged: **formant *trajectories*/velocity** could disambiguate collapsed vowels that are frame-locally identical — the most promising path to label-free routing toward the oracle (0.95 Bark). Manner-vs-place asymmetry explains why purely spectral, frame-local classification hits a wall: the robust cues are *dynamic*, not static.

- [Manner of articulation (Wikipedia)](https://en.wikipedia.org/wiki/Manner_of_articulation) · [Coarticulation as synchronised CV co-onset](https://www.sciencedirect.com/science/article/abs/pii/S0095447021000917)

---

## 5. Perception — the ear, critical bands, and the Bark scale

- The **cochlea** performs a mechanical frequency analysis: the **basilar membrane** is tonotopic — high frequencies resonate at the base, low at the apex. Each point acts as a **band-pass (auditory) filter**.
- **Critical bands:** the ear integrates energy within bands of roughly constant *cochlear distance* (~0.9–1.3 mm each, independent of center frequency). Sounds within one critical band interact (mask each other); across bands they don't.
- **Bark scale (Zwicker, 1961):** 24 Barks = the first 24 critical bands. **Below ~500 Hz ≈ linear in Hz; above ~500 Hz ≈ logarithmic.** This warps Hz into *perceptual* distance.

> **This is why every Goeckoh metric is in Bark, not Hz.** A 100 Hz error at F1=300 is perceptually huge; the same 100 Hz at F2=2300 is nearly inaudible. Measuring formant error in Bark makes "distance" match *what the brain actually hears*. The classifier's nearest-phoneme match, the +33.9% improvement numbers, the remap validation — all Bark-domain. The project even learned to **distrust % Bark improvement for near-canonical vowels** (it explodes when d_before is small) and report **absolute Bark** instead. Pure psychoacoustics driving an engineering discipline.

- [Bark scale (Wikipedia)](https://en.wikipedia.org/wiki/Bark_scale) · [Critical band (Wikipedia)](https://en.wikipedia.org/wiki/Critical_band) · [Bark scale & critical bands (Ansys)](https://ansyshelp.ansys.com/public/Views/Secured/corp/v242/en/Sound_SAS_UG/Sound/UG_SAS/bark_scale_and_critical_bands_179506.html)

---

## 6. LPC — the bridge from physics to code

**Linear Predictive Coding** models each sample as a linear combination of past samples — which is mathematically equivalent to modeling the vocal tract as an **all-pole filter** driven by the source. It *is* the source-filter model in discrete time.

- Estimate filter coefficients (Levinson–Durbin) → the filter captures the **formants** (spectral envelope); the **residual** captures the **source** (buzz/noise).
- **Formants = roots of the LPC polynomial** near the unit circle; peaks of the LPC spectrum.
- **Model order rule:** order ≈ 2 × (expected formants) + 2. Goeckoh uses **order 12** at 16 kHz → comfortably resolves F1–F3 (and headroom).

> This is the literal spec of `DSPEngine`: pre-emphasis + Hamming → LPC (Levinson-Durbin, order 12) → root-finding for F1/F2/F3 → shift toward atlas → **re-synthesize through the corrected all-pole filter** while reusing the original residual (= keep the source, change the filter). The C++ port (`goeckoh_dsp.hpp`) is the same algorithm; the Rust/C++/Python speed shootout is all timing *this* pipeline.

- [LPC (Wikipedia)](https://en.wikipedia.org/wiki/Linear_predictive_coding) · [Formant estimation with LPC (MATLAB)](https://www.mathworks.com/help/signal/ug/formant-estimation-with-lpc-coefficients.html)

---

## 7. Dysarthria — what breaks, acoustically

The disorder Goeckoh targets. Motor-speech impairment → articulatory **undershoot**:
- **Vowel-space compression / centralization** — vowels fail to reach canonical formant targets; the whole quadrilateral shrinks toward the center (schwa).
- **Reduced F2 slope** — sluggish tongue front/back movement flattens formant transitions (the coarticulation cue weakens).
- **Reduced lingual/labial/jaw excursion & velocity**, aberrant timing, unstable formants.
- Clinical metrics: **Vowel Space Area (VSA)** and **Formant Centralization Ratio (FCR)** — both correlate with intelligibility.

> This is *exactly* the wall the project measured: TORGO males produce nearly all vowels in the `IH` region (F1 250–500, F2 1700–2100). **Back vowels are acoustically unrecoverable from F1/F2/F3** because tongue-retraction *and* lip-rounding are both reduced — the cue is physically absent, not merely noisy (F3 identical between intended-front and intended-back groups; intended-/u/ F2 fronted to 1563–1935). The literature's "centralization + reduced F2 slope" is the macroscopic name for Goeckoh's frame-level finding. It also motivates the two real fixes: **(a) prompted/known-word mode** (+72% back, zero detection ambiguity) and **(b) vowel-space remapping** — Lobanov speaker-normalization run *backwards* to expand the compressed polygon onto the true canonical space (back 3.47→2.45 Bark on held-out multi-speaker TORGO).

- [Vowel acoustics in dysarthria (Lansford & Liss, PDF)](https://directory.cci.fsu.edu/files/2014/07/Lansford-and-Liss_2014a.pdf) · [Assessing vowel centralization (PMC)](https://pmc.ncbi.nlm.nih.gov/articles/PMC6194930/) · [Formant Centralization Ratio (ASHA)](https://academy.pubs.asha.org/2018/08/formant-centralization-ratio-a-proposal-for-a-new-acoustic-measure-of-dysarthric-speech/)

---

## 8. The feedback loop — corollary discharge, SIS, and DIVA (the SCH backbone)

This is the neuroscience that makes Goeckoh a *therapy*, not just a vocoder.

### 8.1 Corollary discharge / efference copy
- When the motor cortex issues a speech command, it sends a copy (**efference copy → corollary discharge**) to the **auditory cortex**, which **predicts the sound** the command will produce.
- Direct human evidence: a discharge signal travels from **ventral motor cortex → auditory cortex before speech onset**, reproducible across tasks/patients, and **predicts the degree of auditory suppression** ([PNAS 2024](https://www.pnas.org/doi/10.1073/pnas.2404121121)).

### 8.2 Speaking-Induced Suppression (SIS)
- Self-generated speech elicits a **smaller** auditory-cortex response than identical externally-played sound. Measured as a **reduced N100 (N1) ERP**.
- **The amount of suppression scales with how well the heard sound matches the prediction.** Mismatch → less suppression → a prediction-error signal.

### 8.3 DIVA model (Guenther) — the computational formalization
- Speech = **feedforward** commands (premotor→motor, + cerebellum) *plus* **feedback control**: auditory & somatosensory systems compare output to **sensory predictions** and compute **error signals** that correct future commands.
- **Development (babbling):** Stage 1 — semi-random articulator movements produce auditory/somatosensory feedback that *tunes* the sensory-motor maps. Stage 2 — the learner imitates ambient-language sounds. **Hearing one's own vocalizations is critical for canonical babbling.** Disordered auditory-motor interaction → motor-speech disorders.

> **This is the Self-Correction Hypothesis, stated in the literature's own terms.** Goeckoh's premise — "close the broken corollary-discharge loop; brain hears itself succeed; prediction error collapses; inner speech develops" — maps 1:1 onto: corollary discharge (the prediction), SIS (the comparison), DIVA error signals (the learning). The **200 ms / ~13 ms achieved** latency target exists *because* corollary discharge is **temporally precise**: the corrected voice must arrive inside the window where the auditory cortex is still gated by that pre-onset prediction. Deliver the child's own-voice-corrected formants *within* that window and the heard signal aligns with a *better* target than they produced — the loop re-converges on the corrected articulation while preserving identity (own F0/voice quality). The **breath time-layer** (articulation cleanest at breath onset, decays through the breath; built into `BreathTracker`) is a respiratory-motor refinement of the same DIVA feedforward-planning idea: fresh motor plan + full subglottal pressure = best production = best self-reference target.

- [A corollary discharge circuit in human speech (PNAS 2024)](https://www.pnas.org/doi/10.1073/pnas.2404121121) · [Single-trial speech suppression of auditory cortex (J Neurosci)](https://www.jneurosci.org/content/30/49/16643) · [Inner speech corollary discharge (NeuroImage)](https://www.sciencedirect.com/science/article/abs/pii/S1053811919303246) · [Neurocomputational modeling of speech motor development / DIVA (PMC)](https://pmc.ncbi.nlm.nih.gov/articles/PMC10615680/) · [Auditory-motor interactions in pediatric motor speech disorders](https://pubmed.ncbi.nlm.nih.gov/24491630/)

---

## 9. How it all chains in Goeckoh (one figure in words)

```
LUNGS ──air──► VOCAL FOLDS ──buzz (F0, harmonics, voice quality)──► VOCAL TRACT ──formants──► RADIATED SPEECH
   │  (source: §2)                                       (filter: §3)
   │
   └── controlled by MOTOR CORTEX, which sends a COROLLARY DISCHARGE to AUDITORY CORTEX
            predicting the sound → SIS suppresses the match, flags the error (§8)

GOECKOH inserts itself in the radiated-speech path, in real time:
   mic ─► VAD (voiced only) ─► LPC analysis (§6) ─► extract F1/F2/F3, F0
       ─► Bark-space (§5) nearest-phoneme match vs atlas (§3.2)
       ─► shift FILTER formants toward target  [keep SOURCE: F0/quality/prosody]
       ─► breath-position weighting (§8.3) ─► re-synth ─► speaker
   ...all inside the corollary-discharge window (§8) so the brain hears
      ITSELF, corrected → prediction error collapses → SCH.
```

---

## 10. Open scientific threads the research highlights (project-relevant)

1. **Temporal/coarticulatory cues (§4)** are the strongest *untested* lever for label-free vowel disambiguation — formant trajectories survive where static F1/F2 collapse. The literature's "manner robust / place fragile / coarticulation carries info" supports investing here.
2. **Spectral tilt as the rounding fingerprint (§2.3)** — already validated as the first label-free back-vowel signal; the voice-quality literature says there's more signal in tilt/HNR to extract.
3. **Speaker/age-specific atlases (§3.3)** — children's higher formants mean a single fixed atlas is suboptimal; per-child calibration (which the project flags as under-tested and likely much better than pooled multi-speaker) is *predicted* to help by the developmental vowel-space data.
4. **Latency budget is neuroscience-bounded, not arbitrary (§8)** — the <200 ms target is set by corollary-discharge timing; the achieved ~13 ms is comfortably inside, validating real-time feasibility of the SCH.

---

### Source index
Source–filter: [Oxford RE](https://oxfordre.com/linguistics/display/10.1093/acrefore/9780199384655.001.0001/acrefore-9780199384655-e-894), [MIT 24.915](https://ocw.mit.edu/courses/24-915-linguistic-phonetics-fall-2015/8f4e9d5d8ea634dbbf0e2da8b92675ea_MIT24_915F15_lec4.pdf), [VoiceScience](https://www.voicescience.org/lexicon/source-filter-theory/) ·
F0/pitch: [ScienceInsights](https://scienceinsights.org/what-frequency-is-the-human-voice/), [VoiceScience](https://www.voicescience.org/lexicon/frequency/) ·
Voice quality: [Koffi 2025](https://repository.stcloudstate.edu/cgi/viewcontent.cgi?params=/context/stcloud_ling/article/1155/&path_info=1_Koffi2025ComprehensiveReviewOfJitterShimmerHNR.pdf), [HNR aging](https://www.sciencedirect.com/science/article/abs/pii/S0892199702001236) ·
Formants/vowels: [Hillenbrand 1995](https://www.ling.upenn.edu/courses/cogs501/Hillenbrand.html), [EduHK](https://corpus.eduhk.hk/english_pronunciation/index.php/2-2-formants-of-vowels/), [child vowel space](https://pmc.ncbi.nlm.nih.gov/articles/PMC2597712/) ·
Consonants/coarticulation: [Manner (Wikipedia)](https://en.wikipedia.org/wiki/Manner_of_articulation), [coarticulation](https://www.sciencedirect.com/science/article/abs/pii/S0095447021000917) ·
Psychoacoustics/Bark: [Bark scale](https://en.wikipedia.org/wiki/Bark_scale), [critical band](https://en.wikipedia.org/wiki/Critical_band), [Ansys](https://ansyshelp.ansys.com/public/Views/Secured/corp/v242/en/Sound_SAS_UG/Sound/UG_SAS/bark_scale_and_critical_bands_179506.html) ·
LPC: [Wikipedia](https://en.wikipedia.org/wiki/Linear_predictive_coding), [MATLAB](https://www.mathworks.com/help/signal/ug/formant-estimation-with-lpc-coefficients.html) ·
Dysarthria: [Lansford & Liss](https://directory.cci.fsu.edu/files/2014/07/Lansford-and-Liss_2014a.pdf), [vowel centralization](https://pmc.ncbi.nlm.nih.gov/articles/PMC6194930/), [FCR](https://academy.pubs.asha.org/2018/08/formant-centralization-ratio-a-proposal-for-a-new-acoustic-measure-of-dysarthric-speech/) ·
Feedback loop: [PNAS 2024 corollary discharge](https://www.pnas.org/doi/10.1073/pnas.2404121121), [SIS J Neurosci](https://www.jneurosci.org/content/30/49/16643), [inner speech CD](https://www.sciencedirect.com/science/article/abs/pii/S1053811919303246), [DIVA development](https://pmc.ncbi.nlm.nih.gov/articles/PMC10615680/), [pediatric motor speech](https://pubmed.ncbi.nlm.nih.gov/24491630/)
</content>
</invoke>
