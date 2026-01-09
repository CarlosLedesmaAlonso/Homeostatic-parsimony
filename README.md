# Homeostatic Parsimony  
## Interoception + Relief Gradients (Δdrive) as Sufficient Conditions for Fast Homeostatic Recovery in Minimal RL

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/python-3.8%2B-blue)](https://www.python.org/)
[![Status](https://img.shields.io/badge/status-reproducible-brightgreen)](https://github.com/)

**Author:** Carlos Ledesma-Alonso  
**Affiliation (for identification only):** Universidad Anáhuac (Ph.D. Program in Applied Bioethics)  
*This repository reflects the author’s personal research and does not represent institutional positions.*
**Date:** January 2026  

---

## 1. Executive Summary

This repository contains source code and simulation outputs for a minimal computational study of **homeostatic agency**: the ability of an embodied controller to **recover internal stability** after a critical perturbation.

Using Reinforcement Learning (RL) agents in a controlled ablation study, we show that **interoception** (explicit access to internal state \(h_t\)) combined with a **relief-based value signal**—the **relief gradient**
\[
\Delta D_t = D_{t-1} - D_t
\]
is sufficient for rapid post-shock recovery in this regime. Conversely, the specific **global modulation** scheme tested here (dynamic learning rate/temperature plus a crisis override resembling acute panic) can induce exploratory rigidity and dramatically slower recovery.

A secondary motivation is bioethical: this toy model functions as a **conceptual pressure test** against modular narratives of brain–body replaceability, including speculative proposals of **cephalosomatic anastomosis** (“head transplant”) as if the body were a swappable chassis.

---

## 2. Theoretical Framework

1. **Agency is embodied:** it emerges from the coupling between internal constraints (\(h_t\)) and a control policy (\(\pi\)).
2. **Parsimony principle:** meta-cognitive control is not required for basic recovery if the system values **direction of change** (relief), not only external reward.
3. **Operational biorregulation:** internal variables must be used as a **criterion of value** and coupled to action/learning.  
   *Isolated biorregulators are not enough; operational biorregulation is.*

   In addition, this work is grounded in:
- **Systemic Materialism / Systemism (Mario Bunge):** we adopt Bunge’s systemist stance—i.e., explanation in terms of systems, their organization, and emergent properties—particularly as articulated in *Ontology II: A World of Systems* and in his methodological critique of individualism/holism in social science (“systemism” as the viable alternative).
- **Biorregulation and the Somatic Marker Hypothesis (Antonio Damasio and colleagues):** we use Damasio’s proposal that decision-making and adaptive behavior are guided by **signals arising from bioregulatory processes** (“somatic markers”), and we operationalize this idea computationally as **interoception** (access to internal state \(h_t\)) plus a **relief-gradient value signal** \(\Delta D_t = D_{t-1} - D_t\).

---

## 3. Experimental Design

We designed a micro-world (`SocialEnv`) where agents must manage two internal variables: **Energy** and **Social Integrity** (setpoints at 1.0). At \(t = 500\), a critical **vital shock** is induced for embodied agents, collapsing internal variables to near-failure levels (\(0.1\)).

### Architectures Compared (n = 20 seeds)

| Group | Architecture | Learning Mechanism |
| :--- | :--- | :--- |
| **NO-REG** (baseline) | External-only baseline; internal channel is clamped (non-informative). | External reward only (task reward). |
| **FULL** (modulated) | Embodied + interoception. | External reward + relief + **global modulation + panic override** (low temp, high LR, fixed-action bias). |
| **REG-NO-MOD** (parsimonious) | Embodied + interoception. | External reward + **relief gradient** (\(\Delta D_t\)); **no global parameter modulation**. |

**Important note:** In the current script, `NO-REG` is an external-performance baseline and is **not interpreted as a homeostatic recovery comparator**, because it does not implement operative physiology nor receive an internal shock.

---

## 4. Key Results

Simulation outputs (generated via `v9_final_experiment.py`) show a strong advantage of the parsimonious architecture (`REG-NO-MOD`) over the globally modulated panic architecture (`FULL`).

### Performance Metrics (Mean ± SD)

| Group | Recovery Time (Steps) | Accumulated Suffering (AUC of drive) | Interpretation |
| :--- | :--- | :--- | :--- |
| **FULL** | 257.1 ± 243.0 | 148.4 ± 152.2 | Slow and unstable recovery consistent with rigidity under panic-style control. |
| **REG-NO-MOD** | **10.5 ± 4.0** | **4.9 ± 1.2** | Rapid and robust homeostatic restoration. |
| **NO-REG** | N/A (baseline) | N/A (baseline) | External-only baseline; not a homeostatic recovery comparator in this script. |

### Visualization

Run the script to generate high-resolution figures (saved in the repository root):

- **Figure 1 (Recovery Dynamics):** mean ± SD drive trajectories around the shock.
- **Figure 2 (Recovery Time):** bar comparison of recovery time (FULL vs REG-NO-MOD), including variability across seeds.

---

## 5. Implications for Head Transplants (Conceptual)

This computational model is **not clinical evidence**. It is a minimal systems-level argument:

If survival-level agency depends on **operational biorregulation** (interoception + valuing relief gradients), then extreme brain–body decoupling is not merely a wiring problem. It is a **homeostatic control problem**: replacing the entire somatic substrate implies a maximal interoceptive disturbance and a mismatch between internal signals, value, and adaptive action.

**Somatic mismatch hypothesis (conceptual):** a brain coupled to a radically different body may fail to extract a stable relief gradient and may default to rigid, non-adaptive control modes under acute stress.

---

## 6. Reproduction

1. Clone this repository.
2. Install dependencies:
   ```bash
   pip install torch numpy matplotlib seaborn
Run:

python v9_final_experiment.py

Outputs:

Figure1_Recovery_Dynamics.png

Figure2_Recovery_Time.png

console summary (mean ± SD over seeds)

## 7. Transparency: AI-assisted development

Portions of this project (code refactoring, documentation drafts, and experimental wording) were developed with iterative assistance from Large Language Models (LLMs) (e.g., ChatGPT, Gemini).

All quantitative results reported in this repository are generated by the code provided here, using explicit hyperparameters and fixed random seeds. The full experimental pipeline (simulation + figure generation) is reproducible from the archived release (Zenodo DOI to be added after the first public release).

## 8. Citation

Please cite the Zenodo archive (DOI pending). Suggested format:

Ledesma Alonso, C. (2026). Homeostatic Parsimony: Interoception + Relief Gradients (Δdrive) as Sufficient Conditions for Fast Homeostatic Recovery in Minimal RL. GitHub repository. (Zenodo DOI pending)

(Recommended: add a CITATION.cff file after the Zenodo DOI is minted.)

## References (brief)

**Somatic markers / biorregulation (primary sources)**
- Damasio, A. R. (1996). *The somatic marker hypothesis and the possible functions of the prefrontal cortex.* **Philosophical Transactions of the Royal Society B**, 351(1346), 1413–1420. https://doi.org/10.1098/rstb.1996.0125  
- Bechara, A., Damasio, A. R., Damasio, H., & Anderson, S. W. (1994). *Insensitivity to future consequences following damage to human prefrontal cortex.* **Cognition**, 50(1–3), 7–15. https://doi.org/10.1016/0010-0277(94)90018-3  
- Bechara, A., Tranel, D., Damasio, H., & Damasio, A. R. (1996). *Failure to respond autonomically to anticipated future outcomes following damage to prefrontal cortex.* **Cerebral Cortex**, 6(2), 215–225. https://doi.org/10.1093/cercor/6.2.215  
- Bechara, A., & Damasio, A. R. (2005). *The somatic marker hypothesis: A neural theory of economic decision.* **Games and Economic Behavior**, 52(2), 336–372. https://doi.org/10.1016/j.geb.2004.06.010  
- Damasio, A. (1994). *Descartes’ Error: Emotion, Reason, and the Human Brain.* Putnam.  
- Damasio, A. (2010). *Self Comes to Mind: Constructing the Conscious Brain.* Pantheon.

**Systemism / systemic materialism (Bunge: the specific anchors used here)**
- Bunge, M. (1979). *Treatise on Basic Philosophy, Vol. 4: Ontology II — A World of Systems.* Dordrecht: D. Reidel.  
- Bunge, M. (2000). *Systemism: the alternative to individualism and holism.* **The Journal of Socio-Economics**, 29(2), 147–157. https://doi.org/10.1016/S1053-5357(00)00058-5
