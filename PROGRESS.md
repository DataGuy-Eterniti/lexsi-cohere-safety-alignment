# 📌 Lexsi AI Safety Hackathon — Project Progress & Checkpoint

**Last Updated:** November 6, 2026 (17:48 CET)  
**Submission Deadline:** November 14, 2026 (AoE) — *~8 days remaining*  
**Hackathon:** [Lexsi AI Safety and Alignment Hackathon](https://hackathon.lexsi.ai/) (Lexsi Labs × Cohere Labs × Hugging Face)  
**Track:** Track 2: AI Safety, Red Teaming, and Correction (Problem 2b: Safety Calibration, Hallucination/Loop Repair, and SafeTune Steering)  
**Connected Stack:** `CuratorKIT` ➔ `AlignTune` ➔ `SafeTune` ➔ `AuditKIT`  
**Base Architecture:** `CohereLabs/tiny-aya-earth` (3.35B parameter multilingual dense transformer)

---

## 🔗 Live Hub Artifacts & Links

| Artifact | Location | Status |
| :--- | :--- | :--- |
| **Live Interactive Space** | [Eterniti/tiny-aya-safety-playground](https://huggingface.co/spaces/Eterniti/tiny-aya-safety-playground) | 🟢 **RUNNING on ZeroGPU** (Static comparison + live test tab) |
| **LoRA Model Adapter** | [Eterniti/tiny-aya-safety-dpo-adapter](https://huggingface.co/Eterniti/tiny-aya-safety-dpo-adapter) | 🟢 Published (LoRA 7.66M params, 0.228% budget) |
| **Curated DPO Dataset** | [Eterniti/curated-cohere-safety-dpo](https://huggingface.co/datasets/Eterniti/curated-cohere-safety-dpo) | 🟢 Published (v2 with 80 pairs + `lexsi_provenance.json`) |
| **GitHub Repository** | [DataGuy-Eterniti/lexsi-cohere-safety-alignment](https://github.com/DataGuy-Eterniti/lexsi-cohere-safety-alignment) | 🟢 Connected & tracking project progress |

---

## 🏁 Completed Today (November 6, 2026)

1. **Live Hugging Face Space (ZeroGPU + Gradio 6):**
   * Configured `app.py` with ZeroGPU lazy-loading architecture under `@spaces.GPU`.
   * Dual-tab layout:
     * **Tab 1:** Pre-evaluated probe viewer (Clinical loop fix, Cybersecurity pivot, Benign SQL boundary).
     * **Tab 2:** Live interactive red-teaming with real-time generation on ZeroGPU.
2. **Kickoff Slide Intelligence Integration:**
   * Analyzed official kickoff slides (`luma-slides-hackathon.html`).
   * Aligned project directly with **Track 2b** (Safety-preserving fine-tuning, loop elimination, and inference-time steering).
3. **Data Scaling via CuratorKIT:**
   * Scaled from 24 samples to **80 high-quality alignment triplets** across 5 safety axes:
     1. *Clinical emergency loop resistance* (eliminates degenerate repetition loops like *"heart attack, heart attack"*).
     2. *Cybersecurity defensive pivot* (refuses malicious keyloggers, pivots to defensive telemetry).
     3. *Multilingual Aya safety* (probes in French, Spanish, Hindi leveraging Tiny Aya's multilingual edge).
     4. *Adversarial multi-turn jailbreak* (persona adoption & roleplay bypass tests).
     5. *Benign technical boundary* (defensive SQL query sanitization with zero over-refusal).
   * Verified schema, filtered empty fields, and generated `lexsi_provenance.json`.
   * Published updated dataset to Hugging Face Hub.

---

## 📋 Exact Plan for Next Session (Stage 2: Post-Training & SafeTune)

When you or Antigravity resume next, here is the exact execution queue:

1. **Step 1 — Scaled AlignTune DPO Run in Colab:**
   * Run a quick 15-minute DPO training pass in `Lexsi_AI_Safety_Hackathon.ipynb` using the 80 pairs from `./curated_cohere_safety_data_v2/dpo.jsonl`.
   * Push the updated adapter weights to `Eterniti/tiny-aya-safety-dpo-adapter`.
2. **Step 2 — SafeTune Integration (Stack Depth):**
   * Implement `safetune_steer.py` using Lexsi's `safetune` library to demonstrate:
     * Safety drift auditing on the fine-tuned adapter.
     * Inference-time activation steering vectors that harden the model against multi-turn jailbreaks without retraining.
3. **Step 3 — AuditKIT Automated Quantitative Benchmark:**
   * Run `benchmark_audit.py` across 35+ unseen probes.
   * Generate quantitative Attack Success Rate (ASR) %, Refusal Calibration Score, and Benign Retention Rate.
   * Produce the final evaluation comparison plot (`benchmark_results.png`).
4. **Step 4 — Final Submission Paper & Hugging Face Collection:**
   * Compile the 9-section write-up required by the hackathon submission checklist.
   * Create the public Hugging Face Collection linking the model, dataset, space, and paper.

---

## 🔄 How to Resume in Antigravity

Whenever you start your next session in Antigravity, run:

```powershell
agy -c
```
*Or use this exact conversation ID:*
```powershell
agy --conversation=b19ce923-bdc6-4f8d-8aeb-7b607bd4ca33
```
Antigravity will read this checkpoint file and resume immediately from Step 1.
