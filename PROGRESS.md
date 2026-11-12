# 📌 Lexsi AI Safety Hackathon — Project Progress & Checkpoint

**Last Updated:** November 12, 2026 (09:55 CET)  
**Submission Deadline:** November 14, 2026 (AoE) — *~2.5 days remaining*  
**Hackathon:** [Lexsi AI Safety and Alignment Hackathon](https://hackathon.lexsi.ai/) (Lexsi Labs × Cohere Labs × Hugging Face)  
**Track:** Track 2: AI Safety, Red Teaming, and Correction (Problem 2b: Safety Calibration, Hallucination/Loop Repair, and SafeTune Steering)  
**Connected Stack:** `CuratorKIT` ➔ `AlignTune` ➔ `SafeTune` ➔ `AuditKIT`  
**Base Architecture:** `CohereLabs/tiny-aya-earth` (3.35B parameter multilingual dense transformer)

---

## 🔗 Live Hub Artifacts & Links

| Artifact | Location | Status |
| :--- | :--- | :--- |
| **Live Interactive Space** | [Eterniti/tiny-aya-safety-playground](https://huggingface.co/spaces/Eterniti/tiny-aya-safety-playground) | 🟢 **RUNNING on ZeroGPU (`zero-a10g`)** |
| **LoRA Model Adapter (v2)** | [Eterniti/tiny-aya-safety-dpo-adapter](https://huggingface.co/Eterniti/tiny-aya-safety-dpo-adapter) | 🟢 **Retrained (v2, 160 steps, 2 epochs)** + `lexsi_provenance.json` |
| **Curated DPO Dataset (v2)** | [Eterniti/curated-cohere-safety-dpo](https://huggingface.co/datasets/Eterniti/curated-cohere-safety-dpo) | 🟢 **Published (80 pairs across 5 axes)** + `lexsi_provenance.json` |
| **AuditKIT Evaluation Logs** | [Eterniti/tiny-aya-safety-audit-logs](https://huggingface.co/datasets/Eterniti/tiny-aya-safety-audit-logs) | 🟢 **Published (`RedTeamRunner` 47 checks)** |
| **GitHub Repository** | [DataGuy-Eterniti/lexsi-cohere-safety-alignment](https://github.com/DataGuy-Eterniti/lexsi-cohere-safety-alignment) | 🟢 Updated with verified chart, `README.md`, & `SUBMISSION_ARTICLE.md` |
| **Hugging Face Collection** | [Tiny Aya Safety Alignment & Steering](https://huggingface.co/collections/Eterniti/tiny-aya-safety-alignment-and-steering-6af57502e34895830a3fb0f4) | 🟢 **Published** (all 4 Hub artifacts linked + `cc-by-nc-4.0` licenses set) |
| **9-Section HF Article** | [Calibrated Safety Alignment & Layer-Targeted Steering for Cohere Tiny Aya](https://huggingface.co/blog/Eterniti/tiny-aya-safety-alignment-steering) | 🟢 **Published** on Hugging Face Blog (`SUBMISSION_ARTICLE.md`) |

---

## ✅ Completed Milestones

1. **Stage 1 (Nov 3–6): Data Curation & Live Demo**
   * Scaled DPO dataset via `CuratorKIT` to **80 alignment triplets** across 5 safety axes and pushed to `Eterniti/curated-cohere-safety-dpo`.
   * Deployed `Eterniti/tiny-aya-safety-playground` on Hugging Face ZeroGPU (`zero-a10g`).
2. **Stage 2 (Nov 12): Scaled `AlignTune` DPO v2 Run**
   * Trained 2 epochs (160 steps) on Colab T4 in `06:03`: loss `0.6940 → 0.6392` (min) / `0.6539` (final), reward accuracy `0.400 → 1.000`, peak reward margin `+0.1123`.
   * Pushed updated v2 weights and provenance manifest to `Eterniti/tiny-aya-safety-dpo-adapter`.
3. **Stage 3 (Nov 12): `SafeTune` Steering + `AuditKIT` Quantitative Audit**
   * Ran `AuditKIT` `RedTeamRunner` (47 checks: `39/47 = 83.0%` pass rate) and pushed logs to `Eterniti/tiny-aya-safety-audit-logs`.
   * Extracted Layer-14 refusal direction (`dim=2048`) via `SafeTune`: proved ablation vulnerability (`0/5` cyber refusal) and ran a 5-config steering sweep identifying `layers=[13, 14, 15], strength=2.0` as the Pareto sweet spot (`2/2` persona jailbreaks blocked, `5/5 = 100%` benign retention).
   * Generated verified `benchmark_results.png`, updated `README.md`, and drafted `SUBMISSION_ARTICLE.md`.

---

## 📋 Final Pre-Deadline Checklist (Due Nov 14, AoE)

1. ✅ **Public Hugging Face Collection Created:** All 4 artifacts bundled into [Tiny Aya Safety Alignment & Steering](https://huggingface.co/collections/Eterniti/tiny-aya-safety-alignment-and-steering-6af57502e34895830a3fb0f4), and `cc-by-nc-4.0` license tags set on both the adapter and dataset cards.
2. ✅ **Hugging Face Article Published:** Live at [Calibrated Safety Alignment & Layer-Targeted Steering for Cohere Tiny Aya](https://huggingface.co/blog/Eterniti/tiny-aya-safety-alignment-steering) and linked in `README.md`.
3. **Submit Form:** Submit the public GitHub repo, HF Collection, Space, and HF Article via the official Lexsi hackathon submission form before **November 14, 2026 (AoE)**.
