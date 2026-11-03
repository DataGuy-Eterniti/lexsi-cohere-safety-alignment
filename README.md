# 🛡️ Calibrated Safety Alignment Pipeline for Cohere Tiny Aya
**Lexsi AI Safety and Alignment Hackathon (Nov 2026)**  
*Built with Lexsi Labs Stack (`CuratorKIT` + `AlignTune`), Cohere Labs, and Hugging Face.*

---

## 📌 Executive Summary
Small open-weights models often struggle with two opposing safety failure modes: **dangerous sycophancy/compliance** on hazardous instructions, and **severe over-refusal** on benign queries containing security or medical terminology.

In this project, we built an end-to-end, reproducible cloud-native post-training pipeline that:
1. Synthesizes balanced safety pairs across four critical axes (cybersecurity, chemical/bio, clinical diagnosis, and benign edge-cases).
2. Curates, deduplicates, and validates data provenance using **Lexsi CuratorKIT**.
3. Post-trains a parameter-efficient LoRA adapter on **CohereLabs' `tiny-aya-earth`** using **Lexsi AlignTune** via Direct Preference Optimization (DPO).
4. Demonstrates marked qualitative and quantitative improvements in safety calibration without inducing over-refusal.

---

## 🔗 Artifacts & Open-Source Releases
* **Model Adapter (Hugging Face):** `[INSERT YOUR HF USERNAME]/tiny-aya-safety-dpo-adapter`
* **Curated Dataset & Provenance (Hugging Face):** `[INSERT YOUR HF USERNAME]/curated-cohere-safety-dpo`
* **Base Model:** `CohereLabs/tiny-aya-earth` (3.35B parameters)

---

## 🏗️ Architecture & Pipeline
