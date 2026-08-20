# Merge Transformer Research Plan

## Goal
Merge the from-scratch Transformer architectural research seamlessly into the main fine-tuning paper to create a comprehensive "dual-pronged" study on Vietnamese news summarization.

## Tasks
- [ ] **Task 1: Update Introduction & Abstract** → Verify: Intro clearly states the study explores BOTH from-scratch architectural modifications and pre-trained LLM fine-tuning.
- [ ] **Task 2: Integrate Architecture Theory (Section 3)** → Verify: Create Section 3.2 (`Thiết kế Kiến trúc Tùy chỉnh`). Copy RoPE, RMSNorm, SwiGLU, Weight Tying, and GTF-Penalty equations from partner's paper without breaking existing BART/Qwen text.
- [ ] **Task 3: Integrate Custom Results (Section 5)** → Verify: Create Section 5.1 (`Đánh giá Kiến trúc Tùy chỉnh`) to host the partner's Table 1. Shift the existing LLM results to Section 5.2.
- [ ] **Task 4: Cross-link Findings (Discussion)** → Verify: Connect GTF-Penalty theory to the Trigram Repetition error found in Qwen2.5. Insert the "Mamba vs Transformer" analysis into the discussion/limitations section.
- [ ] **Task 5: Consolidate References** → Verify: Copy all new `\bibitem` entries (RoFormer, LLaMA, PaLM, Mamba, etc.) to the main paper's bibliography.
- [ ] **Task 6: Compile and Review** → Verify: Run `pdflatex main.tex` successfully with no broken refs or layout issues.

## Done When
- [ ] The main paper compiles successfully.
- [ ] All partner's equations, tables, and algorithms are present.
- [ ] The narrative flows logically from custom architectures to pre-trained LLMs.
