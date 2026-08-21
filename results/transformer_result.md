	ROUGE-1		ROUGE-2		ROUGE-L		BLEU		BERTScore F1	
	Vanilla	Penalty	Vanilla	Penalty	Vanilla	Penalty	Vanilla	Penalty	Vanilla	Penalty
Baseline Transformer	34.37	44.61	8.05	11.06	22.13	27.1	0.52	1.41	65.08	70.01
Improved Baseline	46	48.52	10.46	11.78	27.59	27.93	1.82	1.7	68.99	70.12
Soft Prompt Mamba	44.21	45.92	9.38	10.23	26.65	27.03	1.28	1.28	69.37	69.89
Entity Gated Mamba	44.88	47.44	10.23	11.24	27.91	28.31	1.25	1.2	68.49	69.38
Transformer Hypersphere	47.3	48.04	12.84	13.18	28.97	29.01	2.26	2.25	70.15	70.43
Entity Gated Mamba With Label Smoothing	45.73	47	10.59	11.03	27.89	27.89	1.33	1.22	69.15	69.44
Entity Guided Hybrid Mamba	47.85	49.17	12.67	13.4	29.25	29.16	2.26	2.21	70.43	70.85
Entity Guided Pure Transformer	48.79	50	13.19	14.01	29.5	29.74	2.54	2.57	70.34	70.88