# hmmseq

Reconstruct haplotypes in experimental crosses with known founders using genotype likelihoods.

Usage:
```
hmmseq.py build --vcf geno.vcf.gz --error-rate 0.1 --rho 0.5 -o model.npz
hmmseq.py infer --vcf geno.vcf.gz --mom FVB_NJ --dad WSB_EiJ PWK_PhJ \
		--sample offspring_id \
		--model model.npz --min-depth 3 \
		--region chr1 -o result.npz
hmmseq.py summarize -i result.npz --polish
```
Reconstructed haplotypes are in `result.haps.bed` and crossovers in `result.recombs.bed`.

## Dependencies

* Python >= 2.7 (version 3+ okay)
* `cyvcf2`
* `numpy`
* `scipy`