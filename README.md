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

## Rationale

Haplotype inference for experimental crosses -- both relatively simple cases like an F2 or backcross, and the more complex case of multiparental populations like the Collaborative Cross or Diversity Outbred -- is already a well-studied problem.  The usual approach is an HMM whose emission distribution is parameterized by *a priori* known founder genotypes and an error rate, and whose transition matrix is parameterized by the genetic or physical distance between markers.  These HMMs were developed for relatively sparse marker panels, say >0.1 cM between markers.

For haplotype inference from high-throughput sequencing, genotypes are observed on a *much* more dense grid.  `hmmseq` makes the simplifying approximation that transition probabilities can be treated as constant between every pair of markers, without respect to the physical spacing of those markers.  (Of course this is not true -- recombination rates are much more variable at fine scale than coarse -- but without a hotspot-aware recombination map, modelling such variation with any accuracy is hard.)  So a single transition matrix is applied in every inter-marker interval.

Genotypes are also observed with less certainty in sequencing than other methods (eg. SNP arrays), especially in the low-coverage setting.  `hmmseq` therefore uses genotype likelihoods (field `PL` in a VCF file) rather than "hard calls" in its inference procedure.

Depending on the downstream use of the inferred haplotypes, both the probabilistic (forward-backward) and the discrete (Viterbi) solutions of the HMM can be useful.  `hmmseq` stores the complete posterior probabilities as Numpy matrices on disk.  An extra post-processing step applies some heuristics to clean up likely artifacts in the Viterbi solution and uses posterior probabilities to refine the boundaries of haplotype switches.

## Dependencies

* Python >= 2.7 (version 3+ okay)
* `cyvcf2` (from https://github.com/brentp/cyvcf2)
* `numpy`
* `scipy`
