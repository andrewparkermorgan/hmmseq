#! /usr/bin/env python
"""
hmmhaps.py
Simple HMM for haplotype inference from biallelic genotypes in a VCF file.
"""

from __future__ import print_function

import os
import sys
import argparse
import logging
import subprocess as sp
import itertools as it

import cyvcf2 as vcf2
import numpy as np
from math import log, floor
from scipy.misc import logsumexp

## for py2 vs py3 compatibility
if sys.version_info[0] == 2:
	zipper = it.izip
else:
	zipper = zip

parent_parser = argparse.ArgumentParser(description = "Hapotype inference from VCF file via HMM.\n\nRequires: cyvcf2, numpy, scipy and bcftools >= 1.3.")
parser = argparse.ArgumentParser(add_help = False)
parser.add_argument("-N", "--nsites", type = int,
					default = 10000000,
					help = "maximum number of sites to use in one pass [default: %(default)d]")
parser.add_argument("--vcf", type = str,
					help = "vcf file containing parent and offspring genotypes" )
parser.add_argument("-r", "--region", type = str,
					default = "chr19",
					required = False,
					help = "target this genomic region, specified samtools-style like 'chr1:1-10' [default: %(default)s]")

subparsers = parent_parser.add_subparsers(dest = "which_cmd")
parser_build = subparsers.add_parser("build", parents = [parser])
parser_build.add_argument("--mom", nargs = "+", type = str,
							help = "possible maternal genotypes" )
parser_build.add_argument("--dad", nargs = "+", type = str,
							help = "possible paternal genotypes" )
parser_build.add_argument("-e","--error-rate", type = float,
							default = 0.001,
							help = "per-site error rate [default: %(default)f]" )
parser_build.add_argument("-g","--grid_size", type = int,
							default = 250,
							help = "expected spacing between sites [default: %(default)d]" )
parser_build.add_argument("-X", "--X-male", action = "store_true",
							help = "build model for X chromosome in males [NOT IMPLEMENTED]")
parser_build.add_argument("-R", "--rho", type = float,
							default = 0.5,
							help = "recombination rate (in cM/Mb) [default: %(default)f]" )
parser_build.add_argument("-o","--out", type = str,
							default = "model.npz",
							help = "name for output file containing model [default: %(default)s]" )

parser_decode = subparsers.add_parser("infer", parents = [parser])
parser_decode.add_argument("--sample", type = str,
							help = "sample on which to infer haplotypes" )
parser_decode.add_argument("-d","--min-depth", type = int,
							default = 0,
							help = "minimum read depth to accept a site as non-missing [default: %(default)d]" )
parser_decode.add_argument("-b","--bias", type = float,
							default = 0.0,
							help = "model reference-allele bias of this magnitude [default: %(default)f]" )
parser_decode.add_argument("-m","--model", type = str,
							default = "model.npz",
							help = "name for *.npz file containing model [default: %(default)s]" )
parser_decode.add_argument("-o","--out", type = str,
							default = "result.npz",
							help = "name for output file containing results [default: %(default)s]" )

parser_summarize = subparsers.add_parser("summarize", parents = [parser])
parser_summarize.add_argument("-i","--infile", type = str,
							default = "result.npz",
							help = "name for *.npz file inference results [default: %(default)s]" )
parser_summarize.add_argument("--polish", action = "store_true",
							default = False,
							help = "'polish' the Viterbi solution to remove poorly-supported blocks [default: %(default)s]" )
parser_summarize.add_argument("-e","--epsilon", type = float,
							default = 0.9,
							help = "minimum posterior probability required to support a diplotype block [default: %(default)f]" )
parser_summarize.add_argument("-o","--out", type = str,
							default = None,
							help = "prefix for output files; results will be written to {prefix}.haps.bed and {prefix}.recombs.bed [default: use input filename]" )

## set up log trace
logging.basicConfig(level = logging.DEBUG)
logging.StreamHandler(stream = sys.stderr)
logger = logging.getLogger()
np.seterr(all = "ignore") # silently ignore floating-point warnings from Numpy stack

def get_sample_idx(v, samples):
	"""Given sample name, get its column index in VCF."""
	is_present = [ s in v.samples for s in samples ]
	if not all(is_present):
		raise ValueError("One of the specified samples was not found in the VCF.")
	else:
		return [ v.samples.index(s) for s in samples ]

def slurp_command(cmd):
	"""Consume ENTIRE output of a shell command (given as list.)"""
	out = sp.Popen(cmd, stdin = sp.PIPE, stderr = sp.PIPE, stdout = sp.PIPE)
	so, se = out.communicate()
	code = out.returncode
	return so,se,code

def count_sites(vcfpath):
	"""Extract number of sites in VCF from its tabix index."""
	cmd = ["bcftools","index","--nrecords", vcfpath]
	so, se, code = slurp_command(cmd)
	return int(so)

def get_gts(gtypes):
	"""Convert cyvcf2 genotypes (0,1,3,2=missing) to ALT allele dosage (0,1,2,-1=missing)"""
	if 2 in gtypes:
		return [-1]*len(gtypes)
	else:
		return [ ([0,1,3]).index(i) for i in gtypes ]

def unphred(x):
	"""Convert Phred scale to linear scale."""
	p = 10.0**(-1*x/10.0)
	p /= np.sum(p)
	return p

def get_sdp(pidx, site):
	"""Given a vector of homozygous genotypes, compute the integer corresponding to the SDP."""
	geno = np.array(get_gts(site.gt_types[pidx]), dtype = np.int)
	if min(geno) < 0 or np.sum(geno == 1) > 0:
		return None
	else:
		geno = geno/2
		sdp = np.sum(geno*np.power(2, np.arange(geno.shape[0])[::-1]))
		return sdp

def rlencode(x, check = True, dropna = False):
	"""
	Run length encoding.
	Based on http://stackoverflow.com/a/32681075, which is based on the rle
	function from R.
	See https://gist.github.com/nvictus/66627b580c13068589957d6ab0919e66

	Parameters
	----------
	x : 1D array_like
		Input array to encode
	dropna: bool, optional
		Drop all runs of NaNs.

	Returns
	-------
	start positions, run lengths, run values

	"""
	where = np.flatnonzero
	x = np.asarray(x)
	n = len(x)
	if n == 0:
		return (np.array([], dtype=int),
				np.array([], dtype=int),
				np.array([], dtype=x.dtype))

	if check:
		starts = np.r_[0, where(~np.isclose(x[1:], x[:-1], equal_nan=True)) + 1]
	else:
		starts = np.r_[0, where(x[1:] != x[:-1]) + 1]
	lengths = np.diff(np.r_[starts, n])
	values = x[starts]

	if dropna:
		mask = ~np.isnan(values)
		starts, lengths, values = starts[mask], lengths[mask], values[mask]

	return starts, lengths, values


def rldecode(starts, lengths, values, minlength=None):
	"""
	Decode a run-length encoding of a 1D array.

	Parameters
	----------
	starts, lengths, values : 1D array_like
		The run-length encoding.
	minlength : int, optional
		Minimum length of the output array.

	Returns
	-------
	1D array. Missing data will be filled with NaNs.

	"""
	starts, lengths, values = map(np.asarray, (starts, lengths, values))
	# TODO: check validity of rle
	ends = starts + lengths
	n = ends[-1]
	if minlength is not None:
		n = max(minlength, n)
	x = np.full(n, np.nan)
	for lo, hi, val in zip(starts, ends, values):
		x[lo:hi] = val
	return x


def iterruns(x, value=None, **kwargs):
	starts, lengths, values = rlencode(x, **kwargs)
	if value is None:
		ends = starts + lengths
		return zip(starts, ends, values)
	else:
		mask = values == value
		starts, lengths = starts[mask], lengths[mask]
		ends = starts + lengths
		return zip(starts, ends)

def reconstruct(sdps, gobs, tprob, eprob, pi, mask, bias = 0.0):
	"""
	Reconstruct haplotypes by fwd-back and Viterbi algorithms.
	Adapted from https://github.com/churchill-lab/gbrs/blob/master/gbrs/commands.py

	Parameters:
	sdps = SDP at each visited site
	gobs = matrix of observed genotype likelihoods
	tprob = transition matrix (states x states)
	eprob = emission matrix (SDPs x states x genotypes)
	pi = initial probs
	mask = missing-data mask; for entries which are TRUE, skip the site
	bias = reference bias correction
	"""

	## apply reference bias correction to emission probs
	bb = np.exp(eprob)
	bb[:,:,0] += bias
	eprob = np.log(bb/np.sum(bb, 2, keepdims = True))

	## make sure transition probs not too small
	tt = tprob
	tt[ np.logical_not(np.isfinite(tt)) ] = -31

	## Get forward probability
	logger.info("   forward probabilities ...")
	nsites = gobs.shape[0]
	num_genos = tprob.shape[0]
	alpha_c = np.zeros((num_genos, nsites))
	alpha_scaler_c = np.zeros(nsites)
	inits = np.log(pi) + logsumexp(eprob[ sdps[0] ,:,:], b = unphred(gobs[0,:]))
	alpha_c[:, 0] = inits
	normalizer = np.log(sum(np.exp(alpha_c[:, 0])))
	alpha_c[:, 0] -= normalizer # normalization
	alpha_scaler_c[0] = -normalizer
	for i in iter(range(1, nsites)):
		# is PL missing or is the site masked?
		if np.max(gobs[i,:]) > 0 and np.max(gobs[i,:]) < 255 and not mask[i]:
			# PL not missing
			eprob_here = eprob[ sdps[i],:,: ]
			w = logsumexp(eprob_here, axis = 1, b = unphred(gobs[i,:]), keepdims = False)
			this_alpha = np.log(np.exp(alpha_c[:, i-1] + tt).sum(axis=1) + np.nextafter(0, 1)) + w
			alpha_c[:, i] = this_alpha
			normalizer = logsumexp(alpha_c[:, i])
			alpha_c[:, i] -= normalizer  # normalization
			alpha_scaler_c[i] = -normalizer
		else:
			# PL missing
			alpha_c[:, i] = alpha_c[:, i-1]
			alpha_scaler_c[i] = alpha_scaler_c[i-1]
		if i and not i % 10**(floor(log(nsites, 10))-1):
			logger.info("	... {} sites".format(i))

	## Get backward probability
	logger.info("   backward probabilities ...")
	beta_c = np.zeros((num_genos, nsites))
	beta_c[:, -1] = alpha_scaler_c[-1]
	for i in iter(range(nsites-2, -1, -1)):
		if np.max(gobs[i+1]) > 0 and np.max(gobs[i+1,:]) < 255 and not mask[i+1]:
			eprob_here = eprob[ sdps[i+1],:,: ]
			w = logsumexp(eprob_here, axis = 1, b = unphred(gobs[i+1,:]), keepdims = False)
			beta_c[:, i] = np.log(np.exp(tt.transpose() + \
										 beta_c[:, i+1] + \
										 w + \
										 alpha_scaler_c[i]).sum(axis=1))
		else:
			beta_c[:, i] = beta_c[:, i+1]
		if i and not i % 10**(floor(log(nsites, 10))-1):
			logger.info("	... {} sites".format(i))

	## Get forward-backward probability
	gamma_c = np.exp(alpha_c + beta_c)
	normalizer = gamma_c.sum(axis=0)
	gamma_c = gamma_c / normalizer

	## Run Viterbi
	logger.info("	Viterbi algorithm for ML solution ...")
	delta_c = np.zeros((num_genos, nsites))
	delta_c[:, 0] = inits
	for i in iter(range(1, nsites)):
		if np.max(gobs[i]) > 0 and np.max(gobs[i,:]) < 255 and not mask[i]:
			eprob_here = eprob[ sdps[i],:,: ]
			w = logsumexp(eprob_here, axis = 1, b = unphred(gobs[i,:]), keepdims = False)
			delta_c[:, i] = (delta_c[:, i-1] + tt).max(axis=1) + w
		else:
			delta_c[:, i] = delta_c[:, i-1]
		if i and not i % 10**(floor(log(nsites, 10))-1):
			logger.info("	... {} sites".format(i))

	viterbi_states = []
	sid = delta_c[:, nsites-1].argmax()
	viterbi_states.append(sid)
	for i in reversed(list(range(nsites-1))):
		sid = (delta_c[:, i] + tt[sid]).argmax()
		viterbi_states.append(sid)
	viterbi_states.reverse()

	return viterbi_states, gamma_c


def build_model(args):
	"""Built HMM paramter objects."""

	logger.info("--- HMM initialization mode ---")

	## connect to VCF
	logger.info("Connecting to VCF <{}>".format(args.vcf))
	#logger.info("	(parsing with {} threads)".format(args.threads))
	thevcf = vcf2.VCF(args.vcf)
	#thevcf.set_threads(args.threads)

	## define genotype states
	if not args.mom:
		sys.exit("Must supply at least '--mom' for phase-unknown model, or '--mom' and '--dad' for phase-known.")
	if not args.dad:
		logger.info("Building phase-unknown model.")
		logger.info("   parents: {}".format(args.mom))
		try:
			_ = get_sample_idx(thevcf, args.mom)
		except:
			sys.exit("One or more of the specified parents was not found in the VCF file.")
		gstates = list( it.combinations_with_replacement(args.mom, 2) )
		parents = list(args.mom)
	else:
		logger.info("Building phase-known model.")
		logger.info("   maternal: {}".format(args.mom))
		logger.info("   paternal: {}".format(args.dad))
		try:
			_ = get_sample_idx(thevcf, args.mom)
			assert len(_) == len(set(args.mom))
			_ = get_sample_idx(thevcf, args.dad)
			assert len(_) == len(set(args.dad))
		except:
			sys.exit("One or more of the specified parents was not found in the VCF file.")
		gstates = list( it.product(args.mom, args.dad) )
		parents = list(set(args.mom) | set(args.dad))

	gindex = [ get_sample_idx(thevcf, g) for g in gstates ]
	nstates = len(gstates)
	nsites = count_sites(args.vcf)
	if args.nsites:
		nsites = min(args.nsites, nsites)
	logger.info("The model has {} diplotype states.".format(nstates))
	logger.info("Assuming error rate: {}".format(args.error_rate))
	logger.info("Assuming recomb rate (cM/Mb): {}".format(args.rho))
	logger.info("Assuming grid size (bp): {}".format(args.grid_size))

	## define emission probs given an error rate
	def make_emit_probs(eps):
		"""Inelegantly create emission probabilities by brute force."""
		probs = np.zeros((3,3,3), dtype = np.float)
		# ref x ref
		probs[ 0,0,0 ] = 1-eps-eps**2
		probs[ 0,0,1 ] = eps
		probs[ 0,0,2 ] = eps**2
		# ref x het
		probs[ 0,1,0 ] = 0.5-eps
		probs[ 0,1,1 ] = 0.5-eps
		probs[ 0,1,2 ] = 1-2*(0.5-eps)
		# ref x alt
		probs[ 0,2,0 ] = eps
		probs[ 0,2,1 ] = 1-2*eps
		probs[ 0,2,2 ] = eps
		# het x het -- ignore errors here; basically no info
		probs[ 1,1,0 ] = 0.25
		probs[ 1,1,1 ] = 0.50
		probs[ 1,1,2 ] = 0.25
		# het x alt
		probs[ 1,2,0 ] = 1-2*(0.5-eps)
		probs[ 1,2,1 ] = 0.5-eps
		probs[ 1,2,2 ] = 0.5-eps
		# alt x alt
		probs[ 2,2,0 ] = eps**2
		probs[ 2,2,1 ] = eps
		probs[ 2,2,2 ] = 1-eps-eps**2
		# now fold across diagonal
		for ii,jj in it.product(range(0,3), range(0,3)):
			if ii > jj:
				probs[ ii,jj,: ] = probs[ jj,ii,: ]
		for ii,jj in it.product(range(0,3), range(0,3)):
			assert all(probs[ ii,jj,: ] == probs[ jj,ii,: ])

		return probs


	logger.info("Pre-allocating model objects ...")

	## part 1: fill in emission matrix
	## lookup table for emission probs given parental genotypes
	eprobs = make_emit_probs(args.error_rate)
	## now make emission probs per SDP
	emitter = np.ndarray((nsites, nstates, 3), dtype = np.float)
	nparents = len(parents)
	nsdps = 2**nparents
	emitter = np.ndarray((nsdps, nstates, 3), dtype = np.float)
	## loop on SDPs
	for jj in range(0, nsdps):
		## generate the SDP as boolean vector
		sdp_flags = list(map(int, format(jj, "b").zfill(nparents)))
		has_alt = np.array(sdp_flags, dtype = np.bool)
		geno = 2*has_alt
		## fill in emission matrix
		for ii, gs in enumerate(gstates):
			gi = [ parents.index(p) for p in gs ]
			theprobs = eprobs[ geno[ gi[0] ],geno[ gi[1] ],: ]
			emitter[ jj,ii,: ] = np.log(theprobs)

	logger.info("   emission probs: {} SDPs x {} states x {} genotypes [{:0.2f} Mb]".format(emitter.shape[0], emitter.shape[1], emitter.shape[2], float(emitter.nbytes)/(1024*1024)))

	## pre-allocate transition matrix
	xmitter = np.zeros((nstates, nstates), dtype = np.float)
	logger.info("   transition probs: {} states x {} states [{:0.2f} Mb]".format(xmitter.shape[0], xmitter.shape[1], float(emitter.nbytes)/(1024*1024)))

	## part 2: fill in transition matrix
	for s1,s2 in it.combinations_with_replacement(range(0, len(gstates)), 2):
		p1, p2 = gindex[s1], gindex[s2]
		ndiff = 2-len( set(p1) & set(p2) )
		cm = (args.rho/100)*float(args.grid_size)/1.0e6
		if s1 == s2:
			xmitter[ s1,s2 ] = np.log(1.0-cm)
		else:
			xmitter[ s1,s2 ] = np.log(cm)*ndiff
			xmitter[ s2,s1 ] = xmitter[ s1,s2 ]

	logger.info("Saving output to <{}> ...".format(args.out))
	np.savez(args.out, vcf = args.vcf,
				emission = emitter, transition = xmitter,
				error_rate = args.error_rate,
				grid_size = args.grid_size,
				recomb_rate = args.rho,
				parents = parents, states = gstates, istates = gindex)
	logger.info("Done.\n\n")


def infer_haps(args):
	"""Use pre-built model parameters to infer haplotypes for a single individual."""

	logger.info("--- HMM inference mode ---")

	logger.info("Reading model from <{}>".format(args.model))
	mfile = np.load(args.model)
	emat = mfile["emission"]
	tmat = mfile["transition"]
	gstates = mfile["states"]
	gindex = mfile["istates"]
	parents = mfile["parents"]
	nstates = gstates.shape[0]

	logger.info("Model properties:")
	logger.info("   universe of parental haplotypes is: {}".format(parents))
	logger.info("	model has {} diplotype states".format(nstates))
	logger.info("	emission probs have shape {}".format(emat.shape))
	logger.info("	transition probs have shape {}".format(tmat.shape))
	logger.info("	reference bias correction: {}".format(args.bias))
	assert emat.shape[1] == tmat.shape[1]
	assert emat.shape[1] == nstates

	pi = np.ones((gstates.shape[0]), dtype = np.float)
	pi /= pi.sum()
	#logger.info("Initial state probabilities are {}".format(pi))

	## connect to VCF
	logger.info("Connecting to VCF <{}>".format(args.vcf))
	#logger.info("	(parsing with {} threads)".format(args.threads))
	thevcf = vcf2.VCF(args.vcf)
	#thevcf.set_threads(args.threads)
	all_sites = count_sites(args.vcf)
	if args.nsites:
		nsites = min(args.nsites, all_sites)
	else:
		nsites = all_sites
	logger.info("VCF has {} sites (we are using {} of them)".format(all_sites, nsites))

	try:
		sid = get_sample_idx(thevcf, [args.sample])
	except:
		sys.exit("Sample '{}' not found in VCF file.".format(args.sample))

	try:
		pid = get_sample_idx(thevcf, parents)
	except:
		sys.exit("Not all parental haplotypes found in the VCF file.")

	## build observation vector
	logger.info("Traversing VCF file to get genotype likelihoods ...")
	geno = np.zeros((nsites, 3), dtype = np.float)
	sdps = np.zeros((nsites), dtype = np.int)
	pos = np.zeros((nsites), dtype = np.int)
	chrom = np.ndarray((nsites), dtype = object)
	masker = np.full((nsites), False, dtype = np.bool)
	for ii, site in enumerate(thevcf(region = args.region)):

		## quit if we hit max number of sites to examine
		if ii >= args.nsites:
			break

		## extract genotype likelihood, if site passes depth filter
		if max(site.gt_depths[sid], 0) >= args.min_depth:
			geno[ii,:] = np.concatenate(
						(	site.gt_phred_ll_homref[sid],
							site.gt_phred_ll_het[sid],
							site.gt_phred_ll_homalt[sid] ) )
		else:
			geno[ii,:] = np.zeros(3, dtype = np.int)
			masker[ii] = True

		pos[ii] = site.start
		chrom[ii] = site.CHROM
		this_sdp = get_sdp(pid, site)
		if this_sdp is not None:
			sdps[ii] = this_sdp
		else:
			sdps[ii] = -1
			masker[ii] = True

		## progress meter
		if ii and not ii % 10**(floor(log(nsites, 10))-1):
			logger.info("	... {} sites".format(ii))

	logger.info("Done filling genotypes; visited {} sites, of which {} are usable.".format(ii+1, np.sum(np.logical_not(masker))))

	#logger.info("Running Viterbi algorithm ...")
	#rez, usable, D = viterbi(geno, pi , tmat , emat, nstates, 3, nsites, masker, args.bias)
	#logger.info("Total {} sites had usable data.".format(np.sum(usable)))
	#emit_haps(rez, usable, gstates, chrom, pos)
	logger.info("Reconstructing haplotypes ...")
	viterbi_soln, probs = reconstruct(sdps, geno, tmat, emat, pi, masker)
	logger.info("Saving result to <{}> ...".format(args.out))
	np.savez(args.out, sample = args.sample, viterbi = viterbi_soln, probs = probs,
						mask = masker, sdps = sdps,
						chrom = chrom, pos = pos, states = gstates)
	logger.info("Done.\n\n")


def summarize_results(args):
	"""From the full output of haplotype inference, summarize to haplotype blocks and recombination events."""

	logger.info("--- HMM summary mode ---")

	logger.info("Reading inference results from <{}>".format(args.infile))
	infile = np.load(args.infile)
	chrom = infile["chrom"]
	pos = infile["pos"]
	probs = infile["probs"]
	viterbi = infile["viterbi"]
	mask = infile["mask"]
	states = infile["states"]
	sample = infile["sample"]

	## swap orientation of probs so they are sites x states
	probs = probs.T
	nsites = probs.shape[0]
	nstates = states.shape[0]

	## check that dimensions of stuff match
	logger.info("Working on sample: {}".format(sample))
	logger.info("Checking dimensions of input objects ...")
	assert chrom.shape[0] == nsites
	assert pos.shape[0] == nsites
	assert mask.shape[0] == nsites
	assert viterbi.shape[0] == nsites
	assert probs.shape[0] == nsites
	assert probs.shape[1] == nstates
	logger.info("Model properties:")
	logger.info("	model has {} diplotype states".format(nstates))
	logger.info("	posterior probs have shape {}".format(probs.shape))

	## in case of multiple chromosomes, split indices of input on chromosome
	#coords = zipper(chrom,np.arange(chrom.shape[0]))
	#chroms_dict = {}
	#for k,v in it.groupby(coords, lambda x: x[0]):
	#	chroms_dict[k] = list(zip(*v)[1])

	def backtrack(gamma, b, from_s, to_s, eps = 0.9):
		"""Bracket a recombination by walking outward from breakpoint until probability
		has reached a specified threshold on each side."""
		bounds = [b, b+1]
		states = (from_s, to_s)
		## first: backward
		while gamma[ bounds[0],states[0] ] < eps:
			## check if we have hit the boundary of the chromosome
			if bounds[0] == 1 or bounds[1] == gamma.shape[0]-1:
				break
			bounds[0] -= 1
		while gamma[ bounds[1],states[1] ] < eps:
			if bounds[0] == 1 or bounds[1] == gamma.shape[0]-1:
				break
			bounds[1] += 1
		return bounds

	logger.info("Identifying diplotype blocks with Viterbi soln ...")
	## identify transitions in Viterbi solution using run-length encoding
	starts, lens, vit_old = rlencode(viterbi)
	ends = starts+lens
	logger.info("   {} raw blocks".format(len(vit_old)))

	logger.info("Calculating posterior probability under diplotype blocks ...")
	## calculate max prob under each block
	support = np.zeros(starts.shape, dtype = np.float)
	for ii, start, end in zipper(np.arange(len(starts)), starts, ends):
		support[ii] = np.max(probs[ start:end,viterbi[start] ])
		print(chrom[start], pos[start], pos[end-1], support[ii], states[vit_old[ii]])

	## remove low-support blocks
	if args.polish:
		logger.info("Polising diplotype blocks (minimum support: {}) ...".format(args.epsilon))
		unsupported = np.nonzero(support < args.epsilon)[0]
		starts = np.delete(starts, unsupported)
		ends = np.delete(ends, unsupported)
		vit_old = np.delete(vit_old, unsupported)
		support = np.delete(support, unsupported)
		logger.info("   {} blocks remain".format(len(support)))

	logger.info("Pseudo-phasing diplotypes to haplotypes ...")
	## merge consecutive blocks of same state
	vit_new = []
	start_new = []
	end_new = []
	support_new = []
	for istart, ilen, istate in zipper(*rlencode(vit_old)):
		start_new.append(starts[istart])
		end_new.append(ends[istart+ilen-1]-1)
		vit_new.append(istate)
		support_new.append(support[istart])

	## show blocks
	last_s1, last_s2 = None, None
	blocks = []
	for start, end, state, supp in zipper(start_new, end_new, vit_new, support_new):
		## do greedy pre-pseudophasing
		s1, s2 = map(str, (states[state][0], states[state][1]))
		if last_s2 is not None and (s1 == last_s2 or s2 == last_s1):
			s2, s1 = s1, s2
		last_s1, last_s2 = s1, s2
		blocks.append( (chrom[start], pos[start], pos[end], s1, s2, supp) )

	for bl in blocks:
		print(*bl)

	logger.info("Identifying recombination events ...")
	## get boundaries of recombination events
	recombs = []
	nblocks = len(start_new)
	for ii in range(1, nblocks):
		this_chrom = blocks[ii][0]
		prev_state, next_state = vit_new[ii-1], vit_new[ii]
		bounds = backtrack(probs, end_new[ii-1], prev_state, next_state)
		diffhaps = set(states[prev_state]) ^ set(states[next_state])
		if len(diffhaps) == 2:
			## one shared haplotype between states; single recomb, pseudophaseable
			prev_hap = [ h for h in diffhaps if h in states[prev_state] ]
			next_hap = [ h for h in diffhaps if h in states[next_state] ]
			recombs.append( (this_chrom, pos[bounds][0], pos[bounds][1], prev_hap[0], next_hap[0]) )
		else:
			## no shared haplotypes; double recomb, unphaseable, just make phasing arbitrary
			recombs.append( (this_chrom, pos[bounds][0], pos[bounds][1], states[prev_state][0], states[next_state][0]) )
			recombs.append( (this_chrom, pos[bounds][0], pos[bounds][1], states[prev_state][1], states[next_state][1]) )
		#print("^^^ {}: {} -> {} ^^^".format(start_new[ii], vit_new[ii-1], vit_new[ii]))
		#print("^^^ {} ^^^".format(bounds))
		#print((100*probs[ bounds[0]:bounds[1],: ]).astype(np.int))
	logger.info("  {} events".format(len(recombs)))

	## now, finally, we are done -- emit results
	if args.out is None:
		prefix, _ = os.path.splitext(args.infile)
	else:
		prefix = args.out
	hapfile_path = prefix + ".haps.bed"
	recfile_path = prefix + ".recombs.bed"

	logger.info("Writing haplotypes to <{}>".format(hapfile_path))
	with open(hapfile_path, "w") as hapfile:
		## do greedy pseudophasing
		print("# pseudophased haplotypes", file = hapfile)
		blocks_colwise = list(zipper(*blocks))
		for col in (3,4):
			for istart, ilen, ival in zipper(*rlencode(blocks_colwise[col], check = False)):
				print(blocks_colwise[0][istart], blocks_colwise[1][istart],
						blocks_colwise[2][istart+ilen-1], ival, col-3,
						sep = "\t", file = hapfile)

	logger.info("Writing recombinations to <{}>".format(recfile_path))
	with open(recfile_path, "w") as recfile:
		print("# pseudophased recombs", file = recfile)
		for rec in recombs:
			print(*rec, file = recfile, sep = "\t")

	logger.info("Done.\n\n")


if __name__ == "__main__":
	args = parent_parser.parse_args()
	if args.which_cmd == "build":
		build_model(args)
	elif args.which_cmd == "infer":
		infer_haps(args)
	elif args.which_cmd == "summarize":
		summarize_results(args)
