from __future__ import division
"""
these methods are mostly based on Yarin Gal's PhD thesis 
chapter 3.3.1

note:
the dirichlet library is found here:
https://github.com/ericsuh/dirichlet
"""

import cPickle as pkl
import numpy as np
import dirichlet
import matplotlib.pyplot as plt
from tqdm import tqdm

### initialize these
model_name = 'cnn'


def variation_ratio(sampled_probas):
	"""
	Input:
		sampled_probas: tensor shape of (S#samples, B#examples, D#class_dim) 
	output:
		vrs: vector shape of (B#examples)
			computed based on VR(y) = 1-(f_y/S)
			where f_y is the proportion of sampled outputs
			equating the final output

	"""
	vrs = np.zeros((sampled_probas.shape[1]))
	for example_idx in xrange(sampled_probas.shape[1]):
		curr_samples = sampled_probas[:,example_idx]
		curr_preds = np.argmax(curr_samples, 1)
		f_y = np.bincount(curr_preds).max() / sampled_probas.shape[0]
		vrs[example_idx] = f_y
	return vrs

def soft_variation_ratio(sampled_probas):
	raise NotImplementedError()
def point_variation_ratio(point_probas):
	return 1 - np.max(point_probas,1)

def point_predictive_entropy(point_probas):
	"""
	Input:
		point_probas: matrix shape of (B#examples, D#class_dim)
	output:
		pes: vector shape (B#examples)
			just the usual H(y) = - sum( p*logp )
	note: for sampled_probas, just do a mean of samples before passing in
	"""
	return - np.sum( point_probas * np.log(point_probas) , 1)


def mutual_information(sampled_probas):
	"""
	Input:
		sampled_probas: tensor shape of (S#samples, B#examples, D#class_dim) 
	output:
		mis: vector shape of (B#examples)
			MI(y) = H(y) + 1/S * sum_{s,d} (p(y))		
	"""
	mis = np.zeros((sampled_probas.shape[1]))
	for example_idx in xrange(sampled_probas.shape[1]):
		curr_samples = sampled_probas[:, example_idx]
		avg_y = np.mean(curr_samples, 0)
		H_y = - np.sum( avg_y * np.log(avg_y))
		mis[example_idx] = H_y + np.mean(curr_samples * np.log(curr_samples))
	return mis

def dirichlet_fit(sampled_probas, method='fixedpoint'):
	"""
	Input:
		sampled_probas: tensor shape of (S#samples, B#examples, D#class_dim) 
	output:
		das: matrix shape of (B#examples, D#class_dim)
			where each row is the alpha's from a Dirichlet distribution,
			fitted to the sampled_probas per example
	"""
	das = np.zeros((sampled_probas.shape[1], sampled_probas.shape[2]))
	for example_idx in xrange(sampled_probas.shape[1]):
		curr_samples = sampled_probas[:, example_idx]
		alphas = dirichlet.mle(curr_samples, method)
		das[example_idx] = alphas
	return das
def dirichlet_max(sampled_probas):
	"""
	wrapper for dirichlet_fit
	one way to reduce dirichlet alphas to 1 number, 
		take max
	"""
	alphas = dirichlet_fit(sampled_probas)
	return alphas.max(1)
def key_by_f(f):
	dic = {
	'variation_ratio': 'samples',
	'point_predictive_entropy': "point",
	'mutual_information': 'samples',
	'dirichlet_fit': 'samples',
	'dirichlet_max': 'samples',
	'point_variation_ratio': 'point'
	}
	return dic[f]


ens_ys = pkl.load(open('{}_ens_ys.p'.format(model_name),'rb'))
sampled_ys = np.array(pkl.load(open('{}_sampled_ys.p'.format(model_name),'rb')))
adv_ens_ys = pkl.load(open('{}_adv_ens_ys.p'.format(model_name),'rb'))
adv_sampled_ys = np.array(pkl.load(open('{}_adv_sampled_ys.p'.format(model_name),'rb')))
legit_ys = {"point": ens_ys, "samples": sampled_ys}
adv_ys = {'point':adv_ens_ys, "samples": adv_sampled_ys}
f_s = ["variation_ratio", "point_predictive_entropy", "mutual_information", "dirichlet_max", 'point_variation_ratio']

for f in tqdm(f_s):
	legit_spreads = eval(f)(legit_ys[key_by_f(f)])
	adv_spreads = eval(f)(adv_ys[key_by_f(f)])
	plt.hist([legit_spreads, adv_spreads], histtype='bar', normed=1, label=['legit', 'adv'])
	plt.title(f)
	plt.xlabel('measure of spread')
	plt.ylabel('normalized count')
	plt.legend()
	plt.savefig("{}.png".format(f))
	plt.clf()