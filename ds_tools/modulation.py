from __future__ import division
import numpy as np
from sklearn.preprocessing import normalize
import matplotlib.pyplot as plt

'''
KZB refers to the approach proposed by S.M. Khansari-Zadeh and Aude Billard in the paper 
	"A Dynamical System Approach to Realtime Obstacle Avoidance"
HBS refers to the approach proposed by Lukas Huber, Aude Billard and Jean-Jacques Slotine 
	in the paper "Avoidance of Convex and Concave Obstacles with Convergence ensured 
	through Contraction"
'''

def null_space_bases(n):
	'''construct a set of d-1 basis vectors orthogonal to the given d-dim vector n'''
	d = len(n)
	es = []
	for i in range(1, d):
		e = [-n[i]]
		for j in range(1, d):
			if j == i:
				e.append(n[0])
			else:
				e.append(0)
		e = np.hstack(e)
		es.append(e)
	return es

def modulation_HBS(x, x_dot, obs_centers, x_rs, gammas, gamma_grads):
	'''
	gammas is a list of k gamma functions
	obs_centers is a np array of shape k x d
	'''
	K = len(gammas)

	# calculate weight
	x_rels = [x - obs_center for obs_center in obs_centers]
	gamma_vals = np.array([gamma(x_rel) for gamma, x_rel in zip(gammas, x_rels)])
	ms = np.log(gamma_vals - 1)
	logprod = ms.sum()
	bs = np.exp(logprod - ms)
	weights = bs / bs.sum()

	# calculate modulated dynamical system
	x_dot_mods = []
	for oc, x_r, gamma, gamma_grad in zip(obs_centers, x_rs, gammas, gamma_grads):
		M = modulation_single_HBS(x, oc, x_r, gamma, gamma_grad)
		x_dot_mods.append(np.dot(M, x_dot))

	# calculate weighted average of magnitude
	x_dot_mags = [np.linalg.norm(d) for d in x_dot_mods]
	avg_mag = np.dot(weights, x_dot_mags)
	
	# calculate kappa-space dynamical system and weighted average
	kappas = []
	es = null_space_bases(x_dot)
	R = np.vstack([x_dot] + es).T
	R = normalize(R, axis=0, norm='l2')

	for x_dot_mod in x_dot_mods:
		n_x_dot_mod = x_dot_mod / np.linalg.norm(x_dot_mod)
		n_x_dot_mod_cob = np.dot(R.T, n_x_dot_mod) # cob stands for change-of-basis
		assert -1-1e-5 <= n_x_dot_mod_cob[0] <= 1+1e-5, \
			'n_x_dot_mod_cob[0] = %0.2f?'%n_x_dot_mod_cob[0]
		acos = np.arccos(n_x_dot_mod_cob[0])
		kappa = acos * n_x_dot_mod_cob[1:] / np.linalg.norm(n_x_dot_mod_cob[1:])
		kappas.append(kappa)
	avg_kappa = sum([w * kappa for w, kappa in zip(weights, kappas)])

	# map back to task space
	norm = np.linalg.norm(avg_kappa)
	avg_ds_dir = np.array([np.cos(norm)] + list((avg_kappa * np.sin(norm) / norm).flat))
	avg_ds_dir = np.dot(R, avg_ds_dir)

	return avg_mag * avg_ds_dir

def modulation_single_HBS(x, obs_center, x_r, gamma, gamma_grad):
	assert len(x.shape) == 1
	d = x.shape[0]
	x_rel = x - obs_center
	n = gamma_grad(x_rel)
	es = null_space_bases(n)
	r = x - x_r
	E = np.vstack([r] + es).T
	E = normalize(E, axis=0, norm='l2')
	inv_gamma = 1 / abs(gamma(x_rel))
	lambdas = [1 - inv_gamma] + [1 + inv_gamma] * (d-1)
	D = np.diag(lambdas)
	invE = np.linalg.inv(E)
	return np.linalg.multi_dot([E, D, invE])

def modulation_KZB(x, gammas, gamma_grads, obs_centers, safety_margin=1):
	'''
	gammas is a list of k gamma functions
	obs_centers is a np array of shape k x d
	'''
	K = len(gammas)
	x_rels = [(x - obs_center) / safety_margin for obs_center in obs_centers]
	gamma_vals = [gamma(x_rel) for gamma, x_rel in zip(gammas, x_rels)]
	weights = []
	for k in range(K):
		prod = np.prod([(gamma_vals[i]-1)/(gamma_vals[i]+gamma_vals[k]-2) 
			for i in range(K) if i!=k])
		weights.append(prod)
	Ms = []
	for x_rel, gamma, gamma_grad, weight in zip(x_rels, gammas, gamma_grads, weights):
		M = modulation_single_KZB(x_rel, gamma, gamma_grad, weight, safety_margin=1)
		Ms.append(M)
	if len(Ms)==1:
		return Ms[0]
	else:
		return np.linalg.multi_dot(Ms)

def modulation_single_KZB(x_rel, gamma, gamma_grad, weight=1, safety_margin=1):
	assert len(x_rel.shape) == 1 and safety_margin >= 1
	d = x_rel.shape[0]
	x_rel = x_rel / safety_margin
	n = gamma_grad(x_rel)
	es = null_space_bases(n)
	E = np.vstack([n] + es).T
	E = normalize(E, axis=0, norm='l2')
	inv_gamma = weight / abs(gamma(x_rel))
	lambdas = [1 - inv_gamma] + [1 + inv_gamma] * (d-1)
	D = np.diag(lambdas)
	invE = E.T
	return np.linalg.multi_dot([E, D, invE])

def gamma_circle_2d(radius, center):
	'''
	constructs gamma, gamma_grad, and obs_center for a 2d circle 
	with given radius and center
	obs_center also functions as reference point
	'''
	center = np.array(center)
	gamma = lambda x: np.linalg.norm(x) / radius
	gamma_grad = lambda x: x / (np.linalg.norm(x) * radius)
	return gamma, gamma_grad, center

def gamma_rectangle_2d(w, h, center):
	'''
	constructs gamma, gamma_grad, and obs_center for a 2d rectangle
	with given width, height and center
	obs_center also functions as reference point
	'''
	center = np.array(center)
	def gamma(pt):
		x, y = pt
		angle = np.arctan2(y, x)
		first = np.arctan2(h/2, w/2)
		second = np.arctan2(h/2, -w/2)
		third = np.arctan2(-h/2, -w/2)
		fourth = np.arctan2(-h/2, w/2)
		if (first < angle < second) or (third < angle < fourth):
			return 2 * abs(y) / h
		else:
			return 2 * abs(x) / w
	def gamma_grad(pt):
		x, y = pt
		angle = np.arctan2(y, x)
		first = np.arctan2(h/2, w/2)
		second = np.arctan2(h/2, -w/2)
		third = np.arctan2(-h/2, -w/2)
		fourth = np.arctan2(-h/2, w/2)
		if (first < angle < second) or (third < angle < fourth):
			return np.array([0, np.sign(y)*2/h])
		else:
			return np.array([np.sign(x)*2/w, 0])
	return gamma, gamma_grad, center

def gamma_cross_2d(a, b, center):
	center = np.array(center)
	'''
	constructs gamma, gamma_grad, and obs_center for an axis-parallel cross (like the red cross)
	for each "arm", the width is a and length is b
	with given width, height and center
	obs_center also functions as reference point
	'''
	def atan2_pos(x, y):
		ang = np.arctan2(x, y)
		if ang < 0:
			ang = 2 * np.pi + ang
		return ang
	c = a / 2
	d = a / 2 + b
	angles = [(c, d), (1, 1), (d, c), (d, -c), (1, -1), (c, -d), (-c, -d), 
			  (-1, -1), (-d, -c), (-d, c), (-1, 1), (-c, d)]
	angles = [atan2_pos(*x) for x in angles]
	def gamma(pt):
		x, y = pt
		angle = atan2_pos(y, x)
		if angles[0] <= angle < angles[1]:
			return y / c
		elif angles[1] <= angle < angles[2]:
			return x / c
		elif angles[2] <= angle < angles[3]:
			return y / d
		elif angles[3] <= angle < angles[4]:
			return - x / c
		elif angles[4] <= angle < angles[5]:
			return y / c
		elif angles[5] <= angle < angles[6]:
			return - x / d
		elif angles[6] <= angle < angles[7]:
			return - y / c
		elif angles[7] <= angle < angles[8]:
			return - x / c
		elif angles[8] <= angle < angles[9]:
			return - y / d
		elif angles[9] <= angle < angles[10]:
			return  x / c
		elif angles[10] <= angle < angles[11]:
			return - y / c
		else:
			return x / d
	def gamma_grad(pt):
		x, y = pt
		angle = atan2_pos(y, x)
		if angles[0] <= angle < angles[1]:
			return np.array([0, 1 / c])
		elif angles[1] <= angle < angles[2]:
			return np.array([1 / c, 0])
		elif angles[2] <= angle < angles[3]:
			return np.array([0, 1 / d])
		elif angles[3] <= angle < angles[4]:
			return np.array([- 1 / c, 0])
		elif angles[4] <= angle < angles[5]:
			return np.array([0, 1 / c])
		elif angles[5] <= angle < angles[6]:
			return np.array([- 1 / d, 0])
		elif angles[6] <= angle < angles[7]:
			return np.array([0, - 1 / c])
		elif angles[7] <= angle < angles[8]:
			return np.array([- 1 / c, 0])
		elif angles[8] <= angle < angles[9]:
			return np.array([0, - 1 / d])
		elif angles[9] <= angle < angles[10]:
			return  np.array([1 / c, 0])
		elif angles[10] <= angle < angles[11]:
			return np.array([0, - 1 / c])
		else:
			return np.array([1 / d, 0])
	return gamma, gamma_grad, center

def linear_controller(x, x_target, max_norm=0.1):
	x_dot = x_target - x
	n = np.linalg.norm(x_dot)
	if n < max_norm:
		return x_dot
	else:
		return x_dot / n * max_norm

def demo_KZB():
	'''
	demo of the KZB approach. Note that there will be an attractor near the upper-left corner
	of the rectangle, a phenomenon also noted in Fig. 2 of HBS. 
	'''
	x_target = np.array([0.9, 0.8])
	gamma1, gamma_grad1, obs_center1 = gamma_circle_2d(0.15, [0.2, 0.8])
	# gamma2, gamma_grad2, obs_center2 = gamma_circle_2d(0.15, [0.8, 0.2])
	gamma2, gamma_grad2, obs_center2 = gamma_cross_2d(0.1, 0.15, [0.7, 0.3])
	gamma3, gamma_grad3, obs_center3 = gamma_rectangle_2d(0.2, 0.3, [0.6, 0.7])
	gammas = [gamma1, gamma2, gamma3]
	gamma_grads = [gamma_grad1, gamma_grad2, gamma_grad3]
	obs_centers = [obs_center1, obs_center2, obs_center3]
	plt.figure()
	for i in np.linspace(0, 1, 30):
		for j in np.linspace(0, 1, 30):
			x = np.array([i, j])
			if min([g(x - oc) for g, oc in zip(gammas, obs_centers)]) < 1:
				continue
			M = modulation_KZB(x, gammas, gamma_grads, obs_centers)

			x_dot = linear_controller(x, x_target)
			modulated_x_dot = np.dot(M, x_dot) * 0.2
			plt.arrow(i, j, modulated_x_dot[0], modulated_x_dot[1], 
				head_width=0.008, head_length=0.01)

	plt.gca().add_artist(plt.Circle((0.2, 0.8), 0.15))
	# plt.gca().add_artist(plt.Circle((0.8, 0.2), 0.15))
	plt.gca().add_artist(plt.Rectangle((0.50, 0.25), 0.4, 0.1))
	plt.gca().add_artist(plt.Rectangle((0.65, 0.1), 0.1, 0.4))
	plt.gca().add_artist(plt.Rectangle((0.5, 0.55), 0.2, 0.3))
	plt.axis([0, 1, 0, 1])
	plt.gca().set_aspect('equal', adjustable='box')
	plt.plot([x_target[0]], [x_target[1]], 'r*')
	plt.savefig('vector_field_KZB.png', bbox_inches='tight')
	plt.show()

def demo_HBS():
	'''
	demo of the HBS approach with multiple obstacles
	'''
	x_target = np.array([0.9, 0.8])
	gamma1, gamma_grad1, obs_center1 = gamma_circle_2d(0.15, [0.2, 0.8])
	# gamma2, gamma_grad2, obs_center2 = gamma_circle_2d(0.15, [0.8, 0.2])
	gamma2, gamma_grad2, obs_center2 = gamma_cross_2d(0.1, 0.15, [0.7, 0.3])
	gamma3, gamma_grad3, obs_center3 = gamma_rectangle_2d(0.2, 0.3, [0.6, 0.7])
	gammas = [gamma1, gamma2, gamma3]
	gamma_grads = [gamma_grad1, gamma_grad2, gamma_grad3]
	obs_centers = [obs_center1, obs_center2, obs_center3]
	plt.figure()
	for i in np.linspace(0, 1, 30):
		for j in np.linspace(0, 1, 30):
			x = np.array([i, j])
			if min([g(x - oc) for g, oc in zip(gammas, obs_centers)]) < 1:
				continue

			x_dot = linear_controller(x, x_target)
			modulated_x_dot = modulation_HBS(x, x_dot, obs_centers, obs_centers, 
				gammas, gamma_grads) * 0.15

			plt.arrow(i, j, modulated_x_dot[0], modulated_x_dot[1], 
				head_width=0.008, head_length=0.01)

	plt.gca().add_artist(plt.Circle((0.2, 0.8), 0.15))
	# plt.gca().add_artist(plt.Circle((0.8, 0.2), 0.15))
	plt.gca().add_artist(plt.Rectangle((0.50, 0.25), 0.4, 0.1))
	plt.gca().add_artist(plt.Rectangle((0.65, 0.1), 0.1, 0.4))
	plt.gca().add_artist(plt.Rectangle((0.5, 0.55), 0.2, 0.3))
	plt.axis([0, 1, 0, 1])
	plt.gca().set_aspect('equal', adjustable='box')
	plt.plot([x_target[0]], [x_target[1]], 'r*')
	plt.savefig('vector_field_HBS.png', bbox_inches='tight')
	plt.show()

def demo_HBS_single_obs():
	'''
	demo of the HBS approach with a single obstacle 
	(i.e. without weighing and kappa-space interpolation). 
	'''
	x_target = np.array([0.9, 0.8])
	# gamma, gamma_grad, obs_center = gamma_rectangle_2d(0.2, 0.3, [0.6, 0.7])
	gamma, gamma_grad, obs_center = gamma_circle_2d(0.15, [0.2, 0.8])
	plt.figure()
	for i in np.linspace(0, 1, 30):
		for j in np.linspace(0, 1, 30):
			x = np.array([i, j])
			if gamma(x - obs_center) < 1:
				continue
			M = modulation_single_HBS(x, obs_center, obs_center, gamma, gamma_grad)

			x_dot = linear_controller(x, x_target)
			modulated_x_dot = np.dot(M, x_dot) * 0.2
			plt.arrow(i, j, modulated_x_dot[0], modulated_x_dot[1], 
				head_width=0.008, head_length=0.01)

	plt.gca().add_artist(plt.Circle((0.2, 0.8), 0.15))
	# plt.gca().add_artist(plt.Rectangle((0.5, 0.55), 0.2, 0.3))
	plt.axis([0, 1, 0, 1])
	plt.gca().set_aspect('equal', adjustable='box')
	plt.plot([x_target[0]], [x_target[1]], 'r*')
	plt.savefig('vector_field_HBS_single_obs.png', bbox_inches='tight')
	plt.show()

def test_gamma(gamma, xmin, xmax, ymin, ymax, res):
	xs = np.linspace(xmin, xmax, res)
	ys = np.linspace(ymin, ymax, res)
	xs, ys = np.meshgrid(xs, ys)
	combo = zip(xs.flatten(), ys.flatten())
	vals = np.array(map(gamma, combo)).reshape(res, res)
	plt.contour(xs, ys, vals, levels=[1, 2])
	plt.colorbar()
	plt.show()