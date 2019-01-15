import numpy as np 
from math import inf
import matplotlib.pyplot as plt
from gurobipy import *
import random
import os

import copy

"""********************************************************
					Fonctions de gestion
***********************************************************"""

NB_PARAMS = 8
LISTE_PARAMS = ["P_PUISSANCE", "P_COUPLE", "P_POIDS", "P_ACC", "P_PRIX", "P_POLLUTION", "P_DESIGN", "P_CHASSIS"]


"""Minimized params will be multiplied by -1 to tranform the problem
to a maximization problem """
LISTE_MAX = [1, 1, -1, -1, -1, -1, 1, 1]

class  Voiture(): 
	def __init__(self, nom, params):
		self.nom = nom
		self.params = params

		#Reorientation of minimized params :
		for i in range(len(self.params)):
			self.params[i] = self.params[i] * LISTE_MAX[i]

	def normalize_params(self, ideal, nadir): 
		for i in range(len(self.params)) :
			self.params[i] = (self.params[i] - nadir[i]) / (ideal[i] - nadir[i])

	def denormalize_params(self, ideal, nadir) :
		for i in range(len(self.params)) :
			self.params[i] = self.params[i] * (ideal[i] - nadir[i]) + nadir[i]

def read_data(file_name):
	liste_voitures = []
	file = open(file_name, "r")
	line = file.readline()
	while line :
		line = line.rstrip()
		line_split = line.split(" ")

		if(line_split[0].lower() == "nom") : 
			new_params = []
			new_name = line_split[-1]
			
			for i in range(len(LISTE_PARAMS)) :
				line = file.readline()
				line = line.rstrip()
				line_split = line.split(" ")
				new_params += [float(line_split[2])]

			liste_voitures += [Voiture(new_name, new_params)]

		line = file.readline()
	return liste_voitures

def ideal_nadir(liste_voitures):
	ideal = [-inf for i in range(len(LISTE_PARAMS))]
	nadir = [inf for i in range(len(LISTE_PARAMS))]

	for voiture in liste_voitures :
		for i in range(len(LISTE_PARAMS)):
			if voiture.params[i] > ideal[i]:
				ideal[i] = voiture.params[i]
			if voiture.params[i] < nadir[i]:
				nadir[i] = voiture.params[i]

	return ideal, nadir

def normalize_params(liste_voitures, ideal, nadir):
	for voiture in liste_voitures :
		voiture.normalize_params(ideal, nadir)

def plot_normalized_solution(voiture):
	#print(ideal)
	#print(nadir)
	plt.axis([-0.5,NB_PARAMS -0.5, -0.1, 1.1])
	plt.xlabel("critères")
	plt.ylabel("valeurs normalisées")
	x = [i for i in range(0, NB_PARAMS)]
	plt.plot(x, [0 for i in range(len(x))], "r--", label = "nadir")
	plt.plot(x, [1 for i in range(len(x))], "g--", label = "ideal")
	plt.plot(x, voiture.params, "bo", label = "solution")
	#plt.legend(loc = 1)
	plt.title("Solution choisie : "+voiture.nom)
	plt.show()

def plot_param_distribution(liste_voitures, param_index):
	plt.axis([-0.5,len(liste_voitures) -0.5, -0.1, 1.1])
	plt.xlabel("voitures")
	plt.ylabel("valeurs normalisées du paramètre")
	x = [i for i in range(0, len(liste_voitures))]
	
	plt.plot(x, [0 for i in range(len(x))], "r--", label = "nadir")
	plt.plot(x, [1 for i in range(len(x))], "g--", label = "ideal")
	
	y = []
	
	for i in range(len(liste_voitures)):
		y += [liste_voitures[i].params[param_index]]
	
	plt.plot(x, y, "bo", label = LISTE_PARAMS[param_index])
	#plt.legend(loc = 1)
	plt.title("Distribution du paramètre : "+LISTE_PARAMS[param_index])
	plt.show()	

def plot_more_solutions(liste_sols):
	plt.axis([-0.5,NB_PARAMS -0.5, -0.1, 1.1])
	plt.xlabel("critères")
	plt.ylabel("valeurs normalisées")
	x = [i for i in range(0, NB_PARAMS)]
	plt.plot(x, [0 for i in range(len(x))], "r--", label = "nadir")
	plt.plot(x, [1 for i in range(len(x))], "g--", label = "ideal")
	for voiture in liste_sols:
		plt.plot(x, voiture.params, "o")
	#plt.legend(loc = 1)
	ttl = ""
	for voiture in liste_sols :
		ttl += " "+voiture.nom+" "
	plt.title("Comparaison de : "+ ttl)
	plt.show()
"""********************************************************
							Partie 1
***********************************************************"""
liste_voitures = read_data("petites-sportives.txt")
ideal, nadir = ideal_nadir(liste_voitures)
#print(ideal)
#print(nadir)

normalize_params(liste_voitures, ideal, nadir)
for v in liste_voitures :
	print(v.params)

#plot_normalized_solution(liste_voitures[1])
#plot_param_distribution(liste_voitures, 4)
#plot_more_solutions(liste_voitures[:4])
"""********************************************************
							Partie 2
***********************************************************"""

class CSS():
	def __init__(self, set_X, v_poids) :
		self.set_X = set_X
		self.v_poids = v_poids
		self.v_estimate = np.zeros(len(v_poids))
		self.creer_modele()

	def random_poids(self):
		l_w = [random.random() for i in range(len(self.set_X[0].params))]
		s = sum(l_w)
		for i in range(len(l_w)):
			l_w[i] = l_w[i]/s

		return l_w

	def creer_modele(self):
		self.modele = Model("model")
		self.modele.setParam('OutputFlag', False)

		#définition des variables w_i
		for i in range(len(self.set_X[0].params)):
			self.modele.addVar(vtype = GRB.CONTINUOUS, name = "w_"+LISTE_PARAMS[i]) 
		self.modele.update()

		#définition de contraintes portant sur les w_i
		for i in range(len(self.modele.getVars())):
			self.modele.addConstr(self.modele.getVars()[i] >= 0, "c_bound0_"+LISTE_PARAMS[i])
			self.modele.addConstr(self.modele.getVars()[i] <= 1, "c_bound1_"+LISTE_PARAMS[i])

		self.modele.addConstr(quicksum(self.modele.getVars()[i] for i in range(len(self.modele.getVars()))) <= 1, "sum1")
		self.modele.addConstr(quicksum(self.modele.getVars()[i] for i in range(len(self.modele.getVars()))) >= 1, "sum2")
		self.modele.update()

	def is_prefered(self, x, y):
		s = 0

		for i in range(len(x)):
			s += self.v_poids[i] * (x[i] - y[i])

		if s >= 0 :
			return True
		
		return False

	def f(self, x):
		s = 0
		for i in range(len(x)):
			s += self.v_poids[i]*x[i]
		return s

	def f_estim(self, x):
		s = 0
		for i in range(len(x)):
			s += self.v_estimate[i]*x[i]
		return s

	def meilleure_vraie(self):
		f_opt = self.f(self.set_X[0].params)
		x_opt = self.set_X[0].nom
		l_x_opt = [x_opt]

		for x in self.set_X :
			if self.f(x.params) > f_opt :
				f_opt = self.f(x.params)
				x_opt = x.nom
				l_x_opt = [x_opt]
			elif self.f(x.params) == f_opt :
				l_x_opt += [x.nom]

		return f_opt, l_x_opt

	def CSS(self):

		MMR = 2
		sol_opt = None

		while MMR > 0.1 :
			MMR_act, x_p, y_p, w_r = self.MMR()
			MMR = MMR_act
			self.v_estimate = w_r
			sol_opt = x_p

			if self.is_prefered(x_p.params, y_p.params):
				self.modele.addConstr(quicksum(self.modele.getVars()[i]*(x_p.params[i] - y_p.params[i]) for i in range(len(self.modele.getVars()))) >= 0, "cst_"+str(len(self.modele.getVars())))
				self.modele.update()
			if self.is_prefered(y_p.params, x_p.params):
				self.modele.addConstr(quicksum(self.modele.getVars()[i]*(y_p.params[i] - x_p.params[i]) for i in range(len(self.modele.getVars()))) >= 0, "cst_"+str(len(self.modele.getVars())))
				self.modele.update()	
					
			print("xp, yp :",x_p.nom, y_p.nom)
			print(MMR)
			print(w_r)
			print(sol_opt.nom)
			print(self.f_estim(sol_opt.params))
			print("vrai : ", self.meilleure_vraie())



	def MMR(self):
		MR_min = 2
		x_p = None
		l_x_p = []
		y_p = None
		l_y_p = []
		w_p = None 
		l_w_p = []

		for x in self.set_X :
			MR, y, w = self.MR(x)
			print(x, MR, y.nom, w)

			if MR < MR_min :
				MR_min = MR
				l_x_p = [x]
				l_y_p = [y]
				l_w_p = [w]
			elif MR == MR_min :
				l_x_p += [x]
				l_y_p += [y]
				l_w_p += [w]

		index = random.randrange(len(l_x_p))

		x_p = l_x_p[index]
		y_p = l_y_p[index]
		w_p = l_w_p[index]

		return MR_min, x_p, y_p, w_p

	def MR(self, x):
		ind = 0
		if self.set_X[0] == x :
			ind = 1	
		y_max = self.set_X[ind]
		l_y_max = [y_max]
		PMR_max, w_max = self.PMR(x.params, self.set_X[ind].params)
		l_w_max = [w_max]

		for i in range(ind+1, len(self.set_X)):
			if self.set_X[i] != x :
				PMR, w = self.PMR(x.params, self.set_X[i].params)
				if PMR > PMR_max :
					PMR_max = PMR
					y_max = self.set_X[i]
					l_y_max = [y_max]
					w_max = w
					m_w_max = [w_max]
				elif PMR == PMR_max :
					l_y_max += [self.set_X[i]]
					l_w_max += [w_max]

			index = random.randrange(len(l_y_max))


		return PMR_max, l_y_max[index], l_w_max[index]


	def PMR(self, x, y):
		#définition de la fonction objectif :
		self.modele.setObjective(quicksum(self.modele.getVars()[i] * (y[i] - x[i]) for i in range(len(self.modele.getVars()))), GRB.MAXIMIZE)
		self.modele.update()

		self.modele.optimize()
		
		w_sol = [v.x for v in self.modele.getVars()]

		return self.modele.objVal, w_sol
	"""	
	
	def CSS_iteration(self):
		m = Model("model")
		m.setParam( 'OutputFlag', False )

		for x in self.set_X :
			print(x.nom+ " : ")
			s = 0
			for i in range(len(x.params)):
				s += v_poids[i]*x.params[i]
			print("\t",s )
			print("\t", x.params)

		l_vars = []
		for i in range(len(self.set_X[0].params)):
			#print(i)
			m.addVar(vtype = GRB.CONTINUOUS, name = "w_"+LISTE_PARAMS[i])
			m.update()
			#print(m.getVars())

		#print("vars :",m.getVars())
		#print("l_vars :", m.getVars())
		
		for i in range(len(m.getVars())):
			m.addConstr(m.getVars()[i] <= 1, "c_w_"+str(i))
			m.update()

		#c_sum = 0
		#for i in range(len(m.getVars())):
		#	c_sum += m.getVars()[i]

		#m.addConstr(c_sum <= 1, "c_sum")
		#m.addConstr(c_sum >= 1, "c_sum")
		m.addConstr(quicksum(m.getVars()[i] for i in range(len(m.getVars()))) <= 1, "c_sum1")
		m.addConstr(quicksum(m.getVars()[i] for i in range(len(m.getVars()))) >= 1, "c_sum2")
		m.update()

		self.v_estimate = self.random_poids()
		set_constraints = []
		while self.v_estimate !=self.v_poids and len(m.getConstrs()) < 100 :
			print("*************** NEW ITER ***************** \n")
			print("estim :", self.v_estimate)
			print(self.v_poids)

			xp, yp = self.argmin_MR(m)
			#print("xp = ",xp.nom, " yp = ",yp.nom)
			
			if self.is_prefered(xp,yp):
				#print("xp > yp \n")
				coeffs = []
				for i in range(len(xp.params)):
					coeffs += [(xp.params[i] - yp.params[i])]
				#print(quicksum(m.getVars()[i] * coeffs[i] for i in range(len(m.getVars()))))
				m.addConstr(quicksum(m.getVars()[i] * coeffs[i] for i in range(len(m.getVars()))) >= 0, "c_"+str(len(m.getConstrs())))
				m.update()
			elif self.is_prefered(yp, xp):
				#print("yp > xp \n")
				coeffs = []
				for i in range(len(xp.params)):
					coeffs += [(xp.params[i] - yp.params[i])]
				#print(quicksum(m.getVars()[i] * coeffs[i] for i in range(len(m.getVars()))))
				m.addConstr(quicksum(m.getVars()[i] * coeffs[i] for i in range(len(m.getVars()))) <= 0, "c_"+str(len(m.getConstrs())))
				m.update()


	def is_prefered(self, x, y):
		sum = 0
		for i in range(len(self.v_poids)):
			sum += self.v_poids[i]*(x.params[i] - y.params[i])

		if sum >= 0 :
			return True
		return False

	def argmin_MR(self, m):
		#print("IN argmin_MR : \n")
		xp = self.set_X[0]
		l_w_best, val_best, y_best = self.calcul_MR(xp, m)
		for x in self.set_X :

			l_w, val, y = self.calcul_MR(x, m)
			#print(" \t x, val, y : ", x.nom, val, y.nom, "\n")
			if val < val_best :
				#print("\t new best ! \n")
				val_best = val
				l_w_best = l_w
				y_best = y
				xp = x
			#print("\t xp = ", xp.nom)
			#print(val_best)

		return xp, y_best

	def calcul_MR(self, x, m):
		val_best= -1000
		l_w_best = []
		y_best = None
		for y in self.set_X :
			#print("\t MR :",y.nom)
			if x.nom != y.nom :
				l_w, val = self.calcul_PMR(x,y,m)
				if val > val_best :
					val_best = val
					l_w_best = l_w
					y_best = y
		#print(l_w_best, val_best, y_best)
		#print("\t end MR")
		return l_w_best, val_best, y_best
"""
	
def random_poids(N):
	l_w = [random.random() for i in range(N)]
	s = sum(l_w)
	for i in range(len(l_w)):
		l_w[i] = l_w[i]/s

	return l_w

#v_poids = [1/len(LISTE_PARAMS) for i in range(len(LISTE_PARAMS))]
v_poids = random_poids(len(LISTE_PARAMS))
test = CSS(liste_voitures, v_poids)

#test.calcul_MR(liste_voitures[0], None)
#print(liste_voitures[0].params)
#print(liste_voitures[4].params)
#print(test.PMR(liste_voitures[0].params, liste_voitures[4].params))
#test.MR(liste_voitures[0])
#print(test.MMR())
print(test.CSS())
"""********************************************************
						Partie 3 (bonus)
***********************************************************"""
P_MAX = 30
V_MAX = 30

class Objet():
	def __init__(self, poids, l_valeurs):
		self.poids = poids
		self.l_valeurs = l_valeurs


class SacADos():
	def __init__(self, p, n):
		self.p = p
		self.n = n
		self.objets = self.create_objets(p,n)
		self.capacity = int(self.somme_poids()/2)

	def somme_poids(self):
		s = 0
		for o in self.objets :
			s += o.poids

		return s

	def create_objets(self, p, n):
		obj = []
		for i in range(p):
			poids = random.randint(1,P_MAX)
			l_valeurs = []
			for i in range(n):
				l_valeurs += [random.randint(1,V_MAX)]

			obj += [Objet(poids, l_valeurs)]

		return obj

sd = SacADos(5,2)
