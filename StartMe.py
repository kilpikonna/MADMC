import numpy as np 
from math import inf
import matplotlib.pyplot as plt

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
print(ideal)
print(nadir)

normalize_params(liste_voitures, ideal, nadir)
for v in liste_voitures :
	print(v.params)

plot_normalized_solution(liste_voitures[1])
plot_param_distribution(liste_voitures, 4)
plot_more_solutions(liste_voitures[:4])
"""********************************************************
							Partie 2
***********************************************************"""

"""********************************************************
						Partie 3 (bonus)
***********************************************************"""