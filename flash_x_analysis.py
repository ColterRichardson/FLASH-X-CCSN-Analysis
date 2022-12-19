import os
import h5py
import ffmpeg
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import FuncAnimation



class CCSN_1D_CHK:
	"""
	Class that collects the data from the 1d CCSN simulations performed in with the FLASH5/X code included in the  *_chk* files.
	Callable functions include:
		directoryprep - Prepares output directory
		siminfo - Prints information contained in the simulations
		parameterprofile1D - Returns the plotting parameters as
			x = distance from origin
			y = parameter
		quickplotdat - Plots the data provided in the .dat file.
		parevolvesingle - Creates a .gif of the time evolution of a single parameter.   
	"""
	def __init__(self, path):
		self.path = path
		self.plot_var = []
		with open(self.path + "/" + "flash.par") as f:
			line = f.readline()
			while line:
				line = f.readline()
				if line.startswith('basenm') == True:
					self.basenm = line.replace(" ", "").split("=",1)[1].replace("\n","").replace("uhd_","uhd").replace("\"","")
				if line.startswith('plot_var') == True:
					line = line.replace(" ", "").split("=",1)[1]
					line = line.replace("\"","").replace("\n","")
					self.plot_var.append(line)
				if line.startswith("checkpointFileIntervalTime"):
					line = line.replace(" ", "").split("=",1)[1].replace("\n","")
					self.dt = float(line)
				if line.startswith("lrefine_max"):
					line = line.replace(" ", "").split("=",1)[1].replace("\n","")
					self.lmax = int(line)
				if line.startswith("lrefine_min"):
					line = line.replace(" ", "").split("=",1)[1].replace("\n","")
					self.lmin = int(line)
				if line.startswith("xmax"):
					line = line.replace(" ", "").split("=",1)[1].replace("\n","").replace(".","")
					n = float(line.split('e',1)[0])
					p = float(line.split('e',1)[1])     
					self.xmax = n * 10**p
				if line.startswith("nblockx"):
					line = line.replace(" ", "").split("=",1)[1].replace("\n","")
					self.nblockx = int(line)
        
		self.hdf5_chk = {}
		isExist = os.path.exists(self.path + "/Data")
		if not isExist:
			data_path = self.path
		else:
			data_path = self.path + "/Data"
		for files in os.listdir(data_path):
			if "hdf5_chk" in files:
				self.hdf5_chk[files.split("chk_",1)[1]] = h5py.File(data_path + "/" + files,"r")


		self.data = np.genfromtxt(data_path + "/" + self.basenm + ".dat",names=True)

	def directoryprep(self):
		"""
		Creates the Image directory.
		"""
		path = self.path

		# Check whether the specified path exists or not
		isExist = os.path.exists(path + "/Images")
		if not isExist:
			# Create a new directory because it does not exist 
			print("Creating directory at " + path + "/Images")
			os.makedirs(path + "/Images")
		else:
			None

		print("Directory has been prepared for output.")

		return None

	def siminfo(self):
		"""
		Prints the information contained in the hdf5 and .dat files.
		"""
		print("The base name of the simulation is", self.basenm)
		print("\n\n")
		print("The variables available in the hdf5_chk (\"0000\" for example) are","\n",list(self.hdf5_chk["0000"].keys()))
		print("\n\n")
		print("The variables available in the .dat are","\n",self.data.dtype.names)
		print("\n\n")
		return None

	def parameterprofile1D(self, Parameter, Number):
		"""
		Returns the plotting parameters.
		Input:
			'Parameter' - see the keys presented in siminfo
			'Number' - plt_cnt or chk check file
		Output:
			x - position 
			y - parameter
		"""
		param = []
		blks = []
		for i in range(len((self.hdf5_chk[Number]["node type"][:]))):
			if self.hdf5_chk[Number]["node type"][i] == 1:
				blks.append(self.hdf5_chk[Number]["block size"][i,0])
				param.append(self.hdf5_chk[Number][Parameter][i,0,0,:])
			else: None

		x = []
		x1=0
		for i in range(len(blks)):
			dx = blks[i] / len(param[i][:])

			for j in range(len(param[0][:])):

				x.append(x1+ dx * j)
			x1 += blks[i]

		y = []
		for i in range(len(param)):
			for j in range(len(param[i][:])):
				y.append(param[i][j])

		return x, y

	def refinementprofile1D(self, Number, Parameter):
		"""
		Returns the refinement levels.
		Input:
			'Number' - plt_cnt or chk check file
			'Parameter' - Test parameter to base the length of the data on.
		Output:
			x - radius
			dx - x-step as a function of radius
			rmin - minimum x-step
			rmax - maximum x-step
		"""

		param = []
		blks = []
		for i in range(len((self.hdf5_chk[Number]["node type"][:]))):
			if self.hdf5_chk[Number]["node type"][i] == 1:
				blks.append(self.hdf5_chk[Number]["block size"][i,0])
				param.append(self.hdf5_chk[Number][Parameter][i,0,0,:])
			else: None

		x = []
		x1=0
		dxs = []
		for i in range(len(blks)):
			dx = blks[i] / len(param[i][:])

			for j in range(len(param[0][:])):
				x.append(x1+ dx * j)

				dxs.append(dx)

			x1 += blks[i]

		dx = self.xmax / self.nblockx / len(self.hdf5_chk["0000"]["entr"][0,0,0,:])

		# print("Minumum Refinement:\t{:.4f}".format( dx / 2**(self.lmin-1)), "cm")
		# print("Maximum Refinement:\t{:.4f}".format( dx / 2**(self.lmax-1)), "cm")


		rmax = dx / 2**(self.lmin-1)
		rmin = dx / 2**(self.lmax-1)

		return x, dxs, rmin, rmax

	def quickplotdat(self):
		"""
		Prints the data saved in the .dat file of the simulations.
		"""

		path = self.path
		name = self.basenm

		# Check whether the specified path exists or not
		isExist = os.path.exists(path + "/Images")
		if not isExist:
			# Create a new directory because it does not exist 
			os.makedirs(path + "/Images")
		else:
			None

		fig, ax = plt.subplots()
		ax.plot(self.data["time"], self.data["mass"], marker = ".")
		ax.set_ylabel(r"$\mathrm{Mass}$ $\mathrm{[?]}$")
		ax.set_xlabel(r"$\mathrm{Time}$ $\mathrm{[s]}$")
		ax.set_title(name + " Mass")
		ax.grid(which = "both")
		ax.minorticks_on()
		ax.grid(visible =True, which='minor', color='#999999', linestyle='-', alpha=0.2)
		plt.tight_layout()
		plt.savefig(path + "/Images/" + name + "_Mass.png", dpi = 200)
		plt.close()

		fig, ax = plt.subplots()
		ax.plot(self.data["time"], self.data["xmomentum"], marker = ".", label = "x")
		ax.plot(self.data["time"], self.data["ymomentum"], marker = ".", label = "y")
		ax.plot(self.data["time"], self.data["zmomentum"], marker = ".", label = "z")
		ax.set_ylabel(r"$\mathrm{Momentum}$ $\mathrm{[?]}$")
		ax.set_xlabel(r"$\mathrm{Time}$ $\mathrm{[s]}$")
		ax.set_title(name + " Momentum")
		ax.legend()
		ax.grid(which = "both")
		ax.minorticks_on()
		ax.grid(visible =True, which='minor', color='#999999', linestyle='-', alpha=0.2)
		plt.tight_layout()
		plt.savefig(path + "/Images/" + name + "_Momentum.png", dpi = 200)
		plt.close()

		fig, ax = plt.subplots()
		ax.plot(self.data["time"], self.data["E_total"], marker = ".", label = "Total")
		ax.plot(self.data["time"], self.data["E_kinetic"], marker = ".", label = "Kinetic")
		ax.plot(self.data["time"], self.data["E_internal"], marker = ".", label = "Internal")
		ax.set_ylabel(r"$\mathrm{Energy}$ $\mathrm{[?]}$")
		ax.set_xlabel(r"$\mathrm{Time}$ $\mathrm{[s]}$")
		ax.set_title(name + " Energy")
		ax.legend()
		ax.grid(which = "both")
		ax.minorticks_on()
		ax.grid(visible =True, which='minor', color='#999999', linestyle='-', alpha=0.2)
		plt.tight_layout()
		plt.savefig(path + "/Images/" + name + "_Energy.png", dpi = 200)
		plt.close()

		print("The .dat data has been ploted.")

		return None

	def parevolvesingle(self, parameter, N0, Nf):
		"""
		Plots the time evolution of the specified parameter from the initial frame to the final frame.

		Input: 
			'parameter' = ['entr', 'dens', 'pres', 'temp', 'ener', 'velx']
			'N0'        = initial frame to plot (integer)
			'Nf'        = final frame to plot (integer)
		"""

		ccsn = FLASH5_plt_Output_1d(self.path)

		if parameter == "entr":
			Parameter = "Entropy"
			ymin = 0
			ymax = 18
		if parameter == "dens":
			Parameter = "Density"
			ymin = 10**5
			ymax = 10**15
		if parameter == "pres":
			Parameter = "Pressure"
			ymin = float(10**22)
			ymax = float(10**35)
		if parameter == "temp":
			Parameter = "Temperaure"
			ymin = 10**9
			ymax = 10**15
		if parameter == "ener":
			Parameter = "Energy"
			ymin = float(10**17)
			ymax = float(10**20)
		if parameter == "velx":
			Parameter = "Velocity"
			ymin = -0.2*float(10**10)
			ymax = 0.2*float(10**10)
		if parameter == "ye  ":
			Parameter = "Electron-Fraction"
			ymin = 0.4
			ymax = 0.525
		if parameter == "gpot":
			Parameter = "Gravitational-Potential"
			ymin = float(-20 * 10**18)
			ymax = 0
		else:
			None


		x, y = ccsn.parameterprofile1D(parameter, str(N0).zfill(4))

		fig, ax = plt.subplots()

		line, = ax.plot(x,y)

		if parameter == "dens" or parameter == "pres" or parameter == "temp" or parameter == "ener":
			ax.set_yscale('log')
		# elif parameter == "velx":
		#     ax.set_yscale('symlog')
		else:
			ax.set_yscale('linear')

		ax.set_xscale('log')

		ax.set_xlabel(r"$\mathrm{Radius}$ $\mathrm{[cm]}$")
		ax.set_ylabel(r"$\mathrm{" + Parameter + "}$")
		ax.set_title(self.path + " " + Parameter)
		ax.grid(which = "both")
		ax.minorticks_on()
		ax.grid(visible =True, which='minor', color='#999999', linestyle='-', alpha=0.2) 

		ax.set_ylim(ymin, ymax)

		text = ax.annotate("{:.4f}".format(N0*ccsn.dt) + "[s]", (0.87,1.01), xycoords = "axes fraction")

		def animate(i):
			x, y = ccsn.parameterprofile1D(parameter, str(N0+i).zfill(4))
			line.set_xdata(x)
			line.set_ydata(y)

			text.set_text("{:.4f}".format(i*ccsn.dt) + "[s]")
			# ax.set_ylim(np.min(y),np.max(y))

			return [line]

		plt.tight_layout()

		Writer = animation.writers["ffmpeg"]
		writer = Writer(fps=20, metadata=dict(artist='Me'), bitrate=1800)

		anim = FuncAnimation(fig, animate, Nf, blit=False)
		anim.save(self.path + "/Images/" + Parameter + "_evolution_" + str(N0).zfill(4) + "_" + str(Nf).zfill(4) +".mp4", writer = writer, dpi = 300)

		plt.close()
		return None

	def par_quick_plot_single(self, parameter, N0):
    
    
		ccsn = FLASH5_plt_Output_1d(self.path)

		if parameter == "entr":
			Parameter = "Entropy"
			ymin = 0
			ymax = 18
		if parameter == "dens":
			Parameter = "Density"
			ymin = 10**5
			ymax = 10**15
		if parameter == "pres":
			Parameter = "Pressure"
			ymin = float(10**22)
			ymax = float(10**35)
		if parameter == "temp":
			Parameter = "Temperaure"
			ymin = 10**9
			ymax = 10**15
		if parameter == "ener":
			Parameter = "Energy"
			ymin = float(10**15)
			ymax = float(10**20)
		if parameter == "velx":
			Parameter = "Velocity"
			ymin = -0.2*float(10**10)
			ymax = 0.2*float(10**10)
		if parameter == "ye  ":
			Parameter = "Electron Fraction"
			ymin = 0.4
			ymax = 0.525
		if parameter == "gpot":
			Parameter = "Gravitational Potential"
		else:
			None


		x, y, dx = ccsn.parameterprofile1D(parameter, str(N0).zfill(4))

		fig, ax = plt.subplots()

		ax.plot(x,y)


		if parameter == "dens" or parameter == "pres" or parameter == "temp":
			ax.set_yscale('log')
		# elif parameter == "velx":
		#     ax.set_yscale('symlog')
		else:
			ax.set_yscale('linear')


		ax.set_xscale('log')

		ax.set_ylim(ymin, ymax)

		ax.set_xlabel(r"$\mathrm{Radius}$ $\mathrm{[cm]}$")
		ax.set_ylabel(r"$\mathrm{" + Parameter + "}$")
		ax.set_title(self.path + " " + Parameter)
		ax.grid(which = "both")
		ax.minorticks_on()
		ax.grid(visible =True, which='minor', color='#999999', linestyle='-', alpha=0.2)

		ax.annotate("{:.4f}".format(N0*ccsn.dt) + "[s]", (0.87,1.01), xycoords = "axes fraction")

		plt.tight_layout()

		plt.savefig(self.path + "/Images/" + Parameter + "_" + str(N0).zfill(4) +".png", dpi = 300)

		plt.close()
		return None

	def parameter_prep(self, parameter, N0, Nf):
		f = open(self.path + "/" + parameter + ".txt", "w+")
		r = open(self.path + "/" + "radius.txt", "w+")

		for i in range(Nf - N0):
			x, y = self.parameterprofile1D(parameter, str(N0 + i).zfill(4))

			f.write(str(y) + "\n")
			r.write(str(x) + "\n")

		f.close()
		r.close()

		
		
class CCSN_1D_PLT:
	"""
	Class that collects the data from the 1d CCSN simulations performed in with the FLASH5/X code included in the *_plt* files.
	Callable functions include:
		directoryprep - Prepares output directory
		siminfo - Prints information contained in the simulations
		parameterprofile1D - Returns the plotting parameters as
			x = distance from origin
			y = parameter
		quickplotdat - Plots the data provided in the .dat file.
		parevolvesingle - Creates a .gif of the time evolution of a single parameter.   
	"""
	def __init__(self, path):
		self.path = path
		self.plot_var = []
		with open(self.path + "/" + "flash.par") as f:
			line = f.readline()
			while line:
				line = f.readline()
				if line.startswith('basenm') == True:
					self.basenm = line.replace(" ", "").split("=",1)[1].replace("\n","").replace("_","").replace("\"","")
				if line.startswith('plot_var') == True:
					line = line.replace(" ", "").split("=",1)[1]
					line = line.replace("\"","").replace("\n","")
					self.plot_var.append(line)
				if line.startswith("plotFileIntervalTime"):
					line = line.replace(" ", "").split("=",1)[1].replace("\n","")
					self.dt = float(line)
				if line.startswith("lrefine_max"):
					line = line.replace(" ", "").split("=",1)[1].replace("\n","")
					self.lmax = int(line)
				if line.startswith("lrefine_min"):
					line = line.replace(" ", "").split("=",1)[1].replace("\n","")
					self.lmin = int(line)
				if line.startswith("xmax"):
					line = line.replace(" ", "").split("=",1)[1].replace("\n","").replace(".","")
					n = float(line.split('e',1)[0])
					p = float(line.split('e',1)[1])     
					self.xmax = n * 10**p
				if line.startswith("nblockx"):
					line = line.replace(" ", "").split("=",1)[1].replace("\n","")
					self.nblockx = int(line)
        
		self.hdf5_plt = {}
		isExist = os.path.exists(self.path + "/Data")
		if not isExist:
			data_path = self.path
		else:
			data_path = self.path + "/Data"
		for files in os.listdir(data_path):
			if "hdf5_plt" in files:
				self.hdf5_plt[files.split("plt_cnt_",1)[1]] = h5py.File(data_path + "/" + files,"r")


		#self.data = np.genfromtxt(data_path + "/" + self.basenm + ".dat",names=True)
	def directoryprep(self):
		"""
		Creates the Image directory.
		"""
		path = self.path

		# Check whether the specified path exists or not
		isExist = os.path.exists(path + "/Images")
		if not isExist:
			# Create a new directory because it does not exist 
			print("Creating directory at " + path + "/Images")
			os.makedirs(path + "/Images")
		else:
			None

		print("Directory has been prepared for output.")

		return None

	def siminfo(self):
		"""
		Prints the information contained in the hdf5 and .dat files.
		"""
		print("The base name of the simulation is", self.basenm)
		print("\n\n")
		print("The variables available in the hdf5_plt (\"0000\" for example) are","\n",list(self.hdf5_plt["0000"].keys()))
		print("\n\n")
		#print("The variables available in the .dat are","\n",self.data.dtype.names)
		print("\n\n")
		return None

	def parameterprofile1D(self, Parameter, Number):
		"""
		Returns the plotting parameters.
		Input:
			'Parameter' - see the keys presented in siminfo
			'Number' - plt_cnt or chk check file
		Output:
			x - position 
			y - parameter
		"""

		param = []
		blks = []

		for i in range(len((self.hdf5_plt[Number]["node type"][:]))):
			if self.hdf5_plt[Number]["node type"][i] == 1:
				blks.append(self.hdf5_plt[Number]["block size"][i,0])
				param.append(self.hdf5_plt[Number][Parameter][i,0,0,:])
			else: None

		x = []
		#print( blks[0] / ( 2.0 * len(param[0][:] ) ) )
		x1= blks[0] / ( 2.0 * len(param[0][:]) )
		for i in range(len(blks)):
			dx = blks[i] / len(param[i][:])

			for j in range(len(param[0][:])):

				x.append(x1+ dx * j)
			x1 += blks[i]

		y = []
		for i in range(len(param)):
			for j in range(len(param[i][:])):
				y.append(param[i][j])

		return x, y

	def refinementprofile1D(self, Number, Parameter):
		"""
		Returns the refinement levels.
		Input:
			'Number' - plt_cnt or chk check file
			'Parameter' - Test parameter to base the length of the data on.
		Output:
			x - radius
			dx - x-step as a function of radius
			rmin - minimum x-step
			rmax - maximum x-step
		"""

		param = []
		blks = []
		for i in range(len((self.hdf5_plt[Number]["node type"][:]))):
			if self.hdf5_plt[Number]["node type"][i] == 1:
				blks.append(self.hdf5_plt[Number]["block size"][i,0])
				param.append(self.hdf5_plt[Number][Parameter][i,0,0,:])
			else: None

		x = []
		x1=0
		dxs = []
		for i in range(len(blks)):
			dx = blks[i] / len(param[i][:])

			for j in range(len(param[0][:])):
				x.append(x1+ dx * j)

				dxs.append(dx)

			x1 += blks[i]

		dx = self.xmax / self.nblockx / len(self.hdf5_plt["0000"]["entr"][0,0,0,:])

		# print("Minumum Refinement:\t{:.4f}".format( dx / 2**(self.lmin-1)), "cm")
		# print("Maximum Refinement:\t{:.4f}".format( dx / 2**(self.lmax-1)), "cm")


		rmax = dx / 2**(self.lmin-1)
		rmin = dx / 2**(self.lmax-1)

		return x, dxs, rmin, rmax

	def quickplotdat(self):
		"""
		Prints the data saved in the .dat file of the simulations.
		"""

		path = self.path
		name = self.basenm

		# Check whether the specified path exists or not
		isExist = os.path.exists(path + "/Images")
		if not isExist:
			# Create a new directory because it does not exist 
			os.makedirs(path + "/Images")
		else:
			None

		fig, ax = plt.subplots()
		ax.plot(self.data["time"], self.data["mass"], marker = ".")
		ax.set_ylabel(r"$\mathrm{Mass}$ $\mathrm{[?]}$")
		ax.set_xlabel(r"$\mathrm{Time}$ $\mathrm{[s]}$")
		ax.set_title(name + " Mass")
		ax.grid(which = "both")
		ax.minorticks_on()
		ax.grid(visible =True, which='minor', color='#999999', linestyle='-', alpha=0.2)
		plt.tight_layout()
		plt.savefig(path + "/Images/" + name + "_Mass.png", dpi = 200)
		plt.close()

		fig, ax = plt.subplots()
		ax.plot(self.data["time"], self.data["xmomentum"], marker = ".", label = "x")
		ax.plot(self.data["time"], self.data["ymomentum"], marker = ".", label = "y")
		ax.plot(self.data["time"], self.data["zmomentum"], marker = ".", label = "z")
		ax.set_ylabel(r"$\mathrm{Momentum}$ $\mathrm{[?]}$")
		ax.set_xlabel(r"$\mathrm{Time}$ $\mathrm{[s]}$")
		ax.set_title(name + " Momentum")
		ax.legend()
		ax.grid(which = "both")
		ax.minorticks_on()
		ax.grid(visible =True, which='minor', color='#999999', linestyle='-', alpha=0.2)
		plt.tight_layout()
		plt.savefig(path + "/Images/" + name + "_Momentum.png", dpi = 200)
		plt.close()

		fig, ax = plt.subplots()
		ax.plot(self.data["time"], self.data["E_total"], marker = ".", label = "Total")
		ax.plot(self.data["time"], self.data["E_kinetic"], marker = ".", label = "Kinetic")
		ax.plot(self.data["time"], self.data["E_internal"], marker = ".", label = "Internal")
		ax.set_ylabel(r"$\mathrm{Energy}$ $\mathrm{[?]}$")
		ax.set_xlabel(r"$\mathrm{Time}$ $\mathrm{[s]}$")
		ax.set_title(name + " Energy")
		ax.legend()
		ax.grid(which = "both")
		ax.minorticks_on()
		ax.grid(visible =True, which='minor', color='#999999', linestyle='-', alpha=0.2)
		plt.tight_layout()
		plt.savefig(path + "/Images/" + name + "_Energy.png", dpi = 200)
		plt.close()

		print("The .dat data has been ploted.")

		return None

	def parevolvesingle(self, parameter, N0, Nf):
		"""
		Plots the time evolution of the specified parameter from the initial frame to the final frame.

		Input: 
			'parameter' = ['entr', 'dens', 'pres', 'temp', 'ener', 'velx']
			'N0'        = initial frame to plot (integer)
			'Nf'        = final frame to plot (integer)
		"""

		ccsn = FLASH5_plt_Output_1d(self.path)

		if parameter == "entr":
			Parameter = "Entropy"
			ymin = 0
			ymax = 18
		if parameter == "dens":
			Parameter = "Density"
			ymin = 10**5
			ymax = 10**15
		if parameter == "pres":
			Parameter = "Pressure"
			ymin = float(10**22)
			ymax = float(10**35)
		if parameter == "temp":
			Parameter = "Temperaure"
			ymin = 10**9
			ymax = 10**15
		if parameter == "ener":
			Parameter = "Energy"
			ymin = float(10**17)
			ymax = float(10**20)
		if parameter == "velx":
			Parameter = "Velocity"
			ymin = -0.2*float(10**10)
			ymax = 0.2*float(10**10)
		if parameter == "ye  ":
			Parameter = "Electron-Fraction"
			ymin = 0.4
			ymax = 0.525
		if parameter == "gpot":
			Parameter = "Gravitational-Potential"
			ymin = float(-20 * 10**18)
			ymax = 0
		else:
			None

		x, y = ccsn.parameterprofile1D(parameter, str(N0).zfill(4))

		fig, ax = plt.subplots()

		line, = ax.plot(x,y)

		if parameter == "dens" or parameter == "pres" or parameter == "temp" or parameter == "ener":
			ax.set_yscale('log')
		elif parameter == "velx":
			ax.set_yscale('symlog')
		else:
			ax.set_yscale('linear')

		ax.set_xscale('log')

		ax.set_xlabel(r"$\mathrm{Radius}$ $\mathrm{[cm]}$")
		ax.set_ylabel(r"$\mathrm{" + Parameter + "}$")
		ax.set_title(self.path + " " + Parameter)
		ax.grid(which = "both")
		ax.minorticks_on()
		ax.grid(visible =True, which='minor', color='#999999', linestyle='-', alpha=0.2) 

		ax.set_ylim(ymin, ymax)

		text = ax.annotate("{:.4f}".format(N0*ccsn.dt) + "[s]", (0.87,1.01), xycoords = "axes fraction")

		def animate(i):
			x, y = ccsn.parameterprofile1D(parameter, str(N0+i).zfill(4))
			line.set_xdata(x)
			line.set_ydata(y)

			text.set_text("{:.4f}".format(i*ccsn.dt) + "[s]")
			# ax.set_ylim(np.min(y),np.max(y))

			return [line]

		plt.tight_layout()

		Writer = animation.writers["ffmpeg"]
		writer = Writer(fps=20, metadata=dict(artist='Me'), bitrate=1800)

		anim = FuncAnimation(fig, animate, Nf, blit=False)
		anim.save(self.path + "/Images/" + Parameter + "_evolution_" + str(N0).zfill(4) + "_" + str(Nf).zfill(4) +".mp4", writer = writer, dpi = 300)

		plt.close()
		return None

	def par_quick_plot_single(self, parameter, N0):


		ccsn = FLASH5_plt_Output_1d(self.path)

		if parameter == "entr":
			Parameter = "Entropy"
			ymin = 0
			ymax = 18
		if parameter == "dens":
			Parameter = "Density"
			ymin = 10**5
			ymax = 10**15
		if parameter == "pres":
			Parameter = "Pressure"
			ymin = float(10**22)
			ymax = float(10**35)
		if parameter == "temp":
			Parameter = "Temperaure"
			ymin = 10**9
			ymax = 10**15
		if parameter == "ener":
			Parameter = "Energy"
			ymin = float(10**15)
			ymax = float(10**20)
		if parameter == "velx":
			Parameter = "Velocity"
			ymin = -0.2*float(10**10)
			ymax = 0.2*float(10**10)
		if parameter == "ye  ":
			Parameter = "Electron Fraction"
			ymin = 0.4
			ymax = 0.525
		if parameter == "gpot":
			Parameter = "Gravitational Potential"
		else:
			None


		x, y, dx = ccsn.parameterprofile1D(parameter, str(N0).zfill(4))

		fig, ax = plt.subplots()

		ax.plot(x,y)


		if parameter == "dens" or parameter == "pres" or parameter == "temp":
			ax.set_yscale('log')
		# elif parameter == "velx":
		#     ax.set_yscale('symlog')
		else:
			ax.set_yscale('linear')


		ax.set_xscale('log')

		ax.set_ylim(ymin, ymax)

		ax.set_xlabel(r"$\mathrm{Radius}$ $\mathrm{[cm]}$")
		ax.set_ylabel(r"$\mathrm{" + Parameter + "}$")
		ax.set_title(self.path + " " + Parameter)
		ax.grid(which = "both")
		ax.minorticks_on()
		ax.grid(visible =True, which='minor', color='#999999', linestyle='-', alpha=0.2)

		ax.annotate("{:.4f}".format(N0*ccsn.dt) + "[s]", (0.87,1.01), xycoords = "axes fraction")

		plt.tight_layout()

		plt.savefig(self.path + "/Images/" + Parameter + "_" + str(N0).zfill(4) +".png", dpi = 300)

		plt.close()
		return None

	def parameter_prep(self, parameter, N0, Nf):
		f = open(self.path + "/" + parameter + ".txt", "w+")
		r = open(self.path + "/" + "radius.txt", "w+")

		for i in range(Nf - N0):
			x, y = self.parameterprofile1D(parameter, str(N0 + i).zfill(4))

			f.write(str(y) + "\n")
			r.write(str(x) + "\n")

		f.close()
		r.close()
