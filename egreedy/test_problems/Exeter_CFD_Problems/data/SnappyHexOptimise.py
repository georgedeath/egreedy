#! /usr/bin/env python

# imports
import shutil
import os
try:
    from .InitialTEST import InitialTEST#, isCallback
    from .support import RemoveCase, RestoreCase, nostdout, suppress_stdout
except Exception as e:
    print(e)
    from InitialTEST import InitialTEST#, isCallback
    from support import RemoveCase, RestoreCase, nostdout
from PyFoam.Execution.ConvergenceRunner import ConvergenceRunner
from PyFoam.Execution.UtilityRunner import UtilityRunner
from PyFoam.LogAnalysis.BoundingLogAnalyzer import BoundingLogAnalyzer
from PyFoam.Applications.CloneCase import CloneCase
from PyFoam.Applications.Runner import Runner
from PyFoam.Execution.BasicRunner import BasicRunner
from PyFoam.Applications.CopyLastToFirst import CopyLastToFirst
from PyFoam.Applications.ClearCase import ClearCase
from PyFoam.Execution.ParallelExecution import LAMMachine
import subprocess
import numpy as np
from stl import mesh
from stl import stl
import fnmatch
import csv
import os, sys
from contextlib import redirect_stdout
import sys, os, io, pdb
import warnings

# Disable
def blockPrint():
    sys.stdout = open(os.devnull, 'w')

# Restore
def enablePrint():
    sys.stdout = sys.__stdout__

class BasicPitzDailyRun(InitialTEST):

    """
    CFD test problem
    1) Construct the mesh
    2) Run checkMesh on latest mesh
    3) Run steady state case, (no optimisation)
    """

    # class attributes
    # these attributes are likely to be the same accross all instances.
    #Solvers
    solver1="simpleFoam"
    solver2="snappyHexMesh"
    solver3="createPatch"
    solver4="mapFields"
    solver5="surfaceFeatureExtract"

    #utilities
    checkingmesh="checkMesh"

    #CostFunction postprocessing tools
    pCmd="calcPressureDifference"
    mCmd="calcMassFlow"
    stdout = sys.stdout

    def init(self):
        self.setParameters(solver=self.checkingmesh,
                           sizeClass=self.size_class,
                           minimumRunTime=self.min_run_time,
                           casePath=self.case_path)
        self.counter = 1


    def prepare_case(self, source_case, verbose=False):
        if verbose:
            self.__prepare_case(source_case)
        else:
            with nostdout():
                self.__prepare_case(source_case)

    def __prepare_case(self, source_case):
        # remove any previous case directory
        RemoveCase(self.case_path)
        # restore case from source before running for the first time
        RestoreCase(source_case, self.case_path)
        # Run CFD on base case
        self.run()

    def postRunTestCheckConverged(self):
        self.isNotEqual(
            value=self.runInfo()["time"],
            target=self.controlDict()["endTime"],
            message="Reached endTime -> not converged")
        self.shell("cp -r 0 0_orig") #initial fields
        self.shell("pyFoamCopyLastToFirst.py . .")
        self.shell("pyFoamClearCase.py .")
        CloneCase(args=(self.case_path, self.case_path+"pitzDaily_backup")) #Only works with Python3
        self.shell("cp -r 0_orig pitzDaily_backup/") #initial fields
        self.shell("cp -r pitzDaily_backup/constant/polyMesh pitzDaily_backup/constant/polyMesh_backup")

    def SnappyHexMeshrun(self):
        subprocess.call(['rm', '-r', self.case_path+'constant/polyMesh'])
        subprocess.call(['cp', '-r', self.case_path+'pitzDaily_backup/constant/polyMesh_backup', self.case_path+'/constant/polyMesh'])
        subprocess.call(['rm', '-r', self.case_path+'0'])
        subprocess.call(['cp', '-r', self.case_path+'pitzDaily_backup/0_orig', self.case_path+'/0'])
        surface = BasicRunner(argv=[self.solver5,"-case", self.case_path], silent=True)
        surface.start()
        snappy = BasicRunner(argv=[self.solver2,"-overwrite","-case",self.case_path], silent=True)
        snappy.start()
        check = BasicRunner(argv=[self.checkingmesh, "-latestTime","-case", self.case_path], silent=True)
        check.start()
        merge = BasicRunner(argv=[self.solver3, "-overwrite","-case", self.case_path], silent=True) # merge STL with lowerboundary
        merge.start()

    def Optimisationrun(self): #run simpleFoam
        run=ConvergenceRunner(BoundingLogAnalyzer(),argv=[self.solver1,"-case",self.case_path],silent=True)
        run.start()

        subprocess.call(['pyFoamCopyLastToFirst.py',self.case_path, self.case_path], stdout=self.stdout)
        subprocess.call(['pyFoamClearCase.py', self.case_path], stdout=self.stdout)
        subprocess.call(['rm', self.case_path+'0/cellLevel'], stdout=self.stdout)

    def RunUtilities(self, sense='single'):
        # Get the pressure difference (Using an external utility)
        pUtil=UtilityRunner(argv=[self.pCmd,"-case",self.case_path,"-latestTime"],silent=True,logname="Pressure")
        pUtil.add("PressureDifference","Pressure drop = (%f%) between inlet and outlet",idNr=1)
        pUtil.start()
        deltaP=UtilityRunner.get(pUtil,"PressureDifference")[0]

        if sense=="multi":
            # Get the mass flow (Using an external utility)
            mUtil=UtilityRunner(argv=[self.mCmd,"-case",self.case_path,"-latestTime"],silent=True,logname="MassFlow")
            mUtil.add("mass","Flux at outlet = (%f%)",idNr=1)
            mUtil.start()
            massFlow=UtilityRunner.get(mUtil,"mass")[0]

            return float(deltaP), float(massFlow)
        else:
            return float(deltaP)

    def __cost_function(self, sense='single'):
        """
        A method to run CFD for the new shape.

        Kwargs:
            sense (str): whther to return single or multi-objective values.
        """
        self.SnappyHexMeshrun()
        self.Optimisationrun()
        if sense=="single":
            p = self.RunUtilities()
            return p
        elif sense=="multi":
            p, m = self.RunUtilities(sense=sense)
            return p, m
        else:
            print("Invalid input for sense: ", sense)
            print("Available options are: 'single' or 'multi'")
            return None

    def cost_function(self, sense='single', verbose=False):
        """
        A method to run CFD for the new shape.

        Kwargs:
            sense (str): whther to return single or multi-objective values.
        """
        if verbose:
            self.stdout = sys.stdout
            return self.__cost_function(sense=sense)
        else:
            self.stdout = open(os.devnull, 'wb')
            with nostdout():
                res = self.__cost_function(sense=sense)
            return res


class BasicTRun(BasicPitzDailyRun):

    pCmd = 'calcPressureDifferenceDouble'

    def RunUtilities(self, sense='multi'):
        # Get the pressure difference (Using an external utility)
        pUtil=UtilityRunner(argv=[self.pCmd,"-case",self.case_path,"-latestTime"],silent=True,logname="Pressure")
        pUtil.add("PressureDifference","Total pressure drop = (%f%) between Inlet and Outlet1",idNr=1)
        pUtil.add("PressureDifference2","Total pressure drop = (%f%) between Inlet and Outlet2",idNr=1)
        pUtil.start()
        deltaP=UtilityRunner.get(pUtil,"PressureDifference")[0]
        deltaP2=UtilityRunner.get(pUtil,"PressureDifference2")[0]

        if sense=="multi":
            return float(deltaP), float(deltaP2)
        else:
            return max(np.abs(float(deltaP)), np.abs(float(deltaP2))),

    def prepMeshes(self):
        """Callback to prepare the mesh for the case. Default
        behaviour is to run blockMesh on the case"""
        result=self.execute("blockMesh")
        if not result["OK"]:
            self.fatalFail("blockMesh was not able to create a mesh")
        result=self.execute("stitchMesh","-case", self.case_path, "-perfect", "-overwrite", "defaultFaces", "empty_patch1")
        if not result["OK"]:
            self.fatalFail("stitchMesh was not able to create a mesh")

    def Optimisationrun(self): #run simpleFoam
        run=ConvergenceRunner(BoundingLogAnalyzer(),argv=[self.solver1,"-case",self.case_path],silent=True)
        run.start()
        subprocess.call(['pyFoamCopyLastToFirst.py',self.case_path, self.case_path])
        subprocess.call(['pyFoamClearCase.py', self.case_path])


class BasicDuctRun(InitialTEST):

    """
    CFD test problem
    1) Construct the mesh
    2) Run checkMesh on latest mesh
    3) Run steady state case, (no optimisation)
    """

    # class attributes
    # these attributes are likely to be the same accross all instances.
    #Solvers
    solver1="simpleFoam_cp"
    solver2="cartesianMesh"
    solver3="decomposePar"
    solver4="potentialFoam"
    solver5="renumberMesh"
    solver6="reconstructPar"

    #utilities
    checkingmesh="checkMesh"
    machine=LAMMachine(nr=4)
    ncores=4
    #CostFunction postprocessing tools
    pCmd="calcPressureDifference_Kaplan"
    mCmd="calcEnergyLoss"
    stdout = sys.stdout


    def init(self):
        self.setParameters(solver=self.solver1,
                           sizeClass=self.size_class,
                           minimumRunTime=self.min_run_time,
                           casePath=self.case_path)
        self.counter = 1


    def prepare_case(self, source_case, verbose=False):
        if verbose:
            self.__prepare_case(source_case)
        else:
            with nostdout():
                self.__prepare_case(source_case)

    def __prepare_case(self, source_case):
        # remove any previous case directory
        RemoveCase(self.case_path)
        # restore case from source before running for the first time
        RestoreCase(source_case, self.case_path)
        # Run CFD on base case
        self.run()
        self.shell("cp -r 0 0_orig") #initial field

    def postRunTestCheckConverged(self):
        try:
            self.isNotEqual(
                value=self.runInfo()["time"],
                target=self.controlDict()["endTime"],
                message="Reached endTime -> not converged")
        except:
            warnings.warn("Warning: The CFD simulation may not have converged. If this is shown at problem instantiation, then you can safely ignore as no CFD simulation was performed.")
        self.shell("cp -r 0 0_orig") #initial fields

    def SnappyHexMeshrun(self):
        subprocess.call(['rm', '-r', 'constant/polyMesh/*'], cwd=self.case_path)
        subprocess.call(['rm', '-r', '0'], cwd=self.case_path)
        subprocess.call(['cp', '-r', '0_orig', '0'], cwd=self.case_path)
        ###then stl from binary to ascii
        barry = mesh.Mesh.from_file(self.case_path+'constant/triSurface/ribbon.stl')
        barry.save("ribbon.stl", mode=stl.ASCII)
        subprocess.call(['mv', 'ribbon.stl', self.case_path])
        subprocess.call(['surfaceTransformPoints', '-rotate', " ((0 1 0)(0 0 1)) ", 'ribbon.stl', 'ribbon_modified.stl'], stdout=self.stdout, cwd=self.case_path)

        with open(self.case_path+'mergedVolume.stl', 'w') as outfile:
             for infile in (self.case_path+'ribbon_modified.stl', self.case_path+'Hollorfsen_Cervantes_walls_cfmesh.stl'):
                shutil.copyfileobj(open(infile), outfile)

        subprocess.call(['rm', '-r', 'mergedTotal.stl'], cwd=self.case_path)
        subprocess.call(['rm', '-r', 'mergedTotal2.stl'], cwd=self.case_path)
        subprocess.call(['rm', '-r', 'mergedTotal3.stl'], cwd=self.case_path)

        with open(self.case_path+'mergedTotal.stl', 'w') as outfile:
             for infile in (self.case_path+'Hollorfsen_Cervantes_inflow.stl', self.case_path+'RCONE.stl', self.case_path+'outflow_extension.stl',self.case_path+'extended_outflow2.stl', self.case_path+'mergedVolume.stl'):
                shutil.copyfileobj(open(infile), outfile)

        subprocess.call(['surfaceTransformPoints', '-scale', " (0.001 0.001 0.001) ", 'mergedTotal.stl', 'mergedVolume2.stl'], stdout=self.stdout, cwd=self.case_path)

        snappy = BasicRunner(argv=[self.solver2,"-case",self.case_path], silent=False)
        snappy.start()

        decompose = BasicRunner(argv=[self.solver3,"-case",self.case_path], silent=False)
        decompose.start()

    def run_log(self, cmd, cwd, filename):
        logfile = open(cwd+filename, 'w')
        ret_code = subprocess.call(cmd, cwd=cwd, stdout=logfile)
        return ret_code


    def Optimisationrun(self): #run simpleFoam
        #import pdb; pdb.set_trace()
        current = os.getcwd()
        subprocess.call(['mpirun', '-np', str(self.ncores), self.solver4, '-parallel'], cwd=current+'/'+self.case_path, stdout=self.stdout)

        self.run_log(['mpirun', '-np', str(self.ncores) , self.solver1 , '-parallel'], \
                        cwd=current+'/'+self.case_path, \
                        filename='log.txt')

        subprocess.call(['reconstructPar', '-latestTime', '-case', self.case_path], stdout=self.stdout)

    def RunUtilities(self, sense='single'):
        lines = []
        lines2 = []
        if os.path.isdir(self.case_path+"10000"):
            N = 1000
        else:
            N = 1
        subprocess.call(['pyFoamCopyLastToFirst.py',self.case_path, self.case_path])
        subprocess.call(['pyFoamClearCase.py', self.case_path, '--processors-remove', '--keep-postprocessing'])

        # Get the pressure difference (Using an external utility)
        pUtil=UtilityRunner(argv=[self.pCmd,"-case",self.case_path,"-latestTime"],silent=True,logname="Pressure")
        pUtil.add("PressureDifference","Pressure drop = (%f%) between inlet and outlet",idNr=1)
        pUtil.start()
        deltaP=UtilityRunner.get(pUtil,"PressureDifference")[0]

        if sense=="multi":
            # Get the mass flow (Using an external utility)
            mUtil=UtilityRunner(argv=[self.mCmd,"-case",self.case_path,"-latestTime"],silent=True,logname="MassFlow")
            mUtil.add("mass","Flux at outlet = (%f%)",idNr=1)
            mUtil.start()
            massFlow=UtilityRunner.get(mUtil,"mass")[0]

            return -float(deltaP), -float(massFlow)
        else:
            return -float(deltaP)

    def __cost_function(self, sense='single'):
        """
        A method to run CFD for the new shape.

        Kwargs:
            sense (str): whther to return single or multi-objective values.
        """
        self.SnappyHexMeshrun()
        self.Optimisationrun()
        if sense=="single":
            p = self.RunUtilities()
            subprocess.call(['cp', '-r', self.case_path, self.case_path[:-1]+'_'+str(self.counter)+'/'])
            self.counter += 1
            return p
        elif sense=="multi":
            p, m = self.RunUtilities(sense=sense)
            subprocess.call(['cp', '-r', self.case_path, self.case_path[:-1]+'_'+str(self.counter)+'/'])
            self.counter += 1
            return p, m
        else:
            print("Invalid input for sense: ", sense)
            print("Available options are: 'single' or 'multi'")
            return None

    def cost_function(self, sense='single', verbose=False):
        """
        A method to run CFD for the new shape.

        Kwargs:
            sense (str): whther to return single or multi-objective values.
        """
        if verbose:
            self.stdout = sys.stdout
            return self.__cost_function(sense=sense)
        else:
            self.stdout = open(os.devnull, 'wb')
            with nostdout():
                return self.__cost_function(sense=sense)


class BasicHeatExchangerRun(InitialTEST):

    """
    CFD test problem
    1) Construct the mesh
    2) Run checkMesh on latest mesh
    3) Run steady state case, (no optimisation)
    """

    # class attributes
    # these attributes are likely to be the same accross all instances.
    #Solvers
    solver1="heatedFoam"
    solver2="snappyHexMesh"
    solver3="createPatch"
    solver4="mapFields"
    solver5="surfaceFeatureExtract"
    solver6="extrudeMesh"
    #utilities
    checkingmesh="checkMesh"
    solver7="simpleFoam"
    #CostFunction postprocessing tools
    pCmd="calcPressureDifference_heatexchanger"
    tCmd="calcTemperatureDifference"


    def init(self):
        self.setParameters(solver="checkMesh",
                           sizeClass=self.size_class,
                           minimumRunTime=self.min_run_time,
                           casePath=self.case_path)

    def prepare_case(self, source_case, verbose=False):
        if verbose:
            self.__prepare_case(source_case)
        else:
            with nostdout():
                self.__prepare_case(source_case)

    def __prepare_case(self, source_case):
        # remove any previous case directory
        RemoveCase(self.case_path)
        # restore case from source before running for the first time
        RestoreCase(source_case, self.case_path)
        # Run CFD on base case
        self.run()


    def postRunTestCheckConverged(self):
        '''
        self.isNotEqual(
            value=self.runInfo()["time"],
            target=self.controlDict()["endTime"],
            message="Reached endTime -> not converged")
        '''
        self.shell("cp -r 0 0_orig") #initial fields
        #self.shell("pyFoamCopyLastToFirst.py . .")
        #self.shell("pyFoamClearCase.py .")
        CloneCase(args=(self.case_path, self.case_path+"heat_exchange")) #Only works with Python3
        self.shell("cp -r 0_orig heat_exchange/") #initial fields
        self.shell("cp -r heat_exchange/constant/polyMesh heat_exchange/constant/polyMesh_backup")

    def SnappyHexMeshrun(self):
        subprocess.call(['rm', '-r', self.case_path+'constant/polyMesh'])
        subprocess.call(['cp', '-r', self.case_path+'heat_exchange/constant/polyMesh_backup', self.case_path+'/constant/polyMesh'])
        subprocess.call(['rm', '-r', self.case_path+'0'])
        subprocess.call(['cp', '-r', self.case_path+'heat_exchange/0_orig', self.case_path+'0'])
        surface = BasicRunner(argv=[self.solver5,"-case", self.case_path], silent=False)
        surface.start()
        snappy = BasicRunner(argv=[self.solver2,"-overwrite","-case",self.case_path], silent=False)
        snappy.start()
        extrude = BasicRunner(argv=[self.solver6,"-case",self.case_path], silent=False)
        extrude.start()
        check = BasicRunner(argv=[self.checkingmesh, "-latestTime","-case", self.case_path], silent=False)
        check.start()
        #merge = BasicRunner(argv=[self.solver3, "-overwrite","-case", self.case_path], silent=True) # merge STL with lowerboundary
        #merge.start()

    def Optimisationrun(self): #run simpleFoam
        run=ConvergenceRunner(BoundingLogAnalyzer(),argv=[self.solver1,"-case",self.case_path],silent=True)
        run.start()

        subprocess.call(['pyFoamCopyLastToFirst.py',self.case_path, self.case_path])
        subprocess.call(['pyFoamClearCase.py', self.case_path])
        subprocess.call(['rm', self.case_path+'0/cellLevel'])

    def RunUtilities(self, sense='single'):
        # Get the pressure difference (Using an external utility)
        pUtil=UtilityRunner(argv=[self.pCmd,"-case",self.case_path,"-latestTime"],silent=True,logname="Pressure")
        pUtil.add("PressureDifference","Pressure drop = (%f%) between inlet and outlet",idNr=1)
        pUtil.start()
        deltaP=UtilityRunner.get(pUtil,"PressureDifference")[0]

        tUtil=UtilityRunner(argv=[self.tCmd,"-case",self.case_path,"-latestTime"],silent=True,logname="Temperature")
        tUtil.add("TemperatureDifference","Temperature drop = (%f%) between inlet and outlet",idNr=1)
        tUtil.start()
        deltaT=UtilityRunner.get(tUtil,"TemperatureDifference")[0]

        if sense=="multi":

            return float(deltaT), float(deltaP)
        else:
            return float(deltaT)

    def __cost_function(self, sense='single'):
        """
        A method to run CFD for the new shape.

        Kwargs:
            sense (str): whther to return single or multi-objective values.
        """

        self.SnappyHexMeshrun()
        self.Optimisationrun()
        if sense=="single":
            t = self.RunUtilities()
            return p
        elif sense=="multi":
            t, p = self.RunUtilities(sense=sense)
            return t, p
        else:
            print("Invalid input for sense: ", sense)
            print("Available options are: 'single' or 'multi'")
            return None

    def cost_function(self, sense='single', verbose=False):
        """
        A method to run CFD for the new shape.

        Kwargs:
            sense (str): whther to return single or multi-objective values.
        """
        if verbose:
            return self.__cost_function(sense=sense)
        else:
            with nostdout():
                return self.__cost_function(sense=sense)
