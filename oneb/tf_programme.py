pH=3.0
import MDAnalysis as mda
from MDAnalysis.analysis.rms import RMSD
import os
os.environ["MKL_NUM_THREADS"]="1"
os.environ["NUMEXPR_NUM_THREADS"]="1"
os.environ["OMP_NUM_THREADS"]="1"
from openmm.app import *
from openmm import *
from openmm.unit import *
import scipy.spatial
import re
from mergeforce import second_dual_basin_energy
from sys import stdout
from model import *
import random
class sim_CGpH:
    def __init__(self,pH):
        self.sequence = []
        self.pos = []
        self.pH=pH
        self.r = []
        with open('high.gro') as f:
            lines = f.readlines()
        for line in lines:
            if '  CA  ' in line:
                line = line.split()
                self.pos.append(int(line[2]))
                self.sequence.append(re.findall(r'(\D+|\d+)', line[0])[1])
                self.r.append(list(map(float, line[3:])))
        RESI_TYPE = ['ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU', 'GLY', 'HIS', 'ILE', 'LEU', 'LYS', 'MET', 'PHE',
                     'PRO', 'SER', 'THR', 'TRP', 'TYR', 'VAL']
        self.seq = np.zeros([20, len(self.sequence)])
        self.r = np.array(self.r) * 10.0
        self.dismap = scipy.spatial.distance.cdist(self.r, self.r, metric='euclidean')
        for i in range(len(self.sequence)):
            for j in range(len(RESI_TYPE)):
                if self.sequence[i] == RESI_TYPE[j]:
                    self.seq[j][i] = 1.0
        self.lenth = len(self.pos)
        self.pos = np.array(self.pos)
        self.relpos = np.zeros((16, self.lenth, self.lenth))
        for i in range(self.lenth):
            for j in range(self.lenth):
                dij = abs(int(self.pos[i]) - int(self.pos[j]))
                if dij < 16:
                    if dij >= 0:
                        dij = int(dij)
                        self.relpos[dij][i][j] = 1.0
                else:
                    self.relpos[15][i][j] = 1.0
        self.model = CGSolvElec()
        self.model.load_state_dict({k.replace('module.', ''): v for k, v in
                               torch.load('maxcor_model.pkl').items()})
        self.rank = 'cuda'
        self.model.to(self.rank)

    def charge_cal(self,resid):
        disttance = self.dismap[resid]
        index = np.where(disttance < 18)
        l = index[0].shape[0]
        disttance = disttance[index[0]]
        dismapout1 = np.repeat(disttance, l).reshape(l, l)
        seqout = torch.from_numpy(self.seq[:, index[0]][np.newaxis, :]).float()
        dismapout = self.dismap[np.ix_(index[0], index[0])]
        dismapout = torch.from_numpy(np.stack((dismapout1, dismapout1.T, dismapout), axis=-1)[np.newaxis, :]).float()
        relposout = torch.from_numpy(self.relpos[:, index[0][:, None], index[0]][np.newaxis, :]).float()
        resid = np.where(disttance == 0.0)
        res_ids = np.zeros((l, l))
        res_ids[resid[0][0]][resid[0][0]] = 1.0
        res_ids = torch.from_numpy(res_ids[np.newaxis, :]).float()
        addseq = torch.zeros((1, 4))
        with torch.no_grad():
            if seqout[0][3][resid] == 1:
                addseq[0][0] = 1.0
                pred = self.model(seqout.to(self.rank), dismapout.to(self.rank), addseq.to(self.rank), relposout.to(self.rank),
                             res_ids.to(self.rank)).squeeze().detach().cpu().numpy() + 3.67
                return -1.0 / (1.0 + 10 ** (pred - self.pH))
            elif seqout[0][6][resid] == 1:
                addseq[0][1] = 1.0
                pred = self.model(seqout.to(self.rank), dismapout.to(self.rank), addseq.to(self.rank), relposout.to(self.rank),
                             res_ids.to(self.rank)).squeeze().detach().cpu().numpy() + 4.25
                return -1.0 / (1.0 + 10 ** (pred - self.pH))
            elif seqout[0][8][resid] == 1:
                addseq[0][2] = 1.0
                pred = self.model(seqout.to(self.rank), dismapout.to(self.rank), addseq.to(self.rank), relposout.to(self.rank),
                             res_ids.to(self.rank)).squeeze().detach().cpu().numpy() + 6.54
                return 1.0 / (1.0 + 10 ** (self.pH - pred))
            elif seqout[0][11][resid] == 1:
                addseq[0][3] = 1.0
                pred = self.model(seqout.to(self.rank), dismapout.to(self.rank), addseq.to(self.rank), relposout.to(self.rank), res_ids.to(self.rank)).squeeze().detach().cpu().numpy()+10.40
                return 1.0   / (1.0 + 10 ** (self.pH-pred))
            elif seqout[0][1][resid] == 1:
                return 1.0
            else:
                return 0.0

    def add_init_charge(self):
        force = CustomNonbondedForce('0.8682*q1*q2*exp(-0.73555*r)/r')
        force.addPerParticleParameter('q')
        # force.setCutoffDistance(3.0 * nanometers)
        set = []
        set_without_arg = []
        for i in range(218):
            q = self.charge_cal(i)
            if q > 0.0:
                force.addParticle([q])
                if q != 1.0:
                    set_without_arg.append(i)
                set.append(i)
            elif q < 0.0:
                force.addParticle([q])
                set_without_arg.append(i)
                set.append(i)
            else:
                force.addParticle([0.0])
        force.addInteractionGroup(set, set)
        return force, set_without_arg

    def set_new_charge_to_force(self,force ,set, simulation):
        for i in set:
            force.setParticleParameters(i, [self.charge_cal(i)])
        force.updateParametersInContext(simulation.context)
    def simulate_by_temperature(self,totalsteps,temperature):
        system = second_dual_basin_energy('high.top', 'low.top', 'high.xml', 'low.xml')
        dH_force, set = self.add_init_charge()
        system.addForce(dH_force)
        integrator = LangevinIntegrator(temperature * kelvin, 0.1 / picosecond, 0.001 * picoseconds)
        top = GromacsTopFile('high.top')
        platform = Platform.getPlatformByName('CUDA')
        properties = {'Precision': 'single'}
        simulation = Simulation(top.topology, system, integrator, platform, properties)
        simulation.reporters.append(StateDataReporter(stdout, 10 ** 6, step=True, progress=True, remainingTime=True,
                                                      speed=True, totalSteps=totalsteps, separator="\t"))
        gro = GromacsGroFile('low.gro')
        simulation.context.setPositions(gro.positions)
        simulation.reporters.append(XTCReporter("%f/_%f.xtc" % (self.pH,temperature), 2000))
        seed = random.randint(0, 1e6)
        simulation.context.setVelocitiesToTemperature(300 * kelvin, seed)
        for i in range(int(totalsteps/2000)):
            state = simulation.context.getState(getPositions=True)
            r = np.array(state.getPositions(asNumpy=True)) * 10.0
            self.dismap = scipy.spatial.distance.cdist(r, r, metric='euclidean')
            self.set_new_charge_to_force(dH_force, set, simulation)
            simulation.step(2000)
    def simulate(self,totalsteps,name):
        system = second_dual_basin_energy('high.top', 'low.top', 'high.xml', 'low.xml')
        dH_force, set = self.add_init_charge()
        system.addForce(dH_force)
        integrator = LangevinIntegrator(300 * kelvin, 0.01 / picosecond, 0.01 * picoseconds)
        top = GromacsTopFile('high.top')
        platform = Platform.getPlatformByName('CUDA')
        properties = {'Precision': 'single'}
        simulation = Simulation(top.topology, system, integrator, platform, properties)
        simulation.reporters.append(StateDataReporter(stdout, 10 ** 6, step=True, progress=True, remainingTime=True,
                                                      speed=True, totalSteps=totalsteps, separator="\t"))
        gro = GromacsGroFile('high.gro')
        simulation.context.setPositions(gro.positions)
        simulation.reporters.append(XTCReporter("%f/%s.xtc"%(self.pH,name) , 2000))
        seed = random.randint(0, 1e6)
        simulation.context.setVelocitiesToTemperature(300 * kelvin, seed)
        for i in range(int(totalsteps/2000)):
            state = simulation.context.getState(getPositions=True)
            r = np.array(state.getPositions(asNumpy=True)) * 10.0
            self.dismap = scipy.spatial.distance.cdist(r, r, metric='euclidean')
            self.set_new_charge_to_force(dH_force, set, simulation)
            simulation.step(2000)
    def simulate_from_low(self,totalsteps,name):
        system = second_dual_basin_energy('high.top', 'low.top', 'high.xml', 'low.xml')
        dH_force, set = self.add_init_charge()
        system.addForce(dH_force)
        integrator = LangevinIntegrator(300 * kelvin, 0.01 / picosecond, 0.01 * picoseconds)
        top = GromacsTopFile('high.top')
        platform = Platform.getPlatformByName('CUDA')
        properties = {'Precision': 'single'}
        simulation = Simulation(top.topology, system, integrator, platform, properties)
        simulation.reporters.append(StateDataReporter(stdout, 10 ** 6, step=True, progress=True, remainingTime=True,
                                                      speed=True, totalSteps=totalsteps, separator="\t"))
        gro = GromacsGroFile('low.gro')
        simulation.context.setPositions(gro.positions)
        simulation.reporters.append(XTCReporter("%f/%s.xtc"%(self.pH,name) , 2000))
        for i in range(int(totalsteps/2000)):
            state = simulation.context.getState(getPositions=True)
            r = np.array(state.getPositions(asNumpy=True)) * 10.0
            self.dismap = scipy.spatial.distance.cdist(r, r, metric='euclidean')
            self.set_new_charge_to_force(dH_force, set, simulation)
            simulation.step(2000)

sim = sim_CGpH(pH)
sim.simulate(10 ** 9, 'longtraj')