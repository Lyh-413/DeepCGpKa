from openmm.app import *
from openmm import *
from openmm.unit import *
from sys import stdout
import numpy as np
def get_single_basin_force(top,xml):
    high_bonds=CustomBondForce("k*(r-r0)^2")
    high_bonds.addGlobalParameter('k',10000.0)
    high_bonds.addPerBondParameter('r0')
    with open(top,'r') as f:
        top_high=f.readlines()
    for i in range(len(top_high)):
        if '[ atoms ]' in top_high[i]:
            n=i+2
        if '[ bonds ]' in top_high[i]:
            n=i-1-n
            t_b=i+2
        if '[ angles ]' in top_high[i]:
            t_a=i+2
        if '[ dihedrals ]' in top_high[i]:
            t_d=i+2
        if '[ exclusions ]' in top_high[i]:
            t_e=i+2
    while(True):
        try:
            bond_information=top_high[t_b].split()
            id1,id2,r0=int(bond_information[0]),int(bond_information[1]),float(bond_information[3])
            high_bonds.addBond(id1-1,id2-1,[r0])
            t_b=t_b+1
        except:
            break
    high_angles=CustomAngleForce("ktheta*(theta-theta0)^2")
    high_angles.addGlobalParameter('ktheta',20.0)
    high_angles.addPerAngleParameter('theta0')
    while(True):
        try:
            angles_information=top_high[t_a].split()
            id1,id2,id3,r0=int(angles_information[0]),int(angles_information[1]),int(angles_information[2]),float(angles_information[4])*np.pi/180.0
            high_angles.addAngle(id1-1,id2-1,id3-1,[r0])
            t_a=t_a+1
        except:
            break
    high_Torsion=CustomTorsionForce("1.5+cos(theta-theta0)+0.5*cos(3*(theta-theta0))")
    high_Torsion.addPerTorsionParameter('theta0')
    while(True):
        try:
            dihedrals_information=top_high[t_d].split()
            id1,id2,id3,id4,r0=int(dihedrals_information[0]),int(dihedrals_information[1]),int(dihedrals_information[2]),int(dihedrals_information[3]),float(dihedrals_information[5])*np.pi/180.0
            high_Torsion.addTorsion(id1-1,id2-1,id3-1,id4-1,[r0])
            t_d=t_d+2
        except:
            break
    high_exclusions=CustomNonbondedForce("sigma/(r^12)")
    high_exclusions.addGlobalParameter('sigma',0.0000167772)
    high_exclusions.setCutoffDistance(1.5 * nanometers)
    for i in range(n):
        high_exclusions.addParticle()
    while(True):
        try:
            exclude_information=top_high[t_e].split()
            id1,id2=int(exclude_information[0]),int(exclude_information[1])
            high_exclusions.addExclusion(id1-1,id2-1)
            t_e=t_e+1
        except:
            break
    high_contacts=CustomBondForce("(1+a/(r^12))*(1-exp(-(r-r0)^2/(2*sigmaG^2)))")
    high_contacts.addGlobalParameter('a',0.0000167772)
    high_contacts.addGlobalParameter('sigmaG',0.5)
    high_contacts.addPerBondParameter('r0')
    with open(xml,'r') as g:
        xml_high=g.readlines()
    for line in xml_high:
        if '<interaction i' in line:
            index_maohhao=[]
            for i in range(len(line)):
                if line[i]=='"':
                    index_maohhao.append(i)
            idi=int(line[index_maohhao[0]+1:index_maohhao[1]])
            idj=int(line[index_maohhao[2]+1:index_maohhao[3]])
            high_contacts.addBond(idi-1,idj-1,[float(line[index_maohhao[6]+1:index_maohhao[7]])])
    basin_force=CustomCVForce('eb+ea+ed+ex+ec')
    basin_force.addCollectiveVariable('eb',high_bonds)
    basin_force.addCollectiveVariable('ea',high_angles)
    basin_force.addCollectiveVariable('ed',high_Torsion)
    basin_force.addCollectiveVariable('ex',high_exclusions)
    basin_force.addCollectiveVariable('ec',high_contacts)
    return basin_force
def merge_dual_basin_energy(energy1,energy2,delta_V,coupling_constant ):
    dual_basin_force=CustomCVForce('(0.5*(e1+e2+d_v)-(coup^2+0.25*(e1-e2-d_v)^2)^0.5)*4.184')
    dual_basin_force.addGlobalParameter('d_v',delta_V)
    dual_basin_force.addGlobalParameter('coup',coupling_constant)
    dual_basin_force.addCollectiveVariable('e1',energy1)
    dual_basin_force.addCollectiveVariable('e2',energy2)
    return dual_basin_force
def second_dual_basin_energy(hightop,lowtop,highxml,lowxml):
    with open(hightop,'r') as f:
        top_high=f.readlines()
    with open(lowtop,'r') as f:
        top_low=f.readlines()
    for i in range(len(top_high)):
        if '[ atoms ]' in top_high[i]:
            n=i+2
        if '[ bonds ]' in top_high[i]:
            n=i-1-n
            t_b=i+2
        if '[ angles ]' in top_high[i]:
            t_a=i+2
        if '[ dihedrals ]' in top_high[i]:
            t_d=i+2
        if '[ exclusions ]' in top_high[i]:
            t_e=i+2
    high_bonds=CustomBondForce("k*(r-r0)^2")
    high_bonds.addGlobalParameter('k',10000.0*4.184)
    high_bonds.addPerBondParameter('r0')
    print('add bonds')
    while(True):
        try:
            bond_information=top_high[t_b].split()
            id1,id2,r0=int(bond_information[0]),int(bond_information[1]),float(bond_information[3])
            r1=float(top_low[t_b].split()[3])
            high_bonds.addBond(id1-1,id2-1,[(r0+r1)/2])
            t_b=t_b+1
        except:
            break
    high_angles = CustomAngleForce("-4.184*log(exp(-ktheta*(theta-theta0)^2)+exp(-ktheta*(theta-theta1)^2))/3")
    high_angles.addGlobalParameter('ktheta', 20.0 * 3)
    high_angles.addPerAngleParameter('theta0')
    high_angles.addPerAngleParameter('theta1')
    print('add angles')
    while (True):
        try:
            angles_information = top_high[t_a].split()
            id1, id2, id3, r0 = int(angles_information[0]), int(angles_information[1]), int(
                angles_information[2]), float(angles_information[4]) * np.pi / 180.0
            r1 = float(top_low[t_a].split()[4]) * np.pi / 180.0
            high_angles.addAngle(id1 - 1, id2 - 1, id3 - 1, [r0, r1])
            t_a = t_a + 1
        except:
            break
    high_Torsion = CustomTorsionForce(
        "-4.184*log(exp(-3*(1.5+cos(theta-theta0)+0.5*cos(3*(theta-theta0))))+exp(-3*(1.5+cos(theta-theta1)+0.5*cos(3*(theta-theta1)))))/3")
    high_Torsion.addPerTorsionParameter('theta0')
    high_Torsion.addPerTorsionParameter('theta1')
    print('add Torsion')
    while(True):
        try:
            dihedrals_information=top_high[t_d].split()
            id1,id2,id3,id4,r0=int(dihedrals_information[0]),int(dihedrals_information[1]),int(dihedrals_information[2]),int(dihedrals_information[3]),float(dihedrals_information[5])*np.pi/180.0
            r1=float(top_low[t_d].split()[5])*np.pi/180.0
            high_Torsion.addTorsion(id1-1,id2-1,id3-1,id4-1,[r0,r1])
            t_d=t_d+2
        except:
            break
    high_exclusions=CustomNonbondedForce("4.184*sigma/(r^12)")
    high_exclusions.addGlobalParameter('sigma',0.0000167772)
    high_exclusions.setCutoffDistance(2.1 * nanometers)
    print('add exclusion')
    for i in range(n):
        high_exclusions.addParticle()
    check_information=top_high[t_e:]
    t_e2=t_e
    while(True):
        try:
            exclude_information=top_high[t_e].split()
            id1,id2=int(exclude_information[0]),int(exclude_information[1])
            high_exclusions.addExclusion(id1-1,id2-1)
            t_e=t_e+1
        except:
            try:
                exclude_information = top_low[t_e2]
                if exclude_information not in check_information:
                    exclude_information=exclude_information.split()
                    id1,id2=int(exclude_information[0]),int(exclude_information[1])
                    high_exclusions.addExclusion(id1-1,id2-1)
                t_e2=t_e2+1
            except:
                break
    print('add contacts')
    high_contacts=CustomBondForce("0.1*(1+10*a/(r^12))*(1-exp(-(r-r0)^2/(2*sigmaG2^2)))*4.184")
    high_contacts.addGlobalParameter('a',0.0000167772)
    high_contacts.addGlobalParameter('sigmaG2',0.2)
    high_contacts.addPerBondParameter('r0')
    high_contacts_double=CustomBondForce("(1+a/(r^12))*(1-exp(-(r-r0)^2/(2*sigmaG^2)))*(1-exp(-(r-r1)^2/(2*sigmaG^2)))*4.184")
    high_contacts_double.addGlobalParameter('a',0.0000167772)
    high_contacts_double.addGlobalParameter('sigmaG',0.05)
    high_contacts_double.addPerBondParameter('r0')
    high_contacts_double.addPerBondParameter('r1')
    low_contacts=CustomBondForce("0.6*(1+a/(0.6*r^12))*(1-exp(-(r-r0)^2/(2*sigmaG2^2)))*4.184")
    low_contacts.addGlobalParameter('a',0.0000167772)
    low_contacts.addGlobalParameter('sigmaG2',0.2)
    low_contacts.addPerBondParameter('r0')
    with open(highxml,'r') as g:
        xml_high=g.readlines()
    with open(lowxml,'r') as g:
        xml_low=g.readlines()
    check_information=[xml_low[i][:30] for i in range(15,len(xml_low))]
    for line in xml_high:
        if '<interaction i' in line:
            if line[:30] not in check_information:
                index_maohhao=[]
                for i in range(len(line)):
                    if line[i]=='"':
                        index_maohhao.append(i)
                idi=int(line[index_maohhao[0]+1:index_maohhao[1]])
                idj=int(line[index_maohhao[2]+1:index_maohhao[3]])
                high_contacts.addBond(idi-1,idj-1,[float(line[index_maohhao[6]+1:index_maohhao[7]])])
            else:
                index_maohhao=[]
                for i in range(len(line)):
                    if line[i]=='"':
                        index_maohhao.append(i)
                idi=int(line[index_maohhao[0]+1:index_maohhao[1]])
                idj=int(line[index_maohhao[2]+1:index_maohhao[3]])
                for low_line in xml_low:
                    if line[:30]==low_line[:30]:
                        index_maohhao_low = []
                        for i in range(len(low_line)):
                            if low_line[i] == '"':
                                index_maohhao_low.append(i)
                        high_contacts_double.addBond(idi-1,idj-1,[float(line[index_maohhao[6]+1:index_maohhao[7]]),float(low_line[index_maohhao_low[6]+1:index_maohhao_low[7]])])
                        xml_low.remove(low_line)
                        #print('add double_gaussain %d %d'%(idi,idj))
                        break
    for line in xml_low:
        if '<interaction i' in line:
                index_maohhao=[]
                for i in range(len(line)):
                    if line[i]=='"':
                        index_maohhao.append(i)
                idi=int(line[index_maohhao[0]+1:index_maohhao[1]])
                idj=int(line[index_maohhao[2]+1:index_maohhao[3]])
                low_contacts.addBond(idi-1,idj-1,[float(line[index_maohhao[6]+1:index_maohhao[7]])])
    system = System()
    for i in range(218):
        system.addParticle(100.0)
    system.addForce(high_contacts_double)
    system.addForce(high_contacts)
    system.addForce(high_exclusions)
    system.addForce(high_Torsion)
    system.addForce(high_angles)
    system.addForce(high_bonds)
    system.addForce(low_contacts)
    return system
'''
system = System()
for i in range(205):
    system.addParticle(100.0)
system.addForce(merge_dual_basin_energy(get_single_basin_force('high.top','high.xml'),get_single_basin_force('low.top','low.xml'),0.0,20.5))
'''
with open('all.xml', 'w') as f:
    f.write(XmlSerializer.serialize(second_dual_basin_energy('high.top','low.top','high.xml','low.xml')))